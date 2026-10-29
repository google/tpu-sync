# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Cross-node device-to-device KV-cache read benchmark, via PyTorch.

Receiver-initiated pull of a KV cache from one TPU node's HBM into another's,
measured end to end across D2H, H2H over the NIC, and H2D, and verified
byte-for-byte.

Start the SENDER first; it arms the cache and blocks until the receiver is done:

  cd <tpu-raiden>
  PYTHONUNBUFFERED=1 PYTHONPATH=$PWD python3 \
    examples/microbenchmarks/torch_d2d_read_benchmark_runner.py \
      --role=sender \
      --grpc_port=50051 \
      --parallelism=4 \
      --num_blocks=512 \
      --num_layers=8 \
      --block_size=16

Then the RECEIVER, pointed at the sender's VPC address (not localhost):

  cd <tpu-raiden>
  PYTHONUNBUFFERED=1 PYTHONPATH=$PWD python3 \
    examples/microbenchmarks/torch_d2d_read_benchmark_runner.py \
      --role=receiver \
      --peer=10.128.0.241:50051 \
      --parallelism=4 \
      --num_blocks=512 \
      --num_layers=8 \
      --block_size=16

The geometry flags must match on both sides.

See README.md in this directory for how this compares with the other
microbenchmarks here.
"""

import os
import sys
import time
from absl import app
from absl import flags
import numpy as np
import torch
import torch_tpu
import uuid
from typing import Sequence

from tpu_sync.api.torch import kv_cache_manager
from tpu_sync.api.torch import torch_tpu_common_loader
from tpu_sync.rpc import coordination_helper

# Register the TPU device backend with PyTorch c10 dispatcher, then load the
# Raiden Torch extension before any TPU device work, the same order the
# torch image's build-time import check uses. Do not load it with
# RTLD_GLOBAL (the JAX runner needs that, Torch does not): it statically links
# LLVM, and a global load clashes with libtpu's LLVM ("Option ... registered
# more than once").
torch_tpu_common_loader.load_torch_tpu_common()
kv_cache_manager._torch_impl()  # pylint: disable=protected-access
if not hasattr(torch, 'tpu'):
  import torch_tpu._loader as _torch_tpu_loader  # pytype: disable=import-error

  _torch_tpu_loader._init_device('tpu')

_ROLE = flags.DEFINE_string('role', None, 'sender or receiver')
_PEER = flags.DEFINE_string(
    'peer', None, '<IP:Port> of the sender (required on receiver)'
)
_GRPC_PORT = flags.DEFINE_integer('grpc_port', 50051, 'Coordination port')
_NUM_BLOCKS = flags.DEFINE_integer(
    'num_blocks', 512, 'Number of cache blocks to allocate'
)
_BLOCK_SIZE = flags.DEFINE_integer(
    'block_size', 16, 'Number of tokens per block'
)
_NUM_LAYERS = flags.DEFINE_integer(
    'num_layers', 8, 'Number of transformer layers'
)
_PARALLELISM = flags.DEFINE_integer(
    'parallelism', 4, 'Number of parallel TCP streams for H2H'
)
_NUM_SLOTS = flags.DEFINE_integer(
    'num_slots', 2, 'Number of host staging slots'
)
_ENABLE_METRICS = flags.DEFINE_bool(
    'enable_metrics', False, 'Enable Raiden internal C++ metrics'
)
_WINDOW_BLOCKS = flags.DEFINE_integer(
    'window_blocks',
    -1,
    'Window size C in blocks (-1 for auto default, 0 to disable windowing).',
)
_WINDOW_DEPTH = flags.DEFINE_integer(
    'window_depth',
    2,
    'Bounded staging window depth W (default 2 for double-buffering).',
)
_HOST_BLOCKS_TO_ALLOCATE = flags.DEFINE_integer(
    'host_blocks_to_allocate',
    None,
    'Optional explicit host staging block pool size override.',
)
_ITERS = flags.DEFINE_integer(
    'iters', 1, 'Number of measured transfer iterations.'
)
_WARMUP = flags.DEFINE_integer(
    'warmup', 0, 'Number of warmup transfer iterations before measurement.'
)
_POLL_INTERVAL_MS = flags.DEFINE_float(
    'poll_interval_ms',
    0.5,
    'Completion poll sleep interval in milliseconds.',
)
_DTYPE = flags.DEFINE_string(
    'dtype',
    'float32',
    'Data type of cache: float32, bfloat16, fp8 (float8_e4m3fn).',
)


def _resolve_dtype(dtype_name: str) -> tuple[torch.dtype, int]:
  """Resolves the cache dtype name to a torch.dtype and item size in bytes."""
  normalized = dtype_name.lower()
  if normalized in ('float32', 'fp32'):
    return torch.float32, 4
  elif normalized in ('bfloat16', 'bf16'):
    return torch.bfloat16, 2
  elif normalized in ('fp8', 'float8', 'float8_e4m3fn'):
    return torch.float8_e4m3fn, 1
  else:
    raise ValueError(
        f'Unsupported dtype: {dtype_name}. Expected float32, bfloat16, or fp8.'
    )


def _make_base_tensor(
    shape: Sequence[int], device: torch.device
) -> torch.Tensor:
  """Creates the 5D base tensor on device using broadcasted 1D aranges."""
  b = (
      torch.arange(shape[0], device=device, dtype=torch.float32).view(
          -1, 1, 1, 1, 1
      )
      % 16.0
  ) * 0.5
  t = (
      torch.arange(shape[1], device=device, dtype=torch.float32).view(
          1, -1, 1, 1, 1
      )
      % 64.0
  ) * 0.25
  h = (
      torch.arange(shape[2], device=device, dtype=torch.float32).view(
          1, 1, -1, 1, 1
      )
      % 8.0
  ) * 1.0
  kv = (
      torch.arange(shape[3], device=device, dtype=torch.float32).view(
          1, 1, 1, -1, 1
      )
      % 2.0
  ) * 2.0
  d = (
      torch.arange(shape[4], device=device, dtype=torch.float32).view(
          1, 1, 1, 1, -1
      )
      % 128.0
  ) * (1.0 / 128.0)
  return b + t + h + kv + d


def populate_deterministic_cache(
    num_layers: int, shape: Sequence[int], dtype: torch.dtype
) -> list[torch.Tensor]:
  """Populates deterministic cache layers on device."""
  device = torch.device('tpu')
  base = _make_base_tensor(shape, device)
  tensors = []
  for layer_idx in range(num_layers):
    layer_tensor = (base + float(layer_idx % 32)).to(dtype)
    tensors.append(layer_tensor)
  torch.accelerator.synchronize()
  return tensors


def verify_deterministic_cache(
    num_layers: int,
    shape: Sequence[int],
    received_tensors: list[torch.Tensor],
    dtype: torch.dtype,
) -> bool:
  """Verifies data consistency across all cache layers on device."""
  print('Verifying data consistency across all cache layers...')
  device = received_tensors[0].device
  base = _make_base_tensor(shape, device)
  for layer_idx in range(num_layers):
    expected = (base + float(layer_idx % 32)).to(dtype)
    actual = received_tensors[layer_idx]
    if not bool(
        torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    ):
      print(f'Verification FAILED on Layer {layer_idx}!')
      return False
  print('Data consistency verified successfully! 0% corruption.')
  return True


def get_peer_grpc_bns_path(peer_arg: str) -> str:
  """Constructs the gRPC peer address for cloud or internal environments."""
  if not peer_arg:
    raise ValueError('A peer address MUST be provided (e.g. 10.128.0.2:50051)')

  # Fallback for cloud/OSS environments: format raw IP:PORT
  if ':' in peer_arg and not peer_arg.startswith('['):
    parts = peer_arg.rsplit(':', 1)
    if len(parts) == 2 and '.' not in parts[0]:
      return f'[{parts[0]}]:{parts[1]}'
  return peer_arg


def main(_):
  if _ROLE.value not in ('sender', 'receiver'):
    raise ValueError("--role must be 'sender' or 'receiver'")
  if _ENABLE_METRICS.value:
    os.environ['ENABLE_RAIDEN_METRICS'] = 'true'

  device = torch.device('tpu')
  cache_dtype, itemsize = _resolve_dtype(_DTYPE.value)
  # KV-cache shape: (num_blocks, block_size, num_heads, num_kv, head_dim)
  cache_shape = (_NUM_BLOCKS.value, _BLOCK_SIZE.value, 8, 2, 128)

  # Size of a single block in bytes
  block_bytes = int(np.prod(cache_shape[1:])) * itemsize
  payload_bytes = _NUM_LAYERS.value * _NUM_BLOCKS.value * block_bytes
  print(
      f'Config: role={_ROLE.value} parallelism={_PARALLELISM.value} '
      f'num_slots={_NUM_SLOTS.value} num_layers={_NUM_LAYERS.value} '
      f'num_blocks={_NUM_BLOCKS.value} block_size={_BLOCK_SIZE.value} '
      f'dtype={_DTYPE.value}'
  )
  print(
      f'Cache: shape={cache_shape} dtype={_DTYPE.value} '
      f'block={block_bytes / (1024 * 1024):.2f} MiB '
      f'payload={payload_bytes / (1024 * 1024):.2f} MiB'
  )

  if _ROLE.value == 'sender':
    print('Starting H2H Sender Process')
    t0 = time.perf_counter()

    # 1. Populate deterministic cache on TPU
    tpu_src_arr = populate_deterministic_cache(
        _NUM_LAYERS.value, cache_shape, cache_dtype
    )
    print(
        f'Populated {_NUM_LAYERS.value} source layers on device in '
        f'{time.perf_counter() - t0:.2f}s'
    )

    # 2. Start CoordinationServer
    grpc_port = _GRPC_PORT.value
    coordinator_server = coordination_helper.CoordinationServer(port=grpc_port)
    bound_grpc_port = coordinator_server.start()
    print(f'Coordination gRPC server started on port: {bound_grpc_port}')

    # 3. Initialize KVCacheManager and register_read()
    t0 = time.perf_counter()
    sender_manager = kv_cache_manager.KVCacheManager(
        kv_caches=tpu_src_arr,
        local_control_port=0,
        max_blocks=_NUM_BLOCKS.value,
        num_slots=_NUM_SLOTS.value,
        unsafe_skip_buffer_lock=True,
        parallelism=_PARALLELISM.value,
        host_blocks_to_allocate=_HOST_BLOCKS_TO_ALLOCATE.value,
        window_blocks=_WINDOW_BLOCKS.value,
        window_depth=_WINDOW_DEPTH.value,
    )

    print(
        f'Sender KVCacheManager initialized in {time.perf_counter() - t0:.2f}s'
    )

    print('Waiting for Receiver KVCacheManager initialization barrier...')
    coordinator_client = coordination_helper.CoordinationClient(
        server_address=f'localhost:{bound_grpc_port}'
    )
    coordinator_client.collect_replica_info(
        device_id=0, expected_count=2, info=[0]
    )
    print('Receiver ready; arming read registration...')

    transfer_uuid = uuid.uuid4().int & 0xFFFFFFFF
    transfer_req_id = f'perf_test_{transfer_uuid}'

    block_ids = list(range(_NUM_BLOCKS.value))
    total_runs = _WARMUP.value + _ITERS.value
    for run_idx in range(total_runs):
      iter_req_id = f'{transfer_req_id}_{run_idx}'
      iter_uuid = transfer_uuid + run_idx * 10000
      sender_manager.register_read(iter_req_id, iter_uuid, block_ids)
    print(
        f'Armed {len(block_ids)} blocks for read across {total_runs} runs '
        f'(base uuid={transfer_uuid})'
    )

    # 4. Publish metadata to CoordinationServer
    all_endpoints = sender_manager.get_local_endpoints()
    coordinator_server.set_metadata(
        endpoints=all_endpoints,
        transfer_uuid=transfer_uuid,
        transfer_req_id=transfer_req_id,
        block_ids=block_ids,
    )
    print(
        f'Metadata published! Endpoints: {all_endpoints}, req_id:'
        f' {transfer_req_id}, uuid: {transfer_uuid}. Waiting for Receiver...'
    )

    # 5. Wait for receiver to finish

    t0 = time.perf_counter()
    try:
      coordinator_server.wait_for_shutdown()
      print(
          f'Receiver finished after {time.perf_counter() - t0:.2f}s! Shutting'
          ' down Sender coordination server...'
      )
    except KeyboardInterrupt:
      pass
    finally:
      coordinator_server.stop()

  elif _ROLE.value == 'receiver':
    print('Starting H2H Receiver process...')
    # 1. Validate --peer is set
    if not _PEER.value:
      raise ValueError('A peer address MUST be provided')

    # 2. Allocate zero-filled destination tensors on TPU
    t0 = time.perf_counter()
    device_tensors = [
        torch.zeros(cache_shape, dtype=cache_dtype, device=device)
        for _ in range(_NUM_LAYERS.value)
    ]
    torch.accelerator.synchronize()
    print(
        f'Allocated {_NUM_LAYERS.value} destination layers on device in '
        f'{time.perf_counter() - t0:.2f}s'
    )

    # 3. Initialize KVCacheManager
    t0 = time.perf_counter()

    receiver_manager = kv_cache_manager.KVCacheManager(
        kv_caches=device_tensors,
        local_control_port=0,
        max_blocks=_NUM_BLOCKS.value,
        num_slots=_NUM_SLOTS.value,
        unsafe_skip_buffer_lock=True,
        parallelism=_PARALLELISM.value,
        host_blocks_to_allocate=_HOST_BLOCKS_TO_ALLOCATE.value,
        window_blocks=_WINDOW_BLOCKS.value,
        window_depth=_WINDOW_DEPTH.value,
    )

    print(f'Receiver KVCacheManager ready in {time.perf_counter() - t0:.2f}s')

    # 4. Connect CoordinationClient to sender and synchronize at barrier
    resolved_bns_peer = get_peer_grpc_bns_path(_PEER.value)
    print(f'Connecting to peer coordination BNS: {resolved_bns_peer}')
    coordinator_client = coordination_helper.CoordinationClient(
        server_address=resolved_bns_peer
    )

    max_retries = 60
    retry_interval = 5
    barrier_passed = False
    for attempt in range(1, max_retries + 1):
      try:
        coordinator_client.collect_replica_info(
            device_id=1, expected_count=2, info=[1]
        )
        barrier_passed = True
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            f'Attempt {attempt}/{max_retries} waiting for sender barrier'
            f' ({e}). Retrying in {retry_interval}s...'
        )
        time.sleep(retry_interval)

    if not barrier_passed:
      raise RuntimeError(
          f'Failed to synchronize barrier with peer {_PEER.value} after'
          f' {max_retries} attempts.'
      )

    # 5. Retrieve metadata (block_ids, endpoints, transfer_uuid)
    metadata = None
    for attempt in range(1, max_retries + 1):
      try:
        metadata = coordinator_client.get_metadata()
        if metadata.endpoints:
          break
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            f'Attempt {attempt}/{max_retries} waiting for sender metadata'
            f' ({e}). Retrying in 1s...'
        )
      time.sleep(1)

    if metadata is None or not metadata.endpoints:
      raise RuntimeError(
          f'Failed to coordinate with peer {_PEER.value} after {max_retries}'
          ' attempts.'
      )

    src_block_ids = metadata.block_ids
    remote_endpoints = metadata.endpoints
    transfer_uuid = metadata.transfer_uuid
    transfer_req_id = metadata.transfer_req_id

    print(f'Metadata received! Block count: {len(src_block_ids)}')
    print(f'Resolved Peer dynamic H2H endpoints: {remote_endpoints}')

    # 6. Time the transfer (start_read -> poll_stats -> completion)
    print('Executing H2H Read E2E offloading transfer...')
    total_runs = _WARMUP.value + _ITERS.value
    latencies = []
    total_bytes = _NUM_LAYERS.value * len(src_block_ids) * block_bytes
    total_megabytes = total_bytes / (1024 * 1024)

    for run_idx in range(total_runs):
      is_warmup = run_idx < _WARMUP.value
      iter_req_id = f'{transfer_req_id}_{run_idx}'
      iter_uuid = transfer_uuid + run_idx * 10000

      start_time = time.perf_counter()
      receiver_manager.start_read(
          req_id=iter_req_id,
          uuid=iter_uuid,
          remote_endpoint=remote_endpoints,
          remote_block_ids=src_block_ids,
          local_block_ids=src_block_ids,
          parallelism=_PARALLELISM.value,
      )
      start_read_returned = time.perf_counter()

      polls = 0
      completed = False
      while not completed:
        polls += 1
        _, done_recving, failed_recving = receiver_manager.poll_stats()
        if iter_req_id in done_recving:
          completed = True
        elif iter_req_id in failed_recving:
          raise RuntimeError(f'Transfer failed! req_id: {iter_req_id}')
        else:
          time.sleep(_POLL_INTERVAL_MS.value / 1000.0)

      end_time = time.perf_counter()
      elapsed_time = end_time - start_time
      bw_gbps = (total_bytes * 8) / (elapsed_time * 1e9)
      bw_gbs = total_bytes / (elapsed_time * 1e9)

      prefix = '[WARMUP] ' if is_warmup else ''
      print(
          f'{prefix}Run {run_idx + 1}/{total_runs}:'
          f' elapsed={elapsed_time * 1e3:.2f} ms ({bw_gbps:.3f} Gbps,'
          f' {bw_gbs:.3f} GB/s), start_read returned in'
          f' {(start_read_returned - start_time) * 1e3:.1f} ms, waited'
          f' {(end_time - start_read_returned) * 1e3:.1f} ms across {polls}'
          f' poll(s) at {_POLL_INTERVAL_MS.value}ms granularity'
      )
      if not is_warmup:
        latencies.append(elapsed_time)

    # Follow-up F3: poll manager.poll_stats() after completion
    _, final_done, final_failed = receiver_manager.poll_stats()
    print(
        f'Final poll_stats: done_recving={len(final_done)}, '
        f'failed_recving={len(final_failed)}'
    )

    if latencies:
      arr = np.array(latencies)
      rep_elapsed = float(np.median(arr))
      bandwidth_gbps = (total_bytes * 8) / (rep_elapsed * 1e9)
      bandwidth_gbs = total_bytes / (rep_elapsed * 1e9)

      print('\n=== H2H E2E Performance Results ===')
      print(f'Parallelism: {_PARALLELISM.value}')
      print(f'Iterations: {_ITERS.value} (warmup: {_WARMUP.value})')
      print(f'Data Volume Transferred: {total_megabytes:.2f} MB')
      print(
          f'Latency (ms): min={float(np.min(arr)) * 1e3:.2f}'
          f' p50={rep_elapsed * 1e3:.2f} mean={float(np.mean(arr)) * 1e3:.2f}'
          f' p90={float(np.percentile(arr, 90)) * 1e3:.2f}'
          f' max={float(np.max(arr)) * 1e3:.2f}'
      )
      print(f'Elapsed Time (TCP H2H + Copy): {rep_elapsed:.4f} seconds')
      print(f'Bandwidth: {bandwidth_gbps:.3f} Gbps ({bandwidth_gbs:.3f} GB/s)')
      print('===================================\n')
    t0 = time.perf_counter()
    success = verify_deterministic_cache(
        _NUM_LAYERS.value, cache_shape, device_tensors, cache_dtype
    )
    print(f'Verification took {time.perf_counter() - t0:.2f}s')
    if not success:
      print('Signalling failure to peer Sender...')
      coordinator_client.shutdown()
      sys.exit(1)

    # 8. Signal sender shutdown
    print('Signalling completion to peer Sender...')
    coordinator_client.shutdown()
    print('E2E performance test runner completed successfully!')


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
