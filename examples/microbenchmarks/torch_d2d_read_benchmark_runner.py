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

"""Cross-node device-to-device KV-cache read benchmark, via PyTorch.

Receiver-initiated pull of a KV cache from one TPU node's HBM into another's,
measured end to end across D2H, H2H over the NIC, and H2D, and verified
byte-for-byte.

Start the SENDER first; it arms the cache and blocks until the receiver is done:

  cd <tpu-raiden>
  PYTHONUNBUFFERED=1 PYTHONPATH=$PWD python3 \\
    examples/microbenchmarks/torch_d2d_read_benchmark_runner.py \\
      --role=sender \\
      --grpc_port=50051 \\
      --parallelism=20 \\
      --num_blocks=128 \\
      --num_layers=8 \\
      --block_size=8

Then the RECEIVER, pointed at the sender's VPC address (not localhost):

  cd <tpu-raiden>
  PYTHONUNBUFFERED=1 PYTHONPATH=$PWD python3 \\
    examples/microbenchmarks/torch_d2d_read_benchmark_runner.py \\
      --role=receiver \\
      --peer=10.128.0.241:50051 \\
      --parallelism=20 \\
      --num_blocks=128 \\
      --num_layers=8 \\
      --block_size=8

The geometry flags must match on both sides. Only one process per side is
supported.
"""

import os
import sys
import time
import uuid

from absl import app
from absl import flags
import numpy as np
import torch
import torch_tpu

from tpu_sync.api.torch import kv_cache_manager
from tpu_sync.rpc import coordination_helper

_ROLE = flags.DEFINE_string(
    'role', None, 'Role of the task: sender or receiver.'
)
_PEER = flags.DEFINE_string(
    'peer', None, 'IP:PORT of the sender (for receiver).'
)
_GRPC_PORT = flags.DEFINE_integer(
    'grpc_port', 50051, 'Pre-agreed static gRPC coordination port.'
)
_NUM_BLOCKS = flags.DEFINE_integer(
    'num_blocks', 512, 'Number of cache blocks to allocate.'
)
_BLOCK_SIZE = flags.DEFINE_integer('block_size', 2, 'Size of cache blocks.')
_NUM_LAYERS = flags.DEFINE_integer(
    'num_layers', 8, 'Number of transformer layers.'
)
_PARALLELISM = flags.DEFINE_integer(
    'parallelism', 1, 'Number of parallel TCP streams for H2H.'
)
_NUM_SLOTS = flags.DEFINE_integer(
    'num_slots', 2, 'Number of host staging slots to allocate.'
)
_ENABLE_METRICS = flags.DEFINE_boolean(
    'enable_metrics', False, 'Enable internal telemetry metrics collection.'
)


def get_peer_grpc_path(peer_arg: str | None) -> str:
  """Normalizes peer address string into host:port format."""
  if not peer_arg:
    raise ValueError('A peer address MUST be provided (e.g. 10.128.0.2:50051)')
  if ':' in peer_arg and not peer_arg.startswith('['):
    parts = peer_arg.rsplit(':', 1)
    if len(parts) == 2 and '.' not in parts[0]:
      return f'[{parts[0]}]:{parts[1]}'
  return peer_arg


def populate_deterministic_cache(
    _num_blocks: int,
    num_layers: int,
    shape: tuple[int, ...],
    device: torch.device,
) -> list[torch.Tensor]:
  """Creates deterministic numerical patterns across all layer caches."""
  arrs = []
  for layer_idx in range(num_layers):
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + float(
        layer_idx * 1000.0
    )
    t = torch.tensor(base, dtype=torch.float32, device=device)
    arrs.append(t)
  torch.tpu.synchronize()
  return arrs


def verify_deterministic_cache(
    _num_blocks: int,
    num_layers: int,
    shape: tuple[int, ...],
    dst_tpu_arrs: list[torch.Tensor],
) -> bool:
  """Checks layer tensors for byte-for-byte exact equality."""
  print('Verifying data consistency across all cache layers...')
  for layer_idx in range(num_layers):
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(
        shape
    ) + float(layer_idx * 1000.0)
    actual = dst_tpu_arrs[layer_idx].cpu().numpy()
    try:
      np.testing.assert_array_equal(actual, expected)
    except AssertionError as exc:
      print(f'Verification FAILED on Layer {layer_idx}!')
      print(exc)
      return False
  print('Data consistency verified successfully! 0% corruption.')
  return True


def main(_):
  if not _ROLE.value:
    raise ValueError('--role must be specified (sender or receiver)')

  if _ENABLE_METRICS.value:
    os.environ['ENABLE_RAIDEN_METRICS'] = 'true'

  device = torch.device('tpu')
  try:
    _ = torch_tpu
    _ = torch.zeros(1, device=device)
  except Exception as e:
    raise RuntimeError(f'No TPU devices found or torch_tpu failed: {e}')

  cache_shape = (_NUM_BLOCKS.value, 32, _BLOCK_SIZE.value, 8, 128)
  block_bytes = int(np.prod(cache_shape[1:])) * 4
  payload_bytes = _NUM_LAYERS.value * _NUM_BLOCKS.value * block_bytes

  print(
      f'Config: role={_ROLE.value} parallelism={_PARALLELISM.value} '
      f'num_slots={_NUM_SLOTS.value} num_layers={_NUM_LAYERS.value} '
      f'num_blocks={_NUM_BLOCKS.value} block_size={_BLOCK_SIZE.value}'
  )
  print(
      f'Cache: shape={cache_shape} dtype=float32 '
      f'block={block_bytes / (1024 * 1024):.2f} MiB '
      f'payload={payload_bytes / (1024 * 1024):.2f} MiB'
  )

  if _ROLE.value == 'sender':
    print('Starting H2H Sender process...')

    t0 = time.perf_counter()
    tpu_src_arrs = populate_deterministic_cache(
        _NUM_BLOCKS.value, _NUM_LAYERS.value, cache_shape, device
    )
    print(
        f'Populated {_NUM_LAYERS.value} source layers on device in '
        f'{time.perf_counter() - t0:.2f}s'
    )

    grpc_port = _GRPC_PORT.value
    coordination_server = coordination_helper.CoordinationServer(port=grpc_port)
    bound_grpc_port = coordination_server.start()
    print(f'Coordination gRPC server started on port: {bound_grpc_port}')

    t0 = time.perf_counter()
    manager = kv_cache_manager.KVCacheManager(
        kv_caches=tpu_src_arrs,
        local_control_port=0,
        max_blocks=_NUM_BLOCKS.value,
        num_slots=_NUM_SLOTS.value,
        unsafe_skip_buffer_lock=True,
        parallelism=_PARALLELISM.value,
    )
    print(f'KVCacheManager ready in {time.perf_counter() - t0:.2f}s')

    transfer_uuid = uuid.uuid4().int & 0xFFFFFFFF
    transfer_req_id = f'perf_test_{transfer_uuid}'

    total_cache_blocks = _NUM_BLOCKS.value
    block_ids = list(range(total_cache_blocks))
    manager.register_read(transfer_req_id, transfer_uuid, block_ids)
    print(f'Armed {len(block_ids)} blocks for read (uuid={transfer_uuid})')

    all_endpoints = manager.get_local_endpoints()
    coordination_server.set_metadata(
        endpoints=all_endpoints,
        transfer_uuid=transfer_uuid,
        transfer_req_id=transfer_req_id,
        block_ids=block_ids,
    )
    print(
        f'Metadata published! Endpoints: {all_endpoints}, req_id:'
        f' {transfer_req_id}, uuid: {transfer_uuid}. Waiting for Receiver...'
    )

    t0 = time.perf_counter()
    try:
      coordination_server.wait_for_shutdown()
      print(
          f'Receiver finished after {time.perf_counter() - t0:.2f}s! Shutting'
          ' down Sender coordination server...'
      )
    except KeyboardInterrupt:
      pass
    finally:
      coordination_server.stop()

  elif _ROLE.value == 'receiver':
    print('Starting H2H Receiver process...')

    resolved_peer = get_peer_grpc_path(_PEER.value)
    print(f'Connecting to peer coordination server at: {resolved_peer}')
    client = coordination_helper.CoordinationClient(
        server_address=resolved_peer
    )

    max_retries = 15
    metadata = None
    for attempt in range(1, max_retries + 1):
      try:
        metadata = client.get_metadata()
        break
      except Exception as e:
        print(
            f'Attempt {attempt}/{max_retries} waiting for peer network server'
            f' ({e}). Retrying in 5s...'
        )
        time.sleep(5)

    if metadata is None:
      raise RuntimeError(
          f'Failed to coordinate with peer {resolved_peer} after'
          f' {max_retries} attempts.'
      )

    src_block_ids = metadata.block_ids
    remote_endpoints = metadata.endpoints
    transfer_uuid = metadata.transfer_uuid
    transfer_req_id = metadata.transfer_req_id

    print(f'Metadata received! Block count: {len(src_block_ids)}')
    print(f'Resolved Peer dynamic H2H endpoints: {remote_endpoints}')

    t0 = time.perf_counter()
    device_arrs = [
        torch.empty(cache_shape, dtype=torch.float32, device=device)
        for _ in range(_NUM_LAYERS.value)
    ]
    torch.tpu.synchronize()
    print(
        f'Allocated {_NUM_LAYERS.value} destination layers on device in '
        f'{time.perf_counter() - t0:.2f}s'
    )

    t0 = time.perf_counter()
    manager = kv_cache_manager.KVCacheManager(
        kv_caches=device_arrs,
        local_control_port=0,
        max_blocks=_NUM_BLOCKS.value,
        num_slots=_NUM_SLOTS.value,
        unsafe_skip_buffer_lock=True,
        parallelism=_PARALLELISM.value,
    )
    print(f'KVCacheManager ready in {time.perf_counter() - t0:.2f}s')

    print('Executing H2H Read E2E offloading transfer...')
    start_time = time.perf_counter()

    manager.start_read(
        req_id=transfer_req_id,
        uuid=transfer_uuid,
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
      _, done_recving, failed_recving = manager.poll_stats()
      if transfer_req_id in done_recving:
        completed = True
      elif transfer_req_id in failed_recving:
        raise RuntimeError(f'Transfer failed! req_id: {transfer_req_id}')
      else:
        time.sleep(0.01)

    torch.tpu.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(
        f'start_read returned in '
        f'{(start_read_returned - start_time) * 1e3:.1f} ms; '
        f'waited {(end_time - start_read_returned) * 1e3:.1f} ms across '
        f'{polls} poll(s) at 10ms granularity'
    )

    block_byte_size = np.prod(cache_shape[1:]) * 4
    total_bytes = _NUM_LAYERS.value * len(src_block_ids) * block_byte_size
    total_megabytes = total_bytes / (1024 * 1024)
    bandwidth_gbps = (total_bytes * 8) / (elapsed_time * 1e9)

    print('\n=== H2H E2E Performance Results ===')
    print(f'Parallelism: {_PARALLELISM.value}')
    print(f'Data Volume Transferred: {total_megabytes:.2f} MB')
    print(f'Elapsed Time (TCP H2H + Copy): {elapsed_time:.4f} seconds')
    print(
        f'Bandwidth: {bandwidth_gbps:.3f} Gbps '
        f'({total_bytes / elapsed_time / 1e9:.3f} GB/s)'
    )
    print('===================================\n')

    t0 = time.perf_counter()
    success = verify_deterministic_cache(
        _NUM_BLOCKS.value, _NUM_LAYERS.value, cache_shape, device_arrs
    )
    print(f'Verification took {time.perf_counter() - t0:.2f}s')

    if not success:
      print('Signalling failure to peer Sender...')
      client.shutdown()
      sys.exit(1)

    print('Signalling completion to peer Sender...')
    client.shutdown()
    print('E2E performance test runner completed successfully!')

  else:
    raise ValueError(f'Unknown role: {_ROLE.value}')


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
