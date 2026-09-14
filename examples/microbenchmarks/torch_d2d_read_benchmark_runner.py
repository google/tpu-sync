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

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)

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

# Register the TPU device backend with PyTorch c10 dispatcher.
torch_tpu_common_loader.load_torch_tpu_common()
if not hasattr(torch, 'tpu'):
  import torch_tpu._loader as _torch_tpu_loader  # pytype: disable=import-error

  _torch_tpu_loader._init_device('tpu')

FLAGS = flags.FLAGS
flags.DEFINE_string('role', None, 'sender or receiver')
flags.DEFINE_string(
    'peer', None, '<IP:Port> of the sender (required on receiver)'
)
flags.DEFINE_integer('grpc_port', 50051, 'Coordination port')
flags.DEFINE_integer('num_blocks', 512, 'Number of cache blocks to allocate')
flags.DEFINE_integer('block_size', 16, 'Number of tokens per block')
flags.DEFINE_integer('num_layers', 8, 'Number of transformer layers')
flags.DEFINE_integer('parallelism', 4, 'Number of parallel TCP streams for H2H')
flags.DEFINE_integer('num_slots', 2, 'Number of host staging slots')
flags.DEFINE_bool('enable_metrics', False, 'Enable Raiden internal C++ metrics')


def populate_deterministic_cache(
    num_layers: int, shape: Sequence[int]
) -> list[torch.Tensor]:
  device = torch.device('tpu')
  tensors = []
  for layer_idx in range(num_layers):
    # Every layer has distinct values: layer 0 starts at 0, layer 1 at 1000, etc.
    base = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(
        shape
    ) + float(layer_idx * 1000.0)
    tensors.append(base.to(device))
  torch.tpu.synchronize()
  return tensors


def verify_deterministic_cache(
    num_layers: int,
    shape: Sequence[int],
    received_tensors: list[torch.Tensor],
) -> bool:
  print('Verifying data consistency across all cache layers...')
  for layer_idx in range(num_layers):
    expected = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(
        shape
    ) + float(layer_idx * 1000.0)
    actual = received_tensors[layer_idx].cpu()
    if not torch.equal(actual, expected):
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
  if FLAGS.role not in ('sender', 'receiver'):
    raise ValueError("--role must be 'sender' or 'receiver'")
  if FLAGS.enable_metrics:
    os.environ['ENABLE_RAIDEN_METRICS'] = 'true'

  device = torch.device('tpu')
  # KV-cache shape: (num_blocks, num_heads, block_size, num_kv_heads, head_dim)
  cache_shape = (FLAGS.num_blocks, 32, FLAGS.block_size, 8, 128)

  # Size of a single block in bytes (float32 = 4 bytes)
  block_bytes = int(np.prod(cache_shape[1:])) * 4
  payload_bytes = FLAGS.num_layers * FLAGS.num_blocks * block_bytes
  print(
      f'Config: role={FLAGS.role} parallelism={FLAGS.parallelism} '
      f'num_slots={FLAGS.num_slots} num_layers={FLAGS.num_layers} '
      f'num_blocks={FLAGS.num_blocks} block_size={FLAGS.block_size}'
  )
  print(
      f'Cache: shape={cache_shape} dtype=float32 '
      f'block={block_bytes / (1024 * 1024):.2f} MiB '
      f'payload={payload_bytes / (1024 * 1024):.2f} MiB'
  )

  if FLAGS.role == 'sender':
    print('Starting H2H Sender Process')
    t0 = time.perf_counter()

    # 1. Populate deterministic cache on TPU
    tpu_src_arr = populate_deterministic_cache(FLAGS.num_layers, cache_shape)
    print(
        f'Populated {FLAGS.num_layers} source layers on device in '
        f'{time.perf_counter() - t0:.2f}s'
    )

    # 2. Start CoordinationServer
    grpc_port = FLAGS.grpc_port
    coordinator_server = coordination_helper.CoordinationServer(port=grpc_port)
    bound_grpc_port = coordinator_server.start()
    print(f'Coordination gRPC server started on port: {bound_grpc_port}')

    # 3. Initialize KVCacheManager and register_read()
    t0 = time.perf_counter()
    sender_manager = kv_cache_manager.KVCacheManager(
        kv_caches=tpu_src_arr,
        local_control_port=0,
        max_blocks=FLAGS.num_blocks,
        num_slots=FLAGS.num_slots,
        unsafe_skip_buffer_lock=True,
        parallelism=FLAGS.parallelism,
    )

    print(
        f'Sender KVCacheManager initialized in {time.perf_counter() - t0:.2f}s'
    )

    transfer_uuid = uuid.uuid4().int & 0xFFFFFFFF
    transfer_req_id = f'perf_test_{transfer_uuid}'

    block_ids = list(range(FLAGS.num_blocks))
    sender_manager.register_read(transfer_req_id, transfer_uuid, block_ids)
    print(f'Armed {len(block_ids)} blocks for read (uuid={transfer_uuid})')

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

  elif FLAGS.role == 'receiver':
    print('Starting H2H Receiver process...')
    # 1. Validate --peer is set
    if not FLAGS.peer:
      raise ValueError('A peer address MUST be provided')

    # 2. Connect CoordinationClient to sender
    resolved_bns_peer = get_peer_grpc_bns_path(FLAGS.peer)
    print(f'Connecting to peer coordination BNS: {resolved_bns_peer}')
    coordinator_client = coordination_helper.CoordinationClient(
        server_address=resolved_bns_peer
    )

    max_retries = 15
    metadata = None
    for attempt in range(1, max_retries + 1):
      try:
        metadata = coordinator_client.get_metadata()
        break
      except Exception as e:
        print(
            f'Attempt {attempt}/{max_retries} waiting for sender ({e}).'
            'Retrying in 2s...'
        )
        time.sleep(2)

    # 3. Retrieve metadata (block_ids, endpoints, transfer_uuid)
    if metadata is None:
      raise RuntimeError(
          f'Failed to coordinate with peer {FLAGS.peer} after {max_retries}'
          ' attempts.'
      )

    src_block_ids = metadata.block_ids
    remote_endpoints = metadata.endpoints
    transfer_uuid = metadata.transfer_uuid
    transfer_req_id = metadata.transfer_req_id

    print(f'Metadata received! Block count: {len(src_block_ids)}')
    print(f'Resolved Peer dynamic H2H endpoints: {remote_endpoints}')

    # 4. Allocate empty destination tensors on TPU
    t0 = time.perf_counter()
    device_tensors = [
        torch.empty(cache_shape, dtype=torch.float32, device='tpu')
        for _ in range(FLAGS.num_layers)
    ]
    torch.tpu.synchronize()
    print(
        f'Allocated {FLAGS.num_layers} destination layers on device in '
        f'{time.perf_counter() - t0:.2f}s'
    )

    # 5. Initialize KVCacheManager
    t0 = time.perf_counter()

    receiver_manager = kv_cache_manager.KVCacheManager(
        kv_caches=device_tensors,
        local_control_port=0,
        max_blocks=FLAGS.num_blocks,
        num_slots=FLAGS.num_slots,
        unsafe_skip_buffer_lock=True,
        parallelism=FLAGS.parallelism,
    )

    print(f'Receiver KVCacheManager ready in {time.perf_counter() - t0:.2f}s')

    # 6. Time the transfer (start_read -> poll_stats -> completion)
    print('Executing H2H Read E2E offloading transfer...')
    start_time = time.perf_counter()

    receiver_manager.start_read(
        req_id=transfer_req_id,
        uuid=transfer_uuid,
        remote_endpoint=remote_endpoints,
        remote_block_ids=src_block_ids,
        local_block_ids=src_block_ids,
        parallelism=FLAGS.parallelism,
    )

    start_read_returned = time.perf_counter()

    polls = 0
    completed = False
    while not completed:
      polls += 1
      _, done_recving, failed_recving = receiver_manager.poll_stats()
      if transfer_req_id in done_recving:
        completed = True
      elif transfer_req_id in failed_recving:
        raise RuntimeError(f'Transfer failed! req_id: {transfer_req_id}')
      else:
        time.sleep(0.01)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(
        'start_read returned in '
        f'{(start_read_returned - start_time) * 1e3:.1f} ms; '
        f'waited {(end_time - start_read_returned) * 1e3:.1f} ms across '
        f'{polls} poll(s) at 10ms granularity'
    )

    # 7. Compute GB/s bandwidth and verify data
    total_bytes = FLAGS.num_layers * len(src_block_ids) * block_bytes
    total_megabytes = total_bytes / (1024 * 1024)
    bandwidth_gbps = (total_bytes * 8) / (elapsed_time * 1e9)
    bandwidth_gbs = total_bytes / (elapsed_time * 1e9)
    print('\n=== H2H E2E Performance Results ===')
    print(f'Parallelism: {FLAGS.parallelism}')
    print(f'Data Volume Transferred: {total_megabytes:.2f} MB')
    print(f'Elapsed Time (TCP H2H + Copy): {elapsed_time:.4f} seconds')
    print(f'Bandwidth: {bandwidth_gbps:.3f} Gbps ({bandwidth_gbs:.3f} GB/s)')
    print('===================================\n')
    t0 = time.perf_counter()
    success = verify_deterministic_cache(
        FLAGS.num_layers, cache_shape, device_tensors
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
