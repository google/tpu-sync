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

"""Cross-node device-to-device KV-cache read benchmark, via jax.

Receiver-initiated pull of a KV cache from one TPU node's HBM into another's,
measured end to end across D2H, H2H over the NIC, and H2D, and verified
byte-for-byte.

Start the SENDER first; it arms the cache and blocks until the receiver is done:

  cd <tpu-raiden>
  PYTHONUNBUFFERED=1 PYTHONPATH=$PWD python3 \
    examples/microbenchmarks/jax_d2d_read_benchmark_runner.py \
      --role=sender \
      --grpc_port=50051 \
      --parallelism=20 \
      --num_blocks=128 \
      --num_layers=8 \
      --block_size=8

Then the RECEIVER, pointed at the sender's VPC address (not localhost):

  cd <tpu-raiden>
  PYTHONUNBUFFERED=1 PYTHONPATH=$PWD python3 \
    examples/microbenchmarks/jax_d2d_read_benchmark_runner.py \
      --role=receiver \
      --peer=10.128.0.241:50051 \
      --parallelism=20 \
      --num_blocks=128 \
      --num_layers=8 \
      --block_size=8

The geometry flags must match on both sides. --block_size must be a multiple of
the local device count, and only one process per side is supported.

See README.md in this directory for how this compares with the other
microbenchmarks here.
"""

import functools
import os
import sys

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)

import time
import uuid
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

from tpu_sync.api.jax import kv_cache_manager
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
_BLOCK_SIZE = flags.DEFINE_integer('block_size', 4096, 'Size of cache blocks.')
_NUM_LAYERS = flags.DEFINE_integer(
    'num_layers', 60, 'Number of transformer layers.'
)
_PARALLELISM = flags.DEFINE_integer(
    'parallelism', 1, 'Number of parallel TCP streams for H2H.'
)
_NUM_SLOTS = flags.DEFINE_integer(
    'num_slots', 1, 'Number of host staging slots to allocate.'
)
_ENABLE_METRICS = flags.DEFINE_boolean(
    'enable_metrics', False, 'Enable internal telemetry metrics collection.'
)
_COORDINATOR = flags.DEFINE_string(
    'coordinator', None, 'JAX coordinator address'
)
_PROCESS_INDEX = flags.DEFINE_integer('process_index', -1, 'JAX process index')
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_processes', 1, 'Number of JAX processes.'
)
_STAGING_WINDOW_CHUNKS = flags.DEFINE_integer(
    'staging_window_chunks', 0, 'Window chunks (W) for bounded window staging.'
)
_STAGING_CHUNK_BLOCKS = flags.DEFINE_integer(
    'staging_chunk_blocks', 0, 'Chunk size (C) in blocks (0 = auto).'
)
_STAGING_CHUNK_BYTES = flags.DEFINE_integer(
    'staging_chunk_bytes', 0, 'Chunk size target in bytes.'
)
_STAGING_STREAM_BYTES = flags.DEFINE_integer(
    'staging_stream_bytes', 0, 'Stream slice size target in bytes.'
)
_STAGING_ADMISSION = flags.DEFINE_string(
    'staging_admission', 'fail_fast', 'Admission policy: fail_fast or wait.'
)

def setup_shardings(devices):
  num_devices = len(devices)
  axis_shapes = (1, num_devices)
  axis_names = ('data', 'model')
  devices_array = np.array(devices).reshape(axis_shapes)
  mesh = jax.sharding.Mesh(devices_array, axis_names)
  spec = jax.sharding.PartitionSpec(None, None, 'model')

  tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
  host_sharding = jax.sharding.NamedSharding(
      mesh, spec, memory_kind='pinned_host'
  )
  return tpu_sharding, host_sharding

def get_peer_grpc_path(peer_arg: str) -> str:
  # Expects raw "IP:PORT" formatting for cloud environments.
  if not peer_arg:
      raise ValueError("A peer address MUST be provided (e.g. 10.128.0.2:50051)")
  # ensure ipv6 is bracketed if passed (basic check)
  if ':' in peer_arg and not peer_arg.startswith('['):
    parts = peer_arg.rsplit(':', 1)
    if len(parts) == 2 and '.' not in parts[0]: 
      return f'[{parts[0]}]:{parts[1]}'
  return peer_arg

def populate_deterministic_cache(
    num_blocks: int, num_layers: int, shape: tuple[int, ...], sharding
) -> list[jax.Array]:

  @functools.partial(jax.jit, static_argnums=(0,), out_shardings=sharding)
  def _make_layer(shape_tuple: tuple[int, ...], layer_idx: int) -> jax.Array:
    block_vals = jnp.arange(shape_tuple[0], dtype=jnp.float32)[
        :, None, None, None, None
    ] + (layer_idx * 1000.0 + 1.0)
    return jnp.broadcast_to(block_vals, shape_tuple)

  arrs = [_make_layer(shape, layer_idx) for layer_idx in range(num_layers)]
  jax.block_until_ready(arrs)
  return arrs


def verify_deterministic_cache(
    num_blocks: int,
    num_layers: int,
    shape: tuple[int, ...],
    dst_tpu_arrs: list[jax.Array],
    sharding=None,
) -> bool:
  print('Verifying data consistency across all sharded cache layers...')

  @functools.partial(jax.jit, static_argnums=(0,))
  def _check_layer(
      shape_tuple: tuple[int, ...], layer_idx: int, actual: jax.Array
  ) -> jax.Array:
    block_vals = jnp.arange(shape_tuple[0], dtype=jnp.float32)[
        :, None, None, None, None
    ] + (layer_idx * 1000.0 + 1.0)
    expected = jnp.broadcast_to(block_vals, shape_tuple)
    if sharding is not None:
      expected = jax.lax.with_sharding_constraint(expected, sharding)
    return jnp.array_equal(actual, expected)

  for layer_idx in range(num_layers):
    is_equal = bool(_check_layer(shape, layer_idx, dst_tpu_arrs[layer_idx]))
    if not is_equal:
      print(f'Verification FAILED on Layer {layer_idx}!')
      return False
  print('Data consistency verified successfully! 0% corruption.')
  return True


def main(_):
  if not _ROLE.value:
    raise ValueError('--role must be specified')

  if _ENABLE_METRICS.value:
    os.environ['ENABLE_RAIDEN_METRICS'] = 'true'

  if _PROCESS_INDEX.value != -1:
    jax.distributed.initialize(
        coordinator_address=_COORDINATOR.value,
        num_processes=_NUM_PROCESSES.value,
        process_id=_PROCESS_INDEX.value,
    )

  print(f'Initialized JAX distributed mesh: process_index={jax.process_index()}, process_count={jax.process_count()}')

  # This runner assumes ONE process per side. The cross-process machinery it
  # used to carry -- an allgather to collect every process's H2H endpoints for
  # the coordination server, and barriers so the stopwatch bracketed all
  # processes rather than just this one -- has been removed. Without them a
  # multi-process run would publish only process 0's endpoints and time only
  # process 0's transfer, so refuse rather than report a wrong number.
  if jax.process_count() > 1:
    raise RuntimeError(
        f'This runner supports a single process per side, but process_count='
        f'{jax.process_count()}. Restore the multihost allgather/barriers '
        'before running a multi-host sender or receiver.'
    )

  devices = jax.devices('tpu')
  if not devices:
    raise RuntimeError('No TPU devices found.')
  print(f'Initialized JAX. Local TPU chips available: {len(devices)}')

  sorted_devices = sorted(devices, key=lambda d: d.numa_node if hasattr(d, 'numa_node') else -1)
  tpu_sharding, _ = setup_shardings(sorted_devices)

  cache_shape = (_NUM_BLOCKS.value, _BLOCK_SIZE.value, 8, 2, 128)

  # setup_shardings partitions cache axis 2 (8 heads) across every device, so
  # 8 must divide evenly by the device count. Saying so here beats the
  # bare IndivisibleError that XLA raises several frames deeper.
  if 8 % len(devices) != 0:
    raise ValueError(
        'Axis 2 size (8) is not divisible by the '
        f'{len(devices)} local devices it is sharded across. Pass a multiple '
        f'of {len(devices)}.'
    )

  block_bytes = int(np.prod(cache_shape[1:])) * 4
  payload_bytes = _NUM_LAYERS.value * _NUM_BLOCKS.value * block_bytes
  if _STAGING_WINDOW_CHUNKS.value > 0 and _STAGING_CHUNK_BLOCKS.value > 0:
    effective_max_blocks = min(
        _NUM_BLOCKS.value,
        _STAGING_WINDOW_CHUNKS.value * _STAGING_CHUNK_BLOCKS.value,
    )
  else:
    effective_max_blocks = _NUM_BLOCKS.value
  staging_pool_bytes = (
      _NUM_LAYERS.value * effective_max_blocks * _NUM_SLOTS.value * block_bytes
  )
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
  print(
      f'Staging Pool: max_blocks={effective_max_blocks} '
      f'num_slots={_NUM_SLOTS.value} '
      f'total={staging_pool_bytes / (1024 * 1024):.2f} MiB '
      f'({staging_pool_bytes / (1024 ** 3):.3f} GiB)'
  )

  if _ROLE.value == 'sender':
    print('Starting H2H Sender process...')

    t0 = time.perf_counter()
    tpu_src_arrs = populate_deterministic_cache(
        _NUM_BLOCKS.value, _NUM_LAYERS.value, cache_shape, tpu_sharding
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
        max_blocks=effective_max_blocks,
        num_slots=_NUM_SLOTS.value,
        unsafe_skip_buffer_lock=True,
        parallelism=_PARALLELISM.value,
        staging_window_chunks=_STAGING_WINDOW_CHUNKS.value or None,
        staging_chunk_blocks=_STAGING_CHUNK_BLOCKS.value or None,
        staging_chunk_bytes=_STAGING_CHUNK_BYTES.value or None,
        staging_stream_bytes=_STAGING_STREAM_BYTES.value or None,
        staging_admission=_STAGING_ADMISSION.value or None,
    )

    print(f'KVCacheManager ready in {time.perf_counter() - t0:.2f}s')

    transfer_uuid = uuid.uuid4().int & 0xFFFFFFFF
    transfer_req_id = f'perf_test_{transfer_uuid}'

    total_cache_blocks = _NUM_BLOCKS.value
    block_ids = list(range(total_cache_blocks))
    manager.register_read(transfer_req_id, transfer_uuid, block_ids)
    print(f'Armed {len(block_ids)} blocks for read (uuid={transfer_uuid})')

    # Single process per side, so this process's endpoints are the whole set --
    # no allgather, and no packing the JSON into a padded uint8 array to move it
    # through one.
    all_endpoints = manager.get_local_endpoints()

    coordination_server.set_metadata(
        endpoints=all_endpoints,
        transfer_uuid=transfer_uuid,
        transfer_req_id=transfer_req_id,
        block_ids=block_ids,
    )
    print(f'Metadata published! Endpoints: {all_endpoints}, req_id: {transfer_req_id}, uuid: {transfer_uuid}. Waiting for Receiver...')

    t0 = time.perf_counter()
    try:
      coordination_server.wait_for_shutdown()
      print(
          'Receiver finished after '
          f'{time.perf_counter() - t0:.2f}s! Shutting down Sender '
          'coordination server...'
      )
    except KeyboardInterrupt:
      pass
    finally:
      coordination_server.stop()

  elif _ROLE.value == 'receiver':
    print('Starting H2H Receiver process...')

    if not _PEER.value:
      raise ValueError('--peer must be specified for receiver role.')
    resolved_peer = get_peer_grpc_path(_PEER.value)
    print(f'Connecting to peer coordination server at: {resolved_peer}')
    client = coordination_helper.CoordinationClient(server_address=resolved_peer)

    max_retries = 15
    metadata = None
    for attempt in range(1, max_retries + 1):
      try:
        metadata = client.get_metadata()
        break
      except Exception as e:
        print(f'Attempt {attempt}/{max_retries} waiting for peer network server ({e}). Retrying in 5s...')
        time.sleep(5)

    if metadata is None:
      raise RuntimeError(f'Failed to coordinate with peer {resolved_peer} after {max_retries} attempts.')

    src_block_ids = metadata.block_ids
    remote_endpoints = metadata.endpoints
    transfer_uuid = metadata.transfer_uuid
    transfer_req_id = metadata.transfer_req_id

    print(f'Metadata received! Block count: {len(src_block_ids)}')
    print(f'Resolved Peer dynamic H2H endpoints: {remote_endpoints}')

    t0 = time.perf_counter()
    device_arrs = [
        jax.device_put(jnp.empty(cache_shape, dtype=jnp.float32), tpu_sharding)
        for _ in range(_NUM_LAYERS.value)
    ]
    jax.block_until_ready(device_arrs)
    print(
        f'Allocated {_NUM_LAYERS.value} destination layers on device in '
        f'{time.perf_counter() - t0:.2f}s'
    )

    t0 = time.perf_counter()
    manager = kv_cache_manager.KVCacheManager(
        kv_caches=device_arrs,
        local_control_port=0,
        max_blocks=effective_max_blocks,
        num_slots=_NUM_SLOTS.value,
        unsafe_skip_buffer_lock=True,
        parallelism=_PARALLELISM.value,
        staging_window_chunks=_STAGING_WINDOW_CHUNKS.value or None,
        staging_chunk_blocks=_STAGING_CHUNK_BLOCKS.value or None,
        staging_chunk_bytes=_STAGING_CHUNK_BYTES.value or None,
        staging_stream_bytes=_STAGING_STREAM_BYTES.value or None,
        staging_admission=_STAGING_ADMISSION.value or None,
    )
    print(f'KVCacheManager ready in {time.perf_counter() - t0:.2f}s')

    # No barrier before the stopwatch: with one process per side there is
    # nothing to synchronise with, and the pull below is what is being timed.
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
        time.sleep(0.0002)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # start_read is asynchronous, so the split shows how much of the elapsed
    # time was the call itself versus waiting for the transfer to land. The poll
    # loop sleeps 0.2ms per iteration, which quantises short transfers -- if
    # polls is small, the measurement is dominated by that granularity.
    print(
        'start_read returned in '
        f'{(start_read_returned - start_time) * 1e3:.1f} ms; '
        f'waited {(end_time - start_read_returned) * 1e3:.1f} ms across '
        f'{polls} poll(s) at 0.2ms granularity'
    )

    block_byte_size = np.prod(cache_shape[1:]) * 4
    total_bytes = _NUM_LAYERS.value * len(src_block_ids) * block_byte_size

    # One process holds every device, so the local and global figures the
    # multi-host version printed separately were always the same number here.
    total_megabytes = total_bytes / (1024 * 1024)
    bandwidth_gbps = (total_bytes * 8) / (elapsed_time * 1e9)

    print('\n=== H2H E2E Performance Results ===')
    print(f'Parallelism: {_PARALLELISM.value}')
    print(f'Data Volume Transferred: {total_megabytes:.2f} MB')
    print(f'Elapsed Time (TCP H2H + Copy): {elapsed_time:.4f} seconds')
    print(f'Bandwidth: {bandwidth_gbps:.3f} Gbps '
          f'({total_bytes / elapsed_time / 1e9:.3f} GB/s)')
    print('===================================\n')

    if _STAGING_WINDOW_CHUNKS.value > 0:
      stats = manager.staging_stats()
      print('\n=== Host Staging Window Stats ===')
      for k, v in stats.items():
        print(f'{k}: {v}')
      print('=================================\n')

    t0 = time.perf_counter()
    success = verify_deterministic_cache(
        _NUM_BLOCKS.value,
        _NUM_LAYERS.value,
        cache_shape,
        device_arrs,
        tpu_sharding,
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
