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

The geometry flags must match on both sides, and only one process per side is
supported.

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
_COORDINATOR = flags.DEFINE_string(
    'coordinator', None, 'JAX coordinator address'
)
_PROCESS_INDEX = flags.DEFINE_integer('process_index', -1, 'JAX process index')
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_processes', 1, 'Number of JAX processes.'
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


def _resolve_dtype(dtype_name: str) -> tuple[jnp.dtype, int]:
  """Resolves the cache dtype name to a jnp.dtype and item size in bytes."""
  normalized = dtype_name.lower()
  if normalized in ('float32', 'fp32'):
    return jnp.dtype(jnp.float32), 4
  elif normalized in ('bfloat16', 'bf16'):
    return jnp.dtype(jnp.bfloat16), 2
  elif normalized in ('fp8', 'float8', 'float8_e4m3fn'):
    return jnp.dtype(jnp.float8_e4m3fn), 1
  else:
    raise ValueError(
        f'Unsupported dtype: {dtype_name}. Expected float32, bfloat16, or fp8.'
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
    raise ValueError('A peer address MUST be provided (e.g. 10.128.0.2:50051)')
  # ensure ipv6 is bracketed if passed (basic check)
  if ':' in peer_arg and not peer_arg.startswith('['):
    parts = peer_arg.rsplit(':', 1)
    if len(parts) == 2 and '.' not in parts[0]: 
      return f'[{parts[0]}]:{parts[1]}'
  return peer_arg


def _make_layer_fn(shape: tuple[int, ...], sharding, dtype: jnp.dtype):
  """Builds a jitted layer generator with low dynamic range values for FP8."""

  @functools.partial(jax.jit, out_shardings=sharding)
  def _build_layer(layer_offset: jax.Array) -> jax.Array:
    # Use modulo arithmetic to ensure values remain well within the finite
    # dynamic range of low-precision dtypes such as float8_e4m3fn (max 448.0)
    # without overflowing to NaN, while keeping values distinct.
    b = (jax.lax.broadcasted_iota(jnp.float32, shape, 0) % 16.0) * 0.5
    t = (jax.lax.broadcasted_iota(jnp.float32, shape, 1) % 64.0) * 0.25
    h = (jax.lax.broadcasted_iota(jnp.float32, shape, 2) % 8.0) * 1.0
    kv = (jax.lax.broadcasted_iota(jnp.float32, shape, 3) % 2.0) * 2.0
    d = (jax.lax.broadcasted_iota(jnp.float32, shape, 4) % 128.0) * (
        1.0 / 128.0
    )
    arr = b + t + h + kv + d + (layer_offset % 32.0)
    return arr.astype(dtype)

  return _build_layer


def populate_deterministic_cache(
    num_layers: int, shape: tuple[int, ...], sharding, dtype: jnp.dtype
) -> list[jax.Array]:
  """Populates deterministic cache layers on device."""
  build_layer = _make_layer_fn(shape, sharding, dtype)
  arrs = [
      build_layer(jnp.float32(layer_idx)) for layer_idx in range(num_layers)
  ]
  jax.block_until_ready(arrs)
  return arrs


@jax.jit
def _layers_equal(actual: jax.Array, expected: jax.Array) -> jax.Array:
  """Computes exact on-device equality between two arrays."""
  return jnp.all(actual == expected)


def verify_deterministic_cache(
    num_layers: int,
    shape: tuple[int, ...],
    dst_tpu_arrs: list[jax.Array],
    sharding,
    dtype: jnp.dtype,
) -> bool:
  """Verifies cache layers match expected deterministic content on device."""
  print('Verifying data consistency across all sharded cache layers...')
  build_layer = _make_layer_fn(shape, sharding, dtype)
  for layer_idx in range(num_layers):
    expected = build_layer(jnp.float32(layer_idx))
    if not bool(_layers_equal(dst_tpu_arrs[layer_idx], expected)):
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
  elif not jax.distributed.is_initialized():
    try:
      jax.distributed.initialize()
    except ValueError:
      pass

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

  # setup_shardings partitions cache axis 2 (size 8) across every device along
  # 'model', so 8 must divide evenly by the device count.
  if 8 % len(devices) != 0:
    raise ValueError(
        'Axis 2 (size 8) is not divisible by the '
        f'{len(devices)} local devices it is sharded across.'
    )

  cache_dtype, itemsize = _resolve_dtype(_DTYPE.value)
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
    print('Starting H2H Sender process...')

    t0 = time.perf_counter()
    tpu_src_arrs = populate_deterministic_cache(
        _NUM_LAYERS.value, cache_shape, tpu_sharding, cache_dtype
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
        host_blocks_to_allocate=_HOST_BLOCKS_TO_ALLOCATE.value,
        window_blocks=_WINDOW_BLOCKS.value,
        window_depth=_WINDOW_DEPTH.value,
    )

    print(f'KVCacheManager ready in {time.perf_counter() - t0:.2f}s')

    transfer_uuid = uuid.uuid4().int & 0xFFFFFFFF
    transfer_req_id = f'perf_test_{transfer_uuid}'

    total_cache_blocks = _NUM_BLOCKS.value
    block_ids = list(range(total_cache_blocks))
    total_runs = _WARMUP.value + _ITERS.value
    for run_idx in range(total_runs):
      iter_req_id = f'{transfer_req_id}_{run_idx}'
      iter_uuid = transfer_uuid + run_idx * 10000
      manager.register_read(iter_req_id, iter_uuid, block_ids)
    print(
        f'Armed {len(block_ids)} blocks for read across {total_runs} runs '
        f'(base uuid={transfer_uuid})'
    )

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
    if _PEER.value is None:
      raise ValueError('A peer address MUST be provided')

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
    alloc_zero_layer = jax.jit(
        lambda: jnp.zeros(cache_shape, dtype=cache_dtype),
        out_shardings=tpu_sharding,
    )
    device_arrs = [alloc_zero_layer() for _ in range(_NUM_LAYERS.value)]
    jax.block_until_ready(device_arrs)
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
        host_blocks_to_allocate=_HOST_BLOCKS_TO_ALLOCATE.value,
        window_blocks=_WINDOW_BLOCKS.value,
        window_depth=_WINDOW_DEPTH.value,
    )
    print(f'KVCacheManager ready in {time.perf_counter() - t0:.2f}s')

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
      manager.start_read(
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
        _, done_recving, failed_recving = manager.poll_stats()
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
    _, final_done, final_failed = manager.poll_stats()
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
        _NUM_LAYERS.value,
        cache_shape,
        device_arrs,
        tpu_sharding,
        cache_dtype,
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
