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

"""Dynamic performance test runner tool for raw memory offloading collectives."""

import os
import sys

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)

import json
import time
import uuid
from absl import app
from absl import flags
import jax
from jax.experimental import multihost_utils as jax_multihost_utils
import jax.numpy as jnp
import numpy as np
from tpu_sync.api.jax import kv_cache_manager
from tpu_sync.rpc import coordination_helper

_ROLE = flags.DEFINE_string(
    'role', None, 'Role of the task: sender or receiver.'
)
_PEER = flags.DEFINE_string(
    'peer', None, 'Dynamic gRPC address of the peer (for receiver).'
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


def format_ipv6_endpoint(address_str: str) -> str:
  """Encloses raw IPv6 address in square brackets [] for gRPC/sockets."""
  if ':' in address_str and not address_str.startswith('['):
    parts = address_str.rsplit(':', 1)
    if (
        len(parts) == 2 and '.' not in parts[0]
    ):  # Check if it is an IPv6 hex address
      return f'[{parts[0]}]:{parts[1]}'
  return address_str


def get_peer_grpc_bns_path(peer_arg: str) -> str:
  """Constructs the FQDN google:///bns/... path dynamically."""
  if 'google://' in peer_arg:
    return peer_arg

  author = os.environ.get('XM_EXPERIMENT_AUTHOR')
  xid = os.environ.get('XM_XID')
  wid = os.environ.get('XM_WID', '1')
  chubby_cell = os.environ.get('LOCKSERVICE_LOCALITY', 'ok')
  cell = os.environ.get('BORG_CELL', chubby_cell)

  if not author or not xid:
    # Fallback for local workstation testing
    return peer_arg

  peer_role = peer_arg.split(':')[0]
  peer_port_name = 'grpc'  # Pre-allocated port name registered in BNS

  bns_path = f'/bns/{chubby_cell}/borg/{cell}/bns/{author}/{author}_group_{xid}.{wid}.{peer_role}/0:{peer_port_name}'
  resolved_peer_bns = f'google:///{bns_path}'
  print(f'Dynamic peer BNS coordinate resolved: {resolved_peer_bns}')
  return resolved_peer_bns


def populate_deterministic_cache(
    num_blocks: int, num_layers: int, shape: tuple[int, ...], sharding
) -> list[jax.Array]:
  """Populates unique deterministic sequence data across layers on Sender."""
  arrs = []
  for layer_idx in range(num_layers):
    # Generate unique float32 data sequence based on layer index
    base = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + float(
        layer_idx * 1000.0
    )
    arrs.append(jax.device_put(base, sharding))
  jax.block_until_ready(arrs)
  return arrs


def verify_deterministic_cache(
    num_blocks: int,
    num_layers: int,
    shape: tuple[int, ...],
    dst_tpu_arrs: list[jax.Array],
) -> bool:
  """Verifies Receiver's pulled arrays contain exact sequence from Sender."""
  print('Verifying data consistency across all sharded cache layers...')
  for layer_idx in range(num_layers):
    expected = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(
        shape
    ) + float(layer_idx * 1000.0)
    actual = np.asarray(dst_tpu_arrs[layer_idx])
    try:
      np.testing.assert_array_equal(actual, np.asarray(expected))
    except AssertionError as exc:
      print(f'Verification FAILED on Layer {layer_idx}!')
      print(exc)
      return False
  print('Data consistency verified successfully! 0% corruption.')
  return True


def main(_):
  if not _ROLE.value:
    raise ValueError('--role must be specified')

  if _ENABLE_METRICS.value:
    os.environ['ENABLE_RAIDEN_METRICS'] = 'true'

  # Set up JAX and initialize the mesh for distributed TPU environment
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
  print(
      f'Initialized JAX distributed mesh: process_index={jax.process_index()},'
      f' process_count={jax.process_count()}'
  )

  devices = jax.devices('tpu')
  if not devices:
    raise RuntimeError('No TPU devices found.')
  print(f'Initialized JAX. Local TPU chips available: {len(devices)}')

  sorted_devices = sorted(
      devices, key=lambda d: d.numa_node if hasattr(d, 'numa_node') else -1
  )
  tpu_sharding, _ = setup_shardings(sorted_devices)

  # Physical sharding shape matching kv_cache_manager layouts
  # Layout: (num_blocks, head_count, block_size, head_dim)
  # Shape: (_NUM_BLOCKS, 32, _BLOCK_SIZE, 8, 128)
  cache_shape = (_NUM_BLOCKS.value, 32, _BLOCK_SIZE.value, 8, 128)

  if _ROLE.value == 'sender':
    print('Starting H2H Sender process...')

    # Populate unique cache data in TPU memory
    tpu_src_arrs = populate_deterministic_cache(
        _NUM_BLOCKS.value, _NUM_LAYERS.value, cache_shape, tpu_sharding
    )

    # Start background coordination server on Borg-allocated named port
    grpc_port_env = os.environ.get('BORG_PORT_GRPC')
    if grpc_port_env:
      grpc_port = int(grpc_port_env)
      print(f'Borg-allocated named port grpc detected: {grpc_port}')
    else:
      grpc_port = _GRPC_PORT.value
      print(
          'WARNING: BORG_PORT_GRPC not found. Falling back to flag default:'
          f' {grpc_port}'
      )

    if jax.process_index() == 0:
      coordination_server = coordination_helper.CoordinationServer(
          port=grpc_port
      )
      bound_grpc_port = coordination_server.start()
      print(f'Coordination gRPC server started on port: {bound_grpc_port}')
    else:
      coordination_server = None

    # Start KVCacheManager
    manager = kv_cache_manager.KVCacheManager(
        kv_caches=tpu_src_arrs,
        local_control_port=0,
        max_blocks=_NUM_BLOCKS.value,
        num_slots=_NUM_SLOTS.value,
        unsafe_skip_buffer_lock=True,
        parallelism=_PARALLELISM.value,
    )

    # Generate transfer ID and UUID
    transfer_uuid = uuid.uuid4().int & 0xFFFFFFFF
    transfer_req_id = f'perf_test_{transfer_uuid}'

    # Register the read operation
    total_cache_blocks = _NUM_BLOCKS.value
    block_ids = list(range(total_cache_blocks))
    manager.register_read(transfer_req_id, transfer_uuid, block_ids)

    # Get local endpoints
    local_endpoints = manager.get_local_endpoints()

    # Gather endpoints from all hosts to process 0
    local_eps_str = json.dumps(local_endpoints)
    max_len = 4096
    local_eps_bytes = np.zeros(max_len, dtype=np.uint8)
    encoded = local_eps_str.encode('utf-8')
    local_eps_bytes[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)

    local_eps_jax = jnp.array(local_eps_bytes)
    gathered_eps_jax = jax_multihost_utils.process_allgather(local_eps_jax)

    if coordination_server is not None:
      all_endpoints = []
      gathered_eps_np = np.array(gathered_eps_jax)
      for i in range(jax.process_count()):
        str_i = gathered_eps_np[i].tobytes().decode('utf-8').strip('\x00')
        if str_i:
          all_endpoints.extend(json.loads(str_i))

      coordination_server.set_metadata(
          endpoints=all_endpoints,
          transfer_uuid=transfer_uuid,
          transfer_req_id=transfer_req_id,
          block_ids=block_ids,
      )
      print(
          f'Metadata published! Endpoints: {all_endpoints}, '
          f'req_id: {transfer_req_id}, uuid: {transfer_uuid}. '
          'Waiting for Receiver...'
      )

    # Block cleanly until peer Receiver signals transfer completion
    if coordination_server is not None:
      try:
        coordination_server.wait_for_shutdown()
        print('Receiver finished! Shutting down Sender coordination server...')
      except KeyboardInterrupt:
        pass
      finally:
        coordination_server.stop()

    # Global barrier to synchronize all sender hosts
    jax_multihost_utils.sync_global_devices('sender_done')

    if _ENABLE_METRICS.value:
      metrics_str = manager.dump_metrics_to_string()
      print(
          f'\n[Host {jax.process_index()}] --- Detailed Telemetry Metrics ---'
      )
      print(metrics_str)
      with open('/tmp/raiden_metrics.log', 'w') as f:
        f.write(metrics_str)

  elif _ROLE.value == 'receiver':
    print('Starting H2H Receiver process...')
    # Connect and pull dynamic metadata (coordination via native BNS resolving)
    if _PEER.value is None:
      raise ValueError('Receiver role requires a peer address flag (--peer).')
    resolved_bns_peer = get_peer_grpc_bns_path(_PEER.value)
    print(f'Connecting to peer coordination BNS: {resolved_bns_peer}')
    client = coordination_helper.CoordinationClient(
        server_address=resolved_bns_peer
    )

    max_retries = 15
    metadata = None
    for attempt in range(1, max_retries + 1):
      try:
        metadata = client.get_metadata()
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            f'Attempt {attempt}/{max_retries} waiting for peer BNS registration'
            f' ({e}). Retrying in 5s...'
        )
        time.sleep(5)

    if metadata is None:
      raise RuntimeError(
          f'Failed to coordinate with peer {resolved_bns_peer} after'
          f' {max_retries} attempts.'
      )

    src_block_ids = metadata.block_ids
    remote_endpoints = metadata.endpoints
    transfer_uuid = metadata.transfer_uuid
    transfer_req_id = metadata.transfer_req_id

    print(f'Metadata received! Block count: {len(src_block_ids)}')
    print(f'Resolved Peer dynamic H2H endpoints: {remote_endpoints}')

    # Setup local empty device arrays
    device_arrs = [
        jax.device_put(jnp.empty(cache_shape, dtype=jnp.float32), tpu_sharding)
        for _ in range(_NUM_LAYERS.value)
    ]
    jax.block_until_ready(device_arrs)

    # Initialize local KVCacheManager
    manager = kv_cache_manager.KVCacheManager(
        kv_caches=device_arrs,
        local_control_port=0,
        max_blocks=_NUM_BLOCKS.value,
        num_slots=_NUM_SLOTS.value,
        unsafe_skip_buffer_lock=True,
        parallelism=_PARALLELISM.value,
    )

    # Synchronize all receiver hosts before starting transfer
    jax_multihost_utils.sync_global_devices('receiver_transfer_start')

    # Measure E2E transfer performance
    print('Executing H2H Read E2E offloading transfer...')
    start_time = time.perf_counter()

    # Kick off the transfer asynchronously
    manager.start_read(
        req_id=transfer_req_id,
        uuid=transfer_uuid,
        remote_endpoint=remote_endpoints,
        remote_block_ids=src_block_ids,
        local_block_ids=src_block_ids,
        parallelism=_PARALLELISM.value,
    )

    # Robust 10ms polling loop checking poll_stats()
    completed = False
    while not completed:
      _, done_recving, failed_recving = manager.poll_stats()
      if transfer_req_id in done_recving:
        completed = True
      elif transfer_req_id in failed_recving:
        raise RuntimeError(f'Transfer failed! req_id: {transfer_req_id}')
      else:
        time.sleep(0.01)

    # Synchronize all receiver hosts after completing transfer
    jax_multihost_utils.sync_global_devices('receiver_transfer_end')
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Calculate metrics
    # Total data size transferred: layer_count * block_count * block_byte_size
    # Cache block size is float32 (4 bytes)
    block_byte_size = np.prod(cache_shape[1:]) * 4
    total_bytes = _NUM_LAYERS.value * len(src_block_ids) * block_byte_size

    local_bytes = total_bytes * (len(jax.local_devices()) / len(jax.devices()))
    local_megabytes = local_bytes / (1024 * 1024)
    local_bandwidth_gbps = (local_bytes * 8) / (elapsed_time * 1e9)

    global_megabytes = total_bytes / (1024 * 1024)
    global_bandwidth_gbps = (total_bytes * 8) / (elapsed_time * 1e9)

    print(f'\n[Host {jax.process_index()}] --- Local Performance Results ---')
    print(f'Local Data Volume Transferred: {local_megabytes:.2f} MB')
    print(f'Elapsed Time (TCP H2H + Copy): {elapsed_time:.4f} seconds')
    print(f'Local NIC Bandwidth: {local_bandwidth_gbps:.3f} Gbps')
    print('--------------------------------------------------\n')

    if jax.process_index() == 0:
      print('\n=== Global Aggregate Performance Results ===')
      print(f'Total Data Volume Transferred: {global_megabytes:.2f} MB')
      print(f'Elapsed Time (True Parallel E2E): {elapsed_time:.4f} seconds')
      print(f'Global Aggregate Bandwidth: {global_bandwidth_gbps:.3f} Gbps')
      print('============================================\n')

    # Validate data consistency
    success = verify_deterministic_cache(
        _NUM_BLOCKS.value, _NUM_LAYERS.value, cache_shape, device_arrs
    )

    # Synchronize receivers right before shutdown
    jax_multihost_utils.sync_global_devices('receiver_shutdown_barrier')

    if not success:
      # Signal Sender to exit even on failure to avoid dangling resources
      if jax.process_index() == 0:
        client.shutdown()
      sys.exit(1)

    # Signal successful transfer complete to unblock Sender
    if jax.process_index() == 0:
      print('Signalling completion to peer Sender...')
      client.shutdown()
    print('E2E performance test runner completed successfully!')

    if _ENABLE_METRICS.value:
      metrics_str = manager.dump_metrics_to_string()
      print(
          f'\n[Host {jax.process_index()}] --- Detailed Telemetry Metrics ---'
      )
      print(metrics_str)
      with open('/tmp/raiden_metrics.log', 'w') as f:
        f.write(metrics_str)

  else:
    raise ValueError(f'Unknown role: {_ROLE.value}')


if __name__ == '__main__':
  # Bypasses supervisor-injected flags on physical TPU nodes
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
