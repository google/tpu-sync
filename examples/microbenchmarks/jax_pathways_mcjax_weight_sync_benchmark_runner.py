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

"""Multi-host Pathways (FFI) to McJAX WeightSynchronizer benchmark runner."""

import asyncio
from collections import abc
import functools
import gc
import importlib
import ipaddress
import os
import socket
import time
from typing import Any, Optional

from absl import app
from absl import flags
from absl import logging
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from tpu_sync.api.jax import weight_synchronizer
from tpu_sync.frameworks.jax import weight_synchronizer_ffi as raiden_ffi
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "role",
    None,
    ["controller", "source", "destination"],
    "Role of this process in the weight synchronization benchmark.",
    required=True,
)
flags.DEFINE_string(
    "controller_address",
    "localhost:50051",
    "Host:port address of the centralized RaidenController.",
)
flags.DEFINE_string(
    "pathways_target",
    "",
    "gRPC address of the Pathways proxy server (e.g., grpc://localhost:29000).",
)
flags.DEFINE_string(
    "jax_coordinator_address",
    "",
    "Coordinator host:port for multi-host McJAX distributed initialization.",
)
flags.DEFINE_integer(
    "jax_num_processes",
    1,
    "Number of processes in the multi-host McJAX destination cluster.",
)
flags.DEFINE_integer(
    "jax_process_id",
    0,
    "Process index of this worker in the multi-host McJAX destination cluster.",
)
flags.DEFINE_integer(
    "total_src_devices",
    16,
    "Total number of source TPU devices across all Pathways hosts.",
)
flags.DEFINE_integer(
    "total_dst_devices",
    16,
    "Total number of destination TPU devices across all McJAX hosts.",
)
flags.DEFINE_integer(
    "num_src_hosts",
    2,
    "Number of source hosts in the Pathways cluster.",
)
flags.DEFINE_integer(
    "num_iterations",
    10,
    "Number of benchmarked synchronization iterations (after 1 warmup).",
)
flags.DEFINE_integer(
    "num_variables",
    8,
    "Number of weight variables to synchronize.",
)
flags.DEFINE_list(
    "variable_shape",
    ["8192", "8192"],
    "Global shape of each weight variable.",
)
flags.DEFINE_enum(
    "dtype",
    "bfloat16",
    ["float32", "bfloat16"],
    "Data type of the weight variables.",
)
flags.DEFINE_list(
    "src_sharding_spec",
    ["x", "y"],
    "PartitionSpec axes for source weight variables ('None' for replicated).",
)
flags.DEFINE_list(
    "dst_sharding_spec",
    ["x", "y"],
    "PartitionSpec axes for destination weight variables ('None' for"
    " replicated).",
)
flags.DEFINE_list(
    "src_mesh_shape",
    ["4", "4"],
    "2D mesh shape for the Pathways source cluster.",
)
flags.DEFINE_list(
    "dst_mesh_shape",
    ["4", "4"],
    "2D mesh shape for the McJAX destination cluster.",
)
flags.DEFINE_integer(
    "parallelism",
    8,
    "Number of parallel transport streams for weight synchronization.",
)
flags.DEFINE_list(
    "parallelism_list",
    [],
    "Optional comma-separated list of parallelism values to sweep in a single"
    " Pathways session with an in-process RaidenControllerServer on port"
    " (50050 + p).",
)
flags.DEFINE_string(
    "src_log_dir",
    "",
    "Optional directory to write pw_mcjax_src_p{p}.log files during"
    " --parallelism_list sweeps.",
)

_UNIT_NAME = "benchmark_weights"


def _parse_partition_spec(
    spec_list: abc.Sequence[str],
) -> jax.sharding.PartitionSpec:
  """Parses a list of axis names into a JAX PartitionSpec."""
  parsed = [None if s in ("None", "none", "") else s for s in spec_list]
  return jax.sharding.PartitionSpec(*parsed)


def _create_2d_mesh(
    devices: abc.Sequence[jax.Device],
    mesh_shape: tuple[int, ...],
) -> jax.sharding.Mesh:
  """Creates a 2D JAX Mesh with axis names ('x', 'y')."""
  try:
    mesh_devices = mesh_utils.create_device_mesh(mesh_shape, list(devices))
  except (ValueError, RuntimeError):
    mesh_devices = np.array(list(devices)).reshape(mesh_shape)
  axis_names = ("x", "y")[: len(mesh_shape)]
  return jax.sharding.Mesh(mesh_devices, axis_names)


def _resolve_local_ip() -> str:
  """Resolves the primary routable local IP address."""
  ip = "127.0.0.1"
  try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
      sock.connect(("10.255.255.255", 1))
      ip = sock.getsockname()[0]
    finally:
      sock.close()
  except OSError:
    pass
  if ":" in ip and not ip.startswith("["):
    return f"[{ip}]"
  return ip


def _unpack_ip(row: np.ndarray) -> str:
  """Unpacks an IPv4 or IPv6 address from a 6-int32 FFI info row."""
  raw_bytes = b"".join(
      int(x).to_bytes(4, byteorder="little", signed=True) for x in row[:4]
  )
  if raw_bytes[:10] == b"\x00" * 10 and raw_bytes[10:12] == b"\xff\xff":
    return str(ipaddress.IPv4Address(raw_bytes[12:16]))
  addr_str = str(ipaddress.IPv6Address(raw_bytes))
  return f"[{addr_str}]" if ":" in addr_str else addr_str


def _ping_port(addr: str) -> bool:
  """Returns True if the TCP endpoint 'ip:port' accepts connections."""
  ip, port_str = addr.rsplit(":", 1)
  port = int(port_str)
  if ip.startswith("[") and ip.endswith("]"):
    ip = ip[1:-1]
  try:
    sock = socket.create_connection((ip, port), timeout=1.0)
    sock.close()
    return True
  except OSError:
    return False


def _compute_global_shard_index(
    slice_tuple: Optional[tuple[slice, ...]],
    global_shape: abc.Sequence[int],
    sharding_shape: abc.Sequence[int],
) -> int:
  """Computes canonical row-major global shard index matching nd_slice_math."""
  if not slice_tuple:
    return 0
  global_idx = 0
  stride = 1
  for sl, g_dim, m_dim in zip(
      reversed(slice_tuple), reversed(global_shape), reversed(sharding_shape)
  ):
    if m_dim > 1:
      tile_size = g_dim // m_dim
      start = (
          sl.start
          if isinstance(sl, slice) and sl.start is not None
          else (sl if isinstance(sl, int) else 0)
      )
      coord = start // tile_size if tile_size > 0 else 0
      global_idx += coord * stride
      stride *= m_dim
  return global_idx


def _make_variable_tensor(
    shape: tuple[int, ...],
    sharding: jax.sharding.NamedSharding,
    jnp_dtype: Any,
    var_idx: int,
) -> jax.Array:
  """Allocates a deterministic weight tensor on the given NamedSharding."""
  val = float(var_idx + 1) * 0.125
  return jax.jit(
      lambda: jnp.full(shape, val, dtype=jnp_dtype),
      out_shardings=sharding,
  )()


def run_controller() -> None:
  """Runs the centralized RaidenController server."""
  port = int(FLAGS.controller_address.split(":")[-1])
  logging.info(
      "Starting RaidenControllerServer on port %d (parallelism=%d)...",
      port,
      FLAGS.parallelism,
  )
  worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient(
      name_resolver=None
  )
  controller = raiden_controller.RaidenController(
      port=port,
      worker_rpc_client=worker_rpc_client,
  )
  orig_start_transfer = controller.start_transfer
  controller.start_transfer = functools.partial(
      orig_start_transfer,
      parallelism=FLAGS.parallelism,
      skip_d2h=True,
  )
  server = raiden_controller.RaidenControllerServer(controller)
  server.start()

  start_time = time.time()
  while not server._server._stopped:  # pylint: disable=protected-access
    if time.time() - start_time > 1800.0:
      raise RuntimeError(
          "Controller timed out waiting for benchmark completion"
      )
    time.sleep(0.5)

  loop = asyncio.new_event_loop()
  try:
    loop.run_until_complete(worker_rpc_client.shutdown_workers())
  finally:
    loop.close()
  server.stop()
  logging.info("RaidenControllerServer finished cleanly.")


def _measure_and_report_d2h(
    src_vars: abc.Sequence[jax.Array],
    shard_idx: jax.Array,
    src_mesh: jax.sharding.Mesh,
    total_bytes: int,
    parallelism: int,
    src_log_dir: str = "",
) -> None:
  """Benchmarks Pathways FFI D2H transfers and outputs formatted results."""
  d2h_times: list[float] = []
  for _ in range(FLAGS.num_iterations):
    t0 = time.perf_counter()
    d2h_outs = [
        raiden_ffi.d2h(arr, shard_idx, src_mesh, layer_idx=i)
        for i, arr in enumerate(src_vars)
    ]
    jax.block_until_ready(d2h_outs)
    d2h_times.append(time.perf_counter() - t0)

  total_mb = total_bytes / (1024.0 * 1024.0)
  avg_d2h_time = float(np.mean(d2h_times))
  avg_d2h_bw_gbs = (total_bytes / 1e9) / max(avg_d2h_time, 1e-9)
  avg_d2h_bw_gbps = avg_d2h_bw_gbs * 8.0

  report_lines = [
      "==================================================",
      "SOURCE BENCHMARK RESULTS",
      "==================================================",
      f"Total Size:       {total_mb:.2f} MB",
      f"Parallelism:      {parallelism}",
      f"Avg D2H Time:     {avg_d2h_time:.6f} s",
      (
          f"Avg D2H BW:       {avg_d2h_bw_gbs:.2f} GB/s"
          f" ({avg_d2h_bw_gbps:.2f} Gbps)"
      ),
      "==================================================",
  ]
  report_text = "\n".join(report_lines) + "\n"
  print(report_text, end="", flush=True)
  if src_log_dir:
    os.makedirs(src_log_dir, exist_ok=True)
    with open(
        os.path.join(src_log_dir, f"pw_mcjax_src_p{parallelism}.log"), "w"
    ) as f:
      f.write(report_text)


def run_source() -> None:
  """Runs the Pathways FFI source benchmark worker."""
  if FLAGS.pathways_target:
    os.environ.setdefault("JAX_PLATFORMS", "proxy")
    os.environ["JAX_BACKEND_TARGET"] = FLAGS.pathways_target
    try:
      pw_mod = importlib.import_module("pathwaysutils")
      pw_mod.initialize()
    except ModuleNotFoundError:
      pass
    jax.config.update("jax_platforms", "proxy")
    jax.config.update("jax_backend_target", FLAGS.pathways_target)
    logging.info(
        "Configured Pathways backend target: %s", FLAGS.pathways_target
    )

  for _ in range(60):
    if len(jax.devices()) >= FLAGS.total_src_devices:
      break
    time.sleep(2.0)

  src_devices = jax.devices()[: FLAGS.total_src_devices]
  if len(src_devices) != FLAGS.total_src_devices:
    raise ValueError(
        f"Expected {FLAGS.total_src_devices} source devices, got"
        f" {len(src_devices)}"
    )

  src_mesh_shape = tuple(int(x) for x in FLAGS.src_mesh_shape)
  src_mesh = _create_2d_mesh(src_devices, src_mesh_shape)
  src_pspec = _parse_partition_spec(FLAGS.src_sharding_spec)
  src_sharding = jax.sharding.NamedSharding(src_mesh, src_pspec)

  var_shape = tuple(int(x) for x in FLAGS.variable_shape)
  jnp_dtype = jnp.bfloat16 if FLAGS.dtype == "bfloat16" else jnp.float32

  src_vars = [
      _make_variable_tensor(var_shape, src_sharding, jnp_dtype, i)
      for i in range(FLAGS.num_variables)
  ]
  jax.block_until_ready(src_vars)

  slice_byte_sizes = [
      int(np.prod(arr.sharding.shard_shape(arr.shape))) * arr.dtype.itemsize
      for arr in src_vars
  ]
  sizes_sharding = jax.sharding.NamedSharding(
      src_mesh, jax.sharding.PartitionSpec(None)
  )
  slice_byte_sizes_sharded = jax.device_put(
      jnp.array(slice_byte_sizes, dtype=jnp.int32), sizes_sharding
  )

  devices_per_unit = len(src_devices) // max(1, FLAGS.num_src_hosts)
  global_ids = jnp.arange(src_mesh.devices.size, dtype=jnp.int32).reshape(
      src_mesh_shape
  )
  shard_idx = jax.device_put(
      global_ids,
      jax.sharding.NamedSharding(
          src_mesh, jax.sharding.PartitionSpec(*src_mesh.axis_names)
      ),
  )
  source_host_subgrid, _ = raiden_controller.compute_host_subgrid(
      src_mesh_shape, devices_per_unit
  )

  p_values = (
      [int(x) for x in FLAGS.parallelism_list]
      if FLAGS.parallelism_list
      else [FLAGS.parallelism]
  )
  init_parallelism = max(p_values)

  # Warmup and initialize C++ WeightSynchronizer FFI ONCE across all Pathways
  # hosts (with max parallelism so push_pool_ has enough threads for all rungs).
  src_ws_info = raiden_ffi.init_weight_synchronizer_and_d2h(
      device_arrays=src_vars,
      shard_idx=shard_idx,
      mesh=src_mesh,
      slice_byte_sizes=slice_byte_sizes_sharded,
      parallelism=init_parallelism,
      num_layers=len(src_vars),
      listener_port=0,
      num_shards=devices_per_unit,
      host_subgrid=source_host_subgrid,
  )
  jax.block_until_ready(src_ws_info)

  gathered_ws_info = np.asarray(jax.device_get(src_ws_info)).reshape(-1, 6)
  ips: list[str] = []
  listeners: list[str] = []
  for row in gathered_ws_info:
    ip = _unpack_ip(row)
    ips.append(f"{ip}:{row[4]}")
    listeners.append(f"{ip}:{row[5]}")

  unique_listeners: list[str] = []
  for listener in listeners:
    if listener not in unique_listeners:
      unique_listeners.append(listener)

  if len(unique_listeners) != FLAGS.num_src_hosts:
    raise RuntimeError(
        f"Expected {FLAGS.num_src_hosts} unique Pathways host listeners, found"
        f" {len(unique_listeners)}: {unique_listeners}"
    )

  mesh_devices_flat = list(src_mesh.devices.flatten())
  host_units_metadata: list[
      tuple[
          raiden_controller.RaidenId,
          list[str],
          str,
          list[raiden_service_pb2.VariableMetadataProto],
      ]
  ] = []
  for task_idx, listener_addr in enumerate(unique_listeners):
    task_devices = [
        mesh_devices_flat[i]
        for i, l in enumerate(listeners)
        if l == listener_addr
    ]
    task_ips = [ips[i] for i, l in enumerate(listeners) if l == listener_addr]
    task_var_protos = []
    for var_idx, arr in enumerate(src_vars):
      sharding = arr.sharding
      local_shard_shape = sharding.shard_shape(var_shape)
      sharding_shape = [g // l for g, l in zip(var_shape, local_shard_shape)]
      layout = tuple(range(len(var_shape) - 1, -1, -1))
      device_to_slice = sharding.devices_indices_map(var_shape)
      global_shard_indices = [
          _compute_global_shard_index(
              device_to_slice.get(dev), var_shape, sharding_shape
          )
          for dev in task_devices
      ]
      task_var_protos.append(
          raiden_service_pb2.VariableMetadataProto(
              name=f"var_{var_idx}",
              shape=var_shape,
              mesh_shape=sharding_shape,
              layout=layout,
              item_size=arr.dtype.itemsize,
              layer_idx=var_idx,
              global_shard_indices=global_shard_indices,
          )
      )

    task_unit = raiden_controller.RaidenId(
        "pathways_trainer", str(task_idx), _UNIT_NAME
    )
    host_units_metadata.append(
        (task_unit, task_ips, listener_addr, task_var_protos)
    )

  for listener_addr in unique_listeners:
    while not _ping_port(listener_addr):
      time.sleep(0.2)

  total_bytes = sum(
      int(np.prod(arr.shape)) * arr.dtype.itemsize for arr in src_vars
  )

  if FLAGS.parallelism_list:
    worker_rpc_client = None
    for p in p_values:
      worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient(
          name_resolver=None
      )
      if p != p_values[-1]:

        async def _noop_shutdown(
            *unused_args: Any, **unused_kwargs: Any
        ) -> None:
          return None

        worker_rpc_client.shutdown_workers = _noop_shutdown
      _measure_and_report_d2h(
          src_vars=src_vars,
          shard_idx=shard_idx,
          src_mesh=src_mesh,
          total_bytes=total_bytes,
          parallelism=p,
          src_log_dir=FLAGS.src_log_dir,
      )

      ctrl_port = 50050 + p
      controller = raiden_controller.RaidenController(
          port=ctrl_port,
          worker_rpc_client=worker_rpc_client,
      )
      orig_start_transfer = controller.start_transfer
      controller.start_transfer = functools.partial(
          orig_start_transfer,
          parallelism=p,
          skip_d2h=True,
      )
      server = raiden_controller.RaidenControllerServer(controller)
      server.start()

      ctrl_client = raiden_controller.RaidenControllerClientFacade(
          f"127.0.0.1:{ctrl_port}",
          name_resolver=None,
      )
      for (
          task_unit,
          task_ips,
          listener_addr,
          task_var_protos,
      ) in host_units_metadata:
        ctrl_client.register_work_unit(
            task_unit,
            task_ips,
            listener_addr,
            variables=task_var_protos,
        )

      start_time = time.time()
      while not server._server._stopped:  # pylint: disable=protected-access
        if time.time() - start_time > 1800.0:
          raise RuntimeError(
              f"Controller timed out waiting for parallelism={p} completion"
          )
        time.sleep(0.5)

      server.stop()
      time.sleep(2.0)

    if worker_rpc_client is not None:
      loop = asyncio.new_event_loop()
      try:
        loop.run_until_complete(worker_rpc_client.shutdown_workers())
      finally:
        loop.close()
    raiden_ffi.destroy_weight_synchronizer()
    return

  # Single-parallelism mode with external controller_address.
  _measure_and_report_d2h(
      src_vars=src_vars,
      shard_idx=shard_idx,
      src_mesh=src_mesh,
      total_bytes=total_bytes,
      parallelism=FLAGS.parallelism,
      src_log_dir=FLAGS.src_log_dir,
  )

  ctrl_client = raiden_controller.RaidenControllerClientFacade(
      FLAGS.controller_address,
      name_resolver=None,
  )
  for (
      task_unit,
      task_ips,
      listener_addr,
      task_var_protos,
  ) in host_units_metadata:
    ctrl_client.register_work_unit(
        task_unit,
        task_ips,
        listener_addr,
        variables=task_var_protos,
    )

  while any(_ping_port(listener_addr) for listener_addr in unique_listeners):
    time.sleep(0.5)

  raiden_ffi.destroy_weight_synchronizer()


def run_destination() -> None:
  """Runs the multi-host McJAX destination benchmark worker."""
  if FLAGS.jax_coordinator_address and FLAGS.jax_num_processes > 1:
    jax.distributed.initialize(
        coordinator_address=FLAGS.jax_coordinator_address,
        num_processes=FLAGS.jax_num_processes,
        process_id=FLAGS.jax_process_id,
    )

  dst_devices = jax.devices()[: FLAGS.total_dst_devices]
  if len(dst_devices) != FLAGS.total_dst_devices:
    raise ValueError(
        f"Expected {FLAGS.total_dst_devices} destination devices, got"
        f" {len(dst_devices)}"
    )

  dst_mesh_shape = tuple(int(x) for x in FLAGS.dst_mesh_shape)
  dst_mesh = _create_2d_mesh(dst_devices, dst_mesh_shape)
  dst_pspec = _parse_partition_spec(FLAGS.dst_sharding_spec)
  dst_sharding = jax.sharding.NamedSharding(dst_mesh, dst_pspec)

  var_shape = tuple(int(x) for x in FLAGS.variable_shape)
  jnp_dtype = jnp.bfloat16 if FLAGS.dtype == "bfloat16" else jnp.float32

  dst_vars = [
      jax.jit(
          lambda: jnp.zeros(var_shape, dtype=jnp_dtype),
          out_shardings=dst_sharding,
      )()
      for _ in range(FLAGS.num_variables)
  ]
  jax.block_until_ready(dst_vars)

  local_shards = list(dst_vars[0].addressable_shards)
  num_local_dev = len(local_shards)

  variable_protos = []
  for var_idx, arr in enumerate(dst_vars):
    sharding = arr.sharding
    local_shard_shape = sharding.shard_shape(var_shape)
    sharding_shape = [g // l for g, l in zip(var_shape, local_shard_shape)]
    layout = tuple(range(len(var_shape) - 1, -1, -1))
    device_to_slice = sharding.devices_indices_map(var_shape)
    dest_global_shard_indices = [
        _compute_global_shard_index(
            device_to_slice.get(s.device), var_shape, sharding_shape
        )
        for s in arr.addressable_shards
    ]
    variable_protos.append(
        raiden_service_pb2.VariableMetadataProto(
            name=f"var_{var_idx}",
            shape=var_shape,
            mesh_shape=sharding_shape,
            layout=layout,
            item_size=arr.dtype.itemsize,
            layer_idx=var_idx,
            global_shard_indices=dest_global_shard_indices,
        )
    )

  ws = weight_synchronizer.WeightSynchronizer(
      dst_vars,
      local_port=0,
      listener_port=0,
      unsafe_skip_buffer_lock=False,
      parallelism=FLAGS.parallelism,
      auto_h2d=False,
  )

  self_ip = _resolve_local_ip()
  raw_eps = ws.get_local_endpoints()
  shards = [""] * num_local_dev
  if raw_eps:
    flat_devices = list(dst_mesh.devices.flat)
    for i, s in enumerate(local_shards):
      g_idx = flat_devices.index(s.device)
      for ep_info in raw_eps:
        ep_shards = ep_info.get("shards") or []
        ep_global_shards = ep_info.get("global_shards") or ep_shards
        if i in ep_shards or g_idx in ep_global_shards:
          shards[i] = str(ep_info["endpoint"])
          break
      if not shards[i]:
        sub_idx = min(i * len(raw_eps) // num_local_dev, len(raw_eps) - 1)
        shards[i] = str(raw_eps[sub_idx]["endpoint"])
  else:
    shards = [f"{self_ip}:{ws.local_port}"] * num_local_dev

  ctrl_client = raiden_controller.RaidenControllerClientFacade(
      FLAGS.controller_address,
      name_resolver=None,
  )
  unit_id = raiden_controller.RaidenId(
      "mcjax_sampler", str(jax.process_index()), _UNIT_NAME
  )
  ctrl_client.register_work_unit(
      unit_id,
      shards,
      f"{self_ip}:{ws.listener_port}",
      variables=variable_protos,
  )

  # Wait until all source and destination work units are registered.
  src_units: list[raiden_controller.RaidenId] = []
  dst_units: list[raiden_controller.RaidenId] = []
  wait_start = time.time()
  while True:
    metadata_list = ctrl_client.get_metadata()
    src_meta = [
        m for m in metadata_list if m.unit.job_name == "pathways_trainer"
    ]
    dst_meta = [m for m in metadata_list if m.unit.job_name == "mcjax_sampler"]
    src_count = sum(len(m.shards) for m in src_meta)
    dst_count = sum(len(m.shards) for m in dst_meta)
    if (
        src_count >= FLAGS.total_src_devices
        and dst_count >= FLAGS.total_dst_devices
        and src_meta
        and dst_meta
    ):
      src_units = sorted(
          [
              raiden_controller.RaidenId(
                  m.unit.job_name,
                  m.unit.job_replica_id,
                  m.unit.data_name,
                  m.unit.data_replica_idx,
              )
              for m in src_meta
          ],
          key=lambda u: int(u.job_replica_id),
      )
      dst_units = sorted(
          [
              raiden_controller.RaidenId(
                  m.unit.job_name,
                  m.unit.job_replica_id,
                  m.unit.data_name,
                  m.unit.data_replica_idx,
              )
              for m in dst_meta
          ],
          key=lambda u: int(u.job_replica_id),
      )
      break
    if time.time() - wait_start > 1800.0:
      raise RuntimeError(
          f"Timed out waiting for work units: src={src_count}, dst={dst_count}"
      )
    time.sleep(0.5)

  if jax.process_count() > 1:
    multihost_utils.sync_global_devices("all_units_registered")

  net_times: list[float] = []
  h2d_times: list[float] = []

  for it in range(FLAGS.num_iterations + 1):
    sync_uuid = 1000 + it
    if jax.process_count() > 1:
      multihost_utils.sync_global_devices(f"sync_iter_{it}_start")

    t_net_start = time.perf_counter()
    if jax.process_index() == 0:
      ctrl_client.coordinate_transfer(
          src_units=src_units,
          dst_units=dst_units,
          req_id=f"sync_{it}",
          use_block_chunks=True,
          uuid=sync_uuid,
      )
    ws.wait_for_transfer_completion(sync_uuid)
    if jax.process_count() > 1:
      multihost_utils.sync_global_devices(f"sync_iter_{it}_net_done")
    net_duration_s = time.perf_counter() - t_net_start

    t_h2d_start = time.perf_counter()
    ws.h2d()
    for arr in dst_vars:
      arr.block_until_ready()
    if jax.process_count() > 1:
      multihost_utils.sync_global_devices(f"sync_iter_{it}_h2d_done")
    h2d_duration_s = time.perf_counter() - t_h2d_start

    if it == 0:
      for var_idx, arr in enumerate(dst_vars):
        expected_val = float(var_idx + 1) * 0.125
        for shard in arr.addressable_shards:
          if not bool(jnp.allclose(shard.data, expected_val, atol=1e-3)):
            raise AssertionError(
                f"Weight verification failed on variable {var_idx}!"
            )
    else:
      net_times.append(net_duration_s)
      h2d_times.append(h2d_duration_s)

  total_bytes = sum(
      int(np.prod(arr.shape)) * arr.dtype.itemsize for arr in dst_vars
  )
  total_mb = total_bytes / (1024.0 * 1024.0)
  avg_net_time = float(np.mean(net_times))
  avg_net_bw_gbs = (total_bytes / 1e9) / max(avg_net_time, 1e-9)
  avg_net_bw_gbps = avg_net_bw_gbs * 8.0
  avg_h2d_time = float(np.mean(h2d_times))
  avg_h2d_bw_gbs = (total_bytes / 1e9) / max(avg_h2d_time, 1e-9)
  avg_h2d_bw_gbps = avg_h2d_bw_gbs * 8.0

  if jax.process_index() == 0:
    print("==================================================")
    print("DESTINATION BENCHMARK RESULTS")
    print("==================================================")
    print(f"Total Size:           {total_mb:.2f} MB")
    print(f"Parallelism:          {FLAGS.parallelism}")
    print(f"Avg Net Transfer Time:{avg_net_time:.6f} s")
    print(
        f"Avg Net Transfer BW:  {avg_net_bw_gbs:.2f} GB/s"
        f" ({avg_net_bw_gbps:.2f} Gbps)"
    )
    print(f"Avg H2D Time:         {avg_h2d_time:.6f} s")
    print(
        f"Avg H2D BW:           {avg_h2d_bw_gbs:.2f} GB/s"
        f" ({avg_h2d_bw_gbps:.2f} Gbps)"
    )
    print("==================================================", flush=True)
    ctrl_client.shutdown()

  if jax.process_count() > 1:
    multihost_utils.sync_global_devices("destination_benchmark_complete")
  del ws
  gc.collect()


def main(argv: abc.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if FLAGS.role == "controller":
    run_controller()
  elif FLAGS.role == "source":
    run_source()
  elif FLAGS.role == "destination":
    run_destination()
  else:
    raise ValueError(f"Unknown role: {FLAGS.role}")


if __name__ == "__main__":
  app.run(main)
