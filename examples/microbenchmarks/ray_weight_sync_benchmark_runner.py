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

"""Multi-slice Ray PyTorch WeightSynchronizer benchmark runner."""

import argparse
from collections import abc
import datetime
import functools
from importlib import metadata
import json
import math
import os
import pathlib
import re
import socket
import sys
import time
from typing import Any, NamedTuple
import numpy as np
try:
  from tpu_sync.api import weight_synchronizer_manager
  from tpu_sync.rpc import raiden_controller
  from tpu_sync.rpc import raiden_service_pb2
except ImportError:
  from tpu_sync.api import weight_synchronizer_manager
  from tpu_sync.rpc import raiden_controller
  from tpu_sync.rpc import raiden_service_pb2

_AXES, _PRIMES = frozenset({"", "ep", "fsdp", "tp"}), (31, 17, 13, 7, 5, 3)
_TOPO = {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,4,1",
         16: "4,4,1", 32: "4,4,2", 64: "4,4,4", 128: "4,4,4,2"}
_TOPO_V7 = {2: "1,1,1,2", 4: "1,2,1,2", 8: "2,2,1,2",
            16: "2,4,1,2", 32: "4,4,1,2", 64: "4,4,2,2", 128: "4,4,4,2"}


class Mesh(NamedTuple):
  """Logical 3D sharding mesh over (ep, fsdp, tp)."""
  ep: int = 1
  fsdp: int = 1
  tp: int = 1

  @property
  def size(self) -> int:
    return self.ep * self.fsdp * self.tp

  def coords(self, rank: int) -> dict[str, int]:
    if not 0 <= rank < self.size:
      raise ValueError(f"Rank {rank} out of range for size={self.size}")
    r = rank % (self.fsdp * self.tp)
    return {"": 0, "ep": rank // (self.fsdp * self.tp),
            "fsdp": r // self.tp, "tp": r % self.tp}


class TensorSpec(NamedTuple):
  """Global weight shape and logical sharding rule."""
  name: str
  shape: tuple[int, ...]
  axes: tuple[str, ...]

  def partition_spec(self, mesh: Mesh) -> tuple[str, ...]:
    out = []
    for d, ax in zip(self.shape, self.axes):
      deg = int(getattr(mesh, ax, 1))
      rep = deg <= 1 or (ax == "tp" and d < deg)
      if not rep and d % deg:
        raise ValueError(f"{self.name} dim {d} not divisible by {ax}={deg}")
      out.append("" if rep else ax)
    return tuple(out)


class ModelSpec(NamedTuple):
  """Resolved model weight manifest."""
  model_name: str
  dtype: str
  tensors: tuple[TensorSpec, ...]
  itemsize: int = 2

  @property
  def payload_bytes(self) -> int:
    return sum(math.prod(t.shape) for t in self.tensors) * self.itemsize


def parse_mesh(mesh_str: str) -> Mesh:
  dims = {"ep": 1, "fsdp": 1, "tp": 1}
  for part in (p.strip() for p in mesh_str.split(",") if p.strip()):
    k, sep, v = (s.strip() for s in part.partition("="))
    if not sep or k not in dims or int(v) < 1:
      raise ValueError(f"Invalid mesh token {part!r} in {mesh_str!r}")
    dims[k] = int(v)
  return Mesh(**dims)


def load_model_spec(
    *,
    spec_file: str | None = None,
    hf_config: str | None = None,
    sharding_file: str | None = None,
    model_name: str | None = None,
) -> ModelSpec:
  """Loads ModelSpec from an explicit JSON manifest or HuggingFace config."""
  if not (spec_file or hf_config):
    raise ValueError("One of --spec-file or --hf-config is required.")
  if spec_file:
    spec = json.loads(pathlib.Path(spec_file).read_text("utf-8"))
  else:
    spec = {}
  if sharding_file:
    doc = json.loads(pathlib.Path(sharding_file).read_text("utf-8"))
    extra = {"sharding_rules": doc} if isinstance(doc, list) else doc
    spec = {**spec, **extra}
  dt_s = str(spec.get("dtype", "bfloat16"))
  if dt_s not in ("bfloat16", "float32"):
    raise ValueError(f"Unsupported dtype {dt_s!r}")
  isz, nm = (2 if dt_s == "bfloat16" else 4), str(
      model_name or spec.get("model_name")
      or pathlib.Path(spec_file or str(hf_config)).stem)
  if "tensors" in spec and not hf_config:
    ts = tuple(TensorSpec(str(t["name"]), tuple(map(int, t["shape"])),
                          tuple(map(str, t["axes"]))) for t in spec["tensors"])
    if any(len(t.shape) != len(t.axes) or set(t.axes) - _AXES for t in ts):
      raise ValueError("Invalid tensor spec axes.")
    return ModelSpec(nm, dt_s, ts, isz)
  hf = hf_config if hf_config is not None else spec.get("hf_config")
  if hf is None:
    raise ValueError("Spec must define 'tensors' or 'hf_config'.")
  import torch  # pylint: disable=g-import-not-at-top
  import transformers  # pylint: disable=g-import-not-at-top
  is_m = isinstance(hf, abc.Mapping)
  kw = {k: v for k, v in hf.items() if k != "model_type"} if is_m else {}
  cfg = (transformers.AutoConfig.for_model(str(hf["model_type"]), **kw)
         if is_m else transformers.AutoConfig.from_pretrained(str(hf)))
  if spec.get("text_only", True):
    tc = getattr(cfg, "get_text_config", lambda: cfg)()
    cfg = (type(tc)(**kw) if kw and type(tc) is not type(cfg)
           and "text_config" not in kw else tc)
    if "tie_word_embeddings" in kw:
      cfg.tie_word_embeddings = kw["tie_word_embeddings"]
  dt = torch.bfloat16 if dt_s == "bfloat16" else torch.float32
  with torch.device("meta"):
    try:
      mod = transformers.AutoModelForCausalLM.from_config(cfg, dtype=dt)
    except TypeError:
      mod = transformers.AutoModelForCausalLM.from_config(cfg, torch_dtype=dt)
  rules, seen, ts = spec.get("sharding_rules"), set(), []
  for pn, p in mod.named_parameters():
    if id(p) not in seen:
      seen.add(id(p))
      sh, nd = tuple(map(int, p.shape)), p.ndim
      if rules is None:
        ax = (("ep", "fsdp", "tp") if nd == 3
              else (("fsdp", "tp") if nd == 2 else ("",) * nd))
      else:
        ax = next((tuple(map(str, r["axes"])) for r in rules
                   if r.get("rank") in (None, nd)
                   and re.search(str(r["pattern"]), pn)), None)
      if ax is None or len(ax) != nd or set(ax) - _AXES:
        raise ValueError(f"Invalid or unmatched rule for {pn} {sh}")
      ts.append(TensorSpec(pn, sh, ax))
  return ModelSpec(nm, dt_s, tuple(ts), isz)


def shard_bounds_and_index(shape: tuple[int, ...], pspec: tuple[str, ...],
                           mesh: Mesh, rank: int) -> tuple[Any, int]:
  coords, bounds, flat = mesh.coords(rank), [], 0
  for d, ax in zip(shape, pspec):
    deg = int(getattr(mesh, ax, 1))
    c = coords[ax] if deg > 1 else 0
    bounds.append((c * (d // deg), (c + 1) * (d // deg)))
    flat = flat * deg + c
  return tuple(bounds), flat


def build_variable_protos(mspec: ModelSpec, mesh: Mesh, rank: int) -> list[Any]:
  out = []
  for i, t in enumerate(mspec.tensors):
    ps = t.partition_spec(mesh)
    _, flat = shard_bounds_and_index(t.shape, ps, mesh, rank)
    out.append(raiden_service_pb2.VariableMetadataProto(
        name=t.name, shape=t.shape, item_size=mspec.itemsize, layer_idx=i,
        mesh_shape=[int(getattr(mesh, a, 1)) for a in ps], sharding_spec=ps,
        layout=range(len(t.shape) - 1, -1, -1), global_shard_indices=[flat]))
  return out


def generate_synthetic_shard(bounds: tuple[tuple[int, int], ...],
                             tensor_idx: int) -> np.ndarray:
  acc: Any = np.int16((tensor_idx * 43) % 251)
  for d, (st, en) in enumerate(bounds):
    v = (np.arange(st, en, dtype=np.int64) * _PRIMES[d % 6]) % 251
    acc = acc + v.astype(np.int16).reshape(
        [en - st if i == d else 1 for i in range(len(bounds))])
  return np.ascontiguousarray(np.remainder(acc, 251) + 1, dtype=np.int16)


def _read_ws_metric(ws: Any, key: str) -> float:
  if hasattr(ws, "get_metrics"):
    v = ws.get_metrics().get(key)
    if v is not None:
      return float(v)
  impl = getattr(ws, "_impl", None)
  return float(getattr(impl.get_metrics(), key, 0.0)) if impl else 0.0


def summarize_samples(samples: abc.Sequence[float]) -> dict[str, Any]:
  ks = ("p50", "p90", "min", "max", "mean", "stddev")
  if not samples:
    return {**dict.fromkeys(ks, 0.0), "samples": []}
  a = np.asarray(samples, dtype=np.float64)
  vs = (np.percentile(a, 50), np.percentile(a, 90), np.min(a), np.max(a),
        np.mean(a), np.std(a, ddof=1) if len(a) > 1 else 0.0)
  return {**{k: float(v) for k, v in zip(ks, vs)}, "samples": a.tolist()}


def collect_nic_info(sys_net_dir: str = "/sys/class/net") -> dict[str, Any]:
  p = pathlib.Path(sys_net_dir)
  if not p.exists():
    return {"nic_count": 0, "nic_interfaces": "", "nic_mtu": 0}
  skip, phys, fb = ("lo", "docker", "veth", "cbr", "cali", "virbr"), [], []
  for e in sorted(p.iterdir(), key=lambda x: x.name):
    if not e.name.startswith(skip):
      mtu = int((e / "mtu").read_text().strip()) if (e / "mtu").exists() else 0
      (phys if (e / "device").exists() else fb).append((e.name, mtu))
  ch = phys or fb
  return {"nic_count": len(ch), "nic_interfaces": ",".join(n for n, _ in ch),
          "nic_mtu": max((m for _, m in ch), default=0)}


def _local_ip() -> str:
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      return (s.connect(("10.255.255.255", 1)), s.getsockname()[0])[1]
  except OSError:
    return "127.0.0.1"


def assign_slice_ranks(probes: abc.Sequence[abc.Mapping[str, Any]],
                       src_size: int, dst_size: int) -> tuple[Any, Any]:
  """Orders Ray worker probes into source and destination TPU slices."""
  if len(probes) < src_size + dst_size:
    raise ValueError(f"Need {src_size + dst_size} workers, got {len(probes)}")
  kf = lambda x: (str(x["slice_key"]), int(x["node_rank"]), str(x["ip"]))
  by_h: dict[Any, list[abc.Mapping[str, Any]]] = {}
  ordered: list[dict[str, Any]] = []
  for x in sorted(probes, key=lambda x: (*kf(x), int(x["chip_hint"]),
                                         int(x["actor_idx"]))):
    by_h.setdefault(kf(x), []).append(x)
  for g in by_h.values():
    valid_chips = {int(y["chip_hint"]) for y in g if int(y["chip_hint"]) >= 0}
    ok = len(valid_chips) == len(g)
    for i, x in enumerate(g):
      ordered.append({**x, "local_world_size": len(g),
                      "local_chip": int(x["chip_hint"]) if ok else i})

  def _fin(grp: abc.Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    n, g0, tenv = len(grp), grp[0], str(grp[0].get("tpu_topo", ""))
    dims = [int(p) for p in re.split(r"[,xX]", tenv) if p] if tenv else []
    dims.extend([1] * max(0, 3 - len(dims)) if dims else [])
    ext = dims if math.prod(dims) == n else dims[:3] + [2]
    topo = (",".join(map(str, ext)) if dims and math.prod(dims) in (n, n // 2)
            else (_TOPO_V7 if g0.get("is_v7") else _TOPO).get(n, f"{n},1,1"))
    sb = ",".join(f"{g['ip']}:{g['sb_port']}" for g in grp)
    mp = int(g0["master_port"])
    return [{**g, "rank_in_slice": r, "slice_size": n, "sb_addresses": sb,
             "master_addr": str(g0["ip"]), "master_port": mp, "topology": topo}
            for r, g in enumerate(grp)]

  return _fin(ordered[:src_size]), _fin(ordered[src_size : src_size + dst_size])


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
  """Executes the distributed Ray weight-sync benchmark and returns metrics."""
  import ray  # pylint: disable=g-import-not-at-top
  src_m, dst_m = parse_mesh(args.src_mesh), parse_mesh(args.dst_mesh)
  spec = load_model_spec(
      spec_file=args.spec_file, hf_config=args.hf_config,
      sharding_file=args.sharding_file, model_name=args.model_name)
  for t in spec.tensors:
    t.partition_spec(src_m)
    t.partition_spec(dst_m)
  start_t = datetime.datetime.now(datetime.timezone.utc).isoformat()
  ray.init(address="auto", ignore_reinit_error=True)

  @ray.remote(num_cpus=1)
  class _Coord:
    """Hosts WeightSynchronizerManager for transfer coordination."""

    def __init__(self, par: int, gsz: int) -> None:
      self._plan_ms = 0.0
      rpc = raiden_controller.WeightSyncWorkerRpcClient(name_resolver=None)
      self._wsm = weight_synchronizer_manager.WeightSynchronizerManager(
          port=0, worker_rpc_client=rpc, enable_plan_cache=True)
      ctrl = self._wsm._controller  # pylint: disable=protected-access
      orig = ctrl._compute_transfer_schedule  # pylint: disable=protected-access

      async def _timed(*a: Any, **kw: Any) -> Any:
        t0 = time.perf_counter()
        try:
          return await orig(*a, **kw)
        finally:
          self._plan_ms = (time.perf_counter() - t0) * 1000.0

      setattr(ctrl, "_compute_transfer_schedule", _timed)
      ctrl.start_transfer = functools.partial(
          ctrl.start_transfer, parallelism=par, group_size=gsz)
      self._wsm.start_server()

    def step(self, stop: bool = False) -> Any:
      return ((self._wsm.close(), self._plan_ms)[1] if stop
              else f"{_local_ip()}:{self._wsm.port}")

  @ray.remote(resources={"TPU": 1})
  class _Worker:
    """Single-chip TPU worker managing local WeightSynchronizer shards."""

    def __init__(self, idx: int) -> None:
      self._idx = idx
      self._ip = _local_ip()
      self._tensors = []
      self._bounds = []
      self._ws = self._torch = self._dist = None

    def probe(self) -> dict[str, Any]:
      """Returns host, topology, port, NIC, and package version metadata."""
      ports, env = [], os.environ
      for _ in (0, 1):
        with socket.socket() as s:
          ports.append((s.bind(("", 0)), int(s.getsockname()[1]))[1])
      visible = (
          env.get("TPU_VISIBLE_CHIPS") or env.get("TPU_VISIBLE_DEVICES") or "-1"
      )
      cs = visible.split(",")[0]
      sk = (env.get("MEGASCALE_SLICE_ID") or env.get("JOBSET_REPLICA_INDEX")
            or env.get("TPU_WORKER_HOSTNAMES") or "0")
      nr = env.get("TPU_WORKER_ID") or env.get("JOB_COMPLETION_INDEX") or "0"
      pci = pathlib.Path("/sys/bus/pci/devices")
      v7 = pci.exists() and any((d / "device").exists() and (
          d / "device").read_text().strip() == "0x0076" for d in pci.iterdir())
      dists = {(d.metadata["Name"] or "").lower().replace("_", "-"): d.version
               for d in metadata.distributions()}
      vers = {k: dists.get(n, dists.get(alt, "unknown"))
              for k, n, alt in (("tpu_sync", "tpu-sync-torch", "tpu-sync"),
                                ("torch_tpu", "torch-tpu", "torch-tpu"),
                                ("libtpu", "libtpu", "libtpu-nightly"),
                                ("ray", "ray", "ray"))}
      tp = env.get("TORCH_TPU_TOPOLOGY") or env.get("TPU_TOPOLOGY", "")
      return {"actor_idx": self._idx, "ip": self._ip, "is_v7": v7,
              "slice_key": f"{int(sk):04d}" if sk.isdigit() else sk,
              "node_rank": int(nr) if nr.isdigit() else 0, "tpu_topo": tp,
              "chip_hint": int(cs) if cs.lstrip("-").isdigit() else -1,
              "sb_port": ports[0], "master_port": ports[1],
              "nic_info": collect_nic_info(), "versions": vers}

    def setup(self, cfg: abc.Mapping[str, Any], is_src: bool, ms: ModelSpec,
              mesh: Mesh, par: int, wsm_addr: str, xp_id: str):
      """Initializes torch_tpu, allocates HBM shards, and registers unit."""
      for k in [k for k in os.environ if k.startswith("MEGASCALE_")]:
        os.environ.pop(k, None)
      ch, rk = str(cfg["local_chip"]), int(cfg["rank_in_slice"])
      os.environ.update({
          "RANK": str(rk), "WORLD_SIZE": str(cfg["slice_size"]),
          "LOCAL_RANK": ch, "LOCAL_WORLD_SIZE": str(cfg["local_world_size"]),
          "TPU_VISIBLE_DEVICES": ch, "TPU_VISIBLE_CHIPS": ch,
          "MASTER_ADDR": str(cfg["master_addr"]),
          "MASTER_PORT": str(cfg["master_port"]),
          "TORCH_TPU_SLICEBUILDER_ADDRESSES": str(cfg["sb_addresses"]),
          "TORCH_TPU_TOPOLOGY": str(cfg["topology"]),
          "TORCH_TPU_XPROF_SESSION_ID": xp_id})
      import torch  # pylint: disable=g-import-not-at-top
      from torch import distributed  # pylint: disable=g-import-not-at-top
      import torch_tpu  # pylint: disable=g-import-not-at-top,unused-import
      from tpu_sync.api.torch import weight_synchronizer  # pylint: disable=g-import-not-at-top
      self._torch, self._dist = torch, distributed
      if int(cfg["slice_size"]) > 1 and not distributed.is_initialized():
        distributed.init_process_group(backend="tpu_dist")
      dt = torch.bfloat16 if ms.dtype == "bfloat16" else torch.float32
      for i, ts in enumerate(ms.tensors):
        b, _ = shard_bounds_and_index(
            ts.shape, ts.partition_spec(mesh), mesh, rk)
        ten = (torch.from_numpy(generate_synthetic_shard(b, i)).to(dt) if is_src
               else torch.zeros(tuple(e - s for s, e in b), dtype=dt))
        self._bounds.append(b)
        self._tensors.append(ten.to(device=torch.device("tpu")))
      torch.accelerator.synchronize()
      self._ws = weight_synchronizer.WeightSynchronizer(
          [[t] for t in self._tensors], local_port=0, parallelism=par,
          listener_port=0, bind_ip=None, auto_h2d=True,
          global_shard_indices=[rk])
      unit = raiden_controller.RaidenId(
          "trainer" if is_src else "sampler", str(rk), "weights")
      data_eps = []
      if hasattr(self._ws, "get_local_endpoints"):
        for ep in self._ws.get_local_endpoints():
          if isinstance(ep, dict) and ep.get("endpoint"):
            data_eps.append(str(ep["endpoint"]))
          elif isinstance(ep, str):
            data_eps.append(ep)
      if not data_eps:
        data_eps = [f"{self._ip}:{self._ws.local_port}"]
      raiden_controller.RaidenControllerClientFacade(
          wsm_addr, name_resolver=None).register_work_unit(
              unit, data_eps,
              control_plane_rpc_address=f"{self._ip}:{self._ws.listener_port}",
              mesh_shape=[mesh.ep, mesh.fsdp, mesh.tp],
              variables=build_variable_protos(ms, mesh, rk),
              mesh_axes=["ep", "fsdp", "tp"])
      return unit

    def sync(self, uuid: int | None, reset: bool = False):
      """Awaits H2D completion on destination or reads D2H metric on source."""
      if uuid is None:
        v = _read_ws_metric(self._ws, "last_d2h_time_ms")
        return (self._ws.reset_metrics(), v)[1] if reset else v
      t0 = time.perf_counter()
      self._ws.wait_for_transfer_completion(uuid)
      self._torch.accelerator.synchronize()
      res = ((time.perf_counter() - t0) * 1000.0,
             _read_ws_metric(self._ws, "last_h2d_time_ms"))
      if reset:
        for t in self._tensors:
          t.zero_()
        self._torch.accelerator.synchronize()
        self._ws.reset_metrics()
      return res

    def finish(self, check: bool) -> float:
      """Verifies destination shards if requested and shuts down worker."""
      mdiff = 0.0
      if check:
        self._torch.accelerator.synchronize()
        for i, (ten, b) in enumerate(zip(self._tensors, self._bounds)):
          ct = ten.cpu()
          ex = self._torch.from_numpy(
              generate_synthetic_shard(b, i)).to(ct.dtype)
          if ct.numel() > 0 and not self._torch.equal(ct, ex):
            d = self._torch.max(self._torch.abs(ct.float() - ex.float())).item()
            mdiff = max(mdiff, float(d))
      if self._ws is not None:
        self._ws.shutdown()
      self._tensors.clear()
      if self._dist is not None and self._dist.is_initialized():
        self._dist.destroy_process_group()
      return mdiff

  coord = _Coord.remote(args.parallelism, args.group_size)
  workers = [_Worker.remote(i) for i in range(src_m.size + dst_m.size)]
  try:
    wsm_addr = ray.get(coord.step.remote())
    s_cfgs, d_cfgs = assign_slice_ranks(
        ray.get([w.probe.remote() for w in workers]), src_m.size, dst_m.size)
    src_w = [workers[c["actor_idx"]] for c in s_cfgs]
    dst_w = [workers[c["actor_idx"]] for c in d_cfgs]
    xs, xd, par = str(time.time_ns()), str(time.time_ns() + 1), args.parallelism
    units = ray.get(
        [w.setup.remote(c, True, spec, src_m, par, wsm_addr, xs)
         for w, c in zip(src_w, s_cfgs)]
        + [w.setup.remote(c, False, spec, dst_m, par, wsm_addr, xd)
           for w, c in zip(dst_w, d_cfgs)])
    xfer = functools.partial(
        raiden_controller.RaidenControllerClientFacade(
            wsm_addr, name_resolver=None).coordinate_transfer,
        src_units=list(units[:src_m.size]), dst_units=list(units[src_m.size:]),
        use_block_chunks=True, is_sender=True)
    chk, warmup_ms, rows = not args.skip_verification, 0.0, []
    for i in range(-1, args.num_iters):
      uid, t0 = (999 if i < 0 else 1000 + i), time.perf_counter()
      xfer(req_id="warmup" if i < 0 else f"iter_{i}", uuid=uid)
      t_net = (time.perf_counter() - t0) * 1000.0
      d_st = ray.get([w.sync.remote(uid, chk and i < 0) for w in dst_w])
      t_wall = (time.perf_counter() - t0) * 1000.0
      s_st = ray.get([w.sync.remote(None, i < 0) for w in src_w])
      if i < 0:
        warmup_ms = t_wall
      else:
        rows.append((t_wall, t_net, max(d for d, _ in d_st),
                     max(h for _, h in d_st), max(s_st)))
    ray.get([w.finish.remote(False) for w in src_w])
    mdiff = max(ray.get([w.finish.remote(chk) for w in dst_w]), default=0.0)
    plan_ms = ray.get(coord.step.remote(True))
  finally:
    for a in (*workers, coord):
      ray.kill(a)
  st = [summarize_samples([r[j] for r in rows]) for j in range(5)]
  gb_s = (spec.payload_bytes / 1e6) / st[0]["p50"] if st[0]["p50"] > 0 else 0.0
  fmt = lambda m: f"ep={m.ep},fsdp={m.fsdp},tp={m.tp}"
  res = {"model": spec.model_name, "dtype": spec.dtype, "parallelism": par,
         "num_iters": args.num_iters, "payload_bytes": spec.payload_bytes,
         "num_tensors": len(spec.tensors), "src_mesh": fmt(src_m),
         "dst_mesh": fmt(dst_m), "resharded": src_m != dst_m,
         "group_size": args.group_size, "start_time": start_t,
         "end_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "nic_info": s_cfgs[0]["nic_info"], "versions": s_cfgs[0]["versions"],
         "wall_clock_ms": st[0], "goodput_gb_s": gb_s,
         "goodput_gbps": gb_s * 8.0, "plan_warmup_ms": float(plan_ms),
         "warmup_transfer_ms": warmup_ms, "network_ms": st[1],
         "h2d_drain_ms": st[2], "h2d_cpp_ms": st[3], "d2h_cpp_ms": st[4],
         "checksum_verified": int(mdiff == 0.0) if chk else 0,
         "max_abs_diff": float(mdiff)}
  if args.output_json:
    out_p = pathlib.Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
  return res


def main(argv: abc.Sequence[str] | None = None) -> int:
  """CLI entrypoint running benchmark and checking verification/latency."""
  p = argparse.ArgumentParser(description="Ray TPU weight sync benchmark.")
  for f in ("--model-name", "--spec-file", "--hf-config",
            "--sharding-file", "--output-json"):
    p.add_argument(f, type=str, default=None)
  for f in ("--src-mesh", "--dst-mesh"):
    p.add_argument(f, type=str, required=True)
  for f, d in (("--num-iters", 5), ("--parallelism", 8), ("--group-size", 16)):
    p.add_argument(f, type=int, default=d)
  p.add_argument("--max-p50-latency-s", type=float, default=None)
  p.add_argument("--skip-verification", action="store_true")
  args = p.parse_args(argv)
  res = run_benchmark(args)
  print(json.dumps(res, indent=2), flush=True)
  if not args.skip_verification and not res["checksum_verified"]:
    print(f"ERROR: max_abs_diff={res['max_abs_diff']}", file=sys.stderr)
    return 1
  if (args.max_p50_latency_s is not None
      and res["wall_clock_ms"]["p50"] / 1000.0 > args.max_p50_latency_s):
    print("ERROR: p50 latency exceeded --max-p50-latency-s.", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
