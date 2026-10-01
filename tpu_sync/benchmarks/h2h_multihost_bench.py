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

"""Cross-host H2H benchmark driver for BAP: analyze -> record -> gate.

Runs the C++ h2h_benchmark_runner between two hosts of a BAP multi-host job.
BAP's multi-host runner executes this same target on every host of the slice
(one JobSet pod per host, JOB_COMPLETION_INDEX = 0..N-1, a shared job
directory mounted on all of them). Roles:

  worker 0  sender   -- also BAP's "primary": the only host whose artifacts
                        and TensorBoard events get uploaded, so all reporting
                        happens here.
  worker 1  receiver -- runs the C++ receiver per config, serves readiness and
                        the byte-integrity verdicts to the sender over a small
                        TCP line protocol (--peer_port).
  others    idle     -- write a done-marker and exit 0.

Peer discovery, in order: --peers, TPU_WORKER_HOSTNAMES, or a rendezvous file
the receiver drops into the shared job directory (--rendezvous_dir).

Stages:
  analyze  collect the throughput distribution per config, write CSV + summary
           JSON + markdown report + SVG histograms, rank configs by how
           gate-worthy they are (bootstrapping the statistic the gate actually
           uses: the median of --gate_iters iterations). Never gates.
  record   same, plus write h2h_multihost_baselines.json with a floor per
           config (median - k * MAD_sigma, capped at max_margin).
  gate     integrity always gates; throughput gates against the recorded floor
           when --gate_throughput (default) and a floor exists for the config.
"""

import csv
import json
import os
import re
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time

from absl import app
from absl import flags

from tpu_sync.benchmarks import bap_metrics
from tpu_sync.benchmarks import h2h_dist

# Bumped when the stdout / file / peer-protocol contract changes; the e2e test
# keys on it.
_CONTRACT_VERSION = 2

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

_STAGE = flags.DEFINE_enum('stage', 'gate', ['analyze', 'record', 'gate'],
                           'analyze: distribution only; record: write '
                           'baselines; gate: pass/fail.')
_MODE = flags.DEFINE_enum(
    'mode', 'auto', ['auto', 'spmd', 'local'],
    'spmd: every host runs this driver, roles from the worker index. local: '
    'both roles on this machine over loopback. auto: spmd when a worker index '
    'is known (--worker_id, JOB_COMPLETION_INDEX, TPU_WORKER_ID) or when '
    'running under BAP (WORKLOAD_ARTIFACTS_DIR set), else local.')
_SUITE = flags.DEFINE_enum('suite', 'correctness',
                           ['correctness', 'perf', 'both'], 'Config set.')
_CONFIGS = flags.DEFINE_string(
    'configs', '', 'Explicit configs "bs:nb:p,bs:nb:p" (bytes, blocks, '
    'parallelism); overrides --suite.')
_ITERS = flags.DEFINE_integer('iters', 5, 'Timed iterations per runner process.')
_RUNS = flags.DEFINE_integer(
    'runs_per_config', 1, 'Independent runner processes per config; total '
    'samples per config = runs_per_config * iters.')
_GATE_ITERS = flags.DEFINE_integer(
    'gate_iters', 50, 'analyze/record: the --iters the GATE will run with. The '
    'gate compares the median of that many iterations against the floor, so '
    'suitability and p_below_floor are bootstrapped for medians of this size.')

_WORKER_ID = flags.DEFINE_integer('worker_id', -1,
                                  'This host index; overrides the env.')
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', -1, 'Host count. When >= 2 and a rendezvous dir exists, '
    'the sender waits for every other worker\'s done-marker before exiting '
    '(BAP deletes the shared job dir when the primary finishes).')
_PEERS = flags.DEFINE_string(
    'peers', '', 'Comma-separated host list, index-aligned with worker ids; '
    'overrides TPU_WORKER_HOSTNAMES.')
_SENDER_IDX = flags.DEFINE_integer('sender_index', 0, 'Worker that sends.')
_RECEIVER_IDX = flags.DEFINE_integer('receiver_index', 1, 'Worker that receives.')
_RENDEZVOUS_DIR = flags.DEFINE_string(
    'rendezvous_dir', '', 'Shared directory for peer discovery and done '
    'markers. Default: <dirname of $WORKLOAD_ARTIFACTS_DIR>/h2h_rendezvous '
    'when that is set.')

_CONTROL_IFACE = flags.DEFINE_string(
    'control_interface', '', 'Interface for the control-plane handshake. '
    'Default: eth0 in spmd (falls back to --control_ip / the hostname\'s '
    'address when eth0 does not exist), lo in local.')
_CONTROL_IP = flags.DEFINE_string(
    'control_ip', '', 'Control-plane IP of this host when --control_interface '
    'cannot be resolved.')
_DATA_IFACE = flags.DEFINE_string(
    'data_interface', '', 'Comma-separated data-plane interfaces for the '
    'runner. Empty lets the runner auto-discover, which INCLUDES eth0 and any '
    'IPv6-only interface; pin it (eth0 on a single-NIC pod, the DRANET list '
    'on a multi-NIC pod) so record and gate measure the same NICs. '
    'Default lo in local mode.')
_NUMA_NODE = flags.DEFINE_integer('numa_node', -1, 'NUMA node to pin the runner to.')
_CONTROL_PORT = flags.DEFINE_integer(
    'control_port', 9099, 'Base runner control port; config i run r uses '
    'base + i * runs_per_config + r.')
_PEER_PORT = flags.DEFINE_integer(
    'peer_port', 9299, 'Driver-to-driver channel on the receiver host.')
_STARTUP_TIMEOUT_S = flags.DEFINE_integer(
    'startup_timeout_s', 1800, 'Wait for the peer driver, for each receiver '
    'to listen, and for the first sender contact per run. Sized for an '
    'independent image pull + bazel build on the peer.')
_TIMEOUT_S = flags.DEFINE_integer('timeout_s', 1800, 'Per-process hard timeout.')
_RUNNER = flags.DEFINE_string('runner', '', 'Path to h2h_benchmark_runner; '
                              'default: located in the runfiles.')

_OUT_DIR = flags.DEFINE_string(
    'out_dir', '', 'Where CSV/summary/report/SVG/baselines go. Default: '
    '$WORKLOAD_ARTIFACTS_DIR if set, else the current directory.')
_BASELINES = flags.DEFINE_string(
    'baselines', None, 'Baselines JSON read by gate. Default: the copy next to '
    'this file in the runfiles.')
_SIGMA_K = flags.DEFINE_float('sigma_k', 3.5, 'Robust sigmas below the median.')
_MAX_MARGIN = flags.DEFINE_float(
    'max_margin', 0.05, 'Floor is never looser than this fractional drop.')
_GATE_THROUGHPUT = flags.DEFINE_bool(
    'gate_throughput', True, 'Fail when the median drops below the floor.')
_MIN_SAMPLES = flags.DEFINE_integer(
    'min_samples', 30, 'analyze: samples needed to call a config gate-worthy.')

# (block_size_bytes, num_blocks, parallelism) -- see H2H_MULTIHOST_TEST.md.
_CORRECTNESS_CONFIGS = [
    (1048576, 64, 1),   # block mapping / offset / copy
    (1048576, 64, 8),   # races, interleaving, stream partition
    (1048573, 64, 4),   # alignment / boundary / partial write
]
_PERF_CONFIGS = [(2097152, 64, p) for p in (1, 2, 4, 8, 16)]

# Printed by the C++ receiver once its control socket is in accept(), and once
# a sender has been accepted.
_READY_MARKER = 'Waiting for sender connection on control plane'
_CONTACT_MARKER = 'Connection established on control plane'
_INTEG_PASS = 'Data integrity verification PASSED'
_INTEG_FAIL = 'Data integrity verification FAILED'

_RE_P50 = re.compile(r'p50:\s*([0-9.]+)')
_RE_P90 = re.compile(r'p90:\s*([0-9.]+)')
_RE_P99 = re.compile(r'p99:\s*([0-9.]+)')
_RE_MEAN_GBS = re.compile(r'Throughput:\s*([0-9.]+)')
_RE_RAW_MS = re.compile(r'H2H_ITER_MS\s+([0-9.]+)')
_RE_TOTAL_BYTES = re.compile(r'H2H_TOTAL_BYTES\s+(\d+)')
_RE_IFACES = re.compile(r'Interfaces:\s*(\d+)\s*\(active:\s*(\d+)\)')
_RE_AUTOSCALE = re.compile(r'Auto-scaling num_blocks to (\d+)')

# Mirrors kNumLayers / kNumShards in h2h_benchmark_runner.cc; only used when
# the runner does not print H2H_TOTAL_BYTES itself.
_LAYERS = 32
_SHARDS = 1

_RENDEZVOUS_FILE = 'receiver.json'
_READY_YES, _READY_NO, _READY_FAILED = '1', '0', 'X'

# Filled in by main(): the control-plane flags handed to every runner process.
_CONTROL_ARGS = []


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _label(bs, nb, p):
  return f'{bs}B_x{nb}_P{p}'


def _configs():
  if _CONFIGS.value:
    out = []
    for item in _CONFIGS.value.split(','):
      item = item.strip()
      if not item:
        continue
      try:
        bs, nb, p = (int(x) for x in item.split(':'))
      except ValueError:
        raise app.UsageError(
            f'--configs item {item!r} must be bs:nb:p integers')
      if bs <= 0 or nb <= 0 or p <= 0:
        raise app.UsageError(f'--configs item {item!r}: all fields must be > 0')
      out.append((bs, nb, p))
    if not out:
      raise app.UsageError('--configs is empty')
    return out
  if _SUITE.value == 'correctness':
    return list(_CORRECTNESS_CONFIGS)
  if _SUITE.value == 'perf':
    return list(_PERF_CONFIGS)
  return list(_CORRECTNESS_CONFIGS) + list(_PERF_CONFIGS)


def _f(regex, text):
  m = regex.search(text or '')
  return float(m.group(1)) if m else -1.0


def _out_dir():
  d = _OUT_DIR.value or os.environ.get('WORKLOAD_ARTIFACTS_DIR') or os.getcwd()
  os.makedirs(d, exist_ok=True)
  return d


def _rendezvous_dir():
  if _RENDEZVOUS_DIR.value:
    return _RENDEZVOUS_DIR.value
  adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
  if adir:
    # Sibling of the artifacts dir: on the shared BAP job volume, but neither
    # uploaded nor scanned by tb_parser (which lists WORKLOAD_METADATA_DIR).
    return os.path.join(os.path.dirname(os.path.abspath(adir)), 'h2h_rendezvous')
  return ''


def _baselines_path():
  if _BASELINES.value:
    return _BASELINES.value
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'h2h_multihost_baselines.json')


def _iface_ipv4(name):
  """IPv4 address of an interface via SIOCGIFADDR; None if unavailable."""
  if not name:
    return None
  try:
    import fcntl  # pylint: disable=g-import-not-at-top  (Linux only)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
      packed = struct.pack('256s', name.encode('utf-8')[:15])
      return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, packed)[20:24])
    finally:
      s.close()
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def _run_attempt():
  return os.environ.get('GITHUB_RUN_ATTEMPT', '')


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def _discover_topology():
  """Returns (worker_id, hosts). Flags win over the environment."""
  wid = _WORKER_ID.value
  if wid < 0:
    for var in ('JOB_COMPLETION_INDEX', 'TPU_WORKER_ID'):
      v = os.environ.get(var, '').strip()
      if v.isdigit():
        wid = int(v)
        break
  raw = _PEERS.value or os.environ.get('TPU_WORKER_HOSTNAMES', '')
  hosts = [h.strip() for h in raw.split(',') if h.strip()]
  return wid, hosts


def _under_bap():
  return bool(os.environ.get('WORKLOAD_ARTIFACTS_DIR') or
              os.environ.get('TENSORBOARD_OUTPUT_DIR'))


def _detect_mode(worker_id):
  if _MODE.value != 'auto':
    return _MODE.value
  if worker_id >= 0 or _under_bap():
    return 'spmd'
  return 'local'


def _resolve_control_plane(mode):
  """Sets _CONTROL_ARGS and returns this host's control-plane IP."""
  del _CONTROL_ARGS[:]
  if mode == 'local':
    iface = _CONTROL_IFACE.value or 'lo'
    _CONTROL_ARGS.append(f'--control_interface={iface}')
    return '127.0.0.1'
  iface = _CONTROL_IFACE.value or 'eth0'
  if iface == 'lo':
    # Explicit loopback: spmd on one machine (tests, two shells on a laptop).
    _CONTROL_ARGS.append('--control_interface=lo')
    return '127.0.0.1'
  ip = _iface_ipv4(iface)
  if ip:
    _CONTROL_ARGS.append(f'--control_interface={iface}')
    return ip
  explicit = bool(_CONTROL_IP.value)
  ip = _CONTROL_IP.value
  if not ip:
    try:
      ip = socket.gethostbyname(socket.gethostname())
    except OSError:
      ip = ''
  # An inferred loopback address would make the receiver unreachable from the
  # other host; an explicit --control_ip is taken at face value.
  if not ip or (not explicit and ip.startswith('127.')):
    print(f'GATE FAIL: control interface {iface!r} has no IPv4 address and '
          'no routable --control_ip / hostname address is available.',
          file=sys.stderr, flush=True)
    sys.exit(1)
  print(f'[topology] {iface!r} not found; using --control_ip={ip}', flush=True)
  _CONTROL_ARGS.append(f'--control_ip={ip}')
  return ip


# ---------------------------------------------------------------------------
# Runner processes
# ---------------------------------------------------------------------------


def _runfiles_root():
  if os.environ.get('RUNFILES_DIR'):
    return os.environ['RUNFILES_DIR']
  d = os.path.dirname(os.path.abspath(__file__))
  main_root = None
  while d != os.path.dirname(d):
    if d.endswith('.runfiles'):
      return d
    if os.path.basename(d) == '_main':
      main_root = d
    d = os.path.dirname(d)
  return main_root or os.path.dirname(os.path.abspath(__file__))


def _locate_runner():
  if _RUNNER.value:
    return _RUNNER.value
  root = _runfiles_root()
  for dirpath, _, files in os.walk(root, followlinks=True):
    if 'h2h_benchmark_runner' in files:
      cand = os.path.join(dirpath, 'h2h_benchmark_runner')
      if os.access(cand, os.X_OK):
        return cand
  raise FileNotFoundError(f'could not locate h2h_benchmark_runner under {root}')


def _base_argv(cc, bs, nb, p, port):
  # A .py runner (the test fake) is launched through this interpreter so it
  # does not depend on an exec bit surviving the runfiles tree.
  argv = ([sys.executable, cc] if cc.endswith('.py') else [cc])
  argv += [f'--data_interface={_DATA_IFACE.value}',
           f'--peer_control_port={port}', f'--block_size={bs}',
           f'--num_blocks={nb}', f'--parallelism={p}',
           f'--numa_node={_NUMA_NODE.value}', f'--iterations={_ITERS.value}']
  argv += _CONTROL_ARGS
  return argv


def _popen(argv):
  return subprocess.Popen(argv, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          env={**os.environ, 'PYTHONUNBUFFERED': '1'})


class _StreamWatcher(threading.Thread):
  """Drains a process's stdout and flags the readiness marker."""

  def __init__(self, proc, marker):
    super().__init__(daemon=True)
    self._proc = proc
    self._marker = marker
    self._lines = []
    self.ready = threading.Event()

  def run(self):
    try:
      for raw in self._proc.stdout:
        line = raw.decode('utf-8', 'replace')
        self._lines.append(line)
        if self._marker in line:
          self.ready.set()
    finally:
      self.ready.set()  # unblock waiters if the process died instead
      try:
        self._proc.stdout.close()
      except OSError:
        pass

  @property
  def output(self):
    return ''.join(list(self._lines))


def _finish(proc, watcher, timeout):
  try:
    proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
  watcher.join(timeout=60)
  return watcher.output


def _parse_sender(out, bs, nb):
  """Sender stdout -> per-iteration GB/s samples (+ the runner's own summary)."""
  ifm = _RE_IFACES.search(out or '')
  active = int(ifm.group(2)) if ifm else 1
  m = _RE_TOTAL_BYTES.search(out or '')
  if m:
    total_bytes = float(m.group(1))
  else:
    # The runner caps num_blocks so one iteration stays <= 16 GiB per NIC and
    # says so on stdout; honour that or GB/s would be inflated.
    am = _RE_AUTOSCALE.search(out or '')
    nb_eff = int(am.group(1)) if am else nb
    total_bytes = float(active * _LAYERS * _SHARDS * nb_eff * bs)
  p50 = _f(_RE_P50, out)
  raw_gbs = [(total_bytes / 1e9) / (float(ms) / 1000.0)
             for ms in _RE_RAW_MS.findall(out or '') if float(ms) > 0]
  mean_gbs = _f(_RE_MEAN_GBS, out)
  gbs = -1.0
  if raw_gbs:
    gbs = statistics.median(raw_gbs)
  elif p50 > 0:
    gbs = (total_bytes / 1e9) / (p50 / 1000.0)
  elif mean_gbs > 0:
    gbs = mean_gbs
  samples = raw_gbs or ([gbs] if gbs > 0 else [])
  return {'gbs': gbs, 'mean_gbs': mean_gbs, 'p50_ms': p50,
          'p90_ms': _f(_RE_P90, out), 'p99_ms': _f(_RE_P99, out),
          'samples': samples, 'total_bytes': total_bytes,
          'active_ifaces': active if ifm else None}


# ---------------------------------------------------------------------------
# Driver-to-driver channel
# ---------------------------------------------------------------------------


class _PeerServer(threading.Thread):
  """Line protocol served on the receiver host.

      READY <i>  -> "1" once run i's receiver is in accept(),
                    "X" if run i's receiver died before listening,
                    "0" otherwise
      VERDICTS   -> JSON {label: bool} once every run is done, else ""
  """

  def __init__(self, port):
    super().__init__(daemon=True)
    self._lock = threading.Lock()
    self._ready = set()
    self._failed = set()
    self._verdicts = None
    self.last_query = {}   # run index -> time of the last READY query
    self.fetched = threading.Event()
    self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._srv.bind(('0.0.0.0', port))
    self._srv.listen(8)
    self.port = self._srv.getsockname()[1]

  def mark_ready(self, index):
    with self._lock:
      self._ready.add(index)

  def mark_failed(self, index):
    with self._lock:
      self._failed.add(index)

  def set_verdicts(self, verdicts):
    with self._lock:
      self._verdicts = dict(verdicts)

  def queried(self, index):
    with self._lock:
      return index in self.last_query

  def close(self):
    try:
      self._srv.close()
    except OSError:
      pass

  @staticmethod
  def _read_request(conn):
    buf = b''
    while len(buf) < 4096:
      chunk = conn.recv(4096)
      if not chunk:
        break
      buf += chunk
    return buf.decode('utf-8', 'replace').strip()

  def run(self):
    while True:
      try:
        conn, _ = self._srv.accept()
      except OSError:
        return
      with conn:
        try:
          conn.settimeout(10)
          req = self._read_request(conn)
          if req.startswith('READY'):
            idx = int(req.split()[1])
            with self._lock:
              self.last_query[idx] = time.time()
              if idx in self._failed:
                reply = _READY_FAILED
              elif idx in self._ready:
                reply = _READY_YES
              else:
                reply = _READY_NO
            conn.sendall(reply.encode('utf-8'))
          elif req == 'VERDICTS':
            with self._lock:
              payload = self._verdicts
            if payload is not None:
              conn.sendall(json.dumps(payload).encode('utf-8'))
              self.fetched.set()
          else:
            conn.sendall(b'?')
        except (OSError, ValueError, IndexError):
          continue


def _peer_query(host, port, request, timeout=30):
  try:
    with socket.create_connection((host, port), timeout=timeout) as s:
      s.settimeout(timeout)
      s.sendall(request.encode('utf-8'))
      s.shutdown(socket.SHUT_WR)
      chunks = []
      while True:
        chunk = s.recv(65536)
        if not chunk:
          break
        chunks.append(chunk)
    return b''.join(chunks).decode('utf-8')
  except OSError:
    return None


def _backoff(attempt, max_interval):
  """0.5 s, 1 s, 2 s, ... capped at max_interval."""
  return min(0.5 * (2 ** min(attempt, 10)), max_interval)


def _await_peer(host, port, request, accept, timeout, what, interval=5):
  """Polls until `accept(reply)` is true. Returns the reply, or None."""
  deadline = time.time() + timeout
  attempt = 0
  last_report = time.time()
  while time.time() < deadline:
    reply = _peer_query(host, port, request)
    if reply and accept(reply):
      return reply
    if time.time() - last_report >= 60:
      last_report = time.time()
      print(f'[barrier] still waiting for {what} '
            f'({int(deadline - time.time())}s left)', flush=True)
    time.sleep(_backoff(attempt, interval))
    attempt += 1
  print(f'[barrier] TIMEOUT waiting for {what}', flush=True)
  return None


# ---------------------------------------------------------------------------
# Rendezvous dir: receiver.json (receiver publishes, sender polls) and
# worker_<id>.done markers (everyone but the sender writes one at exit).
# ---------------------------------------------------------------------------


def _write_atomic(path, payload):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f'{path}.{os.getpid()}.tmp'
  with open(tmp, 'w') as f:
    f.write(payload)
  os.replace(tmp, path)


def _publish_rendezvous(rdir, info):
  path = os.path.join(rdir, _RENDEZVOUS_FILE)
  _write_atomic(path, json.dumps(info))
  print(f'[rendezvous] published {path}: {info}', flush=True)


def _rendezvous_is_current(info, started_at):
  """A receiver.json from an earlier attempt of the same job is stale."""
  attempt = _run_attempt()
  if attempt and info.get('attempt') and info['attempt'] != attempt:
    return False
  ts = info.get('ts')
  if isinstance(ts, (int, float)) and ts < started_at - 3600:
    return False
  return bool(info.get('ip'))


def _await_rendezvous(rdir, timeout, started_at, interval=5):
  path = os.path.join(rdir, _RENDEZVOUS_FILE)
  deadline = time.time() + timeout
  attempt = 0
  last_report = time.time()
  stale_reported = False
  while time.time() < deadline:
    try:
      with open(path) as f:
        info = json.load(f)
      if _rendezvous_is_current(info, started_at):
        print(f'[rendezvous] found {path}: {info}', flush=True)
        return info
      if not stale_reported:
        stale_reported = True
        print(f'[rendezvous] ignoring stale {path}: {info}', flush=True)
    except (OSError, ValueError):
      pass
    if time.time() - last_report >= 60:
      last_report = time.time()
      print(f'[rendezvous] still waiting for {path} '
            f'({int(deadline - time.time())}s left)', flush=True)
    time.sleep(_backoff(attempt, interval))
    attempt += 1
  print(f'[rendezvous] TIMEOUT waiting for {path}', flush=True)
  return None


def _write_done(rdir, worker_id):
  if not rdir or worker_id < 0:
    return
  try:
    _write_atomic(os.path.join(rdir, f'worker_{worker_id}.done'),
                  json.dumps({'worker_id': worker_id, 'ts': time.time(),
                              'attempt': _run_attempt()}))
  except OSError as e:
    print(f'[barrier] could not write done marker: {e}', file=sys.stderr)


def _await_done(rdir, worker_ids, timeout):
  """Sender-side barrier: wait for every other worker's done marker."""
  if not rdir or not worker_ids:
    return
  started = time.time()
  deadline = started + timeout
  pending = set(worker_ids)
  attempt = 0
  last_report = started
  while pending and time.time() < deadline:
    for wid in list(pending):
      path = os.path.join(rdir, f'worker_{wid}.done')
      try:
        with open(path) as f:
          info = json.load(f)
        if _rendezvous_is_current({**info, 'ip': 'x'}, started):
          pending.discard(wid)
      except (OSError, ValueError):
        pass
    if not pending:
      break
    if time.time() - last_report >= 60:
      last_report = time.time()
      print(f'[barrier] waiting for workers {sorted(pending)} to finish '
            f'({int(deadline - time.time())}s left)', flush=True)
    time.sleep(_backoff(attempt, 5))
    attempt += 1
  if pending:
    print(f'[barrier] workers {sorted(pending)} never wrote a done marker; '
          'continuing (BAP will delete the shared job dir).', flush=True)
  else:
    print('[barrier] all workers done.', flush=True)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


def _schedule(configs):
  """[(idx, label, bs, nb, p, run, port)] -- identical on both hosts."""
  runs = max(1, _RUNS.value)
  out = []
  for i, (bs, nb, p) in enumerate(configs):
    for r in range(runs):
      idx = i * runs + r
      out.append((idx, _label(bs, nb, p), bs, nb, p, r,
                  _CONTROL_PORT.value + idx))
  return out


def _run_receiver(cc, configs, worker_id, own_ip):
  """SPMD receiver. Returns the process exit code."""
  srv = _PeerServer(_PEER_PORT.value)
  srv.start()
  print(f'[receiver] peer channel on {own_ip}:{srv.port}', flush=True)
  rdir = _rendezvous_dir()
  if rdir:
    _publish_rendezvous(rdir, {'ip': own_ip, 'hostname': socket.gethostname(),
                               'peer_port': srv.port, 'worker_id': worker_id,
                               'pid': os.getpid(), 'ts': time.time(),
                               'attempt': _run_attempt()})

  verdicts = {}
  aborted = False
  sched = _schedule(configs)
  for idx, label, bs, nb, p, run, port in sched:
    print(f'[receiver] ({idx + 1}/{len(sched)}) {label} run {run} on :{port}',
          flush=True)
    proc = _popen(_base_argv(cc, bs, nb, p, port) + ['--role=receiver'])
    watcher = _StreamWatcher(proc, _READY_MARKER)
    watcher.start()
    watcher.ready.wait(timeout=_STARTUP_TIMEOUT_S.value)
    if _READY_MARKER not in watcher.output:
      # Died (bind/alloc/interface failure) or hung before listening: tell the
      # sender so it skips this run instead of dialing a dead port.
      if proc.poll() is None:
        proc.kill()
      out = _finish(proc, watcher, 30)
      srv.mark_failed(idx)
      print(f'[receiver] {label} run {run}: runner exited rc={proc.returncode} '
            f'before listening; tail:\n{out[-2000:]}', file=sys.stderr,
            flush=True)
      continue
    srv.mark_ready(idx)

    # Bound the wait for the first sender contact: the C++ receiver blocks in
    # accept() forever, and a sender that never comes must not cost
    # startup+timeout per run.
    contact_deadline = time.time() + _STARTUP_TIMEOUT_S.value
    contacted = False
    while time.time() < contact_deadline:
      if proc.poll() is not None or _CONTACT_MARKER in watcher.output:
        contacted = True
        break
      time.sleep(1)
    if not contacted:
      proc.kill()
      _finish(proc, watcher, 30)
      print(f'[receiver] {label} run {run}: no sender contact within '
            f'{_STARTUP_TIMEOUT_S.value}s'
            f'{" (sender never even asked READY)" if not srv.queried(idx) else ""}'
            '; aborting the remaining runs.', file=sys.stderr, flush=True)
      aborted = True
      break

    # The receiver self-exits after its byte-compare; never kill it early or
    # the check is cut short and reports a false CORRUPT.
    out = _finish(proc, watcher, _TIMEOUT_S.value + _STARTUP_TIMEOUT_S.value)
    ok = (_INTEG_PASS in out) and (_INTEG_FAIL not in out)
    verdicts[label] = verdicts.get(label, True) and ok
    print(f'[receiver] {label} run {run}: integrity='
          f'{"OK" if ok else "CORRUPT"}', flush=True)
    if not ok:
      print(f'--- receiver tail ({label} run {run}) ---\n{out[-4000:]}',
            flush=True)

  srv.set_verdicts(verdicts)
  # The sender may still be inside its own last runner process (up to
  # _TIMEOUT_S after ours), then it asks; size the wait accordingly. After an
  # abort, only a sender that has already talked to us is worth waiting for.
  if aborted:
    wait = 60 if srv.last_query else 5
  else:
    wait = 2 * _TIMEOUT_S.value + _STARTUP_TIMEOUT_S.value
  print(f'[receiver] {"aborted" if aborted else "all runs done"}; waiting up '
        f'to {wait}s for the sender to collect verdicts ...', flush=True)
  fetched = srv.fetched.wait(timeout=wait)
  srv.close()
  _write_done(rdir, worker_id)
  if not fetched:
    print('[receiver] sender never collected the verdicts.', file=sys.stderr)
    return 1
  return 1 if aborted else 0


def _run_sender(cc, configs, peer_ip, peer_port, spawn_receiver=None):
  """Runs every scheduled sender process.

  spawn_receiver is None in spmd mode (the peer driver publishes readiness);
  in local mode it is a callable(argv) -> Popen owning the receiver too.
  Returns (results {label: {...}}, local_verdicts, aborted).
  """
  results = {}
  local_verdicts = {}
  aborted = False
  sched = _schedule(configs)
  for idx, label, bs, nb, p, run, port in sched:
    argv = _base_argv(cc, bs, nb, p, port)
    r = results.setdefault(label, {'samples': [], 'raw': []})
    recv_proc = recv_watcher = None
    if spawn_receiver is None:
      print(f'[sender] ({idx + 1}/{len(sched)}) {label} run {run}: waiting '
            f'for peer receiver {peer_ip}:{peer_port}', flush=True)
      ready = _await_peer(peer_ip, peer_port, f'READY {idx}',
                          lambda rep: rep in (_READY_YES, _READY_FAILED),
                          _STARTUP_TIMEOUT_S.value,
                          f'receiver readiness for {label} run {run}')
      if ready == _READY_FAILED:
        print(f'[sender] {label} run {run}: receiver failed before listening; '
              'skipping run', flush=True)
        r['raw'].append({'run': run, 'rc': None, 'samples': [], 'gbs': -1.0,
                         'mean_gbs': -1.0, 'p50_ms': -1.0, 'p90_ms': -1.0,
                         'p99_ms': -1.0, 'total_bytes': 0,
                         'active_ifaces': None, 'skipped': True})
        continue
      if ready != _READY_YES:
        print(f'[sender] {label}: peer receiver never became ready. If this '
              'is the first run, the peer is probably not running this '
              'driver at all.', file=sys.stderr, flush=True)
        aborted = True
        break
    else:
      print(f'[sender] ({idx + 1}/{len(sched)}) {label} run {run}: local '
            f'receiver on :{port}', flush=True)
      recv_proc = spawn_receiver(argv + ['--role=receiver'])
      recv_watcher = _StreamWatcher(recv_proc, _READY_MARKER)
      recv_watcher.start()
      recv_watcher.ready.wait(timeout=_STARTUP_TIMEOUT_S.value)
      if _READY_MARKER not in recv_watcher.output:
        print(f'[sender] {label}: local receiver never signalled ready.',
              file=sys.stderr, flush=True)
        recv_proc.kill()
        aborted = True
        break

    send_proc = _popen(argv + ['--role=sender', f'--peer_control_ip={peer_ip}'])
    send_watcher = _StreamWatcher(send_proc, _READY_MARKER)
    send_watcher.start()
    out = _finish(send_proc, send_watcher, _TIMEOUT_S.value)
    m = _parse_sender(out, bs, nb)
    m['run'] = run
    m['rc'] = send_proc.returncode

    if recv_proc is not None:
      recv_out = _finish(recv_proc, recv_watcher, _TIMEOUT_S.value)
      ok = (_INTEG_PASS in recv_out) and (_INTEG_FAIL not in recv_out)
      local_verdicts[label] = local_verdicts.get(label, True) and ok
      if not ok:
        print(f'--- receiver tail ({label} run {run}) ---\n{recv_out[-4000:]}',
              flush=True)

    if not m['samples']:
      print(f'[sender] {label} run {run} produced no throughput reading '
            f'(rc={send_proc.returncode}); output:\n{out[-4000:]}', flush=True)
    r['samples'].extend(m['samples'])
    r['raw'].append(m)
  return results, local_verdicts, aborted


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _active_ifaces(r):
  for m in r.get('raw', []):
    if m.get('active_ifaces'):
      return m['active_ifaces']
  return None


def _write_outputs(stage, configs, results, verdicts, out_dir):
  """CSV + summary JSON + markdown + SVG. Returns (per_config, ranking)."""
  labels = [_label(*c) for c in configs]
  csv_path = os.path.join(out_dir, 'h2h_multihost_samples.csv')
  with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    # `rank`: the runner prints H2H_ITER_MS from its SORTED latency vector, so
    # this is the ascending-latency rank within the run, not the time order.
    w.writerow(['config', 'run', 'rank', 'gbs', 'mean_gbs', 'p50_ms', 'p90_ms',
                'p99_ms', 'total_bytes', 'integrity'])
    for label in labels:
      r = results.get(label, {'raw': []})
      integ = int(bool(verdicts.get(label, False)))
      for m in r['raw']:
        for j, g in enumerate(m['samples']):
          w.writerow([label, m['run'], j, f'{g:.4f}', f'{m["mean_gbs"]:.4f}',
                      f'{m["p50_ms"]:.4f}', f'{m["p90_ms"]:.4f}',
                      f'{m["p99_ms"]:.4f}', int(m['total_bytes']), integ])

  per_config = {}
  for label in labels:
    r = results.get(label, {})
    s = r.get('samples', [])
    a = h2h_dist.assess(s, verdicts.get(label, False), _MIN_SAMPLES.value,
                        _MAX_MARGIN.value, k=_SIGMA_K.value,
                        gate_iters=_GATE_ITERS.value)
    a['run_medians'] = [m['gbs'] for m in r.get('raw', []) if m.get('samples')]
    a['active_ifaces'] = _active_ifaces(r)
    per_config[label] = a
  ranking = h2h_dist.rank(per_config)

  summary = {'contract_version': _CONTRACT_VERSION, 'stage': stage,
             'iters': _ITERS.value, 'runs_per_config': max(1, _RUNS.value),
             'gate_iters': _GATE_ITERS.value, 'sigma_k': _SIGMA_K.value,
             'max_margin': _MAX_MARGIN.value, 'configs': per_config,
             'ranking': ranking}
  with open(os.path.join(out_dir, 'h2h_multihost_summary.json'), 'w') as f:
    f.write(h2h_dist.to_json(summary))
  suitable = [c for c in configs if per_config[_label(*c)].get('suitable')]
  suggested = {
      'configs_flag': '--configs=' + ','.join(f'{bs}:{nb}:{p}' for bs, nb, p in suitable),
      'metrics': [f'metrics {{ name: "{_label(*c)}/cpp_gbs" unit: "GB/s" '
                  'stats { stat: MEAN } stats { stat: MEDIAN } }' for c in suitable],
  }
  with open(os.path.join(out_dir, 'h2h_multihost_report.md'), 'w') as f:
    f.write(h2h_dist.render_markdown(stage, per_config, ranking,
                                     _SIGMA_K.value, _MAX_MARGIN.value,
                                     _GATE_ITERS.value, suggested))
  floors = {label: a.get('floor_effective') for label, a in per_config.items()
            if a.get('floor_effective')}
  with open(os.path.join(out_dir, 'h2h_multihost_dist.svg'), 'w') as f:
    f.write(h2h_dist.render_svg(
        {label: results.get(label, {}).get('samples', []) for label in labels},
        floors, title=f'H2H multi-host {stage}: GB/s per iteration'))

  for label in ranking:
    a = per_config[label]
    if 'n' in a:
      print(f'[measured] {label:<22} n={a["n"]:<5} median={a["median"]:8.3f}  '
            f'p10={a["p10"]:8.3f}  p90={a["p90"]:8.3f}  '
            f'cv_robust={a["cv_robust"]:6.3f}  low_tail={a["low_tail_frac"]:5.3f}  '
            f'floor={a["floor_effective"]:8.3f}  '
            f'p_below_floor(median of {_GATE_ITERS.value})={a["p_below_floor"]:.4f}  '
            f'integrity={"OK" if a["integrity"] else "CORRUPT"}  '
            f'{"SUITABLE" if a["suitable"] else "not suitable: " + "; ".join(a["reasons"])}',
            flush=True)
    else:
      print(f'[measured] {label:<22} no samples  '
            f'integrity={"OK" if a.get("integrity") else "CORRUPT"}', flush=True)
  if suitable:
    print(f'\nSuitable configs -> paste into the record and gate registries:\n'
          f'  {suggested["configs_flag"]}', flush=True)
  else:
    print('\nNo config is suitable yet (see the reasons above).', flush=True)
  print(f'\nWrote {csv_path}, h2h_multihost_summary.json, '
        f'h2h_multihost_report.md, h2h_multihost_dist.svg -> {out_dir}',
        flush=True)
  return per_config, ranking


def _write_baselines(configs, results, verdicts, per_config, out_dir):
  cfg = {'contract_version': _CONTRACT_VERSION, 'sigma_k': _SIGMA_K.value,
         'max_margin': _MAX_MARGIN.value, 'iters': _ITERS.value,
         'runs_per_config': max(1, _RUNS.value),
         'gate_iters': _GATE_ITERS.value, 'configs': {}}
  for c in configs:
    label = _label(*c)
    s = results.get(label, {}).get('samples', [])
    a = per_config.get(label, {})
    cfg['configs'][label] = {
        'baseline_gbs': round(statistics.median(s), 3) if s else 0.0,
        'floor_gbs': round(h2h_dist.core_floor(s, _SIGMA_K.value,
                                               _MAX_MARGIN.value), 3) if s else 0.0,
        'floor_uncapped_gbs': round(a['floor_candidates'][f'{_SIGMA_K.value:.1f}'], 3)
                              if a.get('floor_candidates', {}).get(f'{_SIGMA_K.value:.1f}') is not None else 0.0,
        'p_below_floor': a.get('p_below_floor'),
        'suitable': bool(a.get('suitable', False)),
        'integrity': bool(verdicts.get(label, False)),
        'n_samples': len(s),
        'active_ifaces': a.get('active_ifaces'),
    }
  path = os.path.join(out_dir, 'h2h_multihost_baselines.json')
  with open(path, 'w') as f:
    json.dump(cfg, f, indent=2)
  print(f'Recorded {len(cfg["configs"])} baselines+floors -> {path}\n'
        f'Commit it to tpu_sync/benchmarks/h2h_multihost_baselines.json to '
        f'arm the gate.', flush=True)


def _gate(configs, results, verdicts):
  floors = {}
  base = {}
  if _GATE_THROUGHPUT.value:
    try:
      with open(_baselines_path()) as f:
        base = json.load(f)
      floors = {k: float(v.get('floor_gbs', 0.0))
                for k, v in base.get('configs', {}).items()}
    except (OSError, ValueError) as e:
      print(f'[gate] baselines unreadable ({e}); throughput floors disabled.',
            file=sys.stderr, flush=True)
  rec_iters = base.get('gate_iters')
  if floors and rec_iters and rec_iters != _ITERS.value:
    print(f'[gate] WARNING: floors were calibrated for the median of '
          f'{rec_iters} iterations but this run uses --iters={_ITERS.value}; '
          'the flap probability estimate does not apply.', flush=True)

  bad = []
  print('\nH2H multi-host gate\n')
  print(f'{"config":<22} {"median":>9} {"floor":>9}  integrity  verdict')
  print('-' * 72)
  for c in configs:
    label = _label(*c)
    r = results.get(label, {})
    s = r.get('samples', [])
    med = statistics.median(s) if s else -1.0
    has_verdict = label in verdicts
    integ = bool(verdicts.get(label, False))
    floor = floors.get(label, 0.0)
    reasons = []
    if not s:
      reasons.append('NO MEASUREMENT')
    if has_verdict and not integ:
      reasons.append('DATA CORRUPTION')
    elif not has_verdict and s:
      # Bytes moved but the receiver never reported on them (crashed, or the
      # run was skipped on its side): unproven is a failure, but a different one.
      reasons.append('NO VERDICT')
    if _GATE_THROUGHPUT.value and s and floor > 0 and med < floor:
      reasons.append(f'BELOW FLOOR {floor:.3f}')
    rec_nics = base.get('configs', {}).get(label, {}).get('active_ifaces')
    got_nics = _active_ifaces(r)
    if rec_nics and got_nics and rec_nics != got_nics:
      print(f'[gate] WARNING: {label}: floor recorded with {rec_nics} active '
            f'NIC(s), this run used {got_nics}.', flush=True)
    floor_txt = f'{floor:9.3f}' if floor > 0 else '  NO FLOOR'
    print(f'{label:<22} {med:9.3f} {floor_txt}  '
          f'{"OK" if integ else "CORRUPT":<9}  '
          f'{"PASS" if not reasons else "FAIL <-- " + ", ".join(reasons)}',
          flush=True)
    if reasons:
      bad.append(label)
  if bad:
    print(f'\nGATE FAIL on {len(bad)} config(s): {bad}', file=sys.stderr,
          flush=True)
    return 1
  print('\nGATE PASS: all configs byte-exact across hosts'
        + (' and at/above their floors.' if floors else '.'), flush=True)
  return 0


# ---------------------------------------------------------------------------


def main(_):
  started_at = time.time()
  worker_id, hosts = _discover_topology()
  mode = _detect_mode(worker_id)
  configs = _configs()
  stage = _STAGE.value

  if mode == 'local' and flags.FLAGS['data_interface'].using_default_value:
    flags.FLAGS.data_interface = 'lo'
  if mode == 'spmd' and worker_id < 0:
    print('GATE FAIL: spmd mode but no worker index (set --worker_id, or run '
          'under a runner that exports JOB_COMPLETION_INDEX / TPU_WORKER_ID).',
          file=sys.stderr, flush=True)
    sys.exit(1)

  own_ip = _resolve_control_plane(mode)
  cc = _locate_runner()
  rdir = _rendezvous_dir() if mode == 'spmd' else ''
  print(f'[topology] mode={mode} stage={stage} worker_id={worker_id} '
        f'hosts={hosts or "(rendezvous)"} sender={_SENDER_IDX.value} '
        f'receiver={_RECEIVER_IDX.value} control_ip={own_ip} '
        f'rendezvous_dir={rdir or "-"}', flush=True)
  print(f'[topology] configs={[_label(*c) for c in configs]} '
        f'iters={_ITERS.value} runs_per_config={max(1, _RUNS.value)} '
        f'runner={cc}', flush=True)

  if mode == 'spmd':
    if worker_id == _RECEIVER_IDX.value:
      sys.exit(_run_receiver(cc, configs, worker_id, own_ip))
    if worker_id != _SENDER_IDX.value:
      print(f'[topology] worker {worker_id} is neither sender nor receiver; '
            'nothing to do.', flush=True)
      _write_done(rdir, worker_id)
      return
    peer_ip, peer_port = None, _PEER_PORT.value
    if len(hosts) > _RECEIVER_IDX.value:
      peer_ip = hosts[_RECEIVER_IDX.value]
    else:
      if not rdir:
        print('GATE FAIL: no way to find the receiver: give --peers, set '
              'TPU_WORKER_HOSTNAMES, or set --rendezvous_dir / '
              'WORKLOAD_ARTIFACTS_DIR.', file=sys.stderr)
        sys.exit(1)
      info = _await_rendezvous(rdir, _STARTUP_TIMEOUT_S.value, started_at)
      if not info:
        print('GATE FAIL: the receiver never published its address; is worker '
              f'{_RECEIVER_IDX.value} running this same target?',
              file=sys.stderr)
        sys.exit(1)
      peer_ip, peer_port = info['ip'], int(info.get('peer_port', peer_port))
    results, _, aborted = _run_sender(cc, configs, peer_ip, peer_port)
    verdicts = {}
    if aborted:
      print('GATE FAIL: the peer receiver never became ready; run incomplete.',
            file=sys.stderr)
    else:
      raw = _await_peer(peer_ip, peer_port, 'VERDICTS', bool,
                        _TIMEOUT_S.value, 'receiver verdicts')
      if not raw:
        print("GATE FAIL: could not read the receiver's integrity verdicts; "
              'nothing proves the bytes are correct.', file=sys.stderr)
      else:
        verdicts = json.loads(raw)
  else:
    aborted = False
    results, verdicts, _ = _run_sender(cc, configs, '127.0.0.1',
                                       _PEER_PORT.value, _popen)

  out_dir = _out_dir()
  per_config, _ = _write_outputs(stage, configs, results, verdicts, out_dir)

  scalars = {}
  for c in configs:
    label = _label(*c)
    s = results.get(label, {}).get('samples', [])
    if not s:
      continue  # a missing metric beats a -1.0 point on the dashboard
    scalars[f'{label}/cpp_gbs'] = statistics.median(s) if stage == 'gate' else s
  if scalars:
    bap_metrics.emit(scalars)

  rc = None
  total = sum(a.get('n', 0) for a in per_config.values())
  if stage == 'analyze':
    print('\nanalyze: done (no gate).', flush=True)
    rc = 0 if total else 1
  elif stage == 'record':
    _write_baselines(configs, results, verdicts, per_config, out_dir)
    rc = 0 if total else 1
  else:
    rc = _gate(configs, results, verdicts)
  if mode == 'spmd' and aborted:
    rc = 1

  if mode == 'spmd':
    n = _NUM_WORKERS.value if _NUM_WORKERS.value > 0 else len(hosts)
    others = [w for w in range(n) if w != worker_id]
    if len(others) >= 1 and rdir:
      _await_done(rdir, others, _STARTUP_TIMEOUT_S.value)
  sys.exit(rc)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
