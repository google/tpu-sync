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

"""End-to-end test for h2h_multihost_bench.py driven by a fake C++ runner.

Spawns the driver as real subprocesses (worker 0 = sender, worker 1 = receiver)
on one machine, with testdata/fake_h2h_runner.py standing in for the C++
h2h_benchmark_runner, and asserts the CONTRACT v2 stdout strings, files and
exit codes. Runs under bazel and as
`python3 tpu_sync/benchmarks/h2h_multihost_bench_test.py` from the repo root.
"""

import csv
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

from absl.testing import absltest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_DRIVER = os.path.join(_HERE, 'h2h_multihost_bench.py')
_FAKE_RUNNER = os.path.join(_HERE, 'testdata', 'fake_h2h_runner.py')

_CONFIGS = '65536:4:1,65536:4:2'
_LABELS = ('65536B_x4_P1', '65536B_x4_P2')
_TOTAL_BYTES = 65536 * 4 * 32  # block_size * num_blocks * 32 layers * 1 iface
_ITERS = 8
_RUNS = 2
_GATE_ITERS = 50  # driver default; != _ITERS on purpose so the gate warns
_PEERS = '--peers=127.0.0.1,127.0.0.1'
_PROC_TIMEOUT_S = 60
# One contiguous block per test: control ports base..base+6 (config i run r
# listens on base + i * runs + r) and the driver peer channel on base + 7.
_PORT_BLOCK = 8
_CSV_HEADER = ['config', 'run', 'rank', 'gbs', 'mean_gbs', 'p50_ms', 'p90_ms',
               'p99_ms', 'total_bytes', 'integrity']
_SUMMARY_KEYS = ('n', 'median', 'mean', 'stdev', 'mad_sigma', 'cv_robust',
                 'p10', 'p50', 'p90', 'min', 'max', 'low_tail_frac',
                 'integrity', 'floor_candidates', 'cap', 'floor_effective',
                 'gate_iters', 'gate_median_sigma', 'gate_cv', 'gate_p01',
                 'p_below_floor', 'suitable', 'reasons', 'run_medians',
                 'active_ifaces')
_BASELINE_KEYS = ('contract_version', 'sigma_k', 'max_margin', 'iters',
                  'runs_per_config', 'gate_iters', 'configs')
_BASELINE_CFG_KEYS = ('baseline_gbs', 'floor_gbs', 'floor_uncapped_gbs',
                      'p_below_floor', 'suitable', 'integrity', 'n_samples',
                      'active_ifaces')
# Env the driver reads; scrubbed (with every FAKE_H2H_* knob) so each test
# starts from defaults. TENSORBOARD_OUTPUT_DIR must never be set here: it makes
# the driver import tensorflow.
_SCRUB_ENV = ('TENSORBOARD_OUTPUT_DIR', 'WORKLOAD_ARTIFACTS_DIR',
              'TPU_WORKER_HOSTNAMES', 'TPU_WORKER_ID', 'JOB_COMPLETION_INDEX',
              'GITHUB_RUN_ATTEMPT')


def _free_port_block(n):
  """A base port such that base..base+n-1 are all currently bindable.

  Kept below every platform's ephemeral range (Linux 32768+, macOS 49152+) so
  a port that is free now is not taken by an outbound connection later.
  """
  for _ in range(500):
    base = random.randint(20000, 32000)
    socks = []
    try:
      for port in range(base, base + n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socks.append(s)
        s.bind(('127.0.0.1', port))
    except OSError:
      continue
    finally:
      for s in socks:
        s.close()
    return base
  raise RuntimeError('could not find %d consecutive free ports' % n)


def _subprocess_env(extra=None):
  env = {k: v for k, v in os.environ.items()
         if k not in _SCRUB_ENV and not k.startswith('FAKE_H2H_')}
  # Repo root first so `from tpu_sync.benchmarks import ...` resolves; then
  # this test's own sys.path so absl is importable under bazel runfiles.
  env['PYTHONPATH'] = os.pathsep.join([_REPO_ROOT] + [p for p in sys.path if p])
  env['PYTHONUNBUFFERED'] = '1'
  env.update(extra or {})
  return env


def _spmd_args(stage, control_port, peer_port, out_dir, iters=_ITERS,
               runs=_RUNS, configs=_CONFIGS, mode='spmd', num_workers=2,
               startup_timeout_s=45, timeout_s=45):
  args = ['--stage=%s' % stage, '--control_interface=lo',
          '--data_interface=lo', '--runner=%s' % _FAKE_RUNNER,
          '--configs=%s' % configs, '--iters=%d' % iters,
          '--runs_per_config=%d' % runs, '--control_port=%d' % control_port,
          '--peer_port=%d' % peer_port, '--out_dir=%s' % out_dir,
          '--startup_timeout_s=%d' % startup_timeout_s,
          '--timeout_s=%d' % timeout_s]
  if mode:
    args.append('--mode=%s' % mode)
  if num_workers:
    args.append('--num_workers=%d' % num_workers)
  return args


class _Worker:
  """One driver process; stdout/stderr pumped on a thread, killable as a group."""

  def __init__(self, name, args, env):
    self.name = name
    self.cmd = [sys.executable, _DRIVER] + list(args)
    self.timed_out = False
    self.out = ''
    self.err = ''
    self.proc = subprocess.Popen(
        self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        cwd=_REPO_ROOT, start_new_session=True)
    self._thread = threading.Thread(target=self._pump, daemon=True)
    self._thread.start()

  def _pump(self):
    out, err = self.proc.communicate()
    self.out = out.decode('utf-8', 'replace')
    self.err = err.decode('utf-8', 'replace')

  def wait(self, timeout):
    self._thread.join(max(0.0, timeout))
    if self._thread.is_alive():
      self.timed_out = True
      self.kill()
      self._thread.join(10)
    return self.proc.returncode

  def kill(self):
    if self.proc.poll() is None:
      try:
        os.killpg(self.proc.pid, signal.SIGKILL)
      except OSError:
        pass
      try:
        self.proc.kill()
      except OSError:
        pass

  @property
  def rc(self):
    return self.proc.returncode

  @property
  def text(self):
    return self.out + '\n' + self.err

  def describe(self):
    return ('\n===== %s (rc=%s%s)\n$ %s\n--- stdout ---\n%s\n--- stderr ---\n%s'
            % (self.name, self.rc, ', TIMED OUT' if self.timed_out else '',
               ' '.join(self.cmd), self.out[-6000:], self.err[-6000:]))


def _wait_workers(workers, timeout):
  deadline = time.monotonic() + timeout
  try:
    for w in workers:
      w.wait(deadline - time.monotonic())
  finally:
    for w in workers:
      w.kill()


def _dump(*workers):
  return ''.join(w.describe() for w in workers)


def _rank_key(label, cfg):
  """h2h_dist.rank()'s sort key, restated so the test pins it."""
  return (0 if cfg.get('suitable') else 1, 0 if cfg.get('integrity') else 1,
          cfg.get('p_below_floor', 1.0), cfg.get('cv_robust', float('inf')),
          label)


class H2hMultihostBenchTest(absltest.TestCase):
  """Each test owns a tempdir/out_dir and one fresh port block."""

  _class_dir = None
  _baselines_path = None
  _record_control_port = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    for path in (_DRIVER, _FAKE_RUNNER):
      if not os.path.exists(path):
        raise AssertionError('missing %s' % path)
    cls._class_dir = tempfile.mkdtemp(prefix='h2h_multihost_test_')
    cls._record_baselines_once()

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls._class_dir, ignore_errors=True)
    super().tearDownClass()

  @classmethod
  def _record_baselines_once(cls):
    """One record run (peers mode) whose baselines every gate test reuses.

    Gate tests also reuse this control-port base: the fake runner's jitter is
    seeded by (block_size, parallelism, port), so a gate run on the same ports
    reproduces the recorded samples exactly and lands on the baseline.
    """
    out_dir = os.path.join(cls._class_dir, 'record')
    base = _free_port_block(_PORT_BLOCK)
    args = _spmd_args('record', base, base + _PORT_BLOCK - 1, out_dir)
    args.append(_PEERS)
    receiver = _Worker('receiver', args + ['--worker_id=1'], _subprocess_env())
    sender = _Worker('sender', args + ['--worker_id=0'], _subprocess_env())
    _wait_workers([receiver, sender], _PROC_TIMEOUT_S)
    path = os.path.join(out_dir, 'h2h_multihost_baselines.json')
    if sender.rc != 0 or receiver.rc != 0 or not os.path.exists(path):
      raise AssertionError('class-level record run failed' +
                           _dump(sender, receiver))
    cls._baselines_path = path
    cls._record_control_port = base

  def setUp(self):
    super().setUp()
    self.out_dir = self.create_tempdir().full_path
    self._workers = []
    self._t0 = time.monotonic()

  def tearDown(self):
    for w in self._workers:
      w.kill()
    print('[timing] %s %.1fs' % (self.id().split('.')[-1],
                                 time.monotonic() - self._t0),
          file=sys.stderr, flush=True)
    super().tearDown()

  # -- helpers -------------------------------------------------------------

  def _ports(self):
    """(control port base, peer port) from one fresh contiguous block."""
    base = _free_port_block(_PORT_BLOCK)
    return base, base + _PORT_BLOCK - 1

  def _args(self, stage, control_port, peer_port, **kw):
    kw.setdefault('out_dir', self.out_dir)
    return _spmd_args(stage, control_port, peer_port, **kw)

  def _gate_args(self, baselines=None, **kw):
    """Gate on the class baselines, on the class control ports (see above)."""
    _, peer_port = self._ports()
    args = self._args('gate', self._record_control_port, peer_port, **kw)
    args += [_PEERS, '--baselines=%s' % (baselines or self._baselines_path)]
    return args

  def _start(self, name, args, env_extra=None):
    w = _Worker(name, args, _subprocess_env(env_extra))
    self._workers.append(w)
    return w

  def _run_workers(self, specs, timeout=_PROC_TIMEOUT_S):
    """specs: [(name, args, env_extra)]; starts all, waits, kills leftovers."""
    workers = [self._start(n, a, e) for n, a, e in specs]
    _wait_workers(workers, timeout)
    return workers

  def _run_pair(self, args, sender_env=None, receiver_env=None, ids='flag',
                receiver_delay_s=0.0, timeout=_PROC_TIMEOUT_S):
    """Worker 1 (receiver) and worker 0 (sender), concurrently.

    ids='flag' passes --worker_id, ids='env' sets JOB_COMPLETION_INDEX.
    receiver_delay_s > 0 starts the sender first and the receiver that much
    later.
    """
    recv_args, send_args = list(args), list(args)
    recv_env, send_env = dict(receiver_env or {}), dict(sender_env or {})
    if ids == 'flag':
      recv_args.append('--worker_id=1')
      send_args.append('--worker_id=0')
    else:
      recv_env['JOB_COMPLETION_INDEX'] = '1'
      send_env['JOB_COMPLETION_INDEX'] = '0'
    if receiver_delay_s > 0:
      sender = self._start('sender', send_args, send_env)
      time.sleep(receiver_delay_s)
      receiver = self._start('receiver', recv_args, recv_env)
    else:
      receiver = self._start('receiver', recv_args, recv_env)
      sender = self._start('sender', send_args, send_env)
    _wait_workers([receiver, sender], timeout)
    return sender, receiver

  def _assert_rc(self, worker, expected, *context):
    self.assertFalse(worker.timed_out, '%s timed out%s' %
                     (worker.name, _dump(worker, *context)))
    self.assertEqual(worker.rc, expected, '%s exit code %s != %s%s' %
                     (worker.name, worker.rc, expected,
                      _dump(worker, *context)))

  def _assert_in_output(self, needle, worker, *context):
    self.assertIn(needle, worker.text, 'expected %r in %s output%s' %
                  (needle, worker.name, _dump(worker, *context)))

  def _assert_not_in_output(self, needle, worker, *context):
    self.assertNotIn(needle, worker.text, 'unexpected %r in %s output%s' %
                     (needle, worker.name, _dump(worker, *context)))

  def _read_samples(self, out_dir=None):
    """(header, rows) of h2h_multihost_samples.csv."""
    path = os.path.join(out_dir or self.out_dir, 'h2h_multihost_samples.csv')
    self.assertTrue(os.path.exists(path), 'missing %s' % path)
    with open(path, newline='') as f:
      reader = csv.reader(f)
      header = next(reader, [])
      rows = [dict(zip(header, r)) for r in reader]
    return header, rows

  def _assert_samples(self, header, rows, labels, per_config, context):
    self.assertEqual(header, _CSV_HEADER, 'csv header%s' % context)
    for label in labels:
      got = [r for r in rows if r['config'] == label]
      self.assertEqual(len(got), per_config, '%s: %d sample rows, want %d%s'
                       % (label, len(got), per_config, context))
      for r in got:
        self.assertGreater(float(r['gbs']), 0.0, '%r%s' % (r, context))
        self.assertEqual(int(float(r['total_bytes'])), _TOTAL_BYTES,
                         'total_bytes in %r%s' % (r, context))
        self.assertEqual(r['integrity'], '1', '%r%s' % (r, context))
      # `rank` restarts at 0 for every run (ascending-latency rank in-run).
      for run in sorted({r['run'] for r in got}):
        ranks = sorted(int(r['rank']) for r in got if r['run'] == run)
        self.assertEqual(ranks, list(range(len(ranks))),
                         '%s run %s ranks %s%s' % (label, run, ranks, context))

  def _read_json(self, name, out_dir=None):
    path = os.path.join(out_dir or self.out_dir, name)
    self.assertTrue(os.path.exists(path), 'missing %s' % path)
    with open(path) as f:
      return json.load(f)

  # -- tests ---------------------------------------------------------------

  def test_analyze_with_peers(self):
    control_port, peer_port = self._ports()
    args = self._args('analyze', control_port, peer_port) + [_PEERS]
    sender, receiver = self._run_pair(args)
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 0, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('analyze: done (no gate).', sender, receiver)
    self._assert_in_output('[receiver] all runs done', receiver, sender)

    header, rows = self._read_samples()
    self._assert_samples(header, rows, _LABELS, _RUNS * _ITERS, ctx)

    summary = self._read_json('h2h_multihost_summary.json')
    self.assertEqual(summary.get('contract_version'), 2, ctx)
    self.assertEqual(summary.get('stage'), 'analyze', ctx)
    self.assertEqual(summary.get('gate_iters'), _GATE_ITERS, ctx)
    self.assertEqual(set(summary['configs']), set(_LABELS), ctx)
    for label in _LABELS:
      cfg = summary['configs'][label]
      for key in _SUMMARY_KEYS:
        self.assertIn(key, cfg, '%s missing %r: %s%s' % (label, key, cfg, ctx))
      self.assertEqual(cfg['n'], _RUNS * _ITERS, '%s: %s%s' % (label, cfg, ctx))
      self.assertGreater(cfg['median'], 0.0, '%s: %s%s' % (label, cfg, ctx))
      self.assertIs(cfg['integrity'], True, '%s: %s%s' % (label, cfg, ctx))
      self.assertEqual(cfg['gate_iters'], _GATE_ITERS, ctx)
      self.assertEqual(len(cfg['run_medians']), _RUNS, '%s: %s%s' %
                       (label, cfg['run_medians'], ctx))
      self.assertEqual(cfg['active_ifaces'], 1, ctx)
      # Candidates are UNCAPPED median - k*sigma: monotone in k and never
      # above the effective (capped) floor the gate would use.
      fc = cfg['floor_candidates']
      self.assertEqual(set(fc), {'2.5', '3.5', '5.0'}, '%s: %s%s' %
                       (label, fc, ctx))
      self.assertGreaterEqual(fc['2.5'], fc['3.5'], ctx)
      self.assertGreaterEqual(fc['3.5'], fc['5.0'], ctx)
      self.assertLessEqual(fc['3.5'], cfg['floor_effective'] + 1e-9, ctx)
      self.assertGreaterEqual(cfg['floor_effective'], cfg['cap'] - 1e-9, ctx)
      # 16 tight samples < default --min_samples=30: unsuitable for exactly
      # that reason (cv, low tail and flap probability are all fine).
      self.assertIsInstance(cfg['suitable'], bool, ctx)
      self.assertFalse(cfg['suitable'], '%s: %s%s' % (label, cfg, ctx))
      self.assertEqual(cfg['reasons'], ['n=16 < min_samples=30'],
                       '%s: %s%s' % (label, cfg, ctx))
      self.assertLessEqual(cfg['p_below_floor'], 0.001, ctx)

    ranking = summary['ranking']
    self.assertIsInstance(ranking, list, ctx)
    self.assertEqual(set(ranking), set(_LABELS), ctx)
    expected = sorted(_LABELS,
                      key=lambda l: _rank_key(l, summary['configs'][l]))
    self.assertEqual(ranking, expected, 'ranking %s%s' % (ranking, ctx))
    # Both unsuitable and byte-correct, so the key falls through to
    # p_below_floor then cv_robust; the fake's jitter is 0.02/parallelism, so
    # P2 is the tighter config and must come first.
    p1, p2 = summary['configs'][_LABELS[0]], summary['configs'][_LABELS[1]]
    self.assertLessEqual(p2['p_below_floor'], p1['p_below_floor'], ctx)
    self.assertLess(p2['cv_robust'], p1['cv_robust'], ctx)
    self.assertEqual(ranking[0], '65536B_x4_P2', 'ranking %s%s' % (ranking, ctx))

    report = os.path.join(self.out_dir, 'h2h_multihost_report.md')
    self.assertTrue(os.path.exists(report), 'missing %s%s' % (report, ctx))
    with open(report) as f:
      report_text = f.read()
    for label in _LABELS:
      self.assertIn(label, report_text, ctx)
    self.assertIn('p_below_floor', report_text, ctx)

    svg = os.path.join(self.out_dir, 'h2h_multihost_dist.svg')
    self.assertTrue(os.path.exists(svg), 'missing %s%s' % (svg, ctx))
    with open(svg) as f:
      self.assertIn('<svg', f.read(), ctx)

  def test_record_with_file_rendezvous(self):
    control_port, peer_port = self._ports()
    rdv = os.path.join(self.out_dir, 'rdv')
    args = self._args('record', control_port, peer_port)
    args.append('--rendezvous_dir=%s' % rdv)  # no --peers: sender polls rdv
    sender, receiver = self._run_pair(args)
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 0, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('[rendezvous] found', sender, receiver)
    self._assert_in_output('[rendezvous] published', receiver, sender)

    rdv_file = os.path.join(rdv, 'receiver.json')
    self.assertTrue(os.path.exists(rdv_file), 'missing %s%s' % (rdv_file, ctx))
    with open(rdv_file) as f:
      rdv_json = json.load(f)
    for key in ('ip', 'hostname', 'peer_port', 'worker_id', 'pid', 'ts',
                'attempt'):
      self.assertIn(key, rdv_json, '%s%s' % (rdv_json, ctx))
    self.assertEqual(rdv_json['ip'], '127.0.0.1', ctx)
    self.assertEqual(int(rdv_json['peer_port']), peer_port, ctx)
    self.assertEqual(int(rdv_json['worker_id']), 1, ctx)
    self.assertEqual(rdv_json['attempt'], '', ctx)  # GITHUB_RUN_ATTEMPT unset

    header, rows = self._read_samples()
    self._assert_samples(header, rows, _LABELS, _RUNS * _ITERS, ctx)

    baselines = self._read_json('h2h_multihost_baselines.json')
    for key in _BASELINE_KEYS:
      self.assertIn(key, baselines, '%s%s' % (baselines, ctx))
    self.assertEqual(baselines['contract_version'], 2, ctx)
    self.assertEqual(baselines['iters'], _ITERS, ctx)
    self.assertEqual(baselines['runs_per_config'], _RUNS, ctx)
    self.assertEqual(baselines['gate_iters'], _GATE_ITERS, ctx)
    self.assertEqual(set(baselines['configs']), set(_LABELS), ctx)
    for label in _LABELS:
      b = baselines['configs'][label]
      for key in _BASELINE_CFG_KEYS:
        self.assertIn(key, b, '%s missing %r: %s%s' % (label, key, b, ctx))
      self.assertGreater(b['baseline_gbs'], 0.0, '%s: %s%s' % (label, b, ctx))
      self.assertGreater(b['floor_gbs'], 0.0, '%s: %s%s' % (label, b, ctx))
      self.assertLess(b['floor_gbs'], b['baseline_gbs'],
                      '%s: %s%s' % (label, b, ctx))
      # The recorded floor is the capped one; the uncapped is never looser.
      self.assertLessEqual(b['floor_uncapped_gbs'], b['floor_gbs'] + 1e-9,
                           '%s: %s%s' % (label, b, ctx))
      self.assertIs(b['integrity'], True, '%s: %s%s' % (label, b, ctx))
      self.assertIs(b['suitable'], False, '%s: %s%s' % (label, b, ctx))
      self.assertEqual(b['n_samples'], _RUNS * _ITERS,
                       '%s: %s%s' % (label, b, ctx))
      self.assertEqual(b['active_ifaces'], 1, '%s: %s%s' % (label, b, ctx))
    summary = self._read_json('h2h_multihost_summary.json')
    self.assertEqual(summary.get('stage'), 'record', ctx)
    self.assertEqual(summary.get('contract_version'), 2, ctx)

  def test_gate_pass(self):
    sender, receiver = self._run_pair(self._gate_args())
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 0, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('GATE PASS', sender, receiver)
    for bad in ('BELOW FLOOR', 'DATA CORRUPTION', 'NO MEASUREMENT', 'NO FLOOR',
                'GATE FAIL', 'floor recorded with'):
      self._assert_not_in_output(bad, sender, receiver)
    # Floors were recorded with the default --gate_iters=50 but this gate ran
    # --iters=8: a warning, not a failure.
    self._assert_in_output(
        '[gate] WARNING: floors were calibrated for the median of %d '
        'iterations but this run uses --iters=%d' % (_GATE_ITERS, _ITERS),
        sender, receiver)
    header, rows = self._read_samples()
    self._assert_samples(header, rows, _LABELS, _RUNS * _ITERS, ctx)

  def test_gate_fails_below_floor(self):
    sender, receiver = self._run_pair(
        self._gate_args(), sender_env={'FAKE_H2H_SPEED_FACTOR': '0.5'})
    self._assert_rc(sender, 1, receiver)
    self._assert_in_output('BELOW FLOOR', sender, receiver)
    self._assert_in_output('GATE FAIL on 2 config(s)', sender, receiver)
    self._assert_not_in_output('GATE PASS', sender, receiver)
    # The receiver handed over its verdicts, so it is not the failing party.
    self._assert_rc(receiver, 0, sender)

  def test_gate_fails_on_corruption(self):
    sender, receiver = self._run_pair(
        self._gate_args(), receiver_env={'FAKE_H2H_FAIL_INTEGRITY': '1'})
    self._assert_rc(sender, 1, receiver)
    self._assert_in_output('DATA CORRUPTION', sender, receiver)
    self._assert_not_in_output('GATE PASS', sender, receiver)
    self._assert_in_output('integrity=CORRUPT', receiver, sender)
    self._assert_rc(receiver, 0, sender)
    header, rows = self._read_samples()
    self.assertEqual(header, _CSV_HEADER, _dump(sender, receiver))
    self.assertTrue(rows and all(r['integrity'] == '0' for r in rows),
                    'integrity column%s' % _dump(sender, receiver))

  def test_idle_worker_exits_zero(self):
    control_port, peer_port = self._ports()
    rdv = os.path.join(self.out_dir, 'rdv')
    args = self._args('gate', control_port, peer_port, num_workers=3)
    args += ['--peers=127.0.0.1,127.0.0.1,127.0.0.1', '--worker_id=2',
             '--rendezvous_dir=%s' % rdv]
    start = time.monotonic()
    (idle,) = self._run_workers([('idle', args, None)], timeout=30)
    elapsed = time.monotonic() - start
    self._assert_rc(idle, 0)
    self._assert_in_output('nothing to do', idle)
    self.assertLess(elapsed, 20.0, 'idle worker took %.1fs%s' %
                    (elapsed, idle.describe()))
    done = os.path.join(rdv, 'worker_2.done')
    self.assertTrue(os.path.exists(done), 'missing %s%s' % (done, idle.describe()))
    with open(done) as f:
      self.assertEqual(json.load(f).get('worker_id'), 2, idle.describe())

  def test_local_mode_analyze(self):
    control_port, _ = self._ports()
    args = ['--stage=analyze', '--mode=local', '--runner=%s' % _FAKE_RUNNER,
            '--configs=%s' % _CONFIGS, '--iters=%d' % _ITERS,
            '--runs_per_config=%d' % _RUNS, '--control_port=%d' % control_port,
            '--out_dir=%s' % self.out_dir, '--startup_timeout_s=45',
            '--timeout_s=45']
    (local,) = self._run_workers([('local', args, None)])
    self._assert_rc(local, 0)
    self._assert_in_output('[topology] mode=local', local)
    header, rows = self._read_samples()
    self._assert_samples(header, rows, _LABELS, _RUNS * _ITERS, local.describe())
    summary = self._read_json('h2h_multihost_summary.json')
    self.assertEqual(set(summary['configs']), set(_LABELS), local.describe())
    self.assertEqual(summary['contract_version'], 2, local.describe())

  def test_gate_no_floor_for_unknown_config(self):
    with open(self._baselines_path) as f:
      baselines = json.load(f)
    del baselines['configs'][_LABELS[1]]
    partial = os.path.join(self.out_dir, 'partial_baselines.json')
    with open(partial, 'w') as f:
      json.dump(baselines, f)
    sender, receiver = self._run_pair(self._gate_args(baselines=partial))
    self._assert_rc(sender, 0, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('NO FLOOR', sender, receiver)
    self._assert_in_output('GATE PASS', sender, receiver)

  def test_min_samples_satisfied_with_forty_iters(self):
    control_port, peer_port = self._ports()
    args = self._args('analyze', control_port, peer_port, iters=40, runs=1)
    args += [_PEERS, '--min_samples=30']
    sender, receiver = self._run_pair(args)
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 0, receiver)
    self._assert_rc(receiver, 0, sender)
    header, rows = self._read_samples()
    self._assert_samples(header, rows, _LABELS, 40, ctx)
    summary = self._read_json('h2h_multihost_summary.json')
    for label in _LABELS:
      cfg = summary['configs'][label]
      self.assertEqual(cfg['n'], 40, '%s: %s%s' % (label, cfg, ctx))
      # 40 tight samples (<=2% jitter, no outliers) >= min_samples and a
      # median-of-50 that never dips below the floor -> suitable.
      self.assertTrue(cfg['suitable'], '%s: %s%s' % (label, cfg, ctx))
      self.assertEqual(cfg['reasons'], [], '%s: %s%s' % (label, cfg, ctx))
      self.assertLess(cfg['cv_robust'], 0.1, '%s: %s%s' % (label, cfg, ctx))
      self.assertLessEqual(cfg['low_tail_frac'], 0.05, ctx)
      self.assertLessEqual(cfg['p_below_floor'], 0.001, ctx)
      self.assertEqual(cfg['run_medians'], [cfg['run_medians'][0]], ctx)
    self._assert_in_output('SUITABLE', sender, receiver)

  def test_sender_crash_mid_run_gives_no_measurement(self):
    # (j) The C++ sender exit(1)s from a worker thread mid-loop: no results
    # block, so the run yields no samples and the gate reports NO MEASUREMENT.
    control_port, peer_port = self._ports()
    args = self._args('gate', control_port, peer_port, runs=1)
    args += [_PEERS, '--baselines=%s' % self._baselines_path]
    sender, receiver = self._run_pair(
        args, sender_env={'FAKE_H2H_SENDER_DIE_AT_ITER': '3'})
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 1, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('NO MEASUREMENT', sender, receiver)
    self._assert_in_output('GATE FAIL on 2 config(s)', sender, receiver)
    self._assert_in_output('Iteration 3 failed on active manager 0', sender,
                           receiver)
    self._assert_in_output('produced no throughput reading (rc=1)', sender,
                           receiver)
    self._assert_not_in_output('GATE PASS', sender, receiver)
    header, rows = self._read_samples()
    self.assertEqual(header, _CSV_HEADER, ctx)
    self.assertEqual(rows, [], 'expected no sample rows%s' % ctx)
    summary = self._read_json('h2h_multihost_summary.json')
    for label in _LABELS:
      cfg = summary['configs'][label]
      self.assertNotIn('n', cfg, '%s: %s%s' % (label, cfg, ctx))
      self.assertFalse(cfg['suitable'], ctx)
      self.assertIn('no samples', cfg['reasons'], '%s: %s%s' % (label, cfg, ctx))
      self.assertEqual(cfg['run_medians'], [], ctx)
      self._assert_in_output('[measured] %s' % label, sender, receiver)
    self.assertIn('no samples', sender.text, ctx)

  def test_one_failed_run_keeps_the_other(self):
    # (k) Only config 0 run 1 (control port base + 0*runs + 1) dies.
    control_port, peer_port = self._ports()
    dead_port = control_port + 0 * _RUNS + 1
    args = self._args('analyze', control_port, peer_port) + [_PEERS]
    sender, receiver = self._run_pair(
        args, sender_env={'FAKE_H2H_SENDER_DIE_AT_ITER': '3',
                          'FAKE_H2H_SENDER_DIE_ON_PORT': str(dead_port)})
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 0, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('%s run 1 produced no throughput reading (rc=1)'
                           % _LABELS[0], sender, receiver)
    header, rows = self._read_samples()
    self._assert_samples(header, rows, [_LABELS[0]], _ITERS, ctx)
    self._assert_samples(header, rows, [_LABELS[1]], _RUNS * _ITERS, ctx)
    self.assertEqual({r['run'] for r in rows if r['config'] == _LABELS[0]},
                     {'0'}, ctx)
    self.assertEqual({r['run'] for r in rows if r['config'] == _LABELS[1]},
                     {'0', '1'}, ctx)
    summary = self._read_json('h2h_multihost_summary.json')
    c0, c1 = summary['configs'][_LABELS[0]], summary['configs'][_LABELS[1]]
    self.assertEqual(c0['n'], _ITERS, '%s%s' % (c0, ctx))
    self.assertEqual(len(c0['run_medians']), 1, '%s%s' % (c0, ctx))
    self.assertEqual(c1['n'], _RUNS * _ITERS, '%s%s' % (c1, ctx))
    self.assertEqual(len(c1['run_medians']), _RUNS, '%s%s' % (c1, ctx))

  def test_bad_configs_usage_error(self):
    # (l) absl UsageError: non-zero exit and the reason on stderr.
    args = ['--stage=analyze', '--mode=local', '--runner=%s' % _FAKE_RUNNER,
            '--configs=65536:4', '--out_dir=%s' % self.out_dir]
    (bad,) = self._run_workers([('bad', args, None)], timeout=30)
    self.assertFalse(bad.timed_out, bad.describe())
    self.assertNotEqual(bad.rc, 0, bad.describe())
    self._assert_in_output('must be bs:nb:p', bad)
    self.assertFalse(os.path.exists(os.path.join(
        self.out_dir, 'h2h_multihost_samples.csv')), bad.describe())

  def test_auto_mode_from_job_completion_index(self):
    # (m) No --mode / --worker_id: JOB_COMPLETION_INDEX picks spmd and roles.
    control_port, peer_port = self._ports()
    args = self._args('analyze', control_port, peer_port, mode=None) + [_PEERS]
    sender, receiver = self._run_pair(args, ids='env')
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 0, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('[topology] mode=spmd stage=analyze worker_id=0',
                           sender, receiver)
    self._assert_in_output('[topology] mode=spmd stage=analyze worker_id=1',
                           receiver, sender)
    header, rows = self._read_samples()
    self._assert_samples(header, rows, _LABELS, _RUNS * _ITERS, ctx)

  def test_spmd_without_worker_index_fails(self):
    # (n) --mode=spmd with neither --worker_id nor an index in the env.
    control_port, peer_port = self._ports()
    args = self._args('analyze', control_port, peer_port) + [_PEERS]
    (lost,) = self._run_workers([('lost', args, None)], timeout=30)
    self._assert_rc(lost, 1)
    self._assert_in_output('GATE FAIL: spmd mode but no worker index', lost)

  def test_receiver_dead_before_ready_skips_run(self):
    # (o) Config 1's receiver fails to bind: READY answers "X", the sender
    # skips that run at once (no 20-60 s connect retry) and the gate fails it
    # with NO MEASUREMENT while config 0 is unaffected.
    control_port, peer_port = self._ports()
    dead_port = control_port + 1 * 1 + 0  # config 1 run 0, runs=1
    args = self._args('gate', control_port, peer_port, runs=1)
    args += [_PEERS, '--baselines=%s' % self._baselines_path]
    start = time.monotonic()
    sender, receiver = self._run_pair(
        args, receiver_env={'FAKE_H2H_RECEIVER_DIE_BEFORE_READY': '1',
                            'FAKE_H2H_RECEIVER_DIE_ON_PORT': str(dead_port)})
    elapsed = time.monotonic() - start
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 1, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('[sender] %s run 0: receiver failed before '
                           'listening; skipping run' % _LABELS[1], sender,
                           receiver)
    self._assert_in_output('runner exited rc=1 before listening', receiver,
                           sender)
    self._assert_in_output('NO MEASUREMENT', sender, receiver)
    self._assert_in_output('GATE FAIL on 1 config(s)', sender, receiver)
    self.assertLess(elapsed, 30.0, 'took %.1fs%s' % (elapsed, ctx))
    header, rows = self._read_samples()
    self._assert_samples(header, rows, [_LABELS[0]], _ITERS, ctx)
    self.assertEqual([r for r in rows if r['config'] == _LABELS[1]], [], ctx)

  def test_stale_rendezvous_file_is_ignored(self):
    # (p) A receiver.json left by an earlier attempt of the same job (other
    # GITHUB_RUN_ATTEMPT, hour-old ts) must not be dialled.
    control_port, peer_port = self._ports()
    rdv = os.path.join(self.out_dir, 'rdv')
    os.makedirs(rdv)
    wrong_port = control_port + _PORT_BLOCK - 2  # free, nobody listens there
    with open(os.path.join(rdv, 'receiver.json'), 'w') as f:
      json.dump({'ip': '127.0.0.1', 'peer_port': wrong_port, 'ts': 1.0,
                 'attempt': '1'}, f)
    args = self._args('analyze', control_port, peer_port)
    args.append('--rendezvous_dir=%s' % rdv)
    env = {'GITHUB_RUN_ATTEMPT': '2'}
    sender, receiver = self._run_pair(args, sender_env=env, receiver_env=env,
                                      receiver_delay_s=2.0)
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 0, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('[rendezvous] ignoring stale', sender, receiver)
    self._assert_in_output('[rendezvous] found', sender, receiver)
    with open(os.path.join(rdv, 'receiver.json')) as f:
      fresh = json.load(f)
    self.assertEqual(fresh['attempt'], '2', '%s%s' % (fresh, ctx))
    self.assertEqual(int(fresh['peer_port']), peer_port, '%s%s' % (fresh, ctx))
    with open(os.path.join(rdv, 'worker_1.done')) as f:
      self.assertEqual(json.load(f).get('attempt'), '2', ctx)
    header, rows = self._read_samples()
    self._assert_samples(header, rows, _LABELS, _RUNS * _ITERS, ctx)

  def test_sender_waits_for_done_markers(self):
    # (q) When the receiver is discovered via receiver.json in the rendezvous
    # dir and --num_workers=2, the sender leaves only after worker 1 wrote its
    # done marker.
    control_port, peer_port = self._ports()
    rdv = os.path.join(self.out_dir, 'rdv')
    args = self._args('analyze', control_port, peer_port)
    args.append('--rendezvous_dir=%s' % rdv)
    sender, receiver = self._run_pair(args)
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 0, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('[rendezvous] found', sender, receiver)
    done = os.path.join(rdv, 'worker_1.done')
    self.assertTrue(os.path.exists(done), 'missing %s%s' % (done, ctx))
    with open(done) as f:
      marker = json.load(f)
    self.assertEqual(marker.get('worker_id'), 1, '%s%s' % (marker, ctx))
    self.assertEqual(marker.get('attempt'), '', '%s%s' % (marker, ctx))
    self.assertIsInstance(marker.get('ts'), float, '%s%s' % (marker, ctx))
    self._assert_in_output('[barrier] all workers done.', sender, receiver)
    self._assert_not_in_output('[barrier] skipping done-marker wait', sender,
                               receiver)
    self._assert_not_in_output('never wrote a done marker', sender, receiver)
    self.assertFalse(os.path.exists(os.path.join(rdv, 'worker_0.done')), ctx)

  def test_sender_with_peers_skips_done_markers(self):
    # When peers come from the hosts list (--peers / TPU_WORKER_HOSTNAMES), the
    # rendezvous dir is not verified shared across pods, so the sender must not
    # block on done markers from workers 1..N-1.
    control_port, peer_port = self._ports()
    rdv = os.path.join(self.out_dir, 'rdv')
    startup_timeout_s = 120
    args = self._args('analyze', control_port, peer_port, num_workers=4,
                      startup_timeout_s=startup_timeout_s)
    args += [_PEERS, '--rendezvous_dir=%s' % rdv]
    start = time.monotonic()
    sender, receiver = self._run_pair(args)
    elapsed = time.monotonic() - start
    ctx = _dump(sender, receiver)
    self._assert_rc(sender, 0, receiver)
    self._assert_rc(receiver, 0, sender)
    self._assert_in_output('[barrier] skipping done-marker wait', sender,
                           receiver)
    self._assert_not_in_output('never wrote a done marker', sender, receiver)
    self._assert_not_in_output('[barrier] all workers done.', sender, receiver)
    self.assertLess(elapsed, startup_timeout_s / 2,
                    'took %.1fs%s' % (elapsed, ctx))

  def test_receiver_aborts_without_sender(self):
    # (r) No sender ever contacts the receiver: it kills its runner after
    # --startup_timeout_s, aborts the remaining runs and exits 1 promptly.
    control_port, peer_port = self._ports()
    args = self._args('gate', control_port, peer_port, startup_timeout_s=3,
                      timeout_s=3)
    args += [_PEERS, '--worker_id=1']
    start = time.monotonic()
    (receiver,) = self._run_workers([('receiver', args, None)], timeout=30)
    elapsed = time.monotonic() - start
    self._assert_rc(receiver, 1)
    self.assertIn('no sender contact within 3s', receiver.err,
                  receiver.describe())
    self._assert_in_output('aborting the remaining runs', receiver)
    self._assert_in_output('[receiver] aborted; waiting up to', receiver)
    self._assert_in_output('sender never collected the verdicts', receiver)
    self.assertLess(elapsed, 20.0, 'took %.1fs%s' % (elapsed, receiver.describe()))

  def test_autoscaled_num_blocks_sets_total_bytes(self):
    # The runner caps one iteration at 16 GiB per NIC ("Auto-scaling
    # num_blocks to N ...") and the driver must size GB/s from N, not from
    # the requested --num_blocks, or throughput would be inflated 2x here.
    control_port, _ = self._ports()
    args = ['--stage=analyze', '--mode=local', '--runner=%s' % _FAKE_RUNNER,
            '--configs=1048576:1024:1', '--iters=2', '--runs_per_config=1',
            '--control_port=%d' % control_port, '--out_dir=%s' % self.out_dir,
            '--startup_timeout_s=45', '--timeout_s=45']
    (local,) = self._run_workers([('local', args, None)])
    self._assert_rc(local, 0)
    header, rows = self._read_samples()
    self.assertEqual(header, _CSV_HEADER, local.describe())
    self.assertEqual(len(rows), 2, local.describe())
    for r in rows:
      self.assertEqual(r['config'], '1048576B_x1024_P1', local.describe())
      self.assertEqual(int(float(r['total_bytes'])), 512 * 32 * 1048576,
                       'total_bytes in %r%s' % (r, local.describe()))


if __name__ == '__main__':
  absltest.main()
