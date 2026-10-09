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

import json
import multiprocessing
import os
import subprocess
import sys

from absl import app  # pylint: disable=unused-import  # used by the OSS entry point below
from absl import flags
import numpy as np

from tpu_sync.benchmarks import bap_metrics
from tpu_sync.benchmarks import perf_core

_BASELINES = flags.DEFINE_string(
    'baselines', None,
    'Path to h2d_d2h_gating_baselines.json. Default: alongside this binary.')
_RECORD = flags.DEFINE_bool(
    'record', False,
    'Re-measure and OVERWRITE baselines/floors instead of gating.')
_ITERS = flags.DEFINE_integer(
    'iters', None,
    'Override iters from the baselines file. Use a large value when recording '
    '(the floor depends on a MAD/sigma estimate, which needs many samples to be '
    'stable); the gate itself runs the smaller value in the baselines file.')
_FRAMEWORK = flags.DEFINE_enum(
    'framework',
    'jax',
    ['all', 'jax', 'torch'],
    'Framework to benchmark: "jax", "torch", or "all" (both JAX and PyTorch).',
)
_ENFORCE = flags.DEFINE_bool(
    'enforce',
    None,
    'Override the "enforce" setting in the baselines file.',
)

# The perf floor (NOT the correctness check) can be bypassed per-PR by putting
# one of these tags in the CL description / commit message.
_SKIP_TAGS = ('[skip-perf-gate]', '[skip-h2d-d2h-gating]')


def _baselines_path(framework):
  """Baselines file for one framework.

  JAX and torch exercise different transfer code, so their floors are not
  interchangeable and never share a file.
  """
  if _BASELINES.value:
    return _BASELINES.value
  return os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      f'h2d_d2h_gating_{framework}_baselines.json')


def _opted_out():
  """True if the CL description, commit message, or environment requests bypass."""
  for var in ('SKIP_PERF_GATE', 'TPU_RAIDEN_SKIP_PERF_GATE'):
    if os.environ.get(var, '').lower() in ('1', 'true', 'yes'):
      return var
  for cmd in (
      ['hg', 'log', '-r', '.', '-T', '{desc}'],
      ['git', 'log', '-1', '--format=%B'],
  ):
    try:
      res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
      if res.returncode == 0:
        msg = res.stdout.lower()
        for tag in _SKIP_TAGS:
          if tag in msg:
            return tag
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return None


def _core_floor(samples, k):
  """Per-config gate floor: median - k robust-sigmas (MAD-based).

      floor = median - k * MAD_sigma

  MAD_sigma = 1.4826 * median(|x - median|) is an outlier-resistant standard
  deviation, so the low tail does not drag the bound down.
  """
  x = np.asarray(samples, float)
  med = np.median(x)
  sigma = 1.4826 * np.median(np.abs(x - med))
  return float(med - k * sigma)


def _write_tb_metrics(results, framework):
  """Log per-config throughput to TENSORBOARD_OUTPUT_DIR so BAP ingests it.

  Each tag MUST have a matching metrics{name:...} in benchmark_registry.pbtxt.
  """
  scalars = {}
  for c, r in results:
    tag_suffix = (
        f"{c['dtype']}_L{c['num_layers']}_{'x'.join(map(str, c['shape']))}"
    )
    # Symmetrical framework-qualified metric tags
    scalars[f'{framework}_{tag_suffix}/d2h_gbps'] = r['d2h_gbps']
    scalars[f'{framework}_{tag_suffix}/h2d_gbps'] = r['h2d_gbps']

    # Legacy un-prefixed alias for JAX to maintain continuity with existing
    # MLCompass regression monitoring (go/tpu-sync-oss-mlcompass).
    if framework == 'jax':
      scalars[f'{tag_suffix}/d2h_gbps'] = r['d2h_gbps']
      scalars[f'{tag_suffix}/h2d_gbps'] = r['h2d_gbps']
  bap_metrics.emit(scalars)


def _measure_configs(configs, fw, iters, warmup):
  """Measure every config once under one framework.

  The round-trip byte-equality check runs inside perf_core.measure(), before
  any floor comparison, so a corrupt or no-op transfer raises here regardless
  of the skip tag or "enforce".
  """
  results = []
  for c in configs:
    r = perf_core.measure(
        shape=c['shape'],
        num_layers=c['num_layers'],
        dtype=c['dtype'],
        shard_axis=c.get('shard_axis', 2),
        iters=iters,
        warmup=warmup,
        framework=fw,
    )
    results.append((c, r))
    print(
        f"[{fw} measured] {c['dtype']} L{c['num_layers']} "
        f"{'x'.join(map(str, c['shape']))}  "
        f"d2h {r['d2h_gbps']:.1f}  h2d {r['h2d_gbps']:.1f} Gbps"
    )
  return results


def _record_output_path(path):
  """Where recorded baselines can actually be retrieved from.

  Under a test runner the source tree is a sandbox copy that is deleted
  afterwards, so the result has to go where the harness collects artifacts.
  """
  for var in ('WORKLOAD_ARTIFACTS_DIR', 'TEST_UNDECLARED_OUTPUTS_DIR'):
    directory = os.environ.get(var)
    if directory:
      return os.path.join(directory, os.path.basename(path))
  return path


def _record_baselines(cfg, results, sigma_k, path, fw, iters):
  """Overwrite this framework's baselines and floors from these measurements."""
  for c, r in results:
    for d in ('d2h', 'h2d'):
      c[f'baseline_{d}'] = round(r[f'{d}_gbps'], 1)
      c[f'floor_{d}'] = round(_core_floor(r[f'{d}_gbps_all'], sigma_k), 1)
  out_path = _record_output_path(path)
  with open(out_path, 'w') as f:
    json.dump(cfg, f, indent=2)
  print(
      f'Recorded {fw} baselines+floors '
      f'(iters={iters}, sigma_k={sigma_k}) -> {out_path}'
  )
  # Also echo it: on a remote runner the log is the one artifact always kept.
  print(json.dumps(cfg, indent=2))


def _gate(results, fw, iters, sigma_k):
  """Print the comparison table; return how many directions are below floor."""
  print(
      f'\n{fw.upper()} perf gate: median of {iters} iters vs per-config floor'
      f' (median - {sigma_k} robust-sigmas / MAD)\n'
  )
  print(
      f"{'config':30}{'dir':4}{'baseline':>9}{'floor':>9}{'median':>9}"
      f"{'drop':>7}  verdict"
  )
  fails = 0
  for c, r in results:
    label = f"{c['dtype']} L{c['num_layers']} {'x'.join(map(str, c['shape']))}"
    for d in ('d2h', 'h2d'):
      base = float(c[f'baseline_{d}'])
      floor = float(c[f'floor_{d}'])
      med = r[f'{d}_gbps']
      ok = med >= floor
      fails += not ok
      print(
          f'{label:30}{d:4}{base:9.1f}{floor:9.1f}{med:9.1f}'
          f'{(base-med)/base*100:6.1f}% '
          f" {'PASS' if ok else 'FAIL <-- REGRESSION'}"
      )
  return fails


def main(_):
  fw_choice = _FRAMEWORK.value.lower()
  frameworks = ['jax', 'torch'] if fw_choice == 'all' else [fw_choice]
  if _BASELINES.value and len(frameworks) > 1:
    raise ValueError(
        '--baselines names a single file but --framework=all needs one per '
        'framework. Run each framework separately.'
    )

  total_configs = 0
  blocking_fails = 0
  reported_fails = 0

  for fw in frameworks:
    path = _baselines_path(fw)
    with open(path) as f:
      cfg = json.load(f)
    sigma_k = float(cfg.get('sigma_k', 3.5))     # robust sigmas below median
    warmup = int(cfg.get('warmup', 3))
    iters = int(cfg.get('iters', 20))
    if _ITERS.value is not None:
      iters = _ITERS.value

    results = _measure_configs(cfg['configs'], fw, iters, warmup)

    if _RECORD.value:
      _record_baselines(cfg, results, sigma_k, path, fw, iters)
      continue

    # per-config throughput to TB for the BAP dashboard (gate mode only)
    _write_tb_metrics(results, framework=fw)
    total_configs += len(results)
    fails = _gate(results, fw, iters, sigma_k)
    enforce = (
        _ENFORCE.value
        if _ENFORCE.value is not None
        else cfg.get('enforce', True)
    )
    if enforce:
      blocking_fails += fails
    else:
      reported_fails += fails

  if _RECORD.value:
    return

  print()
  if reported_fails:
    print(
        f'{reported_fails} direction(s) below the floor in a framework with'
        ' "enforce": false -- reported, NOT blocking'
    )
  if not blocking_fails:
    print(
        f'GATE PASS: all {total_configs} configs across {frameworks}'
        ' at/above their floor'
    )
    return

  # A perf-floor regression, blocking unless the author opted out via a tag.
  msg = f'GATE FAIL: {blocking_fails} direction(s) below the floor'
  opt = _opted_out()
  if opt:
    print(msg + f'   [report-only: {opt} bypass detected, NOT blocking]')
    return
  print(
      msg
      + f'   (to bypass: add {_SKIP_TAGS[0]} to your CL description, or pass'
      ' --enforce=false / set SKIP_PERF_GATE=1)',
      file=sys.stderr,
  )
  sys.exit(1)


def _parse_known_flags(args):
  """Ignore flags owned by the test runner rather than this binary."""
  return flags.FLAGS(args, known_only=True)


if __name__ == '__main__':
  # The torch path measures from worker processes (perf_core._measure_torch),
  # and hermetic Python has no interpreter for spawn to re-exec. handle_main
  # supplies one and routes spawned children back into the worker.
  multiprocessing.set_start_method('spawn', force=True)
  app.run(main, flags_parser=_parse_known_flags)
