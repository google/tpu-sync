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

"""Distribution statistics, gate floors, config ranking and SVG plots for the
H2H benchmarks.

Stdlib only: the benchmark pip hub has no matplotlib, and the driver must not
pull anything heavy before the timed transfers. Everything here is pure
functions over lists of GB/s samples so it can be unit-tested without a TPU.

Two levels of statistics matter:
  * per-iteration samples: what the runner measures (spread = cv_robust,
    low_tail_frac);
  * the gate statistic: the gate compares the MEDIAN of `gate_iters`
    iterations against the floor, and a median is far tighter than a single
    iteration. `assess()` bootstraps that median from the pooled samples and
    reports p_below_floor -- the estimated probability that a healthy gate run
    trips the floor (i.e. the gate's false-alarm rate).
"""

import html
import json
import math
import random
import statistics

# A config is "suitable" for a throughput gate when all of these hold. They are
# deliberately conservative: a gate that flaps is worse than no gate.
DEFAULT_MAX_CV_ROBUST = 0.10     # MAD-sigma / median, per iteration
DEFAULT_MAX_LOW_TAIL = 0.05      # fraction of samples below 0.8 * median
DEFAULT_MAX_P_BELOW_FLOOR = 0.001  # bootstrap false-alarm rate of the gate
FLOOR_K_CANDIDATES = (2.5, 3.5, 5.0)
BOOTSTRAP_ROUNDS = 2000


def pct(xs, q):
  """q-th percentile (0..100) with linear interpolation. xs non-empty."""
  xs = sorted(xs)
  if len(xs) == 1:
    return xs[0]
  pos = (len(xs) - 1) * (q / 100.0)
  lo = int(pos)
  hi = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mad_sigma(xs):
  """1.4826 * median(|x - median|): an outlier-resistant stddev estimate."""
  if len(xs) < 2:
    return 0.0
  med = statistics.median(xs)
  return 1.4826 * statistics.median([abs(x - med) for x in xs])


def core_floor(samples, k, max_margin):
  """Gate floor: max(median - k * MAD_sigma, median * (1 - max_margin)).

  Identical to h2h_cpp_gate._core_floor (h2d_d2h_benchmark_gating uses the
  uncapped median - k * sigma). The cap means the floor is a flat max_margin
  drop unless the config is tight enough that k * cv_robust < max_margin.
  """
  med = statistics.median(samples)
  return max(med - k * mad_sigma(samples), med * (1.0 - max_margin))


def bootstrap_medians(samples, gate_iters, rounds=BOOTSTRAP_ROUNDS, seed=0):
  """Medians of `rounds` resamples of size gate_iters (with replacement)."""
  if not samples or gate_iters <= 0:
    return []
  rng = random.Random(seed)
  return [statistics.median(rng.choices(samples, k=gate_iters))
          for _ in range(rounds)]


def summarize(samples):
  """Descriptive stats for one config. Returns {} for no samples."""
  if not samples:
    return {}
  med = statistics.median(samples)
  sigma = mad_sigma(samples)
  low_tail = sum(1 for x in samples if x < 0.8 * med) / len(samples)
  return {
      'n': len(samples),
      'median': med,
      'mean': statistics.fmean(samples),
      'stdev': statistics.pstdev(samples) if len(samples) > 1 else 0.0,
      'mad_sigma': sigma,
      'cv_robust': (sigma / med) if med > 0 else float('inf'),
      'p10': pct(samples, 10),
      'p50': med,
      'p90': pct(samples, 90),
      'min': min(samples),
      'max': max(samples),
      'low_tail_frac': low_tail,
  }


def assess(samples, integrity, min_samples, max_margin, k=3.5, gate_iters=50,
           max_cv=DEFAULT_MAX_CV_ROBUST, max_low_tail=DEFAULT_MAX_LOW_TAIL,
           max_p_below_floor=DEFAULT_MAX_P_BELOW_FLOOR, bootstrap=True):
  """summarize() plus floors, the bootstrapped gate statistic and a verdict.

  Keys added: floor_candidates (UNCAPPED median - k*sigma for each k in
  FLOOR_K_CANDIDATES, so the effect of k is visible), cap, floor_effective
  (= core_floor with the requested k, what the gate uses), gate_iters,
  gate_median_sigma, gate_cv, gate_p01 (1st percentile of bootstrapped medians),
  p_below_floor, suitable, reasons.
  """
  s = summarize(samples)
  reasons = []
  if not integrity:
    reasons.append('integrity FAILED')
  if not s:
    reasons.append('no samples')
    return {'integrity': bool(integrity), 'suitable': False,
            'reasons': reasons, 'floor_candidates': {}}
  med, sigma = s['median'], s['mad_sigma']
  s['integrity'] = bool(integrity)
  s['floor_candidates'] = {f'{kk:.1f}': med - kk * sigma
                           for kk in FLOOR_K_CANDIDATES}
  s['cap'] = med * (1.0 - max_margin)
  s['floor_effective'] = core_floor(samples, k, max_margin)
  if bootstrap:
    meds = bootstrap_medians(samples, gate_iters)
    gsig = mad_sigma(meds)
    s['gate_iters'] = gate_iters
    s['gate_median_sigma'] = gsig
    s['gate_cv'] = (gsig / med) if med > 0 else float('inf')
    s['gate_p01'] = pct(meds, 1) if meds else None
    s['p_below_floor'] = (sum(1 for m in meds if m < s['floor_effective']) /
                          len(meds)) if meds else 1.0
  else:
    s['gate_iters'] = gate_iters
    s['gate_median_sigma'] = None
    s['gate_cv'] = None
    s['gate_p01'] = None
    s['p_below_floor'] = None

  if s['n'] < min_samples:
    reasons.append(f'n={s["n"]} < min_samples={min_samples}')
  if s['cv_robust'] > max_cv:
    reasons.append(f'cv_robust={s["cv_robust"]:.3f} > {max_cv}')
  if s['low_tail_frac'] > max_low_tail:
    reasons.append(f'low_tail_frac={s["low_tail_frac"]:.3f} > {max_low_tail}')
  if s['p_below_floor'] is not None and s['p_below_floor'] > max_p_below_floor:
    reasons.append(f'p_below_floor={s["p_below_floor"]:.4f} > '
                   f'{max_p_below_floor} (gate would flap)')
  s['suitable'] = not reasons
  s['reasons'] = reasons
  return s


def rank(per_config):
  """Best-first: suitable, byte-correct, lowest flap probability, lowest cv.

  A config whose integrity check failed is a bug to fix, not a gate candidate,
  so it sorts below every byte-correct config regardless of its spread.
  """
  def key(item):
    label, a = item
    return (0 if a.get('suitable') else 1,
            0 if a.get('integrity') else 1,
            a.get('p_below_floor', 1.0),
            a.get('cv_robust', float('inf')),
            label)
  return [label for label, _ in sorted(per_config.items(), key=key)]


def render_markdown(stage, per_config, ranking, k, max_margin, gate_iters=50,
                    suggested=None):
  """Markdown report: one row per config, the ranking, and (when `suggested`
  = {'configs_flag': str, 'metrics': [str]}) the registry lines to paste into
  the record and gate registries."""
  out = [f'# H2H multi-host {stage}', '',
         f'floor = max(median - {k} * MAD_sigma, median * (1 - {max_margin})); '
         f'the gate compares the median of {gate_iters} iterations against it. '
         'floor(k=...) columns are UNCAPPED candidates; `floor` is the '
         'effective (capped) value; `p_below_floor` is the bootstrapped '
         f'probability that a healthy median-of-{gate_iters} run trips it.',
         '',
         '| config | n | median GB/s | MAD-sigma | cv_robust | p10 | p90 | '
         'low-tail | floor(k=2.5) | floor(k=3.5) | floor(k=5) | cap | floor | '
         f'gate_cv | p_below_floor | integrity | suitable |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for label in ranking:
    a = per_config[label]
    if 'n' not in a:
      out.append(f'| {label} | 0 | - | - | - | - | - | - | - | - | - | - | - | '
                 f'- | - | {"OK" if a.get("integrity") else "CORRUPT"} | no |')
      continue
    fc = a['floor_candidates']
    out.append(
        f'| {label} | {a["n"]} | {a["median"]:.3f} | {a["mad_sigma"]:.3f} | '
        f'{a["cv_robust"]:.3f} | {a["p10"]:.3f} | {a["p90"]:.3f} | '
        f'{a["low_tail_frac"]:.3f} | {fc["2.5"]:.3f} | {fc["3.5"]:.3f} | '
        f'{fc["5.0"]:.3f} | {a["cap"]:.3f} | {a["floor_effective"]:.3f} | '
        f'{a["gate_cv"]:.4f} | {a["p_below_floor"]:.4f} | '
        f'{"OK" if a["integrity"] else "CORRUPT"} | '
        f'{"**yes**" if a["suitable"] else "no"} |')
  out += ['', '## Ranking (best gate candidates first)', '']
  for i, label in enumerate(ranking, 1):
    a = per_config[label]
    why = 'suitable' if a.get('suitable') else '; '.join(a.get('reasons', []))
    out.append(f'{i}. `{label}` — {why}')
  if suggested is not None:
    out += ['', '## Next step: put the suitable configs into record and gate', '']
    if suggested['metrics']:
      out += ['In `benchmark_registry_h2h_multihost_record.pbtxt` and '
              '`benchmark_registry_h2h_multihost_gate.pbtxt`, replace `--suite=...` in '
              '`runtime_flags` with:', '', '```', suggested['configs_flag'], '```', '',
              'and replace the `metrics { ... }` lines with:', '', '```']
      out += suggested['metrics'] + ['```']
    else:
      out.append('No config is suitable yet: collect more samples (raise `--iters` / '
                 '`--runs_per_config`) or investigate the `reasons` column first.')
  out.append('')
  return '\n'.join(out)


def _hist(samples, bins):
  lo, hi = min(samples), max(samples)
  if hi <= lo:
    hi = lo + 1e-9
  width = (hi - lo) / bins
  counts = [0] * bins
  for x in samples:
    i = min(int((x - lo) / width), bins - 1)
    counts[i] += 1
  return lo, hi, counts


def render_svg(per_config_samples, per_config_floor=None, title='',
               bins=30, panel_w=820, panel_h=170):
  """One histogram panel per config; median (blue) and floor (red) lines.

  per_config_samples: {label: [gbs, ...]}; per_config_floor: {label: gbs}.
  Returns the SVG document as a string.
  """
  per_config_floor = per_config_floor or {}
  labels = list(per_config_samples.keys())
  n_panels = max(1, len(labels))
  total_h = 40 + n_panels * panel_h
  left, right, top, bottom = 60, 20, 34, 30
  plot_w = panel_w - left - right
  plot_h = panel_h - top - bottom

  out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_w}" '
         f'height="{total_h}" viewBox="0 0 {panel_w} {total_h}" '
         f'font-family="Helvetica, Arial, sans-serif" font-size="11">',
         f'<rect width="100%" height="100%" fill="white"/>',
         f'<text x="{panel_w / 2:.0f}" y="22" text-anchor="middle" '
         f'font-size="15">{html.escape(title)}</text>']

  for idx, label in enumerate(labels):
    samples = [x for x in per_config_samples[label] if x > 0]
    y0 = 40 + idx * panel_h
    out.append(f'<text x="{left}" y="{y0 + 14}" font-size="12" '
               f'font-weight="bold">{html.escape(label)}</text>')
    if not samples:
      out.append(f'<text x="{left + plot_w / 2:.0f}" y="{y0 + top + plot_h / 2:.0f}" '
                 f'text-anchor="middle" fill="#999">no samples</text>')
      continue
    lo, hi, counts = _hist(samples, bins)
    cmax = max(counts) or 1
    med = statistics.median(samples)
    floor = per_config_floor.get(label)
    # Extend the x-range so the floor line is visible when it lies below min.
    xlo = min(lo, floor) if floor else lo
    xhi = hi
    span = max(xhi - xlo, 1e-9)

    def sx(v, xlo=xlo, span=span):
      return left + (v - xlo) / span * plot_w

    base_y = y0 + top + plot_h
    out.append(f'<line x1="{left}" y1="{base_y}" x2="{left + plot_w}" '
               f'y2="{base_y}" stroke="#444"/>')
    bw = (hi - lo) / bins
    for i, c in enumerate(counts):
      if c == 0:
        continue
      x = sx(lo + i * bw)
      w = max(sx(lo + (i + 1) * bw) - x, 1.0)
      h = c / cmax * plot_h
      out.append(f'<rect x="{x:.1f}" y="{base_y - h:.1f}" width="{w:.1f}" '
                 f'height="{h:.1f}" fill="#7aa6d9" stroke="white" '
                 f'stroke-width="0.5"/>')
    mx = sx(med)
    out.append(f'<line x1="{mx:.1f}" y1="{y0 + top}" x2="{mx:.1f}" '
               f'y2="{base_y}" stroke="#1f4e9c" stroke-width="2"/>')
    out.append(f'<text x="{mx + 4:.1f}" y="{y0 + top + 12}" fill="#1f4e9c">'
               f'median {med:.2f}</text>')
    if floor:
      fx = sx(floor)
      out.append(f'<line x1="{fx:.1f}" y1="{y0 + top}" x2="{fx:.1f}" '
                 f'y2="{base_y}" stroke="#c0392b" stroke-width="2" '
                 f'stroke-dasharray="5,3"/>')
      out.append(f'<text x="{fx + 4:.1f}" y="{y0 + top + 26}" fill="#c0392b">'
                 f'floor {floor:.2f}</text>')
    for v in (xlo, xlo + span / 2, xhi):
      out.append(f'<text x="{sx(v):.1f}" y="{base_y + 14}" '
                 f'text-anchor="middle" fill="#555">{v:.2f}</text>')
    out.append(f'<text x="{left + plot_w}" y="{y0 + 14}" text-anchor="end" '
               f'fill="#555">n={len(samples)}  p10={pct(samples, 10):.2f}  '
               f'p90={pct(samples, 90):.2f}  GB/s</text>')
  out.append('</svg>')
  return '\n'.join(out)


def to_json(obj):
  """json.dumps with inf/nan made JSON-safe."""
  def fix(v):
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
      return None
    if isinstance(v, dict):
      return {k: fix(x) for k, x in v.items()}
    if isinstance(v, list):
      return [fix(x) for x in v]
    return v
  return json.dumps(fix(obj), indent=2)
