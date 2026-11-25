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

"""Prepares an unpacked tpu_sync_jax wheel directory for multi-ABI repacking."""

import pathlib
import re
import sys

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_JAX_REQ_RE = re.compile(
    r"^Requires-Dist:\s*(jax|jaxlib|libtpu)(?=[\s;=<>!~\[(]|$)"
)
_FORBIDDEN_SOS = (
    "tpu_sync/frameworks/jax/_kv_cache_manager_ffi.so",
    "tpu_sync/frameworks/jax/_weight_synchronizer_ffi.so",
    "tpu_sync/solutions/mooncake/jax/_mooncake_provider.so",
)
_SHIM_CODE = '''"""Runtime ABI-dispatch loader for _tpu_raiden_jax."""

import importlib.machinery as _mach
import importlib.metadata as _meta
import importlib.util as _util
import pathlib
import re
import sys


def _load() -> object:
  raw = None
  for dist in ("jaxlib", "jax"):
    try:
      raw = _meta.version(dist)
      break
    except _meta.PackageNotFoundError:
      pass
  if raw is None:
    raise ImportError("tpu_sync_jax requires jax/jaxlib to be installed.")
  pkg = pathlib.Path(__file__).resolve().parent
  m = re.match(r"^(\\d+)\\.(\\d+)\\.(\\d+)$", raw.strip())
  so = pkg / f"_tpu_raiden_jax_{m[1]}_{m[2]}_{m[3]}.so" if m else None
  if so is None or not so.is_file():
    avail = [
        p.stem.removeprefix("_tpu_raiden_jax_").replace("_", ".")
        for p in sorted(pkg.glob("_tpu_raiden_jax_*.so"))
    ]
    raise ImportError(
        f"JAX {raw!r} has no bundled _tpu_raiden_jax ABI ({', '.join(avail)})."
    )
  loader = _mach.ExtensionFileLoader("_tpu_raiden_jax", str(so))
  spec = _util.spec_from_file_location(
      __name__, so, loader=loader, submodule_search_locations=None
  )
  mod = _util.module_from_spec(spec)
  prev = sys.modules.get(__name__)
  sys.modules[__name__] = mod
  try:
    loader.exec_module(mod)
  except BaseException:
    sys.modules.pop(__name__, None)
    if prev is not None:
      sys.modules[__name__] = prev
    raise
  if __package__ in sys.modules:
    setattr(sys.modules[__package__], "_tpu_raiden_jax", mod)
  return mod


_load()
'''


def _parse_ver(v: str) -> tuple[int, int, int]:
  m = _VERSION_RE.match(v)
  if not m:
    raise ValueError(f"Invalid X.Y.Z JAX version: {v!r}")
  return int(m[1]), int(m[2]), int(m[3])


def compute_jax_requires_dist(versions: list[str]) -> str:
  """Computes the PEP 508 jax[tpu] Requires-Dist constraint for versions."""
  parsed = sorted({_parse_ver(v): v for v in versions}.items())
  if not parsed:
    raise ValueError("versions must not be empty")
  if len(parsed) == 1:
    return f"jax[tpu]=={parsed[0][1]}"
  (lo_maj, lo_min, lo_patch), lo_s = parsed[0]
  (hi_maj, hi_min, _), hi_s = parsed[-1]
  specs = [f">={lo_s}", f"<={hi_s}"]
  if lo_maj == hi_maj:
    parsed_vers = {ver for ver, _ in parsed}
    kept_minors = {ver[1] for ver in parsed_vers}
    for m in range(lo_min, hi_min + 1):
      if m not in kept_minors:
        specs.append(f"!={lo_maj}.{m}.*")
      else:
        p_start = lo_patch + 1 if m == lo_min else 0
        p_end = max(p for _, mn, p in parsed_vers if mn == m)
        for p in range(p_start, p_end):
          if (lo_maj, m, p) not in parsed_vers:
            specs.append(f"!={lo_maj}.{m}.{p}")
  return "jax[tpu]" + ",".join(specs)


def rewrite_metadata(metadata_text: str, versions: list[str]) -> str:
  """Rewrites jax/jaxlib/libtpu Requires-Dist headers in wheel METADATA."""
  req = f"Requires-Dist: {compute_jax_requires_dist(versions)}"
  header, sep, body = metadata_text.partition("\n\n")
  out, done = [], False
  for line in header.splitlines():
    if _JAX_REQ_RE.match(line):
      if not done:
        out.append(req)
        done = True
    else:
      out.append(line)
  if not done:
    out.append(req)
  return "\n".join(out) + (
      sep + body if sep else ("\n" if metadata_text.endswith("\n") else "")
  )


def prepare_unpacked_jax_wheel(
    pkg_dir: pathlib.Path, versions: list[str]
) -> None:
  """Injects per-ABI JAX loader shim and rewrites METADATA in pkg_dir."""
  ordered = [v for _, v in sorted({_parse_ver(v): v for v in versions}.items())]
  if not ordered:
    raise ValueError("versions must not be empty")
  for rel in _FORBIDDEN_SOS:
    if (pkg_dir / rel).exists():
      raise RuntimeError(f"Forbidden standalone extension in wheel: {rel}")
  jax_dir = pkg_dir / "tpu_sync/frameworks/jax"
  base_so = jax_dir / "_tpu_raiden_jax.so"
  if not base_so.is_file():
    raise RuntimeError(f"Missing base extension: {base_so}")
  base_so.unlink()
  expected = {f"_tpu_raiden_jax_{v.replace('.', '_')}.so" for v in ordered}
  actual = {p.name for p in jax_dir.glob("_tpu_raiden_jax_*.so")}
  if actual != expected:
    raise RuntimeError(
        f"Variant .so mismatch: {sorted(actual)} != {sorted(expected)}"
    )
  (jax_dir / "_tpu_raiden_jax.py").write_text(_SHIM_CODE, encoding="utf-8")
  metas = list(pkg_dir.glob("*.dist-info/METADATA"))
  if len(metas) != 1:
    raise RuntimeError(f"Expected 1 .dist-info/METADATA, found {len(metas)}")
  metas[0].write_text(
      rewrite_metadata(metas[0].read_text("utf-8"), ordered), "utf-8"
  )


def main(argv: list[str]) -> int:
  """CLI entry point for preparing an unpacked tpu_sync_jax wheel."""
  if len(argv) < 2:
    raise SystemExit(
        "usage: pack_jax_wheel.py <unpacked_wheel_dir> <jax_ver> [...]"
    )
  prepare_unpacked_jax_wheel(pathlib.Path(argv[0]), argv[1:])
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
