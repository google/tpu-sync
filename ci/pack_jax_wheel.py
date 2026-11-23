#!/usr/bin/env python3

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
"""Packs per-JAX-version _tpu_raiden_jax.so variants into a multi-ABI wheel.

Reads a stage directory containing one base wheel from
//ci/wheel:raiden_jax_wheel and one or more _tpu_raiden_jax_<M>_<m>_<p>.so
variants, replaces the unversioned .so with the versioned variants, injects
self-contained compatibility shims for _tpu_raiden_jax and its consolidated FFI
submodules, updates METADATA Requires-Dist, and rewrites RECORD.

Uses only the Python standard library so it runs on stock python3 in both local
builds and lightweight CI pack jobs.
"""

import argparse
import base64
import csv
import hashlib
import io
import os
import pathlib
import re
import sys
import zipfile

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_STAGED_SO_RE = re.compile(r"^_tpu_raiden_jax_(\d+)_(\d+)_(\d+)\.so$")
_BASE_EXTENSION_PATH = "tpu_sync/frameworks/jax/_tpu_raiden_jax.so"
_FORBIDDEN_EXTENSION_PATHS = frozenset({
    "tpu_sync/frameworks/jax/_kv_cache_manager_ffi.so",
    "tpu_sync/frameworks/jax/_weight_synchronizer_ffi.so",
    "tpu_sync/solutions/mooncake/jax/_mooncake_provider.so",
})
_JAX_REQUIRES_RE = re.compile(r"^Requires-Dist:\s*(jax|jaxlib|libtpu)\b")

# Self-contained ABI dispatcher shipped inside multi-ABI wheels. Never imports
# jax/jaxlib (uses importlib.metadata) so callers can preload _tpu_raiden_jax
# before importing jax to register Raiden's XLA protobuf descriptors ahead of
# libtpu.
_TPU_RAIDEN_JAX_SHIM = '''"""ABI-dispatching shim for the consolidated _tpu_raiden_jax extension."""

import importlib.machinery
import importlib.metadata
import importlib.util
import pathlib
import re
import sys

_EXACT_VERSION_RE = re.compile(r"(\\d+)\\.(\\d+)\\.(\\d+)")
_VARIANT_RE = re.compile(r"^_tpu_raiden_jax_(\\d+_\\d+_\\d+)\\.so$")


def _running_jax_version():
  for dist_name in ("jaxlib", "jax"):
    try:
      return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
      continue
  return None


def _load_variant():
  pkg_dir = pathlib.Path(__file__).resolve().parent
  built = [
      m.group(1)
      for so in pkg_dir.glob("_tpu_raiden_jax_*.so")
      if (m := _VARIANT_RE.match(so.name)) is not None
  ]
  if not built:
    raise ImportError(
        f"tpu_raiden found no versioned _tpu_raiden_jax_*.so in {pkg_dir}."
    )
  supported = ", ".join(
      s.replace("_", ".")
      for s in sorted(built, key=lambda s: tuple(int(p) for p in s.split("_")))
  )
  version = _running_jax_version()
  if version is None:
    raise ImportError(
        "tpu_raiden could not determine the installed jaxlib/jax version, so "
        "it cannot select a compatible JAX extension. Extensions are bundled "
        f"for JAX: {supported}."
    )
  m = _EXACT_VERSION_RE.fullmatch(version)
  suffix = f"{m.group(1)}_{m.group(2)}_{m.group(3)}" if m is not None else None
  if suffix not in built:
    raise ImportError(
        "tpu_raiden has no JAX extension compatible with JAX "
        f"{version}: JAX/XLA C++ ABI requires an exact version match. "
        f"Extensions are bundled for JAX: {supported}."
    )
  path = pkg_dir / f"_tpu_raiden_jax_{suffix}.so"
  loader = importlib.machinery.ExtensionFileLoader("_tpu_raiden_jax", str(path))
  spec = importlib.util.spec_from_file_location(
      __name__, path, loader=loader, submodule_search_locations=None
  )
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not create module spec for {__name__} at {path}.")
  module = importlib.util.module_from_spec(spec)
  previous = sys.modules.get(__name__)
  sys.modules[__name__] = module
  try:
    spec.loader.exec_module(module)
  except BaseException:
    if previous is None:
      sys.modules.pop(__name__, None)
    else:
      sys.modules[__name__] = previous
    raise
  if __package__ and __package__ in sys.modules:
    setattr(sys.modules[__package__], "_tpu_raiden_jax", module)
  return module


_load_variant()
'''

_WHEEL_SHIMS = {
    "tpu_sync/frameworks/jax/_tpu_raiden_jax.py": _TPU_RAIDEN_JAX_SHIM,
    "tpu_sync/frameworks/jax/_weight_synchronizer_ffi.py": (
        '"""Compatibility shim routing _weight_synchronizer_ffi to'
        ' _tpu_raiden_jax."""\n\n'
        "import sys\n"
        "from . import _tpu_raiden_jax\n\n"
        "sys.modules[__name__] = _tpu_raiden_jax.weight_synchronizer_ffi\n"
    ),
    "tpu_sync/frameworks/jax/_kv_cache_manager_ffi.py": (
        '"""Compatibility shim routing _kv_cache_manager_ffi to'
        ' _tpu_raiden_jax."""\n\n'
        "import sys\n"
        "from . import _tpu_raiden_jax\n\n"
        "sys.modules[__name__] = _tpu_raiden_jax.kv_cache_manager_ffi\n"
    ),
}


def _parse_version(version: str) -> tuple[int, int, int]:
  match = _VERSION_RE.match(version)
  if match is None:
    raise ValueError(
        f"Invalid JAX version {version!r}; expected MAJOR.MINOR.PATCH"
    )
  return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def discover_stage_dir(
    stage_dir: pathlib.Path,
) -> tuple[pathlib.Path, list[tuple[str, pathlib.Path]]]:
  """Finds the single base wheel and versioned .so variants in ``stage_dir``."""
  wheels = sorted(stage_dir.glob("tpu_sync_jax-*.whl"))
  if len(wheels) != 1:
    raise ValueError(
        f"Expected exactly 1 base wheel in {stage_dir}, found {wheels}"
    )
  sos = sorted(stage_dir.glob("_tpu_raiden_jax_*.so"))
  if not sos:
    raise ValueError(
        f"No staged _tpu_raiden_jax_*.so files found in {stage_dir}"
    )
  variants: dict[str, pathlib.Path] = {}
  for so in sos:
    match = _STAGED_SO_RE.match(so.name)
    if match is None:
      raise ValueError(f"Unexpected staged extension filename: {so.name}")
    version = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    variants[version] = so
  ordered = sorted(variants.items(), key=lambda kv: _parse_version(kv[0]))
  return wheels[0], ordered


def compute_jax_requires_dist(versions: list[str]) -> str:
  """Returns the Requires-Dist value for the bundled JAX versions."""
  ordered = sorted(versions, key=_parse_version)
  if len(ordered) == 1:
    return f"jax[tpu]=={ordered[0]}"
  return f"jax[tpu]>={ordered[0]},<={ordered[-1]}"


def rewrite_metadata(metadata_text: str, versions: list[str]) -> str:
  """Rewrites single-version jax/jaxlib/libtpu pins to jax[tpu] bounds."""
  if "\r\n\r\n" in metadata_text:
    line_sep = "\r\n"
    header_block, body = metadata_text.split("\r\n\r\n", 1)
    has_body = True
  elif "\n\n" in metadata_text:
    line_sep = "\n"
    header_block, body = metadata_text.split("\n\n", 1)
    has_body = True
  else:
    line_sep = "\n"
    header_block, body = metadata_text, ""
    has_body = False

  new_headers: list[str] = []
  inserted_idx: int | None = None
  for line in header_block.splitlines():
    if _JAX_REQUIRES_RE.match(line):
      if inserted_idx is None:
        inserted_idx = len(new_headers)
      continue
    new_headers.append(line)

  jax_req_line = f"Requires-Dist: {compute_jax_requires_dist(versions)}"
  if inserted_idx is None:
    new_headers.append(jax_req_line)
  else:
    new_headers.insert(inserted_idx, jax_req_line)

  rebuilt_headers = line_sep.join(new_headers)
  if has_body:
    return rebuilt_headers + line_sep + line_sep + body
  return rebuilt_headers + line_sep


def _record_entry(arcname: str, data: bytes) -> tuple[str, str, str]:
  digest = (
      base64.urlsafe_b64encode(hashlib.sha256(data).digest())
      .rstrip(b"=")
      .decode("ascii")
  )
  return (arcname, f"sha256={digest}", str(len(data)))


def _make_zipinfo(
    arcname: str,
    mode: int,
    date_time: tuple[int, int, int, int, int, int] = (2026, 1, 1, 0, 0, 0),
) -> zipfile.ZipInfo:
  info = zipfile.ZipInfo(filename=arcname, date_time=date_time)
  info.compress_type = zipfile.ZIP_DEFLATED
  info.external_attr = (mode & 0xFFFF) << 16
  return info


def pack_jax_wheel(
    base_wheel: pathlib.Path,
    variants: list[tuple[str, pathlib.Path]],
    output_dir: pathlib.Path,
    pins_file: pathlib.Path | None = None,
) -> pathlib.Path:
  """Repacks base_wheel with versioned JAX extensions and compatibility shims."""
  if not base_wheel.name.endswith(".whl"):
    raise ValueError(f"Expected .whl filename, got {base_wheel.name!r}")
  versions = [v for v, _ in variants]
  jax_req = compute_jax_requires_dist(versions)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_name = base_wheel.name
  out_path = output_dir / out_name
  tmp_path = output_dir / f".{out_name}.tmp.{os.getpid()}"

  preserved: list[tuple[zipfile.ZipInfo, bytes]] = []
  dist_info_prefix: str | None = None
  found_base_ext = False

  with zipfile.ZipFile(base_wheel, "r") as zin:
    for info in zin.infolist():
      name = info.filename
      if name in _FORBIDDEN_EXTENSION_PATHS:
        raise RuntimeError(
            f"Base wheel {base_wheel} unexpectedly contains unconsolidated"
            f" extension {name}; expected consolidation into"
            f" {_BASE_EXTENSION_PATH}."
        )
      if name == _BASE_EXTENSION_PATH:
        found_base_ext = True
        continue
      if name in _WHEEL_SHIMS:
        continue
      if name.endswith(".dist-info/RECORD"):
        dist_info_prefix = name.rsplit("/", 1)[0]
        continue
      data = zin.read(name)
      if name.endswith(".dist-info/METADATA"):
        dist_info_prefix = name.rsplit("/", 1)[0]
        data = rewrite_metadata(data.decode("utf-8"), versions).encode("utf-8")
      preserved.append((info, data))

  if not found_base_ext:
    raise RuntimeError(
        f"Base wheel {base_wheel} is missing {_BASE_EXTENSION_PATH}."
    )
  if dist_info_prefix is None:
    raise RuntimeError(f"Base wheel {base_wheel} is missing .dist-info.")

  record_rows: list[tuple[str, str, str]] = []
  record_path = f"{dist_info_prefix}/RECORD"

  try:
    with zipfile.ZipFile(
        tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zout:
      for orig_info, data in preserved:
        new_info = zipfile.ZipInfo(
            filename=orig_info.filename, date_time=orig_info.date_time
        )
        new_info.compress_type = zipfile.ZIP_DEFLATED
        new_info.external_attr = orig_info.external_attr
        zout.writestr(new_info, data)
        record_rows.append(_record_entry(orig_info.filename, data))

      for version, so_path in variants:
        major, minor, patch = _parse_version(version)
        arcname = f"tpu_sync/frameworks/jax/_tpu_raiden_jax_{major}_{minor}_{patch}.so"
        so_bytes = so_path.read_bytes()
        zout.writestr(_make_zipinfo(arcname, 0o100755), so_bytes)
        record_rows.append(_record_entry(arcname, so_bytes))

      for arcname, content in sorted(_WHEEL_SHIMS.items()):
        shim_bytes = content.encode("utf-8")
        zout.writestr(_make_zipinfo(arcname, 0o100644), shim_bytes)
        record_rows.append(_record_entry(arcname, shim_bytes))

      record_buf = io.StringIO()
      writer = csv.writer(record_buf, lineterminator="\n")
      for row in record_rows:
        writer.writerow(row)
      writer.writerow((record_path, "", ""))
      zout.writestr(
          _make_zipinfo(record_path, 0o100644),
          record_buf.getvalue().encode("utf-8"),
      )

    os.replace(tmp_path, out_path)
  except BaseException:
    if tmp_path.exists():
      tmp_path.unlink()
    raise

  if pins_file is not None:
    pins_file.parent.mkdir(parents=True, exist_ok=True)
    pins_file.write_text(f"{jax_req}\n", encoding="utf-8")

  size_mib = out_path.stat().st_size / (1024 * 1024)
  print(
      f"Packed {out_path.name} ({size_mib:.2f} MiB) with JAX ABIs: "
      f"{', '.join(versions)} ({jax_req})"
  )
  return out_path


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--stage-dir",
      required=True,
      type=pathlib.Path,
      help=(
          "Directory containing one base tpu_sync_jax-*.whl and one or more"
          " _tpu_raiden_jax_<ver>.so variants."
      ),
  )
  parser.add_argument(
      "--output-dir",
      required=True,
      type=pathlib.Path,
      help="Directory where the repacked wheel will be written.",
  )
  parser.add_argument(
      "--pins-file",
      type=pathlib.Path,
      default=None,
      help="Optional path to write the wheel's JAX Requires-Dist pin.",
  )
  args = parser.parse_args(argv)
  base_wheel, variants = discover_stage_dir(args.stage_dir)
  pack_jax_wheel(base_wheel, variants, args.output_dir, args.pins_file)
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
