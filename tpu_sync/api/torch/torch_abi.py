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

"""Torch-ABI variant selection for the raiden torch extension.

The torch extension links libtorch symbols and torch_tpu's per-torch-version
glue library, so one binary serves exactly one torch ABI generation. A
multi-ABI wheel bundles one extension per supported torch release, named
``_tpu_raiden_torch_<major>_<minor>_<patch>.so``, and this module picks the
variant matching the installed torch at first use.

Selection mirrors torch_tpu's glue policy: variants are built only for the
torch releases that broke ABI compatibility with the previous one, so each
variant serves every release from its own version up to the next variant's,
and the newest variant serves everything after it. The choice is therefore
the newest bundled variant whose version is <= the running torch.
"""

import importlib.machinery
import importlib.util
import pathlib
import re
import sys


def running_torch_suffix() -> str | None:
  """Returns the installed torch's version as a suffix ("2_11_0"), or None."""
  try:
    # pylint: disable=g-import-not-at-top
    import torch
    # pylint: enable=g-import-not-at-top
  except ImportError:
    return None
  m = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__)
  if m is None:
    return None
  return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"


def _parse(suffix: str) -> tuple[int, ...]:
  return tuple(int(p) for p in suffix.split("_"))


def bundled_suffixes(package_dir: pathlib.Path, stem: str) -> list[str]:
  """Version suffixes of the variants bundled next to ``stem`` in the wheel."""
  pattern = re.compile(re.escape(stem) + r"_(\d+_\d+_\d+)\.so$")
  return [
      m.group(1)
      for so in package_dir.glob(f"{stem}_*.so")
      if (m := pattern.match(so.name)) is not None
  ]


def resolve_suffix(running: str | None, built: list[str]) -> str:
  """Chooses the bundled variant for the running torch.

  Raises:
    ImportError: if the running torch version cannot be determined or
      predates every bundled variant.
  """
  supported = ", ".join(s.replace("_", ".") for s in sorted(built, key=_parse))
  if running is None:
    raise ImportError(
        "tpu_raiden could not determine the installed torch version, so it "
        "cannot select a compatible torch extension. Extensions are bundled "
        f"for torch: {supported}."
    )
  candidates = [b for b in built if _parse(b) <= _parse(running)]
  if not candidates:
    raise ImportError(
        "tpu_raiden has no torch extension compatible with torch "
        f"{running.replace('_', '.')}: it predates the oldest bundled "
        f"variant. Extensions are bundled for torch: {supported}."
    )
  return max(candidates, key=_parse)


def _preload_torch_tpu_glue(suffix: str | None) -> None:
  """Preloads matching torch_tpu glue library into RTLD_GLOBAL so symbols resolve."""
  try:
    import torch_tpu  # pylint: disable=g-import-not-at-top
    import ctypes  # pylint: disable=g-import-not-at-top
    tpu_dir = pathlib.Path(torch_tpu.__file__).resolve().parent
    candidates = []
    if suffix:
      candidates.append(tpu_dir / f"libpywrap_{suffix}_common.so")
      candidates.append(tpu_dir / "common" / f"glue_{suffix}" / f"libpywrap_{suffix}_common.so")
    candidates.append(tpu_dir / "common" / "libpywrap_torch_tpu_common.so")
    for glue_path in candidates:
      if glue_path.exists():
        ctypes.CDLL(str(glue_path), mode=ctypes.RTLD_GLOBAL)
        break
  except Exception:  # pylint: disable=broad-except
    pass


def load_extension(package: str, stem: str):
  """Imports the extension ``<package>.<stem>``, dispatching on torch ABI.

  A source checkout (or a single-ABI wheel) carries an unversioned
  ``<stem>.so`` and imports normally. A multi-ABI wheel carries only
  versioned variants; the matching one is loaded under the unversioned
  module name, so ``PyInit_<stem>`` resolves and importers of the
  unversioned name keep working.
  """
  name = f"{package}.{stem}"
  if name in sys.modules:
    return sys.modules[name]

  running_suffix = running_torch_suffix()
  _preload_torch_tpu_glue(running_suffix)

  pkg = importlib.import_module(package)
  # Works for regular and namespace packages alike (__file__ is None for the
  # latter); __path__ always carries the package directory.
  package_dir = pathlib.Path(next(iter(pkg.__path__))).resolve()
  built = bundled_suffixes(package_dir, stem)
  if not built:
    # Source checkout / single-ABI layout: the unversioned extension.
    return importlib.import_module(name)

  # Version-suffixed variants take precedence over an unversioned <stem>.so:
  # environments upgraded in place from a pre-variant install can carry a
  # stale unversioned extension alongside the wheel's variants.
  suffix = resolve_suffix(running_suffix, built)
  path = package_dir / f"{stem}_{suffix}.so"

  # The variant file still exports PyInit_<stem>; the loader's module name
  # (not the filename) determines the init symbol CPython looks up.
  loader = importlib.machinery.ExtensionFileLoader(stem, str(path))
  spec = importlib.util.spec_from_file_location(
      name, path, loader=loader, submodule_search_locations=None
  )
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  try:
    spec.loader.exec_module(module)
  except BaseException:
    del sys.modules[name]
    raise
  setattr(pkg, stem, module)
  return module

