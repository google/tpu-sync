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

"""Local-scope loader for the torch_tpu common shared library."""

import ctypes
import importlib
import os
import pathlib
import re

import torch_tpu

from tpu_sync.api.torch import torch_abi

_torch_tpu_loader = importlib.import_module("torch_tpu._loader")


_LOADED = False


def load_torch_tpu_common() -> None:
  """Loads torch_tpu and its common library without global XLA symbols."""
  global _LOADED
  if _LOADED:
    return

  _torch_tpu_loader.load()
  # Wheel layout keeps the common library under torch_tpu/common/; a source
  # checkout / bazel runfiles tree keeps it at its build location,
  # torch_tpu/csrc/common/ (upstream source-layout consolidation). Probe the
  # package path both as imported and fully resolved: bazel runfiles are
  # symlink farms whose resolve() escapes to the source tree, where build
  # outputs do not exist.
  raw_pkg = pathlib.Path(torch_tpu.__file__).parent
  pkgs = [raw_pkg]
  if raw_pkg.resolve() != raw_pkg:
    pkgs.append(raw_pkg.resolve())
  commons = [p / "common" for p in pkgs]
  commons += [p / "csrc" / "common" for p in pkgs]
  lib = None
  source_layout = False
  for common in commons:
    candidate = common / "libpywrap_torch_tpu_common.so"
    if candidate.exists():
      lib = candidate
      source_layout = common.name == "common" and common.parent.name == "csrc"
      break
    # Per-torch-version glue layout: .../glue_<v>/libpywrap_<v>_common.so.
    # Load the glue matching the installed torch; its SONAME is what the
    # raiden extension's NEEDED entry names.
    built = [
        m.group(1)
        for d in common.glob("glue_*")
        if (m := re.match(r"glue_(\d+_\d+_\d+)$", d.name)) is not None
    ]
    if built:
      suffix = torch_abi.resolve_suffix(torch_abi.running_torch_suffix(), built)
      lib = common / f"glue_{suffix}" / f"libpywrap_{suffix}_common.so"
      break
  del source_layout  # Layouts differ only in where the library was found.
  if lib is not None and lib.exists():
    # RTLD_LOCAL on purpose: the extension reaches this library through its
    # own NEEDED entry (ld.so reuses the already-loaded object by soname), so
    # nothing here needs to enter the global scope. Loading it globally would
    # let its gRPC/XLA copies preempt the extension's statically linked ones
    # (shared registries, duplicate registrations, abort).
    ctypes.CDLL(str(lib), mode=os.RTLD_LOCAL | os.RTLD_NOW)
  _LOADED = True
