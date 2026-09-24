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

"""Fault and latency injection API for ``_tpu_raiden_torch``."""

from typing import Any, Mapping, Sequence

_IMPL = None


def _impl():
  """Returns the fault_injection submodule of the torch-backed extension."""
  global _IMPL
  if _IMPL is None:
    # pylint: disable=g-import-not-at-top
    from tpu_sync.api.torch import torch_abi
    from tpu_sync.api.torch import torch_tpu_common_loader

    torch_tpu_common_loader.load_torch_tpu_common()
    ext = torch_abi.load_extension(
        "tpu_sync.frameworks.torch",
        "_tpu_raiden_torch",
    )
    # pylint: enable=g-import-not-at-top
    _IMPL = getattr(ext, "fault_injection")
  return _IMPL


def inject_faults(rules: Sequence[Mapping[str, Any]]) -> None:
  """Installs fault injection rules, replacing any active ones.

  Args:
    rules: Sequence of rule dicts, e.g. ``[{"hook":
      "transfer_recv_session.pull.request", "action": "fail", "probability":
      1.0}]``.
  """
  # Wrapped rather than delegated: nanobind's nb::list caster rejects tuples
  # and other sequences, so the argument has to be materialized as a list.
  _impl().inject_faults(list(rules))


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
  """Delegates reset_faults, get_hit_count, ... to the extension lazily."""
  if name.startswith("_"):
    raise AttributeError(name)
  return getattr(_impl(), name)
