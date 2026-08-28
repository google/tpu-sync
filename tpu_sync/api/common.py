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

"""Shared common types and enums for TPU Raiden Python APIs."""

import dataclasses
import enum


@dataclasses.dataclass(unsafe_hash=True)
class _PurePythonRaidenId:
  """Identifier for the work unit in Raiden owning a sharded set of data."""

  job_name: str = ""
  job_replica_id: str = ""
  data_name: str = ""
  data_replica_idx: int = 0

  def empty(self) -> bool:
    return (
        not self.job_name
        and not self.job_replica_id
        and not self.data_name
        and self.data_replica_idx == 0
    )


try:
  from tpu_sync.common import _raiden_id  # pylint: disable=g-import-not-at-top

  RaidenId = _raiden_id.RaidenId
except (ImportError, ModuleNotFoundError, AttributeError):
  RaidenId = _PurePythonRaidenId


class BlockStatus(enum.Enum):
  """Represents the residency status and location of a KV cache block."""

  # Residency only. Whether a block is pinned is tracked separately by the
  # LRU pin count, not by its status.
  INIT = 0  # Unallocated / Empty block slot in the directory.
  REMOTE = 1  # Discovered on a remote peer; not present in local memory.
  HBM = 2  # Resident in local TPU HBM device memory only (not on host).
  HOST = 3  # Resident in local Host DRAM only (not in HBM).
  HOST_AND_HBM = (
      4  # Resident in both local Host DRAM and TPU HBM device memory.
  )
