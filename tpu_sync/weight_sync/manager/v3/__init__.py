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

"""V3 Raiden weight-sync controller: thin Python layer over the C++ engine.

Public surface::

  RaidenControllerV3            local controller (plan/execute/serve)
  RaidenControllerClientFacade  client for a remote controller server
  WeightSyncConfig / TransferOptions / TransferPlan / RaidenFuture
  SamplerCommand / HostCommand  read-only plan views
  slice-math helpers            thin wrappers over the C++ reshard planner

Transfers use striped seeding: the Trainer pushes ``num_stripes`` contiguous
bundle ranges, each to ``seed_replication`` sampler replicas, and the samplers
then complete each other with pulls that a per-transfer pull scheduler inside
the C++ controller grants while the transfer runs; it is internal and
intentionally not exposed here.

Shared proto/ID helpers live in ``..manager.controller_types`` and are
re-exported here only where they are part of the V3 call signature
(``NameResolver``, ``RaidenMemoryType``).
"""

from tpu_sync.api.common import RaidenId
from tpu_sync.weight_sync.manager.v3 import controller
from tpu_sync.weight_sync.manager.v3 import remote
from tpu_sync.weight_sync.manager.v3 import slice_math
from tpu_sync.weight_sync.manager.v3 import types

HostCommand = types.HostCommand
NDSlice = types.NDSlice
NameResolver = types.NameResolver
RaidenControllerClientFacade = remote.RaidenControllerClientFacade
RaidenControllerV3 = controller.RaidenControllerV3
RaidenFuture = types.RaidenFuture
RaidenMemoryType = types.RaidenMemoryType
RemoteTransferResult = remote.RemoteTransferResult
SamplerCommand = types.SamplerCommand
SynchronizerLike = types.SynchronizerLike
TransferOptions = types.TransferOptions
TransferPlan = types.TransferPlan
TransferStatus = types.TransferStatus
VariableBundleSpec = types.VariableBundleSpec
WeightSyncConfig = types.WeightSyncConfig

compute_seed_sampler_count = slice_math.compute_seed_sampler_count
generate_strided_copy_chunks = slice_math.generate_strided_copy_chunks
generate_strided_copy_chunks_tile_aware = (
    slice_math.generate_strided_copy_chunks_tile_aware
)
get_global_indices = slice_math.get_global_indices
intersect_nd_slices = slice_math.intersect_nd_slices
is_nd_slice_tile_aligned = slice_math.is_nd_slice_tile_aligned
partition_variable_bundles = slice_math.partition_variable_bundles
to_physical = slice_math.to_physical

__all__ = [
    "HostCommand",
    "NDSlice",
    "NameResolver",
    "RaidenControllerClientFacade",
    "RaidenControllerV3",
    "RaidenFuture",
    "RaidenId",
    "RaidenMemoryType",
    "RemoteTransferResult",
    "SamplerCommand",
    "SynchronizerLike",
    "TransferOptions",
    "TransferPlan",
    "TransferStatus",
    "VariableBundleSpec",
    "WeightSyncConfig",
    "compute_seed_sampler_count",
    "generate_strided_copy_chunks",
    "generate_strided_copy_chunks_tile_aware",
    "get_global_indices",
    "intersect_nd_slices",
    "is_nd_slice_tile_aligned",
    "partition_variable_bundles",
    "to_physical",
]
