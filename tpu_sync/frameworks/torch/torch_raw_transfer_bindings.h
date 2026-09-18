// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_RAW_TRANSFER_BINDINGS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_RAW_TRANSFER_BINDINGS_H_

#include "nanobind/nanobind.h"

namespace raiden {

// Registers the raw-transfer surface (RawHostBuffer, PjRtCopyFuture,
// PreparedTorchRawTransfer, transfer_{d2h,h2d}[_batch][_async],
// await_all/is_ready) on `m`. Called by the standalone _torch_raw_transfer
// extension and by _tpu_raiden_torch, which mounts it as its `raw_transfer`
// submodule so the wheel ships a single ABI-dispatched extension.
__attribute__((visibility("default"))) void BindTorchRawTransfer(
    nanobind::module_& m);

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_TORCH_RAW_TRANSFER_BINDINGS_H_
