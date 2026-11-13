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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_SOLUTIONS_MOONCAKE_TORCH_MOONCAKE_PROVIDER_BINDINGS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_SOLUTIONS_MOONCAKE_TORCH_MOONCAKE_PROVIDER_BINDINGS_H_

#include "nanobind/nanobind.h"

namespace raiden::mooncake {

// Registers the torch half of the Mooncake TPU buffer provider on `m`:
// export_tensor(), which turns a TPU tensor into a provider handle, and the
// live_buffer_count(), live_dma_mapping_count() and mark_process_exiting()
// controls of tpu_sync/solutions/mooncake/mooncake_provider.h.
// _tpu_raiden_torch mounts it as its `mooncake_provider` submodule, and
// defines the mc_tpu_buffer_provider_get that serves the handles' table.
void BindMooncakeProvider(nanobind::module_& m);

}  // namespace raiden::mooncake

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_SOLUTIONS_MOONCAKE_TORCH_MOONCAKE_PROVIDER_BINDINGS_H_
