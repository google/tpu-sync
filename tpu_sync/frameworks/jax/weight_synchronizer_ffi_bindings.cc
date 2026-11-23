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

#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi_bindings.h"

#include <cstddef>

#include <nanobind/nanobind.h>
#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi_internal.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace nb = nanobind;

namespace tpu_raiden::weight_sync {
namespace {

nb::list PrepareExtendedInfo(nb::list gathered_info_list, nb::list device_ids,
                             nb::list r_starts, nb::list r_ends,
                             nb::list c_starts, nb::list c_ends) {
  size_t n = nb::len(device_ids);
  nb::list extended_info;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      extended_info.append(gathered_info_list[i * 5 + j]);
    }
    extended_info.append(device_ids[i]);
    extended_info.append(r_starts[i]);
    extended_info.append(r_ends[i]);
    extended_info.append(c_starts[i]);
    extended_info.append(c_ends[i]);
  }
  return extended_info;
}

}  // namespace

void BindWeightSynchronizerFfi(nb::module_& m) {
  m.def("destroy_weight_synchronizer", []() {
    WeightSynchronizerFfiManager::Default().Clear(/*delete_instances=*/true);
    ClearSharedWsMap();
  });

  m.def(
      "is_listener_active",
      [](int shard_idx = 0) -> bool {
        if (const auto* ws =
                WeightSynchronizerFfiManager::Default().GetSynchronizerForShard(
                    shard_idx);
            ws != nullptr) {
          return ws->is_listener_active();
        }
        return WeightSynchronizerFfiManager::Default().HasActiveListener();
      },
      nb::arg("shard_idx") = 0);

  m.def("prepare_extended_info", &PrepareExtendedInfo);
}

}  // namespace tpu_raiden::weight_sync
