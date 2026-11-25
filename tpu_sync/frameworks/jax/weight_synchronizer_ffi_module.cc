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

#include <nanobind/nanobind.h>
#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi_internal.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace nb = nanobind;

NB_MODULE(_weight_synchronizer_ffi, m) {
  m.def("destroy_weight_synchronizer", []() {
    tpu_raiden::weight_sync::WeightSynchronizerFfiManager::Default().Clear(
        /*delete_instances=*/true);
    tpu_raiden::weight_sync::ClearSharedWsMap();
  });

  m.def(
      "is_listener_active",
      [](int shard_idx = 0) -> bool {
        if (const auto* ws =
                tpu_raiden::weight_sync::WeightSynchronizerFfiManager::Default()
                    .GetSynchronizerForShard(shard_idx);
            ws != nullptr) {
          return ws->is_listener_active();
        }
        return tpu_raiden::weight_sync::WeightSynchronizerFfiManager::Default()
            .HasActiveListener();
      },
      nb::arg("shard_idx") = 0);
}
