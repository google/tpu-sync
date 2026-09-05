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
#include <nanobind/ndarray.h>
#include "tpu_sync/frameworks/jax/weight_synchronizer.h"
#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi.h"

namespace nb = nanobind;

namespace {

nb::list prepare_extended_info(nb::list gathered_info_list, nb::list device_ids,
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

NB_MODULE(_weight_synchronizer_ffi, m) {
  m.def("destroy_weight_synchronizer", []() {
    for (size_t i = 0; i < tpu_raiden::weight_sync::kMaxShards; ++i) {
      if (tpu_raiden::weight_sync::g_weight_synchronizers[i] != nullptr) {
        delete tpu_raiden::weight_sync::g_weight_synchronizers[i];
        tpu_raiden::weight_sync::g_weight_synchronizers[i] = nullptr;
      }
    }
  });

  m.def(
      "is_listener_active",
      [](int shard_idx = 0) -> bool {
        if (shard_idx >= 0 &&
            static_cast<size_t>(shard_idx) <
                tpu_raiden::weight_sync::kMaxShards &&
            tpu_raiden::weight_sync::g_weight_synchronizers[shard_idx] !=
                nullptr) {
          return tpu_raiden::weight_sync::g_weight_synchronizers[shard_idx]
              ->is_listener_active();
        }
        for (size_t i = 0; i < tpu_raiden::weight_sync::kMaxShards; ++i) {
          if (tpu_raiden::weight_sync::g_weight_synchronizers[i] != nullptr &&
              tpu_raiden::weight_sync::g_weight_synchronizers[i]
                  ->is_listener_active()) {
            return true;
          }
        }
        return false;
      },
      nb::arg("shard_idx") = 0);

  m.def("prepare_extended_info", &prepare_extended_info);
}
