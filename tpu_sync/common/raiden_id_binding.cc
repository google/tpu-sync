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

#include <string>

#include "absl/strings/str_format.h"
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>  // IWYU pragma: keep
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include "tpu_sync/common/raiden_id.h"

namespace nb = nanobind;

NB_MODULE(_raiden_id, m) {
  nb::class_<tpu_raiden::RaidenId>(m, "RaidenId")
      .def(nb::init<std::string, std::string, std::string, int>(),
           nb::arg("job_name") = "", nb::arg("job_replica_id") = "",
           nb::arg("data_name") = "", nb::arg("data_replica_idx") = 0)
      .def_rw("job_name", &tpu_raiden::RaidenId::job_name)
      .def_rw("job_replica_id", &tpu_raiden::RaidenId::job_replica_id)
      .def_rw("data_name", &tpu_raiden::RaidenId::data_name)
      .def_rw("data_replica_idx", &tpu_raiden::RaidenId::data_replica_idx)
      .def("empty", &tpu_raiden::RaidenId::empty)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def("__hash__",
           [](const tpu_raiden::RaidenId& id) {
             return tpu_raiden::RaidenIdHash()(id);
           })
      .def("__repr__",
           [](const tpu_raiden::RaidenId& id) {
             return absl::StrFormat(
                 "RaidenId(job_name='%s', job_replica_id='%s', data_name='%s', "
                 "data_replica_idx=%d)",
                 id.job_name, id.job_replica_id, id.data_name,
                 id.data_replica_idx);
           })
      .def("__str__", [](const tpu_raiden::RaidenId& id) {
        return absl::StrFormat(
            "RaidenId(job_name='%s', job_replica_id='%s', data_name='%s', "
            "data_replica_idx=%d)",
            id.job_name, id.job_replica_id, id.data_name, id.data_replica_idx);
      });
}
