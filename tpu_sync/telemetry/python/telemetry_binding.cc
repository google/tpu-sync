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

#include "tpu_sync/telemetry/python/telemetry_binding.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>  // IWYU pragma: keep
#include <nanobind/stl/map.h>  // IWYU pragma: keep
#include <nanobind/stl/optional.h>  // IWYU pragma: keep
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include <nanobind/stl/vector.h>  // IWYU pragma: keep
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

namespace nb = nanobind;

void BindTelemetryApi(nb::module_& m) {
  nb::enum_<MetricType>(m, "MetricType")
      .value("COUNTER", MetricType::kCounter)
      .value("GAUGE", MetricType::kGauge)
      .value("HISTOGRAM", MetricType::kHistogram);

  nb::class_<MetricMetadata>(m, "MetricMetadata")
      .def_prop_ro(
          "name",
          [](const MetricMetadata& self) { return std::string(self.name); })
      .def_prop_ro("description",
                   [](const MetricMetadata& self) {
                     return std::string(self.description);
                   })
      .def_ro("type", &MetricMetadata::type)
      .def_prop_ro("buckets",
                   [](const MetricMetadata& self) {
                     return std::vector<double>(self.buckets.begin(),
                                                self.buckets.end());
                   })
      .def_prop_ro("label_names",
                   [](const MetricMetadata& self) {
                     return std::vector<std::string>(self.label_names.begin(),
                                                     self.label_names.end());
                   })
      .def("__repr__",
           [](const MetricMetadata& self) {
             std::string type_str;
             switch (self.type) {
               case MetricType::kCounter:
                 type_str = "MetricType.COUNTER";
                 break;
               case MetricType::kGauge:
                 type_str = "MetricType.GAUGE";
                 break;
               case MetricType::kHistogram:
                 type_str = "MetricType.HISTOGRAM";
                 break;
             }
             std::string buckets_str =
                 absl::StrCat("[", absl::StrJoin(self.buckets, ", "), "]");
             std::string labels_str =
                 self.label_names.empty()
                     ? "[]"
                     : absl::StrCat("['",
                                    absl::StrJoin(self.label_names, "', '"),
                                    "']");
             return absl::StrCat("MetricMetadata(name='", self.name,
                                 "', description='", self.description,
                                 "', type=", type_str,
                                 ", buckets=", buckets_str,
                                 ", label_names=", labels_str, ")");
           })
      .def(nb::self == nb::self)
      .def(nb::self != nb::self);

  m.attr("ALL_METRICS") =
      std::vector<MetricMetadata>(std::begin(metric_metadata::kAllMetrics),
                                  std::end(metric_metadata::kAllMetrics));

  m.def(
      "configure_telemetry",
      [](const std::optional<std::vector<std::string>>& backends) {
        if (!backends.has_value()) {
          if (absl::Status status = RaidenMetricStore::GetGlobalMetricStore()
                                        .InitializeFromEnvironment();
              !status.ok()) {
            throw nb::value_error(
                absl::StrCat("Failed to initialize from environment: ",
                             status.message())
                    .c_str());
          }
          return;
        }
        std::vector<absl::string_view> backend_views(backends->begin(),
                                                     backends->end());
        if (absl::Status status =
                RaidenMetricStore::GetGlobalMetricStore()
                    .InitializeFromBackendNames(backend_views);
            !status.ok()) {
          throw nb::value_error(std::string(status.message()).c_str());
        }
      },
      nb::arg("backends") = nb::none(),
      nb::call_guard<nb::gil_scoped_release>(),
      "Configures active C++ telemetry backends given an optional sequence of "
      "backend names (e.g. list or tuple). If omitted or None, initializes "
      "from the TPU_RAIDEN_TELEMETRY_BACKENDS environment variable.");

  m.def(
      "get_raiden_metrics_prometheus_text",
      []() -> std::string {
        return RaidenMetricStore::GetGlobalMetricStore().GetTextSnapshot();
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "Exports Prometheus text snapshot of TPU Raiden metrics.");

  m.def(
      "get_metric_metadata",
      []() -> std::vector<MetricMetadata> {
        return RaidenMetricStore::GetGlobalMetricStore().GetMetricMetadata();
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "Returns metadata for all registered metrics.");

  m.def(
      "get_and_reset_metric_samples",
      []() -> std::map<std::string, std::vector<double>> {
        return RaidenMetricStore::GetGlobalMetricStore()
            .GetAndResetMetricSamples();
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "Extracts and resets buffered metric samples across all backends.");
}

}  // namespace tpu_raiden::telemetry
