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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>  // IWYU pragma: keep
#include <nanobind/stl/optional.h>  // IWYU pragma: keep
#include <nanobind/stl/pair.h>  // IWYU pragma: keep
#include <nanobind/stl/shared_ptr.h>  // IWYU pragma: keep
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include <nanobind/stl/string_view.h>  // IWYU pragma: keep
#include <nanobind/stl/tuple.h>  // IWYU pragma: keep
#include <nanobind/stl/vector.h>  // IWYU pragma: keep
#include "xla/pjrt/status_casters.h"
#include "tpu_sync/core/raiden_future.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/frameworks/jax/kv_cache_manager.h"
#include "tpu_sync/frameworks/jax/nb_statusor.h"  // IWYU pragma: keep
#include "tpu_sync/frameworks/jax/raw_transfer_internal.h"
#include "tpu_sync/frameworks/jax/weight_synchronizer.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/kv_cache/kv_cache_store_wrapper.h"
#include "tpu_sync/telemetry/python/telemetry_binding.h"

namespace nb = nanobind;

using ::tpu_raiden::jax::WeightSynchronizer;

namespace tpu_raiden {
namespace kv_cache {
namespace {

std::vector<std::string> ToStdStringVector(
    const std::vector<nb::bytes>& bytes_vec) {
  std::vector<std::string> str_vec;
  str_vec.reserve(bytes_vec.size());
  for (const auto& b : bytes_vec) {
    str_vec.push_back(std::string(b.c_str(), b.size()));
  }
  return str_vec;
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden

using ::tpu_raiden::kv_cache::ToStdStringVector;

NB_MODULE(_tpu_raiden_jax, m) {
  // =========================================================================
  // 1. Bind KVCacheManager
  // =========================================================================

  // Bind the new Future class as RaidenFuture to maintain duck-typing
  // compatibility.
  nb::class_<tpu_raiden::RaidenFuture>(m, "RaidenFuture")
      .def("Await",
           [](tpu_raiden::RaidenFuture& self) {
             nb::gil_scoped_release release;
             absl::Status status = self.Await();
             if (!status.ok()) {
               throw std::runtime_error("Async copy failed: " +
                                        std::string(status.message()));
             }
           })
      .def("wait",
           [](tpu_raiden::RaidenFuture& self) {
             nb::gil_scoped_release release;
             absl::Status status = self.Await();
             if (!status.ok()) {
               throw std::runtime_error("Async copy failed: " +
                                        std::string(status.message()));
             }
           })
      .def("IsReady", &tpu_raiden::RaidenFuture::IsReady)
      .def("is_ready", &tpu_raiden::RaidenFuture::IsReady);

  nb::class_<tpu_raiden::kv_cache::jax::KVCacheManager>(m, "KVCacheManager")
      .def(nb::init<nb::list, std::optional<int>, std::optional<int>, bool, int,
                    int, std::optional<std::string>, std::optional<std::string>,
                    int64_t>(),
           nb::arg("device_arrays"), nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = nb::none(),
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("parallelism") = 1, nb::arg("raiden_worker_port") = 0,
           nb::arg("raiden_controller_address") = nb::none(),
           nb::arg("worker_id") = nb::none(), nb::arg("node_id") = 0)
      .def(nb::init<nanobind::list, int64_t, int64_t, int64_t, int64_t, double,
                    bool, int, int, std::optional<std::string>,
                    std::optional<std::string>>(),
           nb::arg("kv_caches"), nb::arg("node_id") = 0,
           nb::arg("local_control_port"), nb::arg("max_blocks"),
           nb::arg("num_slots"), nb::arg("timeout_s") = 120.0,
           nb::arg("unsafe_skip_buffer_lock") = true,
           nb::arg("parallelism") = 4, nb::arg("raiden_worker_port") = 0,
           nb::arg("raiden_controller_address") = nb::none(),
           nb::arg("worker_id") = nb::none())

      // Use lambdas to wrap the returned raiden::PjRtCopyFuture into
      // KVCacheManagerFuture
      .def(
          "h2d",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self,
             const std::vector<int64_t>& src_offsets,
             const std::vector<int64_t>& dst_offsets,
             const std::vector<int64_t>& copy_sizes)
              -> absl::StatusOr<tpu_raiden::RaidenFuture> {
            auto res = self.H2d(src_offsets, dst_offsets, copy_sizes);
            if (!res.ok()) return res.status();
            return tpu_raiden::RaidenFuture{std::move(res.value())};
          },
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("dst_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list())

      .def(
          "d2h",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self,
             const std::vector<int64_t>& src_offsets,
             const std::vector<int64_t>& dst_offsets,
             const std::vector<int64_t>& copy_sizes)
              -> absl::StatusOr<tpu_raiden::RaidenFuture> {
            auto res = self.D2h(src_offsets, dst_offsets, copy_sizes);
            if (!res.ok()) return res.status();
            return tpu_raiden::RaidenFuture{std::move(res.value())};
          },
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("dst_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list())

      .def(
          "d2h_auto_allocate",
          xla::ValueOrThrowWrapper(
              [](tpu_raiden::kv_cache::jax::KVCacheManager& self,
                 const std::vector<int64_t>& src_offsets,
                 const std::vector<int64_t>& copy_sizes)
                  -> absl::StatusOr<
                      std::pair<std::vector<int>, tpu_raiden::RaidenFuture>> {
                auto res = self.D2hAutoAllocate(src_offsets, copy_sizes);
                if (!res.ok()) return res.status();
                return std::make_pair(
                    res.value().first,
                    tpu_raiden::RaidenFuture{std::move(res.value().second)});
              }),
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list())

      .def(
          "h2h_write",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids)
              -> absl::StatusOr<
                  std::pair<std::vector<int>, tpu_raiden::RaidenFuture>> {
            auto res = self.H2hWrite(peer, src_block_ids);
            if (!res.ok()) return res.status();
            return std::make_pair(
                res.value().first,
                tpu_raiden::RaidenFuture{std::move(res.value().second)});
          },
          nb::arg("peer"), nb::arg("src_block_ids"))

      .def(
          "h2h_read",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids)
              -> absl::StatusOr<
                  std::pair<std::vector<int>, tpu_raiden::RaidenFuture>> {
            auto res = self.H2hRead(peer, src_block_ids);
            if (!res.ok()) return res.status();
            return std::make_pair(
                res.value().first,
                tpu_raiden::RaidenFuture{std::move(res.value().second)});
          },
          nb::arg("peer"), nb::arg("src_block_ids"))

      .def("local_port", &tpu_raiden::kv_cache::jax::KVCacheManager::local_port)
      .def("get_raiden_worker_port",
           &tpu_raiden::kv_cache::jax::KVCacheManager::GetRaidenWorkerPort)
      .def("get_host_pointer",
           static_cast<const uint8_t* (
               tpu_raiden::kv_cache::jax::KVCacheManager::*)(size_t, size_t)
                           const>(
               &tpu_raiden::kv_cache::jax::KVCacheManager::GetHostPointer),
           nb::arg("layer_idx"), nb::arg("shard_idx"))
      .def(
          "read_host_memory",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self, size_t layer_idx,
             size_t shard_idx, size_t num_floats) -> nb::list {
            const uint8_t* ptr = self.GetHostPointer(layer_idx, shard_idx);
            nb::list lst;
            if (ptr) {
              const float* fptr = reinterpret_cast<const float*>(ptr);
              for (size_t i = 0; i < num_floats; ++i) {
                lst.append(fptr[i]);
              }
            }
            return lst;
          },
          nb::arg("layer_idx"), nb::arg("shard_idx"), nb::arg("num_floats"))
      .def_prop_ro("num_layers",
                   &tpu_raiden::kv_cache::jax::KVCacheManager::num_layers)
      .def_prop_ro("num_shards",
                   &tpu_raiden::kv_cache::jax::KVCacheManager::num_shards)
      .def_prop_ro("slice_byte_size",
                   &tpu_raiden::kv_cache::jax::KVCacheManager::slice_byte_size)
      .def_prop_ro(
          "num_block_arrays",
          &tpu_raiden::kv_cache::jax::KVCacheManager::num_block_arrays)
      .def("block_bytes",
           &tpu_raiden::kv_cache::jax::KVCacheManager::block_bytes,
           nb::arg("block_array_idx"))
      .def_prop_ro(
          "local_control_port",
          &tpu_raiden::kv_cache::jax::KVCacheManager::local_control_port)
      .def("get_local_endpoints",
           [](const tpu_raiden::kv_cache::jax::KVCacheManager& self) {
             auto eps = self.get_local_endpoints();
             nb::list py_eps;
             for (const auto& ep : eps) {
               nb::dict d;
               d["endpoint"] = ep.endpoint;
               d["shards"] = ep.shards;
               py_eps.append(d);
             }
             return py_eps;
           })
      .def("notify_for_read",
           &tpu_raiden::kv_cache::jax::KVCacheManager::NotifyForRead,
           nb::arg("req_id"), nb::arg("uuid"), nb::arg("block_ids"))
      .def(
          "start_read",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self,
             const std::string& req_id, uint64_t uuid,
             nb::object remote_endpoint,
             const std::vector<int64_t>& remote_block_ids,
             const std::vector<int64_t>& local_block_ids, int parallelism,
             std::optional<std::vector<int64_t>> local_host_block_ids) {
            if (nb::isinstance<nb::str>(remote_endpoint)) {
              self.StartRead(req_id, uuid,
                             nb::cast<std::string>(remote_endpoint),
                             remote_block_ids, local_block_ids, parallelism,
                             local_host_block_ids);
            } else if (nb::isinstance<nb::list>(remote_endpoint)) {
              std::vector<tpu_raiden::RaidenTransferEndpoint> descriptors;
              nb::list ep_list = nb::cast<nb::list>(remote_endpoint);
              for (size_t i = 0; i < ep_list.size(); ++i) {
                nb::dict d = nb::cast<nb::dict>(ep_list[i]);
                tpu_raiden::RaidenTransferEndpoint desc;
                desc.endpoint = nb::cast<std::string>(d["endpoint"]);
                desc.shards = nb::cast<std::vector<int64_t>>(d["shards"]);
                descriptors.push_back(std::move(desc));
              }
              self.StartRead(req_id, uuid, descriptors, remote_block_ids,
                             local_block_ids, parallelism,
                             local_host_block_ids);
            } else {
              throw std::runtime_error(
                  "remote_endpoint must be str or list of dicts");
            }
          },
          nb::arg("req_id"), nb::arg("uuid"), nb::arg("remote_endpoint"),
          nb::arg("remote_block_ids"), nb::arg("local_block_ids"),
          nb::arg("parallelism") = 1,
          nb::arg("local_host_block_ids") = nb::none())
      .def("complete_read",
           [](tpu_raiden::kv_cache::jax::KVCacheManager& self) {
             auto [done_sending, done_recving, failed_recving] =
                 self.CompleteReadRaw();
             return nb::make_tuple(done_sending, done_recving, failed_recving);
           })
      .def("unlock_blocks",
           &tpu_raiden::kv_cache::jax::KVCacheManager::UnlockBlocks,
           nb::arg("block_ids"))
      .def("dump_metrics_to_string",
           &tpu_raiden::kv_cache::jax::KVCacheManager::DumpMetricsToString);

  // =========================================================================
  // 2. Bind WeightSynchronizer
  // =========================================================================
  nb::class_<WeightSynchronizer>(m, "WeightSynchronizer")
      .def(nb::init<nb::list, std::optional<int>, int, bool, std::optional<int>,
                    std::optional<std::string>, bool>(),
           nb::arg("jax_arrays"), nb::arg("local_port") = nb::none(),
           nb::arg("parallelism") = 1,
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("listener_port") = nb::none(),
           nb::arg("bind_ip") = nb::none(), nb::arg("auto_h2d") = false)

      .def(
          "D2h",
          [](WeightSynchronizer& self) {
            auto status_or_future = self.D2h();
            if (!status_or_future.ok()) {
              throw std::runtime_error(
                  "Weight sync D2H failed: " +
                  std::string(status_or_future.status().message()));
            }
            absl::Status status = status_or_future.value().Await();
            if (!status.ok()) {
              throw std::runtime_error("Weight sync D2H copy failed: " +
                                       std::string(status.message()));
            }
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "H2d",
          [](WeightSynchronizer& self) {
            auto status_or_future = self.H2d();
            if (!status_or_future.ok()) {
              throw std::runtime_error(
                  "Weight sync H2D failed: " +
                  std::string(status_or_future.status().message()));
            }
            absl::Status status = status_or_future.value().Await();
            if (!status.ok()) {
              throw std::runtime_error("Weight sync H2D copy failed: " +
                                       std::string(status.message()));
            }
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "set_skip_tiling",
          [](WeightSynchronizer& self, bool skip_all) {
            self.SetSkipTiling(skip_all);
          },
          nb::arg("skip_all"))
      .def(
          "set_skip_tiling",
          [](WeightSynchronizer& self, const std::vector<bool>& skip_tiling) {
            self.SetSkipTiling(skip_tiling);
          },
          nb::arg("skip_tiling"))
      .def(
          "bind_weights",
          [](WeightSynchronizer& self, nb::list jax_arrays) {
            absl::Status status = self.BindWeights(jax_arrays);
            if (!status.ok()) {
              throw std::runtime_error("Bind weights failed: " +
                                       std::string(status.message()));
            }
          },
          nb::arg("jax_arrays"))

      .def(
          "get_host_buffer",
          [](WeightSynchronizer& self, size_t layer_idx, size_t shard_idx) {
            const uint8_t* ptr = self.GetHostBufferPtr(layer_idx, shard_idx);
            if (!ptr) {
              throw std::runtime_error("Invalid layer or shard index");
            }
            size_t size = self.slice_byte_size() + 256 * 1024;
            size_t shape[1] = {size};
            return nb::ndarray<uint8_t, nb::numpy, nb::c_contig>(
                const_cast<uint8_t*>(ptr), 1, shape,
                nb::handle() /* view only, no ownership copy */
            );
          },
          nb::arg("layer_idx") = 0, nb::arg("shard_idx") = 0)
      .def_prop_ro("local_port", &WeightSynchronizer::local_port)
      .def_prop_ro("listener_port", &WeightSynchronizer::listener_port)
      .def_prop_ro("is_listener_active",
                   &WeightSynchronizer::is_listener_active)
      .def("get_local_endpoints",
           [](const WeightSynchronizer& self) {
             auto eps = self.get_local_endpoints();
             nb::list py_eps;
             for (const auto& ep : eps) {
               nb::dict d;
               d["endpoint"] = ep.endpoint;
               d["shards"] = ep.shards;
               py_eps.append(d);
             }
             return py_eps;
           })
      .def_prop_ro("num_layers", &WeightSynchronizer::num_layers)
      .def_prop_ro("num_shards", &WeightSynchronizer::num_shards)
      .def_prop_ro("slice_byte_size", &WeightSynchronizer::slice_byte_size)
      .def("get_metrics", &WeightSynchronizer::GetMetrics);

  nb::class_<tpu_raiden::weight_sync::WeightSyncMetrics>(m, "WeightSyncMetrics")
      .def_ro("last_d2h_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_d2h_time_ms)
      .def_ro("last_h2d_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_h2d_time_ms)
      .def_ro("last_h2h_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_h2h_time_ms)
      .def_ro("last_staging_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_staging_time_ms)
      .def_ro("last_tiling_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_tiling_time_ms)
      .def_ro("last_detiling_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_detiling_time_ms)
      .def_ro("last_total_push_resharded_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::
                  last_total_push_resharded_time_ms)
      .def_ro("last_d2h_bytes",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_d2h_bytes)
      .def_ro("last_h2d_bytes",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_h2d_bytes)
      .def_ro("last_h2h_bytes",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_h2h_bytes)
      .def_ro("last_tiled_bytes",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_tiled_bytes)
      .def_ro("last_detiled_bytes",
              &tpu_raiden::weight_sync::WeightSyncMetrics::last_detiled_bytes)
      .def_ro("total_d2h_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::total_d2h_time_ms)
      .def_ro("total_h2d_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::total_h2d_time_ms)
      .def_ro("total_h2h_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::total_h2h_time_ms)
      .def_ro(
          "total_staging_time_ms",
          &tpu_raiden::weight_sync::WeightSyncMetrics::total_staging_time_ms)
      .def_ro("total_tiling_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::total_tiling_time_ms)
      .def_ro(
          "total_detiling_time_ms",
          &tpu_raiden::weight_sync::WeightSyncMetrics::total_detiling_time_ms)
      .def_ro("total_push_resharded_time_ms",
              &tpu_raiden::weight_sync::WeightSyncMetrics::
                  total_push_resharded_time_ms)
      .def_ro("total_d2h_bytes",
              &tpu_raiden::weight_sync::WeightSyncMetrics::total_d2h_bytes)
      .def_ro("total_h2d_bytes",
              &tpu_raiden::weight_sync::WeightSyncMetrics::total_h2d_bytes)
      .def_ro("total_h2h_bytes",
              &tpu_raiden::weight_sync::WeightSyncMetrics::total_h2h_bytes)
      .def_ro("total_tiled_bytes",
              &tpu_raiden::weight_sync::WeightSyncMetrics::total_tiled_bytes)
      .def_ro("total_detiled_bytes",
              &tpu_raiden::weight_sync::WeightSyncMetrics::total_detiled_bytes)
      .def_ro("d2h_call_count",
              &tpu_raiden::weight_sync::WeightSyncMetrics::d2h_call_count)
      .def_ro("push_resharded_call_count",
              &tpu_raiden::weight_sync::WeightSyncMetrics::
                  push_resharded_call_count);

  // =========================================================================
  // 3. Bind KVCacheStore
  // =========================================================================
  nb::class_<tpu_raiden::kv_cache::RaidenId>(m, "RaidenId")
      .def(nb::init<std::string, std::string, std::string, int>(),
           nb::arg("job_name") = "", nb::arg("job_replica_id") = "",
           nb::arg("data_name") = "", nb::arg("data_replica_idx") = 0)
      .def_rw("job_name", &tpu_raiden::kv_cache::RaidenId::job_name)
      .def_rw("job_replica_id", &tpu_raiden::kv_cache::RaidenId::job_replica_id)
      .def_rw("data_name", &tpu_raiden::kv_cache::RaidenId::data_name)
      .def_rw("data_replica_idx",
              &tpu_raiden::kv_cache::RaidenId::data_replica_idx)
      .def("empty", &tpu_raiden::kv_cache::RaidenId::empty)
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def("__hash__",
           [](const tpu_raiden::kv_cache::RaidenId& id) {
             return tpu_raiden::RaidenIdHash()(id);
           })
      .def("__repr__",
           [](const tpu_raiden::kv_cache::RaidenId& id) {
             return absl::StrCat(
                 "RaidenId(job_name='", id.job_name, "', job_replica_id='",
                 id.job_replica_id, "', data_name='", id.data_name,
                 "', data_replica_idx=", id.data_replica_idx, ")");
           })
      .def("__str__", [](const tpu_raiden::kv_cache::RaidenId& id) {
        return absl::StrCat(
            "RaidenId(job_name='", id.job_name, "', job_replica_id='",
            id.job_replica_id, "', data_name='", id.data_name,
            "', data_replica_idx=", id.data_replica_idx, ")");
      });

  nb::enum_<tpu_raiden::kv_cache::BlockStatus>(m, "BlockStatus")
      .value("INIT", tpu_raiden::kv_cache::BlockStatus::INIT)
      .value("REMOTE", tpu_raiden::kv_cache::BlockStatus::REMOTE)
      .value("HBM", tpu_raiden::kv_cache::BlockStatus::HBM)
      .value("HOST", tpu_raiden::kv_cache::BlockStatus::HOST)
      .value("HOST_AND_HBM", tpu_raiden::kv_cache::BlockStatus::HOST_AND_HBM);

  nb::class_<tpu_raiden::kv_cache::RaidenBlockId>(m, "RaidenBlockId")
      .def(nb::init<tpu_raiden::kv_cache::RaidenId, int,
                    tpu_raiden::kv_cache::BlockStatus>(),
           nb::arg("raiden_id") = tpu_raiden::kv_cache::RaidenId(),
           nb::arg("host_block_id") = -1,
           nb::arg("status") = tpu_raiden::kv_cache::BlockStatus::INIT)
      .def(nb::init<tpu_raiden::kv_cache::RaidenId, int, int,
                    tpu_raiden::kv_cache::BlockStatus>(),
           nb::arg("raiden_id") = tpu_raiden::kv_cache::RaidenId(),
           nb::arg("host_block_id") = -1, nb::arg("device_block_id") = -1,
           nb::arg("status") = tpu_raiden::kv_cache::BlockStatus::INIT)
      .def_rw("raiden_id", &tpu_raiden::kv_cache::RaidenBlockId::raiden_id)
      .def_rw("host_block_id",
              &tpu_raiden::kv_cache::RaidenBlockId::host_block_id)
      .def_rw("device_block_id",
              &tpu_raiden::kv_cache::RaidenBlockId::device_block_id)
      .def_rw("status", &tpu_raiden::kv_cache::RaidenBlockId::status);

  nb::class_<tpu_raiden::kv_cache::KVCacheStoreWrapper>(m, "KVCacheStore")
      .def(nb::init<size_t, std::string, tpu_raiden::kv_cache::RaidenId, int,
                    int64_t, std::string, int, int, std::string>(),
           nb::arg("capacity"), nb::arg("global_registry_address") = "",
           nb::arg("raiden_id") = tpu_raiden::kv_cache::RaidenId(),
           // No defaults: every KVCacheStore has a controller and a
           // publishable address, so 0/"" are not
           // valid values a caller can fall into by omission.
           nb::arg("num_shards"), nb::arg("shard_size_bytes") = 0,
           nb::arg("store_server_ip"), nb::arg("raiden_controller_port") = 0,
           // When expected_worker_count > 0, constructor blocks until that
           // many workers have registered with the controller (times out and
           // throws after RAIDEN_EXPECTED_WORKERS_TIMEOUT_S, default 120s).
           nb::arg("expected_worker_count") = 0,
           // KV pool group this store's KVTransferSpec is published under
           // (see global_registry.proto); empty falls back to
           // raiden_id.job_name.
           nb::arg("kv_pool_group") = "",
           // Release the GIL during construction so concurrent in-process
           // Python threads (e.g., worker registration threads) can run and
           // avoid deadlocking on the expected_worker_count barrier.
           nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "raiden_id",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
            return (*self).raiden_id();
          },
          "Returns the RaidenId associated with this store.")
      .def_prop_ro("raiden_controller_address",
                   [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
                     return self->raiden_controller_address();
                   })
      .def_prop_ro("store_server_address",
                   [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
                     return self->store_server_address();
                   })
      .def(
          "lookup",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes, bool enable_global,
             bool pin_found) {
            auto hashes = ToStdStringVector(block_hashes);
            auto res = self->Lookup(hashes, enable_global, pin_found);
            if (!res.ok()) {
              throw std::runtime_error(absl::StrCat(
                  "KVCacheStore lookup failed: ", res.status().message()));
            }
            std::vector<
                std::pair<nb::bytes, tpu_raiden::kv_cache::RaidenBlockId>>
                py_res;
            py_res.reserve(res.value().size());
            for (const auto& pair : res.value()) {
              py_res.push_back(std::make_pair(
                  nb::bytes(pair.first.data(), pair.first.size()),
                  pair.second));
            }
            return py_res;
          },
          nb::arg("block_hashes"), nb::arg("enable_global") = false,
          nb::arg("pin_found") = true,
          "Checks the LRU directory for cached block hashes. Returns a list of "
          "all matched replica pairs prior to the first miss. Pins every local "
          "hit unless pin_found is false.")
      .def(
          "insert",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes,
             const std::vector<tpu_raiden::kv_cache::RaidenBlockId>& slices,
             bool on_host) -> bool {
            auto hashes = ToStdStringVector(block_hashes);
            absl::Status status = self->Insert(hashes, slices, on_host);
            // A REMOTE slice is a caller bug, not a full cache: surface it as
            // ValueError instead of an indistinguishable False.
            if (absl::IsInvalidArgument(status)) {
              throw std::invalid_argument(std::string(status.message()));
            }
            return status.ok();
          },
          nb::arg("block_hashes"), nb::arg("slices"), nb::arg("on_host"))
      .def("capacity",
           [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
             return self->capacity();
           })
      .def(
          "release",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes) {
            auto hashes = ToStdStringVector(block_hashes);
            self->Release(hashes);
          },
          nb::arg("block_hashes"))
      .def(
          "save",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes,
             const std::optional<tpu_raiden::kv_cache::RaidenId>&
                 dst_raiden_id) -> bool {
            auto hashes = ToStdStringVector(block_hashes);
            return self->Save(hashes, dst_raiden_id).ok();
          },
          nb::arg("block_hashes"), nb::arg("dst_raiden_id") = nb::none())
      .def(
          "load",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes,
             const std::vector<int>& device_block_ids) -> bool {
            auto hashes = ToStdStringVector(block_hashes);
            return self->Load(hashes, device_block_ids).ok();
          },
          nb::arg("block_hashes"), nb::arg("device_block_ids"))
      .def(
          "load",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes,
             const std::vector<tpu_raiden::kv_cache::RaidenBlockId>& slices,
             const std::vector<int>& device_block_ids) -> bool {
            auto hashes = ToStdStringVector(block_hashes);
            return self->Load(hashes, slices, device_block_ids).ok();
          },
          nb::arg("block_hashes"), nb::arg("slices"),
          nb::arg("device_block_ids"))
      .def("poll_save_status",
           [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
             // Poll drains whatever futures completed and can make blocking
             // RPCs on the way -- one verdict call per outstanding remote
             // save. Holding the GIL across that stalls every Python thread,
             // not just this one.
             //
             // Released around the C++ call ONLY. The nb::bytes below are
             // Python objects, so building them and letting std::make_tuple
             // copy the vectors touches CPython refcounts, which must happen
             // with the GIL held. This is why nb::call_guard is wrong here:
             // it would span the whole lambda, including that.
             // Five vectors. `existing` and `unregistered` annotate REMOTE
             // save failures rather than being outcomes of their own, and are
             // empty for local saves. See KVCacheStore::PollSaveStatus.
             std::vector<std::string> done, failed, pending, existing,
                 unregistered;
             {
               nb::gil_scoped_release release;
               auto res = self->PollSaveStatus();
               done = std::move(res.done);
               failed = std::move(res.failed);
               pending = std::move(res.pending);
               existing = std::move(res.existing);
               unregistered = std::move(res.unregistered);
             }
             std::vector<nb::bytes> py_done, py_failed, py_pending, py_existing,
                 py_unregistered;
             py_done.reserve(done.size());
             for (const auto& h : done) {
               py_done.push_back(nb::bytes(h.data(), h.size()));
             }
             py_failed.reserve(failed.size());
             for (const auto& h : failed) {
               py_failed.push_back(nb::bytes(h.data(), h.size()));
             }
             py_pending.reserve(pending.size());
             for (const auto& h : pending) {
               py_pending.push_back(nb::bytes(h.data(), h.size()));
             }
             py_existing.reserve(existing.size());
             for (const auto& h : existing) {
               py_existing.push_back(nb::bytes(h.data(), h.size()));
             }
             py_unregistered.reserve(unregistered.size());
             for (const auto& h : unregistered) {
               py_unregistered.push_back(nb::bytes(h.data(), h.size()));
             }
             return std::make_tuple(py_done, py_failed, py_pending, py_existing,
                                    py_unregistered);
           })
      .def("poll_load_status",
           [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
             // Released around the C++ call ONLY. The subsequent nb::bytes
             // initialization happens with the GIL held (nb::call_guard
             // is wrong here since it would span the whole lambda).
             // See poll_save_status for the detailed stall analysis.
             std::vector<std::string> done, failed, pending;
             {
               nb::gil_scoped_release release;
               auto res = self->PollLoadStatus();
                done = std::move(res.done);
                failed = std::move(res.failed);
                pending = std::move(res.pending);
             }
             std::vector<nb::bytes> py_done, py_failed, py_pending;
             py_done.reserve(done.size());
             for (const auto& h : done) {
               py_done.push_back(nb::bytes(h.data(), h.size()));
             }
             py_failed.reserve(failed.size());
             for (const auto& h : failed) {
               py_failed.push_back(nb::bytes(h.data(), h.size()));
             }
             py_pending.reserve(pending.size());
             for (const auto& h : pending) {
               py_pending.push_back(nb::bytes(h.data(), h.size()));
             }
             return std::make_tuple(py_done, py_failed, py_pending);
           })
      .def(
          "read_remote",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes,
             const std::vector<tpu_raiden::kv_cache::RaidenBlockId>& slices,
             const std::vector<int32_t>& device_block_ids) -> bool {
            auto hashes = ToStdStringVector(block_hashes);
            return self->ReadRemote(hashes, slices, device_block_ids).ok();
          },
          nb::arg("block_hashes"), nb::arg("slices"),
          nb::arg("device_block_ids"))
      .def("poll_remote_read_status",
           [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
             // Released around the C++ call ONLY. The subsequent nb::bytes
             // initialization happens with the GIL held (nb::call_guard
             // is wrong here since it would span the whole lambda).
             // See poll_save_status for the detailed stall analysis.
             std::vector<std::string> done, failed, pending;
             {
               nb::gil_scoped_release release;
               std::tie(done, failed, pending) = self->PollRemoteReadStatus();
             }
             std::vector<nb::bytes> py_done, py_failed, py_pending;
             py_done.reserve(done.size());
             for (const auto& h : done) {
               py_done.push_back(nb::bytes(h.data(), h.size()));
             }
             py_failed.reserve(failed.size());
             for (const auto& h : failed) {
               py_failed.push_back(nb::bytes(h.data(), h.size()));
             }
             py_pending.reserve(pending.size());
             for (const auto& h : pending) {
               py_pending.push_back(nb::bytes(h.data(), h.size()));
             }
             return std::make_tuple(py_done, py_failed, py_pending);
           });

  tpu_raiden::telemetry::BindTelemetryApi(m);
}
