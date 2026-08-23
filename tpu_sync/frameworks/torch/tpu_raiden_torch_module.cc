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

#include "ATen/ops/from_blob.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/core/raiden_future.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/frameworks/torch/kv_cache_manager.h"
#include "tpu_sync/frameworks/torch/pool_layout_nanobind.h"
#include "tpu_sync/frameworks/torch/torch_nanobind_utils.h"
#include "tpu_sync/frameworks/torch/weight_synchronizer.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/kv_cache/kv_cache_store_wrapper.h"
#include "tpu_sync/kv_cache/reshard/reshard_client.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/telemetry/python/telemetry_binding.h"

namespace nb = nanobind;

using ::tpu_raiden::torch::KVCacheManager;
using ::tpu_raiden::torch::WeightSynchronizer;

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

// C++-owned reshard client behind the facade-compatible api/torch shim. The
// shim normalizes Python objects to plain tuples / dicts (private contract);
// every encoding decision lives in reshard::ReshardClient. Conversions run
// with the GIL; transport I/O releases it.
namespace reshardpb2 {

namespace nb = nanobind;

using UnitTuple = std::tuple<std::string, std::string, std::string, int32_t>;

tpu_raiden::kv_cache::RaidenId UnitOf(const UnitTuple& t) {
  tpu_raiden::kv_cache::RaidenId unit;
  unit.job_name = std::get<0>(t);
  unit.job_replica_id = std::get<1>(t);
  unit.data_name = std::get<2>(t);
  unit.data_replica_idx = std::get<3>(t);
  return unit;
}

int64_t DictInt(nb::dict d, const char* key, int64_t fallback) {
  if (!d.contains(key)) return fallback;
  return nb::cast<int64_t>(d[key]);
}

std::string DictStr(nb::dict d, const char* key, const char* fallback) {
  if (!d.contains(key)) return fallback;
  return nb::cast<std::string>(d[key]);
}

std::vector<reshard::ClientPoolSpec> ManifestOf(nb::handle sequence) {
  std::vector<reshard::ClientPoolSpec> out;
  for (nb::handle item : sequence) {
    nb::dict d = nb::cast<nb::dict>(item);
    reshard::ClientPoolSpec pool;
    pool.tag = DictStr(d, "tag", "");
    pool.storage_index = DictInt(d, "storage_index", 0);
    pool.base_offset_bytes = DictInt(d, "base_offset_bytes", 0);
    pool.block_stride_bytes = DictInt(d, "block_stride_bytes", 0);
    pool.num_blocks = DictInt(d, "num_blocks", 0);
    pool.dtype_tag = DictStr(d, "dtype_tag", "");
    if (d.contains("regions")) {
      for (nb::handle region_item : d["regions"]) {
        nb::dict r = nb::cast<nb::dict>(region_item);
        reshard::ClientPoolRegion region;
        region.name = DictStr(r, "name", "");
        region.offset_bytes = DictInt(r, "offset_bytes", 0);
        region.stride_bytes = DictInt(r, "stride_bytes", 0);
        region.unit_bytes = DictInt(r, "unit_bytes", 0);
        region.num_units = DictInt(r, "num_units", 0);
        region.units_per_stride = DictInt(r, "units_per_stride", 1);
        pool.regions.push_back(std::move(region));
      }
    }
    out.push_back(std::move(pool));
  }
  return out;
}

std::vector<reshard::ClientPoolSpans> SpansOf(nb::handle sequence) {
  std::vector<reshard::ClientPoolSpans> out;
  for (nb::handle item : sequence) {
    nb::dict d = nb::cast<nb::dict>(item);
    reshard::ClientPoolSpans entry;
    entry.tag = DictStr(d, "tag", "");
    for (nb::handle block : d["block_ids"]) {
      entry.block_ids.push_back(nb::cast<int64_t>(block));
    }
    entry.declared_bytes = DictInt(d, "declared_bytes", 0);
    entry.dst_space_version =
        static_cast<int32_t>(DictInt(d, "dst_space_version", 0));
    for (nb::handle span_item : d["spans"]) {
      auto t = nb::cast<std::tuple<int64_t, int64_t, int64_t, int64_t,
                                   int64_t, int64_t, int64_t, int64_t>>(
          span_item);
      reshard::ClientByteSpan span;
      span.src_block_ordinal = std::get<0>(t);
      span.src_offset_bytes = std::get<1>(t);
      span.dst_block_index = std::get<2>(t);
      span.dst_offset_bytes = std::get<3>(t);
      span.size_bytes = std::get<4>(t);
      span.src_stride_bytes = std::get<5>(t);
      span.dst_stride_bytes = std::get<6>(t);
      span.count = std::get<7>(t);
      entry.spans.push_back(span);
    }
    out.push_back(std::move(entry));
  }
  return out;
}

void ThrowIfError(const absl::Status& status) {
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }
}

struct ClientHandle {
  std::unique_ptr<reshard::ReshardClient> client;
};

}  // namespace reshardpb2
}  // namespace kv_cache
}  // namespace tpu_raiden

using ::tpu_raiden::kv_cache::ToStdStringVector;

NB_MODULE(_tpu_raiden_torch, m) {
  // =========================================================================
  // 1. Bind RaidenFuture
  // =========================================================================
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
      .def("is_ready", &tpu_raiden::RaidenFuture::IsReady)
      // Non-blocking success check; valid once is_ready() is true. True if the
      // transfer is still pending or completed successfully, False if a ready
      // event carries an error. Lets pollers distinguish success from failure
      // without calling the blocking Await() (which can deadlock the live
      // model).
      .def("ok",
           [](tpu_raiden::RaidenFuture& self) { return self.PollError().ok(); })
      // Non-blocking error message ("" when ok); pair with ok() for logging.
      .def("error_message", [](tpu_raiden::RaidenFuture& self) {
        absl::Status status = self.PollError();
        return status.ok() ? std::string() : std::string(status.message());
      });

  // =========================================================================
  // 2. Bind KVCacheManager
  // =========================================================================
  auto manager_cls = nb::class_<KVCacheManager>(m, "KVCacheManager");
  manager_cls
      .def(nb::init<const std::vector<std::vector<at::Tensor>>&,
                    std::optional<int>, std::optional<int>, bool, int, int,
                    std::optional<std::string>, std::optional<std::string>,
                    int64_t>(),
           nb::arg("device_tensors"), nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = nb::none(),
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("parallelism") = 1, nb::arg("raiden_worker_port") = 0,
           nb::arg("raiden_controller_address") = nb::none(),
           nb::arg("worker_id") = nb::none(), nb::arg("node_id") = 0)
      .def(nb::init<const std::vector<at::Tensor>&, int64_t, int64_t, int64_t,
                    int64_t, double, bool, int, std::optional<int>, int,
                    std::optional<std::string>, std::optional<std::string>>(),
           nb::arg("kv_caches"), nb::arg("node_id"),
           nb::arg("local_control_port"), nb::arg("max_blocks"),
           nb::arg("num_slots"), nb::arg("timeout_s") = 120.0,
           nb::arg("unsafe_skip_buffer_lock") = true,
           nb::arg("parallelism") = 4, nb::arg("listener_port") = nb::none(),
           nb::arg("raiden_worker_port") = 0,
           nb::arg("raiden_controller_address") = nb::none(),
           nb::arg("worker_id") = nb::none())
      .def(nb::init<size_t, size_t, size_t, int64_t, std::optional<int>,
                    std::optional<int>, int, int, std::optional<std::string>,
                    std::optional<std::string>>(),
           nb::arg("num_layers"), nb::arg("num_shards"),
           nb::arg("slice_byte_size"), nb::arg("node_id"),
           nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = nb::none(),
           nb::arg("parallelism") = 1, nb::arg("raiden_worker_port") = 0,
           nb::arg("raiden_controller_address") = nb::none(),
           nb::arg("worker_id") = nb::none())
      .def("node_id", &KVCacheManager::node_id)
      .def("get_raiden_worker_port", &KVCacheManager::GetRaidenWorkerPort)
      .def(
          "register_active_plan",
          [](KVCacheManager& self, uint64_t uuid,
             const nb::bytes& request_bytes, bool is_sender) {
            tpu_sync::rpc::StartTransferRequest request;
            if (!request.ParseFromArray(request_bytes.c_str(),
                                        request_bytes.size())) {
              throw std::runtime_error(
                  "KVCacheManager register_active_plan failed: invalid "
                  "StartTransferRequest bytes");
            }
            absl::Status status =
                self.RegisterActivePlan(uuid, request, is_sender);
            if (!status.ok()) {
              throw std::runtime_error(
                  "KVCacheManager register_active_plan failed: " +
                  std::string(status.message()));
            }
          },
          nb::arg("uuid"), nb::arg("request_bytes"), nb::arg("is_sender"))
      .def(
          "unregister_active_plan",
          [](KVCacheManager& self, uint64_t uuid) {
            absl::Status status = self.UnregisterActivePlan(uuid);
            if (!status.ok()) {
              throw std::runtime_error(
                  "KVCacheManager unregister_active_plan failed: " +
                  std::string(status.message()));
            }
          },
          nb::arg("uuid"))
      .def(
          "push_registered_plan",
          [](KVCacheManager& self, uint64_t uuid, const std::string& peer,
             const std::vector<int>& src_block_ids,
             const std::vector<int>& dst_block_ids, int layer_idx,
             int parallelism) {
            absl::Status status =
                self.PushRegisteredPlan(uuid, peer, src_block_ids,
                                        dst_block_ids, layer_idx, parallelism);
            if (!status.ok()) {
              throw std::runtime_error(
                  "KVCacheManager push_registered_plan failed: " +
                  std::string(status.message()));
            }
          },
          nb::arg("uuid"), nb::arg("peer"), nb::arg("src_block_ids"),
          nb::arg("dst_block_ids"), nb::arg("layer_idx") = -1,
          nb::arg("parallelism") = 1, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "read_block_bytes",
          [](KVCacheManager& self, size_t layer_idx, int block_id) {
            auto status_or = self.ReadBlockBytes(layer_idx, block_id);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager read_block_bytes failed: " +
                  std::string(status_or.status().message()));
            }
            const std::string& data = status_or.value();
            return nb::bytes(data.data(), data.size());
          },
          nb::arg("layer_idx"), nb::arg("block_id"))
      .def(
          "write_block_bytes",
          [](KVCacheManager& self, size_t layer_idx, int block_id,
             const nb::bytes& payload) {
            std::string payload_str(payload.c_str(), payload.size());
            absl::Status status =
                self.WriteBlockBytes(layer_idx, block_id, payload_str);
            if (!status.ok()) {
              throw std::runtime_error(
                  "KVCacheManager write_block_bytes failed: " +
                  std::string(status.message()));
            }
          },
          nb::arg("layer_idx"), nb::arg("block_id"), nb::arg("payload"))
      .def(
          "RegisterRecv",
          [](KVCacheManager& self, uint64_t uuid, const std::string& req_id,
             int expected_block_count) {
            absl::Status status =
                self.RegisterRecv(uuid, req_id, expected_block_count);
            if (!status.ok()) {
              throw std::runtime_error("KVCacheManager RegisterRecv failed: " +
                                       std::string(status.message()));
            }
          },
          nb::arg("uuid"), nb::arg("req_id"), nb::arg("expected_block_count"))
      .def(
          "H2d",
          [](KVCacheManager& self,
             const std::vector<int64_t>& src_offsets_major_dim,
             const std::vector<int64_t>& dst_offsets_major_dim,
             const std::vector<int64_t>& copy_sizes_major_dim) {
            auto status_or =
                self.H2d(src_offsets_major_dim, dst_offsets_major_dim,
                         copy_sizes_major_dim);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager H2d failed: " +
                  std::string(status_or.status().message()));
            }
            return tpu_raiden::RaidenFuture{std::move(status_or.value())};
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "D2h",
          [](KVCacheManager& self,
             const std::vector<int64_t>& src_offsets_major_dim,
             const std::vector<int64_t>& dst_offsets_major_dim,
             const std::vector<int64_t>& copy_sizes_major_dim) {
            auto status_or =
                self.D2h(src_offsets_major_dim, dst_offsets_major_dim,
                         copy_sizes_major_dim);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager D2h failed: " +
                  std::string(status_or.status().message()));
            }
            return tpu_raiden::RaidenFuture{std::move(status_or.value())};
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "D2hAutoAllocate",
          [](KVCacheManager& self,
             const std::vector<int64_t>& src_offsets_major_dim,
             const std::vector<int64_t>& copy_sizes_major_dim) {
            auto status_or = self.D2hAutoAllocate(src_offsets_major_dim,
                                                  copy_sizes_major_dim);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager D2hAutoAllocate failed: " +
                  std::string(status_or.status().message()));
            }
            return std::make_pair(
                status_or.value().first,
                tpu_raiden::RaidenFuture{std::move(status_or.value().second)});
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{})
      .def(
          "H2hWrite",
          [](KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids) {
            auto status_or = self.H2hWrite(std::move(peer), src_block_ids);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager H2hWrite failed: " +
                  std::string(status_or.status().message()));
            }
            return std::make_pair(
                status_or.value().first,
                tpu_raiden::RaidenFuture{std::move(status_or.value().second)});
          },
          nb::arg("peer"), nb::arg("src_block_ids"),
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "H2hRead",
          [](KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids) {
            auto status_or = self.H2hRead(std::move(peer), src_block_ids);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager H2hRead failed: " +
                  std::string(status_or.status().message()));
            }
            return std::make_pair(
                status_or.value().first,
                tpu_raiden::RaidenFuture{std::move(status_or.value().second)});
          },
          nb::arg("peer"), nb::arg("src_block_ids"),
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("local_port", &KVCacheManager::local_port)
      .def_prop_ro("num_layers", &KVCacheManager::num_layers)
      .def_prop_ro("num_shards", &KVCacheManager::num_shards)
      .def_prop_ro("slice_byte_size", &KVCacheManager::slice_byte_size)
      .def_prop_ro("num_block_arrays", &KVCacheManager::num_block_arrays)
      .def("block_bytes", &KVCacheManager::block_bytes,
           nb::arg("block_array_idx"))
      .def_prop_ro("local_control_port", &KVCacheManager::local_control_port)
      .def("get_local_endpoints",
           [](const KVCacheManager& self) {
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
      .def_prop_ro("listener_port", &KVCacheManager::listener_port)
      .def_prop_ro("is_listener_active", &KVCacheManager::is_listener_active)
      .def_prop_ro("transfer_address", &KVCacheManager::transfer_address)
      .def_prop_ro("listener_address", &KVCacheManager::listener_address)

      .def("notify_for_read", &KVCacheManager::NotifyForRead, nb::arg("req_id"),
           nb::arg("uuid"), nb::arg("block_ids"))
      .def(
          "start_read",
          [](KVCacheManager& self, const std::string& req_id, uint64_t uuid,
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
      .def("complete_read", [](KVCacheManager& self) {
        auto [done_sending, done_recving, failed_recving] =
            self.CompleteReadRaw();
        return nb::make_tuple(done_sending, done_recving, failed_recving);
      });
  tpu_raiden::torch_bindings::BindPoolApi<tpu_raiden::RaidenFuture>(
      manager_cls);

  // =========================================================================
  // 3. Bind WeightSynchronizer
  // =========================================================================
  nb::class_<WeightSynchronizer>(m, "WeightSynchronizer")
      .def(nb::init<const std::vector<std::vector<at::Tensor>>&,
                    std::optional<int>, int, std::optional<int>,
                    std::optional<std::string>, bool, bool>(),
           nb::arg("device_tensors"), nb::arg("local_port") = nb::none(),
           nb::arg("parallelism") = 1, nb::arg("listener_port") = nb::none(),
           nb::arg("bind_ip") = nb::none(),
           nb::arg("unsafe_skip_buffer_lock") = true,
           nb::arg("auto_h2d") = false)
      .def(
          "PushWeights",
          [](WeightSynchronizer& self, const std::vector<std::string>& peers) {
            absl::Status s = self.PushWeights(peers);
            if (!s.ok()) {
              throw std::runtime_error(
                  "WeightSynchronizer PushWeights failed: " +
                  std::string(s.message()));
            }
          },
          nb::arg("peers"), nb::call_guard<nb::gil_scoped_release>())

      .def(
          "D2h",
          [](WeightSynchronizer& self) {
            auto status_or_future = self.D2h();
            if (!status_or_future.ok()) {
              throw std::runtime_error(
                  "WeightSynchronizer D2H failed: " +
                  std::string(status_or_future.status().message()));
            }
            absl::Status status = status_or_future.value().Await();
            if (!status.ok()) {
              throw std::runtime_error("WeightSynchronizer D2H copy failed: " +
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
                  "WeightSynchronizer H2D failed: " +
                  std::string(status_or_future.status().message()));
            }
            absl::Status status = status_or_future.value().Await();
            if (!status.ok()) {
              throw std::runtime_error("WeightSynchronizer H2D copy failed: " +
                                       std::string(status.message()));
            }
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "get_host_buffer",
          [](WeightSynchronizer& self, size_t layer_idx,
             size_t shard_idx) -> at::Tensor {
            const uint8_t* ptr = self.GetHostBufferPtr(layer_idx, shard_idx);
            if (!ptr) {
              throw std::runtime_error("Invalid layer or shard index");
            }
            size_t size = self.slice_byte_size() + 256 * 1024;
            return at::from_blob(const_cast<uint8_t*>(ptr),
                                 {static_cast<int64_t>(size)}, at::kByte);
          },
          nb::arg("layer_idx") = 0, nb::arg("shard_idx") = 0)
      .def_prop_ro("local_port", &WeightSynchronizer::local_port)
      .def_prop_ro("listener_port", &WeightSynchronizer::listener_port)
      .def_prop_ro("is_listener_active",
                   &WeightSynchronizer::is_listener_active)
      .def_prop_ro("num_layers", &WeightSynchronizer::num_layers)
      .def_prop_ro("num_shards", &WeightSynchronizer::num_shards)
      .def_prop_ro("slice_byte_size", &WeightSynchronizer::slice_byte_size);

  // =========================================================================
  // 4. Bind KVCacheStore
  // =========================================================================
  nb::class_<tpu_raiden::kv_cache::RaidenId>(m, "RaidenId")
      .def(nb::init<std::string, std::string, std::string, int>(),
           nb::arg("job_name") = "", nb::arg("job_replica_id") = "",
           nb::arg("data_name") = "", nb::arg("data_replica_idx") = 0)
      .def_rw("job_name", &tpu_raiden::kv_cache::RaidenId::job_name)
      .def_rw("job_replica_id", &tpu_raiden::kv_cache::RaidenId::job_replica_id)
      .def_rw("data_name", &tpu_raiden::kv_cache::RaidenId::data_name)
      .def_rw("data_replica_idx",
              &tpu_raiden::kv_cache::RaidenId::data_replica_idx);

  nb::enum_<tpu_raiden::kv_cache::BlockStatus>(m, "BlockStatus")
      .value("INIT", tpu_raiden::kv_cache::BlockStatus::INIT)
      .value("REMOTE", tpu_raiden::kv_cache::BlockStatus::REMOTE)
      .value("HBM", tpu_raiden::kv_cache::BlockStatus::HBM)
      .value("HOST", tpu_raiden::kv_cache::BlockStatus::HOST)
      .value("HOST_AND_HBM", tpu_raiden::kv_cache::BlockStatus::HOST_AND_HBM)
      .value("STORAGE", tpu_raiden::kv_cache::BlockStatus::STORAGE);

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
      // 0 when this store does not host a reshard service (regular stores);
      // set on stores built via create_reshard_store.
      .def_prop_ro("reshard_service_port",
                   [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
                     return self->reshard_service()
                                ? self->reshard_service()->port()
                                : 0;
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
             // Poll drains whatever futures completed and can make blocking
             // RPCs on the way -- every poll entry point drains every
             // operation kind, so even this read-status poll makes one
             // verdict call per outstanding remote save. Holding the GIL
             // across that stalls every Python thread,
             // not just this one.
             //
             // Released around the C++ call ONLY. The nb::bytes below are
             // Python objects, so building them and letting std::make_tuple
             // copy the vectors touches CPython refcounts, which must happen
             // with the GIL held. This is why nb::call_guard is wrong here:
             // it would span the whole lambda, including that.
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
           })
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
             // Released around the C++ call ONLY. The subsequent nb::bytes
             // initialization happens with the GIL held (nb::call_guard
             // is wrong here since it would span the whole lambda).
             // See poll_remote_read_status for the detailed stall analysis.
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
             // See poll_remote_read_status for the detailed stall analysis.
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
           });

  // C++-owned reshard client. The facade-compatible surface is provided by
  // tpu_raiden/api/torch/reshard_client.py on top of this binding.
  nb::class_<::tpu_raiden::kv_cache::reshardpb2::ClientHandle>(m,
                                                               "ReshardClient")
      .def(
          "__init__",
          [](::tpu_raiden::kv_cache::reshardpb2::ClientHandle* self,
             const std::string& address) {
            new (self)::tpu_raiden::kv_cache::reshardpb2::ClientHandle();
            self->client =
                std::make_unique<tpu_raiden::kv_cache::reshard::ReshardClient>(
                    address);
          },
          nb::arg("controller_address"))
      .def(
          "register_work_unit",
          [](::tpu_raiden::kv_cache::reshardpb2::ClientHandle& self,
             const ::tpu_raiden::kv_cache::reshardpb2::UnitTuple& unit,
             const std::vector<std::string>& shards,
             const std::string& control_plane_rpc_address,
             const std::vector<int64_t>& mesh_shape,
             const std::vector<int32_t>& layout,
             const std::vector<int64_t>& global_shape, int32_t itemsize,
             bool has_pool_manifest, nb::object pool_manifest,
             std::optional<std::string> layout_fingerprint,
             std::optional<int64_t> page_tokens,
             std::optional<int32_t> transfer_parallelism,
             std::optional<int32_t> transfer_rank, bool has_variables,
             const std::vector<nb::bytes>& variables) {
            tpu_raiden::kv_cache::reshard::RegisterWorkUnitArgs args;
            args.unit = ::tpu_raiden::kv_cache::reshardpb2::UnitOf(unit);
            args.shards = shards;
            args.control_plane_rpc_address = control_plane_rpc_address;
            args.mesh_shape = mesh_shape;
            args.layout = layout;
            args.global_shape = global_shape;
            args.itemsize = itemsize;
            args.has_pool_manifest = has_pool_manifest;
            if (has_pool_manifest) {
              args.pool_manifest =
                  ::tpu_raiden::kv_cache::reshardpb2::ManifestOf(pool_manifest);
            }
            args.layout_fingerprint = std::move(layout_fingerprint);
            args.page_tokens = page_tokens;
            args.transfer_parallelism = transfer_parallelism;
            args.transfer_rank = transfer_rank;
            args.has_variables = has_variables;
            for (const nb::bytes& payload : variables) {
              args.variables.emplace_back(payload.c_str(), payload.size());
            }
            nb::gil_scoped_release release;
            ::tpu_raiden::kv_cache::reshardpb2::ThrowIfError(
                self.client->RegisterWorkUnit(args));
          })
      .def("register_request_blocks",
           [](::tpu_raiden::kv_cache::reshardpb2::ClientHandle& self,
              const std::string& req_id, int64_t uuid,
              const ::tpu_raiden::kv_cache::reshardpb2::UnitTuple& unit,
              const std::vector<int64_t>& block_ids, nb::object pool_spans) {
             tpu_raiden::kv_cache::RaidenId unit_id =
                 ::tpu_raiden::kv_cache::reshardpb2::UnitOf(unit);
             std::vector<tpu_raiden::kv_cache::reshard::ClientPoolSpans> spans =
                 ::tpu_raiden::kv_cache::reshardpb2::SpansOf(pool_spans);
             nb::gil_scoped_release release;
             ::tpu_raiden::kv_cache::reshardpb2::ThrowIfError(
                 self.client->RegisterRequestBlocks(req_id, uuid, unit_id,
                                                    block_ids, spans));
           })
      .def("release_request_blocks",
           [](::tpu_raiden::kv_cache::reshardpb2::ClientHandle& self,
              const std::string& req_id, int64_t uuid) {
             nb::gil_scoped_release release;
             ::tpu_raiden::kv_cache::reshardpb2::ThrowIfError(
                 self.client->ReleaseRequestBlocks(req_id, uuid));
           })
      .def("complete_request_blocks",
           [](::tpu_raiden::kv_cache::reshardpb2::ClientHandle& self,
              const std::string& req_id, int64_t uuid,
              const ::tpu_raiden::kv_cache::reshardpb2::UnitTuple& unit) {
             tpu_raiden::kv_cache::RaidenId unit_id =
                 ::tpu_raiden::kv_cache::reshardpb2::UnitOf(unit);
             nb::gil_scoped_release release;
             ::tpu_raiden::kv_cache::reshardpb2::ThrowIfError(
                 self.client->CompleteRequestBlocks(req_id, uuid, unit_id));
           })
      .def("cancel_request_blocks_if_unclaimed",
           [](::tpu_raiden::kv_cache::reshardpb2::ClientHandle& self,
              const std::string& req_id, int64_t uuid) {
             nb::gil_scoped_release release;
             absl::StatusOr<bool> result =
                 self.client->CancelRequestBlocksIfUnclaimed(req_id, uuid);
             ::tpu_raiden::kv_cache::reshardpb2::ThrowIfError(result.status());
             return *result;
           })
      .def("start_transfer",
           [](::tpu_raiden::kv_cache::reshardpb2::ClientHandle& self,
              const std::vector<::tpu_raiden::kv_cache::reshardpb2::UnitTuple>&
                  src_units,
              const std::vector<::tpu_raiden::kv_cache::reshardpb2::UnitTuple>&
                  dst_units,
              const std::string& req_id, bool use_block_chunks, bool is_sender,
              int64_t expected_block_count, int64_t uuid,
              const std::string& dst_controller_address,
              const std::string& src_controller_address, int32_t dst_mem_type,
              bool has_dst_device_block_ids,
              const std::vector<int64_t>& dst_device_block_ids,
              bool has_transfer_pool_tags,
              const std::vector<std::string>& transfer_pool_tags,
              const std::vector<int64_t>& dst_block_counts,
              const std::vector<int64_t>& dst_skip_bytes) {
             tpu_raiden::kv_cache::reshard::StartTransferArgs args;
             for (const auto& unit : src_units) {
               args.src_units.push_back(
                   ::tpu_raiden::kv_cache::reshardpb2::UnitOf(unit));
             }
             for (const auto& unit : dst_units) {
               args.dst_units.push_back(
                   ::tpu_raiden::kv_cache::reshardpb2::UnitOf(unit));
             }
             args.req_id = req_id;
             args.use_block_chunks = use_block_chunks;
             args.is_sender = is_sender;
             args.expected_block_count = expected_block_count;
             args.uuid = uuid;
             args.dst_controller_address = dst_controller_address;
             args.src_controller_address = src_controller_address;
             args.dst_mem_type = dst_mem_type;
             args.has_dst_device_block_ids = has_dst_device_block_ids;
             args.dst_device_block_ids = dst_device_block_ids;
             args.has_transfer_pool_tags = has_transfer_pool_tags;
             args.transfer_pool_tags = transfer_pool_tags;
             args.dst_block_counts = dst_block_counts;
             args.dst_skip_bytes = dst_skip_bytes;
             nb::gil_scoped_release release;
             absl::StatusOr<bool> result = self.client->StartTransfer(args);
             ::tpu_raiden::kv_cache::reshardpb2::ThrowIfError(result.status());
             return *result;
           })
      .def("get_transfer_status",
           [](::tpu_raiden::kv_cache::reshardpb2::ClientHandle& self,
              const std::string& req_id, int64_t uuid) {
             nb::gil_scoped_release release;
             absl::StatusOr<int32_t> result =
                 self.client->GetTransferStatus(req_id, uuid);
             ::tpu_raiden::kv_cache::reshardpb2::ThrowIfError(result.status());
             return *result;
           })
      .def("get_metadata",
           [](::tpu_raiden::kv_cache::reshardpb2::ClientHandle& self) {
             std::vector<std::string> serialized;
             {
               nb::gil_scoped_release release;
               absl::StatusOr<std::vector<std::string>> result =
                   self.client->GetMetadata();
               ::tpu_raiden::kv_cache::reshardpb2::ThrowIfError(
                   result.status());
               serialized = *std::move(result);
             }
             nb::list out;
             for (const std::string& payload : serialized) {
               out.append(nb::bytes(payload.data(), payload.size()));
             }
             return out;
           })
      .def("shutdown",
           [](::tpu_raiden::kv_cache::reshardpb2::ClientHandle& self) {
             nb::gil_scoped_release release;
             absl::StatusOr<bool> result = self.client->Shutdown();
             ::tpu_raiden::kv_cache::reshardpb2::ThrowIfError(result.status());
             return *result;
           });
  // Capability marker for the vLLM connector's version-skew probe: present
  // and true iff start_transfer accepts the dst_skip_bytes clip.
  m.attr("reshard_client_supports_dst_skip_bytes") = true;

  // NOTE: KVCacheStoreWrapper is already bound above as "KVCacheStore";
  // nanobind keys class bindings by C++ type, so a second nb::class_ for the
  // same type is silently skipped ("type already registered") and the module
  // would end up without a ReshardStore attribute. Expose the reshard-store
  // factory as a free function returning the existing binding instead; the
  // Python ReshardStore facade wraps it.
  m.def(
      "create_reshard_store",
      [](tpu_raiden::kv_cache::RaidenId raiden_id, std::string store_server_ip,
         int raiden_controller_port, int reshard_service_port) {
        auto store = tpu_raiden::kv_cache::KVCacheStore::CreateReshardStore(
            std::move(raiden_id), store_server_ip, raiden_controller_port,
            reshard_service_port);
        if (!store.ok()) {
          throw std::runtime_error(
              absl::StrCat("ReshardStore initialization failed: ",
                           store.status().message()));
        }
        return tpu_raiden::kv_cache::KVCacheStoreWrapper(std::move(*store));
      },
      nb::arg("raiden_id"), nb::arg("store_server_ip"),
      nb::arg("raiden_controller_port") = 0,
      nb::arg("reshard_service_port") = 0,
      nb::call_guard<nb::gil_scoped_release>());

  tpu_raiden::telemetry::BindTelemetryApi(m);
}
