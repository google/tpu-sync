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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>  // IWYU pragma: keep
#include <nanobind/stl/optional.h>  // IWYU pragma: keep
#include <nanobind/stl/pair.h>  // IWYU pragma: keep
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include <nanobind/stl/string_view.h>  // IWYU pragma: keep
#include <nanobind/stl/tuple.h>  // IWYU pragma: keep
#include <nanobind/stl/vector.h>  // IWYU pragma: keep
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/rpc/controller_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/manager/v3/controller_v3.h"
#include "tpu_sync/weight_sync/manager/v3/dynamic_pull_engine.h"
#include "tpu_sync/weight_sync/manager/v3/entity_registry.h"
#include "tpu_sync/weight_sync/manager/v3/logical_reshard_planner.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"
#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

namespace nb = nanobind;

void ThrowIfError(const absl::Status& status) {
  if (status.ok()) {
    return;
  }
  if (status.code() == absl::StatusCode::kInvalidArgument) {
    throw nb::value_error(std::string(status.message()).c_str());
  }
  throw std::runtime_error(std::string(status.message()));
}

template <typename T>
T UnwrapOrThrow(absl::StatusOr<T> status_or) {
  ThrowIfError(status_or.status());
  return *std::move(status_or);
}

RaidenId ParseRaidenIdBytes(const nb::bytes& raw) {
  tpu_sync::rpc::RaidenIdProto proto;
  if (!proto.ParseFromString(absl::string_view(raw.c_str(), raw.size()))) {
    throw nb::value_error("Failed to parse RaidenIdProto bytes");
  }
  return RaidenIdFromProto(proto);
}

std::vector<RaidenId> ParseRaidenIdBytesList(
    const std::vector<nb::bytes>& raw_list) {
  std::vector<RaidenId> units;
  units.reserve(raw_list.size());
  for (const nb::bytes& b : raw_list) {
    units.push_back(ParseRaidenIdBytes(b));
  }
  return units;
}

absl::flat_hash_map<int32_t, bool> MapToFlatHashMap(
    const std::map<int32_t, bool>& m) {
  absl::flat_hash_map<int32_t, bool> out;
  out.reserve(m.size());
  for (const auto& [k, v] : m) {
    out[k] = v;
  }
  return out;
}

// Wraps a Python callable `fn(endpoint: str, control_request_bytes: bytes)
// -> Optional[bytes]` as the controller's worker RPC transport. A `None`
// return is treated as a successful empty `ControlResponse`; a `bytes` return
// is parsed as a serialized `ControlResponse`. The GIL is acquired only for
// the duration of the Python call, so this is safe to invoke from C++ server
// and transfer threads.
DynamicPullEngine::RpcSenderFn MakePyRpcSender(nb::object fn) {
  return [fn = std::move(fn)](absl::string_view endpoint,
                              const tpu_sync::rpc::ControlRequest& req)
             -> absl::StatusOr<tpu_sync::rpc::ControlResponse> {
    std::string req_bytes = req.SerializeAsString();
    nb::gil_scoped_acquire acquire;
    try {
      nb::object res = fn(std::string(endpoint),
                          nb::bytes(req_bytes.data(), req_bytes.size()));
      tpu_sync::rpc::ControlResponse resp;
      if (nb::isinstance<nb::bytes>(res)) {
        nb::bytes b = nb::cast<nb::bytes>(res);
        if (!resp.ParseFromString(absl::string_view(b.c_str(), b.size()))) {
          return absl::InternalError(
              "Failed to parse ControlResponse bytes from py_rpc_sender");
        }
      } else if (res.is_none()) {
        resp.set_success(true);
      } else {
        return absl::InternalError("py_rpc_sender must return bytes or None");
      }
      return resp;
    } catch (const std::exception& e) {
      return absl::InternalError(e.what());
    }
  };
}

// Wraps a Python callable `resolver(endpoint: str) -> str` in front of a
// dedicated C++ `ControlPipeClient`: every worker-bound RPC first resolves the
// logical endpoint in Python and is then dialed natively.
DynamicPullEngine::RpcSenderFn MakePyResolvingRpcSender(nb::object resolver) {
  ControlPipeConfig cfg;
  cfg.backend_type = ResolveControlPipeBackendType();
  cfg.enable_tcp_connection_pooling = true;
  std::shared_ptr<ControlPipeClient> client = CreateControlPipeClient(cfg);
  return [resolver = std::move(resolver), client](
             absl::string_view endpoint,
             const tpu_sync::rpc::ControlRequest& req)
             -> absl::StatusOr<tpu_sync::rpc::ControlResponse> {
    std::string resolved;
    {
      nb::gil_scoped_acquire acquire;
      try {
        resolved = nb::cast<std::string>(resolver(std::string(endpoint)));
      } catch (const std::exception& e) {
        return absl::InternalError(e.what());
      }
    }
    return client
        ->Call<tpu_sync::rpc::ControlRequest, tpu_sync::rpc::ControlResponse>(
            resolved, req, absl::Seconds(120));
  };
}

}  // namespace

NB_MODULE(_controller_v3, m) {
  m.doc() =
      "C++ V3 Weight Sync Controller with Dynamic Bundle Pull and Source "
      "Promotion.";

  nb::class_<NdSlice>(m, "NdSlice")
      .def(nb::init<>())
      .def(
          "__init__",
          [](NdSlice* self, const std::vector<int64_t>& offsets,
             const std::vector<int64_t>& sizes) {
            new (self) NdSlice{offsets, sizes};
          },
          nb::arg("offsets"), nb::arg("sizes"))
      .def_rw("offsets", &NdSlice::offsets)
      .def_rw("sizes", &NdSlice::sizes)
      .def("is_empty", &NdSlice::IsEmpty)
      .def("num_elements", &NdSlice::NumElements)
      .def("__eq__", &NdSlice::operator==);

  nb::class_<VariableBundleSpec>(m, "VariableBundleSpec")
      .def(nb::init<>())
      .def_rw("bundle_id", &VariableBundleSpec::bundle_id)
      .def_rw("layer_indices", &VariableBundleSpec::layer_indices)
      .def_rw("total_bytes", &VariableBundleSpec::total_bytes)
      .def_rw("layer_byte_sizes", &VariableBundleSpec::layer_byte_sizes);

  m.def("compute_seed_sampler_count",
        &LogicalReshardPlanner::ComputeSeedSamplerCount,
        nb::arg("num_dst_replicas"), nb::arg("broadcast_host_ratio"),
        nb::arg("trainer_hosts"), nb::arg("sampler_hosts"),
        "Computes bandwidth-matched seed sampler count n_seed.");

  m.def(
      "get_global_indices",
      [](int64_t idx, const std::vector<int64_t>& shape) {
        return LogicalReshardPlanner::GetGlobalIndices(idx, shape);
      },
      nb::arg("idx"), nb::arg("shape"));

  m.def("intersect_nd_slices", &LogicalReshardPlanner::IntersectNdSlices,
        nb::arg("slice_a"), nb::arg("slice_b"));

  m.def(
      "to_physical",
      [](const NdSlice& slice, const std::vector<int64_t>& layout) {
        return LogicalReshardPlanner::ToPhysical(slice, layout);
      },
      nb::arg("slice"), nb::arg("layout"));

  m.def(
      "is_nd_slice_tile_aligned",
      [](const NdSlice& src_slice, const NdSlice& dst_slice,
         const NdSlice& intersection, const std::vector<int64_t>& src_layout,
         const std::vector<int64_t>& dst_layout) {
        return LogicalReshardPlanner::IsNdSliceTileAligned(
            src_slice, dst_slice, intersection, src_layout, dst_layout);
      },
      nb::arg("src_slice"), nb::arg("dst_slice"), nb::arg("intersection"),
      nb::arg("src_layout"), nb::arg("dst_layout"));

  m.def(
      "generate_strided_copy_chunks",
      [](const NdSlice& src_slice, const NdSlice& dst_slice,
         const NdSlice& intersection, const std::vector<int64_t>& src_layout,
         const std::vector<int64_t>& dst_layout, int64_t itemsize) {
        std::vector<StridedCopyChunk> chunks =
            LogicalReshardPlanner::GenerateStridedCopyChunks(
                src_slice, dst_slice, intersection, src_layout, dst_layout,
                itemsize);
        std::vector<
            std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>>
            out;
        out.reserve(chunks.size());
        for (const StridedCopyChunk& c : chunks) {
          out.emplace_back(c.src_offset_bytes, c.dst_offset_bytes, c.size_bytes,
                           c.src_stride_bytes, c.dst_stride_bytes, c.count);
        }
        return out;
      },
      nb::arg("src_slice"), nb::arg("dst_slice"), nb::arg("intersection"),
      nb::arg("src_layout"), nb::arg("dst_layout"), nb::arg("itemsize"));

  m.def(
      "generate_strided_copy_chunks_tile_aware",
      [](const NdSlice& src_slice, const NdSlice& dst_slice,
         const NdSlice& intersection, const std::vector<int64_t>& src_layout,
         const std::vector<int64_t>& dst_layout, int64_t itemsize) {
        std::vector<StridedCopyChunk> chunks =
            LogicalReshardPlanner::GenerateStridedCopyChunksTileAware(
                src_slice, dst_slice, intersection, src_layout, dst_layout,
                itemsize);
        std::vector<
            std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>>
            out;
        out.reserve(chunks.size());
        for (const StridedCopyChunk& c : chunks) {
          out.emplace_back(c.src_offset_bytes, c.dst_offset_bytes, c.size_bytes,
                           c.src_stride_bytes, c.dst_stride_bytes, c.count);
        }
        return out;
      },
      nb::arg("src_slice"), nb::arg("dst_slice"), nb::arg("intersection"),
      nb::arg("src_layout"), nb::arg("dst_layout"), nb::arg("itemsize"));

  m.def(
      "partition_variable_bundles",
      [](int32_t num_layers, const std::vector<int64_t>& dst_layer_shard_bytes,
         int32_t num_bundle_groups) {
        return LogicalReshardPlanner::PartitionVariableBundles(
            num_layers, dst_layer_shard_bytes, num_bundle_groups);
      },
      nb::arg("num_layers"), nb::arg("dst_layer_shard_bytes"),
      nb::arg("num_bundle_groups") = 8);

  nb::class_<HostTransferCommand>(m, "HostTransferCommand",
                                  "Read-only view of one per-host "
                                  "StartTransferRequest dispatch command.")
      .def_prop_ro("unit_bytes",
                   [](const HostTransferCommand& c) {
                     std::string s =
                         RaidenIdToProto(c.unit).SerializeAsString();
                     return nb::bytes(s.data(), s.size());
                   })
      .def_prop_ro("host_idx",
                   [](const HostTransferCommand& c) { return c.host_idx; })
      .def_prop_ro(
          "control_endpoint",
          [](const HostTransferCommand& c) { return c.control_endpoint; })
      .def("request_bytes", [](const HostTransferCommand& c) {
        std::string s = c.request.SerializeAsString();
        return nb::bytes(s.data(), s.size());
      });

  nb::class_<SamplerHostCommand>(
      m, "SamplerHostCommand",
      "Read-only view of the receiver command of one sampler host and the "
      "bundles the Trainer seeds onto it.")
      .def_prop_ro("replica_idx",
                   [](const SamplerHostCommand& c) { return c.replica_idx; })
      .def_prop_ro("command",
                   [](const SamplerHostCommand& c) { return c.command; })
      .def_prop_ro("seeded_bundles",
                   [](const SamplerHostCommand& c) { return c.seeded_bundles; })
      .def_prop_ro("seeded_layers",
                   [](const SamplerHostCommand& c) { return c.seeded_layers; });

  nb::class_<MaterializedTransferPlan>(
      m, "MaterializedTransferPlan",
      "Read-only view of a C++ materialized V3 transfer plan.")
      .def_prop_ro("req_id",
                   [](const MaterializedTransferPlan& p) { return p.req_id; })
      .def_prop_ro("uuid",
                   [](const MaterializedTransferPlan& p) { return p.uuid; })
      .def_prop_ro("num_stripes",
                   [](const MaterializedTransferPlan& p) {
                     return p.seed_layout.num_stripes;
                   })
      .def_prop_ro("seed_replication",
                   [](const MaterializedTransferPlan& p) {
                     return p.seed_layout.replication;
                   })
      .def_prop_ro("dst_layer_shard_bytes",
                   [](const MaterializedTransferPlan& p) {
                     return p.dst_layer_shard_bytes;
                   })
      .def_prop_ro("skip_tiling_by_layer",
                   [](const MaterializedTransferPlan& p) {
                     nb::list out;
                     for (bool b : p.skip_tiling_by_layer) {
                       out.append(nb::bool_(b));
                     }
                     return out;
                   })
      .def_prop_ro("dst_mem_type",
                   [](const MaterializedTransferPlan& p) {
                     return static_cast<int32_t>(p.dst_mem_type);
                   })
      .def_prop_ro(
          "parallelism",
          [](const MaterializedTransferPlan& p) { return p.parallelism; })
      .def_prop_ro(
          "variable_bundles",
          [](const MaterializedTransferPlan& p) { return p.variable_bundles; })
      .def_prop_ro(
          "sampler_commands",
          [](const MaterializedTransferPlan& p) { return p.sampler_commands; })
      .def_prop_ro("trainer_wave_commands",
                   [](const MaterializedTransferPlan& p) {
                     return p.trainer_wave_commands;
                   })
      .def(
          "describe_plan",
          [](const MaterializedTransferPlan& p) {
            const SeedLayout& layout = p.seed_layout;
            nb::dict out;
            out["num_stripes"] = layout.num_stripes;
            out["seed_replication"] = layout.replication;
            out["stripe_bundles"] = nb::cast(layout.stripe_bundles);
            out["stripe_seeds"] = nb::cast(layout.stripe_seeds);
            nb::list waves;
            for (const auto& wave : layout.trainer_waves) {
              nb::list pushes;
              for (const auto& [replica, stripe] : wave) {
                pushes.append(nb::make_tuple(replica, stripe));
              }
              waves.append(pushes);
            }
            out["trainer_waves"] = waves;
            return out;
          },
          "Returns the seed layout as plain Python data: `num_stripes`, "
          "`seed_replication`, `stripe_bundles`, `stripe_seeds` and "
          "`trainer_waves` (lists of `(replica, stripe)`). The pulls are "
          "scheduled while the transfer runs.");

  nb::class_<RaidenControllerV3>(m, "RaidenControllerV3")
      .def(
          "__init__",
          [](RaidenControllerV3* self, int port, double broadcast_host_ratio,
             int32_t num_bundle_groups,
             int32_t max_concurrent_uploads_per_source, bool enable_plan_cache,
             int32_t num_stripes, int32_t seed_replication,
             int32_t grant_batch_size, int64_t lease_timeout_ms,
             int64_t long_poll_timeout_ms, double request_registry_ttl_s,
             int64_t transfer_timeout_ms, nb::object py_rpc_sender,
             nb::object py_endpoint_resolver) {
            RaidenControllerV3::Options opts;
            opts.port = port;
            opts.broadcast_host_ratio = broadcast_host_ratio;
            opts.num_bundle_groups = num_bundle_groups;
            opts.max_concurrent_uploads_per_source =
                max_concurrent_uploads_per_source;
            opts.enable_plan_cache = enable_plan_cache;
            opts.num_stripes = num_stripes;
            opts.seed_replication = seed_replication;
            opts.grant_batch_size = grant_batch_size;
            opts.lease_timeout_ms = lease_timeout_ms;
            opts.long_poll_timeout_ms = long_poll_timeout_ms;
            opts.request_registry_ttl_s = request_registry_ttl_s;
            opts.transfer_timeout_ms = transfer_timeout_ms;
            if (!py_rpc_sender.is_none()) {
              opts.custom_rpc_sender = MakePyRpcSender(py_rpc_sender);
            } else if (!py_endpoint_resolver.is_none()) {
              opts.custom_rpc_sender =
                  MakePyResolvingRpcSender(py_endpoint_resolver);
            }
            new (self) RaidenControllerV3(opts);
          },
          nb::arg("port") = 0, nb::arg("broadcast_host_ratio") = 1.0,
          nb::arg("num_bundle_groups") = 8,
          nb::arg("max_concurrent_uploads_per_source") = 8,
          nb::arg("enable_plan_cache") = true, nb::arg("num_stripes") = 0,
          nb::arg("seed_replication") = 2, nb::arg("grant_batch_size") = 8,
          nb::arg("lease_timeout_ms") = 30000,
          nb::arg("long_poll_timeout_ms") = 5000,
          nb::arg("request_registry_ttl_s") = 600.0,
          nb::arg("transfer_timeout_ms") = 600000,
          nb::arg("py_rpc_sender") = nb::none(),
          nb::arg("py_endpoint_resolver") = nb::none(),
          "Constructs the C++ controller. `py_rpc_sender(endpoint: str, "
          "control_request_bytes: bytes) -> Optional[bytes]` replaces the "
          "built-in worker RPC transport for all worker-bound RPCs; "
          "`py_endpoint_resolver(endpoint: str) -> str` rewrites worker "
          "endpoints before the built-in transport dials them.")
      .def("start_server",
           [](RaidenControllerV3& self) {
             absl::StatusOr<int> port;
             {
               nb::gil_scoped_release release;
               port = self.StartServer();
             }
             return UnwrapOrThrow(std::move(port));
           })
      .def("stop_server",
           [](RaidenControllerV3& self) {
             nb::gil_scoped_release release;
             self.StopServer();
           })
      .def_prop_ro("port", &RaidenControllerV3::port)
      .def_prop_rw("controller_address",
                   &RaidenControllerV3::controller_address,
                   [](RaidenControllerV3& self, absl::string_view addr) {
                     self.set_controller_address(addr);
                   })
      .def_prop_rw("broadcast_host_ratio",
                   &RaidenControllerV3::broadcast_host_ratio,
                   &RaidenControllerV3::set_broadcast_host_ratio)
      .def_prop_rw("num_bundle_groups", &RaidenControllerV3::num_bundle_groups,
                   &RaidenControllerV3::set_num_bundle_groups)
      .def_prop_rw("num_stripes", &RaidenControllerV3::num_stripes,
                   &RaidenControllerV3::set_num_stripes)
      .def_prop_rw("seed_replication", &RaidenControllerV3::seed_replication,
                   &RaidenControllerV3::set_seed_replication)
      .def_prop_rw("request_registry_ttl_s",
                   &RaidenControllerV3::request_registry_ttl_s,
                   &RaidenControllerV3::set_request_registry_ttl_s)
      .def_prop_rw("max_concurrent_uploads_per_source",
                   &RaidenControllerV3::max_concurrent_uploads_per_source,
                   &RaidenControllerV3::set_max_concurrent_uploads_per_source)
      .def_prop_rw("grant_batch_size", &RaidenControllerV3::grant_batch_size,
                   &RaidenControllerV3::set_grant_batch_size)
      .def_prop_rw("lease_timeout_ms", &RaidenControllerV3::lease_timeout_ms,
                   &RaidenControllerV3::set_lease_timeout_ms)
      .def_prop_rw("long_poll_timeout_ms",
                   &RaidenControllerV3::long_poll_timeout_ms,
                   &RaidenControllerV3::set_long_poll_timeout_ms)
      .def_prop_rw("transfer_timeout_ms",
                   &RaidenControllerV3::transfer_timeout_ms,
                   &RaidenControllerV3::set_transfer_timeout_ms)
      .def(
          "register_work_unit_bytes",
          [](RaidenControllerV3& self, const nb::bytes& req_bytes) {
            tpu_sync::rpc::RegisterWorkUnitRequest req;
            if (!req.ParseFromString(
                    absl::string_view(req_bytes.c_str(), req_bytes.size()))) {
              throw nb::value_error(
                  "Failed to parse RegisterWorkUnitRequest bytes");
            }
            ThrowIfError(self.RegisterWorkUnit(req));
          },
          nb::arg("req_bytes"))
      .def(
          "attach_host_bytes",
          [](RaidenControllerV3& self, const nb::bytes& unit_bytes,
             absl::string_view control_address,
             const std::optional<std::vector<std::string>>& host_shards) {
            RaidenId unit = ParseRaidenIdBytes(unit_bytes);
            std::vector<std::string> shards =
                host_shards.value_or(std::vector<std::string>{});
            ThrowIfError(self.AttachHost(unit, control_address, shards));
          },
          nb::arg("unit_bytes"), nb::arg("control_address"),
          nb::arg("host_shards") = nb::none())
      .def(
          "has_unit_bytes",
          [](const RaidenControllerV3& self, const nb::bytes& unit_bytes) {
            return self.entity_registry().HasUnit(
                ParseRaidenIdBytes(unit_bytes));
          },
          nb::arg("unit_bytes"))
      .def("get_registered_units_bytes",
           [](const RaidenControllerV3& self) {
             std::vector<nb::bytes> out;
             for (const RaidenId& u :
                  self.entity_registry().GetRegisteredUnits()) {
               std::string s = RaidenIdToProto(u).SerializeAsString();
               out.emplace_back(s.data(), s.size());
             }
             return out;
           })
      .def("get_all_metadata_bytes",
           [](const RaidenControllerV3& self) {
             std::vector<tpu_sync::rpc::RegisterWorkUnitRequest> metadata =
                 self.entity_registry().GetAllMetadata();
             std::vector<nb::bytes> out;
             out.reserve(metadata.size());
             for (const auto& item : metadata) {
               std::string s = item.SerializeAsString();
               out.emplace_back(s.data(), s.size());
             }
             return out;
           })
      .def(
          "build_materialized_plan_bytes",
          [](RaidenControllerV3& self, const std::string& req_id, uint64_t uuid,
             const std::vector<nb::bytes>& src_unit_bytes_list,
             const std::vector<nb::bytes>& dst_unit_bytes_list,
             int32_t dst_mem_type, bool skip_d2h, int32_t parallelism,
             const std::map<int32_t, bool>& skip_tiling, bool use_cached_plan) {
            std::vector<RaidenId> src_units =
                ParseRaidenIdBytesList(src_unit_bytes_list);
            std::vector<RaidenId> dst_units =
                ParseRaidenIdBytesList(dst_unit_bytes_list);
            absl::flat_hash_map<int32_t, bool> skip_tiling_map =
                MapToFlatHashMap(skip_tiling);
            absl::StatusOr<MaterializedTransferPlan> plan;
            {
              nb::gil_scoped_release release;
              plan = self.BuildMaterializedPlan(
                  req_id, uuid, src_units, dst_units,
                  static_cast<tpu_sync::rpc::MemoryType>(dst_mem_type),
                  skip_d2h, parallelism, skip_tiling_map, use_cached_plan);
            }
            return UnwrapOrThrow(std::move(plan));
          },
          nb::arg("req_id"), nb::arg("uuid"), nb::arg("src_unit_bytes_list"),
          nb::arg("dst_unit_bytes_list"),
          nb::arg("dst_mem_type") =
              static_cast<int32_t>(tpu_sync::rpc::MEMORY_TYPE_DRAM),
          nb::arg("skip_d2h") = false, nb::arg("parallelism") = 1,
          nb::arg("skip_tiling") = std::map<int32_t, bool>{},
          nb::arg("use_cached_plan") = true)
      .def(
          "execute_materialized_transfer_sync_bytes",
          [](RaidenControllerV3& self, const std::string& req_id,
             uint64_t expected_uuid,
             std::optional<int64_t> transfer_timeout_ms) {
            std::optional<absl::Duration> transfer_timeout;
            if (transfer_timeout_ms.has_value()) {
              transfer_timeout = absl::Milliseconds(*transfer_timeout_ms);
            }
            absl::Status status;
            {
              nb::gil_scoped_release release;
              status = self.ExecuteMaterializedTransferSync(
                  req_id, expected_uuid, /*rpc_sender_override=*/nullptr,
                  transfer_timeout);
            }
            ThrowIfError(status);
          },
          nb::arg("req_id"), nb::arg("expected_uuid") = 0,
          nb::arg("transfer_timeout_ms") = nb::none(),
          "Executes the plan stored under `req_id` against the destination "
          "units it was materialized for. A non-zero `expected_uuid` must "
          "match the stored plan's uuid. `transfer_timeout_ms` overrides the "
          "controller's `transfer_timeout_ms` for this execution.")
      .def(
          "wait_for_transfer",
          [](RaidenControllerV3& self, const std::string& req_id,
             std::optional<double> timeout_s) {
            std::optional<absl::Duration> timeout;
            if (timeout_s.has_value()) timeout = absl::Seconds(*timeout_s);
            absl::Status status;
            {
              nb::gil_scoped_release release;
              status = self.WaitForTransfer(req_id, timeout);
            }
            ThrowIfError(status);
          },
          nb::arg("req_id"), nb::arg("timeout_s") = nb::none(),
          "Waits for transfer `req_id`. Without `timeout_s` it waits until "
          "shortly after the transfer's deadline. A timed-out wait does not "
          "cancel the transfer.")
      .def(
          "get_transfer_status",
          [](RaidenControllerV3& self, const std::string& req_id) {
            return static_cast<int>(self.GetTransferStatus(req_id));
          },
          nb::arg("req_id"))
      .def(
          "handle_control_request_bytes",
          [](RaidenControllerV3& self, const nb::bytes& req_bytes) {
            tpu_sync::rpc::ControlRequest req;
            if (!req.ParseFromString(
                    absl::string_view(req_bytes.c_str(), req_bytes.size()))) {
              throw nb::value_error("Failed to parse ControlRequest bytes");
            }
            tpu_sync::rpc::ControlResponse resp;
            {
              nb::gil_scoped_release release;
              resp = self.HandleControlRequest(req);
            }
            std::string out = resp.SerializeAsString();
            return nb::bytes(out.data(), out.size());
          },
          nb::arg("req_bytes"))
      .def(
          "get_pull_stats",
          [](const RaidenControllerV3& self, const std::string& req_id) {
            std::vector<PullShardStats> stats =
                UnwrapOrThrow(self.GetPullStats(req_id));
            nb::list out;
            for (const PullShardStats& s : stats) {
              nb::dict shard;
              shard["host"] = s.host;
              shard["requests"] = s.requests;
              shard["grants"] = s.grants;
              shard["completions"] = s.completions;
              shard["failures"] = s.failures;
              shard["expired_leases"] = s.expired_leases;
              shard["long_poll_timeouts"] = s.long_poll_timeouts;
              shard["suspended_hosts"] = s.suspended_hosts;
              shard["unhealthy_sources"] = s.unhealthy_sources;
              shard["parked_requests"] = s.parked_requests;
              shard["done_hosts"] = s.done_hosts;
              out.append(shard);
            }
            return out;
          },
          nb::arg("req_id"),
          "Returns the pull scheduler counters of the current (or last) "
          "execution of `req_id`, one dict per host index: `host`, "
          "`requests`, `grants`, `completions`, `failures`, "
          "`expired_leases`, `long_poll_timeouts`, `suspended_hosts`, "
          "`unhealthy_sources`, `parked_requests` and `done_hosts`.")
      .def("clear_plan_cache", &RaidenControllerV3::ClearPlanCache)
      .def("get_plan_cache_size", &RaidenControllerV3::GetPlanCacheSize)
      .def("get_transfer_record_count",
           &RaidenControllerV3::GetTransferRecordCount)
      .def(
          "get_retained_plan_uuids",
          [](RaidenControllerV3& self) {
            nb::dict out;
            for (const auto& [req_id, uuid] : self.GetRetainedPlanUuids()) {
              out[nb::str(req_id.data(), req_id.size())] = uuid;
            }
            return out;
          },
          "Returns {req_id: uuid} for every retained record holding a "
          "materialized plan (after evicting expired records).")
      .def("get_plan_materialization_count",
           &RaidenControllerV3::GetPlanMaterializationCountForTest);
}

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden
