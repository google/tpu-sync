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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_POOL_LAYOUT_NANOBIND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_POOL_LAYOUT_NANOBIND_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tpu_sync/kv_cache/pool_layout.h"

namespace tpu_raiden {
namespace torch_bindings {

namespace nb = nanobind;

using RegionTuple =
    std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t>;
// (tag, storage_index, base_offset_bytes, block_stride_bytes, num_blocks,
//  regions, dtype_tag)
using PoolTuple = std::tuple<std::string, int64_t, int64_t, int64_t, int64_t,
                             std::vector<RegionTuple>, std::string>;

inline void ThrowIfNotOk(const absl::Status& status,
                         const std::string& context) {
  if (!status.ok()) {
    throw std::runtime_error(absl::StrCat(context, ": ", status.message()));
  }
}

inline std::vector<kv_cache::PoolSpec> PoolsFromTuples(
    const std::vector<PoolTuple>& entries) {
  std::vector<kv_cache::PoolSpec> pools;
  pools.reserve(entries.size());
  for (const auto& [tag, storage_index, base_offset_bytes, block_stride_bytes,
                    num_blocks, region_entries, dtype_tag] : entries) {
    if (storage_index < 0) {
      throw std::invalid_argument(
          absl::StrCat("pool ", tag, " storage_index must be non-negative"));
    }
    kv_cache::PoolSpec pool;
    pool.tag = tag;
    pool.storage_index = static_cast<size_t>(storage_index);
    pool.base_offset_bytes = base_offset_bytes;
    pool.block_stride_bytes = block_stride_bytes;
    pool.num_blocks = num_blocks;
    pool.regions.reserve(region_entries.size());
    for (const auto& [name, offset_bytes, stride_bytes, unit_bytes, num_units,
                      units_per_stride] : region_entries) {
      pool.regions.push_back(kv_cache::RegionSpec{
          .name = name,
          .offset_bytes = offset_bytes,
          .stride_bytes = stride_bytes,
          .unit_bytes = unit_bytes,
          .num_units = num_units,
          .units_per_stride = units_per_stride,
      });
    }
    pool.dtype_tag = dtype_tag;
    pools.push_back(std::move(pool));
  }
  return pools;
}

inline nb::dict RegionToDict(const kv_cache::RegionSpec& region) {
  nb::dict result;
  result["name"] = region.name;
  result["offset_bytes"] = region.offset_bytes;
  result["stride_bytes"] = region.stride_bytes;
  result["unit_bytes"] = region.unit_bytes;
  result["num_units"] = region.num_units;
  result["units_per_stride"] = region.units_per_stride;
  return result;
}

inline nb::dict PoolSpecToDict(const kv_cache::PoolSpec& pool) {
  nb::dict result;
  result["tag"] = pool.tag;
  result["storage_index"] = pool.storage_index;
  result["base_offset_bytes"] = pool.base_offset_bytes;
  result["block_stride_bytes"] = pool.block_stride_bytes;
  result["num_blocks"] = pool.num_blocks;
  nb::list regions;
  for (const kv_cache::RegionSpec& region : pool.regions) {
    regions.append(RegionToDict(region));
  }
  result["regions"] = regions;
  result["dtype_tag"] = pool.dtype_tag;
  return result;
}

inline nb::dict PoolBlockRefToDict(const kv_cache::PoolBlockRef& ref) {
  nb::dict result;
  result["ptr"] = reinterpret_cast<uintptr_t>(ref.ptr);
  result["block_stride_bytes"] = ref.block_stride_bytes;
  result["tag"] = ref.pool->tag;
  result["dtype_tag"] = ref.pool->dtype_tag;
  result["pool_idx"] = ref.pool_idx;
  result["shard_idx"] = ref.shard_idx;
  result["block_id"] = ref.block_id;
  nb::list regions;
  for (const kv_cache::RegionSpec& region : ref.pool->regions) {
    regions.append(RegionToDict(region));
  }
  result["regions"] = regions;
  return result;
}

// Binds the pool admission API onto a manager binding. Shared by the host and
// torch modules so the surface never diverges. The bound class must derive
// from kv_cache::KVCacheManagerBase.
template <typename FutureT, typename ClassT>
void BindPoolApi(ClassT& cls) {
  using ManagerT = typename ClassT::Type;
  cls.def(
         "register_pools_native",
         [](ManagerT& self, const std::vector<PoolTuple>& pools,
            int64_t staging_leases,
            const std::vector<int64_t>& staging_blocks_per_pool) {
           std::vector<kv_cache::PoolSpec> specs = PoolsFromTuples(pools);
           // Bounded host staging hints:
           // one entry per pool, or empty for the full-mirror default.
           if (!staging_blocks_per_pool.empty()) {
             if (staging_blocks_per_pool.size() != specs.size()) {
               throw std::invalid_argument(
                   absl::StrCat("staging_blocks_per_pool has ",
                                staging_blocks_per_pool.size(), " entries for ",
                                specs.size(), " pools"));
             }
             for (size_t i = 0; i < specs.size(); ++i) {
               specs[i].staging_blocks_per_request = staging_blocks_per_pool[i];
             }
           }
           ThrowIfNotOk(self.RegisterPools(std::move(specs), staging_leases),
                        "KVCacheManager register_pools failed");
         },
         nb::arg("pools"), nb::arg("staging_leases") = 0,
         nb::arg("staging_blocks_per_pool") = std::vector<int64_t>{})
      .def("pool_staging_summary_native",
           [](const ManagerT& self) {
             nb::dict result;
             nb::list storages;
             bool any_bounded = false;
             int64_t bounded_bytes = 0;
             int64_t full_bytes = 0;
             for (const auto& s : self.PoolStagingSummary()) {
               nb::dict d;
               d["storage_index"] = s.storage_index;
               d["bounded"] = s.bounded;
               d["stride_bytes"] = s.stride_bytes;
               d["num_slots"] = s.num_slots;
               d["blocks_per_lease"] = s.blocks_per_lease;
               d["host_bytes_per_shard"] = s.host_bytes_per_shard;
               d["free_slots"] = s.free_slots;
               storages.append(d);
               any_bounded |= s.bounded;
               (s.bounded ? bounded_bytes : full_bytes) +=
                   s.host_bytes_per_shard;
             }
             result["mode"] = any_bounded ? "bounded" : "full";
             result["leases"] = self.pool_staging_leases();
             result["bounded_storage_bytes_per_shard"] = bounded_bytes;
             result["full_storage_bytes_per_shard"] = full_bytes;
             result["storages"] = storages;
             return result;
           })
      .def(
          "get_pool_block_ref_native",
          [](ManagerT& self, size_t pool_idx, size_t shard_idx,
             int64_t block_id) {
            auto status_or =
                self.GetPoolBlockRef(pool_idx, shard_idx, block_id);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  absl::StrCat("KVCacheManager get_pool_block_ref failed: ",
                               status_or.status().message()));
            }
            return PoolBlockRefToDict(status_or.value());
          },
          nb::arg("pool_idx"), nb::arg("shard_idx") = 0,
          nb::arg("block_id") = 0)
      .def(
          "pool_indices_with_tag_native",
          [](ManagerT& self, const std::string& tag) {
            return self.PoolIndicesWithTag(tag);
          },
          nb::arg("tag"))
      .def("num_pools", [](ManagerT& self) { return self.num_pools(); })
      .def("has_explicit_pools",
           [](ManagerT& self) { return self.has_explicit_pools(); })
      .def(
          "pool_spec_native",
          [](ManagerT& self, size_t pool_idx) {
            const kv_cache::PoolSpec* pool = self.pool(pool_idx);
            if (pool == nullptr) {
              throw std::out_of_range(
                  absl::StrCat("pool index out of range: ", pool_idx));
            }
            return PoolSpecToDict(*pool);
          },
          nb::arg("pool_idx"))
      .def(
          "get_block_host_pointer",
          [](ManagerT& self, size_t layer_idx, size_t shard_idx, int block_id) {
            auto status_or =
                self.GetBlockHostPointerValue(layer_idx, shard_idx, block_id);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  absl::StrCat("KVCacheManager get_block_host_pointer failed: ",
                               status_or.status().message()));
            }
            return status_or.value();
          },
          nb::arg("layer_idx"), nb::arg("shard_idx") = 0,
          nb::arg("block_id") = 0)
      .def("layer_block_byte_size", &ManagerT::LayerBlockByteSize,
           nb::arg("layer_idx"))
      .def(
          "d2h_pool_blocks",
          [](ManagerT& self, size_t pool_idx,
             const std::vector<int64_t>& block_ids,
             std::optional<size_t> shard_idx, std::optional<uint64_t> uuid) {
            auto status_or =
                self.D2hPoolBlocks(pool_idx, block_ids, shard_idx, uuid);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  absl::StrCat("KVCacheManager d2h_pool_blocks failed: ",
                               status_or.status().message()));
            }
            return FutureT{std::move(status_or).value()};
          },
          nb::arg("pool_idx"), nb::arg("block_ids"),
          nb::arg("shard_idx") = nb::none(), nb::arg("uuid") = nb::none(),
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "h2d_pool_blocks",
          [](ManagerT& self, size_t pool_idx,
             const std::vector<int64_t>& block_ids,
             std::optional<size_t> shard_idx, std::optional<uint64_t> uuid) {
            auto status_or =
                self.H2dPoolBlocks(pool_idx, block_ids, shard_idx, uuid);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  absl::StrCat("KVCacheManager h2d_pool_blocks failed: ",
                               status_or.status().message()));
            }
            return FutureT{std::move(status_or).value()};
          },
          nb::arg("pool_idx"), nb::arg("block_ids"),
          nb::arg("shard_idx") = nb::none(), nb::arg("uuid") = nb::none(),
          nb::call_guard<nb::gil_scoped_release>());
}

}  // namespace torch_bindings
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_POOL_LAYOUT_NANOBIND_H_
