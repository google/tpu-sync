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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#endif
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/core/controller/worker_service_server.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/tpu_utils.h"

namespace xla {
class PjRtBuffer;
}  // namespace xla

namespace tpu_raiden {
class MetricsCollector;

namespace kv_cache {
namespace jax {

struct UnpackedCache {
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers;
#ifndef WITHOUT_PYTHON
  nanobind::list device_arrays;
#endif
};

class NumaAwareKVCacheManager {
 public:
  NumaAwareKVCacheManager(const NumaAwareKVCacheManager&) = delete;
  NumaAwareKVCacheManager& operator=(const NumaAwareKVCacheManager&) = delete;
  NumaAwareKVCacheManager(NumaAwareKVCacheManager&&) = default;
  NumaAwareKVCacheManager& operator=(NumaAwareKVCacheManager&&) = default;

#ifndef WITHOUT_PYTHON
  // JAX sharded constructor E2E (cache-only by default)
  NumaAwareKVCacheManager(
      nanobind::list device_arrays,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1,
      int64_t node_id = 0);

  // New transfer-enabled constructor (flat list of arrays, single shard per
  // layer)
  NumaAwareKVCacheManager(nanobind::list kv_caches, int64_t node_id,
                          int64_t local_control_port, int64_t max_blocks,
                          int64_t num_slots, double timeout_s,
                          bool unsafe_skip_buffer_lock, int parallelism);
#endif

  // FFI metadata constructor (cache-only by default)
  NumaAwareKVCacheManager(size_t num_layers, size_t num_shards,
                          size_t slice_byte_size, std::optional<int> local_port,
                          std::optional<int> host_blocks_to_allocate,
                          int parallelism = 1);

  // Test-only constructor for sub-manager mock injection
  explicit NumaAwareKVCacheManager(
      std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> sub_managers);

  ~NumaAwareKVCacheManager();

#ifndef WITHOUT_PYTHON
  nanobind::list kv_caches() const {
    return device_arrays_.value_or(nanobind::list());
  }
#endif

  // Forwarding methods delegating to sub_managers_
  size_t num_layers() const;
  size_t num_shards() const;
  size_t slice_byte_size() const;
  size_t num_block_arrays() const;
  size_t block_bytes(size_t block_array_idx) const;

  std::optional<int> local_port() const;
  int local_control_port() const;
  int64_t node_id() const;

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx);
  const uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) const;
  size_t GetHostSize(size_t layer_idx, size_t shard_idx);

  int64_t NotifyForRead(const std::string& req_id, uint64_t uuid,
                        const std::vector<int64_t>& block_ids);

  std::vector<int32_t> RenewRemoteLeases(const std::string& remote_endpoint,
                                         const std::vector<uint64_t>& uuids);
  std::vector<int32_t> CancelRemoteLeases(const std::string& remote_endpoint,
                                          const std::vector<uint64_t>& uuids);

  void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::string& remote_endpoint,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  std::vector<RaidenTransferEndpoint> get_local_endpoints() const;
  std::vector<RaidenTransferEndpoint> get_local_data_endpoints() const;

  void SetSubmanagerShardsForTesting(
      const std::vector<std::vector<int64_t>>& assignment) {
    submanager_to_global_shards_ = assignment;
  }

  void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  CompleteReadRaw();

  absl::Status UnlockBlocks(const std::vector<int>& block_ids);

  std::string DumpMetricsToString() const;

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt);

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt);

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(const std::vector<int64_t>& src_offsets_major_dim = {},
                  const std::vector<int64_t>& copy_sizes_major_dim = {});

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1);

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1);

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string peer, const std::vector<int>& src_block_ids);

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int>& src_block_ids);

  // Receiver-initiated pull into EXPLICIT local host blocks. Unlike H2hRead,
  // which auto-allocates its destination, this lands the data in the blocks
  // the caller already reserved -- which is what a store-level read needs,
  // since the store commits those ids into its directory.
  absl::StatusOr<raiden::PjRtCopyFuture> H2hReadExplicit(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids);

  // Receiver-initiated pull: remote host src -> local host staging -> local
  // device dst. Fans out to every sub-manager with the endpoint whose shard
  // tags match that sub-manager's global shards (same matching idiom as the
  // vector H2hRead/H2hWrite above).
  absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int64_t>& src_host_offsets,
      const std::vector<int64_t>& dst_host_offsets,
      const std::vector<int64_t>& dst_device_offsets,
      const std::vector<int64_t>& copy_sizes);

 private:
#ifndef WITHOUT_PYTHON
  NumaAwareKVCacheManager(UnpackedCache&& cache, std::optional<int> local_port,
                          std::optional<int> host_blocks_to_allocate,
                          bool unsafe_skip_buffer_lock, int parallelism,
                          int64_t node_id = 0);

  NumaAwareKVCacheManager(UnpackedCache&& cache, int64_t node_id,
                          int64_t local_control_port, int64_t max_blocks,
                          int64_t num_slots, double timeout_s,
                          bool unsafe_skip_buffer_lock, int parallelism);

  std::optional<nanobind::list> device_arrays_;
#endif

  void InitSubManagers(
      const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
      std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
      bool unsafe_skip_buffer_lock, int parallelism, int64_t node_id,
      int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
      double timeout_s);

  static constexpr uint64_t k48BitMask = 0xFFFFFFFFFFFFULL;
  std::atomic<uint64_t> global_seq_counter_{1};

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> sub_managers_;
  std::vector<std::pair<int, int>> global_shard_to_submanager_;
  std::vector<std::vector<int64_t>> submanager_to_global_shards_;
  size_t total_num_shards_ = 0;

  std::map<std::string, int> done_sending_counts_;
  std::map<std::string, int> done_recving_counts_;
  std::map<std::string, int> req_expected_counts_;
  std::set<std::string> failed_recving_set_;
  std::shared_ptr<MetricsCollector> metrics_collector_;
};

class KVCacheManager {
 public:
  KVCacheManager(const KVCacheManager&) = delete;
  KVCacheManager& operator=(const KVCacheManager&) = delete;
  KVCacheManager(KVCacheManager&&) = default;
  KVCacheManager& operator=(KVCacheManager&&) = default;

#ifndef WITHOUT_PYTHON
  // JAX sharded constructor E2E (cache-only by default)
  KVCacheManager(
      nanobind::list device_arrays,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1,
      int raiden_worker_port = 0,
      std::optional<std::string> raiden_controller_address = std::nullopt,
      std::optional<std::string> worker_id = std::nullopt, int64_t node_id = 0);

  // New transfer-enabled constructor (flat list of arrays, single shard per
  // layer)
  KVCacheManager(
      nanobind::list kv_caches, int64_t node_id, int64_t local_control_port,
      int64_t max_blocks, int64_t num_slots, double timeout_s,
      bool unsafe_skip_buffer_lock, int parallelism, int raiden_worker_port = 0,
      std::optional<std::string> raiden_controller_address = std::nullopt,
      std::optional<std::string> worker_id = std::nullopt);
#endif

  // FFI metadata constructor (cache-only by default)
  KVCacheManager(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
      int parallelism = 1, int raiden_worker_port = 0,
      std::optional<std::string> raiden_controller_address = std::nullopt,
      std::optional<std::string> worker_id = std::nullopt);

  // Test-only constructor for sub-manager mock injection
  explicit KVCacheManager(
      std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> sub_managers,
      int raiden_worker_port = 0,
      std::optional<std::string> raiden_controller_address = std::nullopt,
      std::optional<std::string> worker_id = std::nullopt);

  ~KVCacheManager();

  NumaAwareKVCacheManager* numa_manager() const { return numa_manager_.get(); }
  int GetRaidenWorkerPort() const;

#ifndef WITHOUT_PYTHON
  nanobind::list kv_caches() const { return numa_manager_->kv_caches(); }
#endif

  // Forwarding methods delegating to sub_managers_
  size_t num_layers() const { return numa_manager_->num_layers(); }
  size_t num_shards() const { return numa_manager_->num_shards(); }
  size_t slice_byte_size() const { return numa_manager_->slice_byte_size(); }
  size_t num_block_arrays() const { return numa_manager_->num_block_arrays(); }
  size_t block_bytes(size_t block_array_idx) const {
    return numa_manager_->block_bytes(block_array_idx);
  }

  std::optional<int> local_port() const { return numa_manager_->local_port(); }
  int local_control_port() const { return numa_manager_->local_control_port(); }
  int64_t node_id() const { return numa_manager_->node_id(); }

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) {
    return numa_manager_->GetHostPointer(layer_idx, shard_idx);
  }
  const uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) const {
    return numa_manager_->GetHostPointer(layer_idx, shard_idx);
  }
  size_t GetHostSize(size_t layer_idx, size_t shard_idx) {
    return numa_manager_->GetHostSize(layer_idx, shard_idx);
  }

  int64_t NotifyForRead(const std::string& req_id, uint64_t uuid,
                        const std::vector<int64_t>& block_ids) {
    return numa_manager_->NotifyForRead(req_id, uuid, block_ids);
  }

  std::vector<int32_t> RenewRemoteLeases(const std::string& remote_endpoint,
                                         const std::vector<uint64_t>& uuids) {
    return numa_manager_->RenewRemoteLeases(remote_endpoint, uuids);
  }

  std::vector<int32_t> CancelRemoteLeases(const std::string& remote_endpoint,
                                          const std::vector<uint64_t>& uuids) {
    return numa_manager_->CancelRemoteLeases(remote_endpoint, uuids);
  }

  void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::string& remote_endpoint,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt) {
    numa_manager_->StartRead(req_id, uuid, remote_endpoint, remote_block_ids,
                             local_block_ids, parallelism,
                             local_host_block_ids);
  }

  std::vector<RaidenTransferEndpoint> get_local_endpoints() const {
    return numa_manager_->get_local_endpoints();
  }

  std::vector<RaidenTransferEndpoint> get_local_data_endpoints() const {
    return numa_manager_->get_local_data_endpoints();
  }

  void SetSubmanagerShardsForTesting(
      const std::vector<std::vector<int64_t>>& assignment) {
    numa_manager_->SetSubmanagerShardsForTesting(assignment);
  }

  void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt) {
    numa_manager_->StartRead(req_id, uuid, remote_descriptors, remote_block_ids,
                             local_block_ids, parallelism,
                             local_host_block_ids);
  }

  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  CompleteReadRaw() {
    return numa_manager_->CompleteReadRaw();
  }

  // Unlocks host staging blocks (allocated+locked by D2hAutoAllocate) across
  // all sub-managers, making them reclaimable. `block_ids` are the chunk ids
  // returned by D2hAutoAllocate; sub-managers allocate in lockstep so the same
  // ids are unlocked on each.
  absl::Status UnlockBlocks(const std::vector<int>& block_ids) {
    return numa_manager_->UnlockBlocks(block_ids);
  }

  std::string DumpMetricsToString() const {
    return numa_manager_->DumpMetricsToString();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt) {
    return numa_manager_->H2d(src_offsets_major_dim, dst_offsets_major_dim,
                              copy_sizes_major_dim, slot_idx, layer_idx,
                              shard_idx);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt) {
    return numa_manager_->D2h(src_offsets_major_dim, dst_offsets_major_dim,
                              copy_sizes_major_dim, slot_idx, layer_idx,
                              shard_idx);
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(const std::vector<int64_t>& src_offsets_major_dim = {},
                  const std::vector<int64_t>& copy_sizes_major_dim = {}) {
    return numa_manager_->D2hAutoAllocate(src_offsets_major_dim,
                                          copy_sizes_major_dim);
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1) {
    return numa_manager_->H2hWrite(std::move(peer), src_block_ids,
                                   dst_block_ids, uuid, layer_idx);
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1) {
    return numa_manager_->H2hWrite(remote_descriptors, src_block_ids,
                                   dst_block_ids, uuid, layer_idx);
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string peer, const std::vector<int>& src_block_ids) {
    return numa_manager_->H2hRead(std::move(peer), src_block_ids);
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int>& src_block_ids) {
    return numa_manager_->H2hRead(remote_descriptors, src_block_ids);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int64_t>& src_host_offsets,
      const std::vector<int64_t>& dst_host_offsets,
      const std::vector<int64_t>& dst_device_offsets,
      const std::vector<int64_t>& copy_sizes) {
    return numa_manager_->H2dRead(remote_descriptors, src_host_offsets,
                                  dst_host_offsets, dst_device_offsets,
                                  copy_sizes);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2hReadExplicit(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids) {
    return numa_manager_->H2hReadExplicit(remote_descriptors, src_block_ids,
                                          dst_block_ids);
  }

 private:
  void StartGrpcServer(
      int raiden_worker_port,
      std::optional<std::string> raiden_controller_address = std::nullopt,
      std::optional<std::string> worker_id = std::nullopt);

  std::unique_ptr<NumaAwareKVCacheManager> numa_manager_;
  std::unique_ptr<tpu_raiden::controller::WorkerServiceServer>
      private_grpc_server_;
};

}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_
