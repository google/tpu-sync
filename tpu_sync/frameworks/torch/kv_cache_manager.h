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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_KV_CACHE_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_KV_CACHE_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "torch_tpu/csrc/eager/tensor_to_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {
namespace controller {
class WorkerServiceServer;
}
namespace kv_cache {
class KVCacheListener;
}  // namespace kv_cache
}  // namespace tpu_raiden

namespace tpu_raiden {
namespace torch {

class TorchKVCacheManager : public KVCacheManagerWithTransfer {
 public:
  // PyTorch sharded constructor E2E (cache-only by default)
  TorchKVCacheManager(
      const std::vector<std::vector<at::Tensor>>& device_tensors,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1,
      int64_t node_id = 0);

  // New transfer-enabled constructor (flat list of tensors, single shard per
  // layer)
  TorchKVCacheManager(const std::vector<at::Tensor>& kv_caches, int64_t node_id,
                      int64_t local_control_port, int64_t max_blocks,
                      int64_t num_slots, double timeout_s,
                      bool unsafe_skip_buffer_lock, int parallelism = 4,
                      std::optional<int> listener_port = std::nullopt);

  // Metadata-only constructor for host-memory Stage 1 resharding tests.
  TorchKVCacheManager(size_t num_layers, size_t num_shards,
                      size_t slice_byte_size, int64_t node_id,
                      std::optional<int> local_port,
                      std::optional<int> host_blocks_to_allocate,
                      int parallelism);

  ~TorchKVCacheManager() override;

  // Transfer-specific methods merged from TransferEngine
  std::vector<int64_t> RegisterKvCache(
      const std::vector<at::Tensor>& kv_caches);

  void RegisterHostBuffers(int64_t node_id);

  const std::vector<at::Tensor>& kv_caches() const { return kv_caches_; }

  std::optional<int> listener_port() const;
  bool is_listener_active() const;

  std::string transfer_address() const;
  std::string listener_address() const;

  absl::Status PushRegisteredPlan(uint64_t uuid, const std::string& peer,
                                  const std::vector<int>& src_block_ids,
                                  const std::vector<int>& dst_block_ids,
                                  int layer_idx = -1, int parallelism = 1);

  absl::StatusOr<std::string> ReadBlockBytes(size_t layer_idx, int block_id,
                                             size_t shard_idx = 0);

  absl::Status WriteBlockBytes(size_t layer_idx, int block_id,
                               absl::string_view payload, size_t shard_idx = 0);

 private:
  // Buffers unpacked from a 2D tensor list, together with the owning
  // DeviceBufferRefs that must outlive their use (see UnpackTorchTensor).
  struct UnpackedLayers {
    std::vector<std::vector<raiden::RaidenBufferHandle>> buffers;
    std::vector<torch_tpu::DeviceBufferRef> refs;
    xla::PjRtClient* client = nullptr;
    std::vector<int64_t> logical_dimensions;
    size_t logical_slice_byte_size = 0;
    size_t logical_physical_size = 0;
    bool has_logical_metadata = false;
  };
  static UnpackedLayers UnpackLayers(
      const std::vector<std::vector<at::Tensor>>& device_tensors,
      bool unsafe_skip_buffer_lock = false);

  // Delegated-to constructor for BOTH public ctors. Moves the keep-alive refs
  // into buffer_refs_ so the materialized device buffers survive for this
  // manager's lifetime. `kv_caches` is retained for the disagg path's
  // kv_caches() accessor (empty for the offload path).
  TorchKVCacheManager(UnpackedLayers unpacked, std::optional<int> local_port,
                      std::optional<int> host_blocks_to_allocate,
                      bool unsafe_skip_buffer_lock, int parallelism,
                      int64_t node_id, int64_t local_control_port,
                      int64_t max_blocks, int64_t num_slots, double timeout_s,
                      std::vector<at::Tensor> kv_caches);

  std::vector<at::Tensor> kv_caches_;
  // Keep-alives for the materialized device buffers backing the manager.
  std::vector<torch_tpu::DeviceBufferRef> buffer_refs_;
  std::unique_ptr<tpu_raiden::kv_cache::KVCacheListener> listener_;
};

class KVCacheManager {
 public:
  KVCacheManager(
      const std::vector<std::vector<at::Tensor>>& device_tensors,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1,
      int raiden_worker_port = 0,
      std::optional<std::string> raiden_controller_address = std::nullopt,
      std::optional<std::string> worker_id = std::nullopt, int64_t node_id = 0);

  KVCacheManager(
      const std::vector<at::Tensor>& kv_caches, int64_t node_id,
      int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
      double timeout_s = 120.0, bool unsafe_skip_buffer_lock = true,
      int parallelism = 4, std::optional<int> listener_port = std::nullopt,
      int raiden_worker_port = 0,
      std::optional<std::string> raiden_controller_address = std::nullopt,
      std::optional<std::string> worker_id = std::nullopt);

  KVCacheManager(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      int64_t node_id, std::optional<int> local_port,
      std::optional<int> host_blocks_to_allocate, int parallelism = 1,
      int raiden_worker_port = 0,
      std::optional<std::string> raiden_controller_address = std::nullopt,
      std::optional<std::string> worker_id = std::nullopt);

  ~KVCacheManager();

  int GetRaidenWorkerPort() const;

  TorchKVCacheManager* torch_manager() const { return torch_manager_.get(); }

  std::vector<int64_t> RegisterKvCache(
      const std::vector<at::Tensor>& kv_caches) {
    return torch_manager_->RegisterKvCache(kv_caches);
  }

  void RegisterHostBuffers(int64_t node_id) {
    torch_manager_->RegisterHostBuffers(node_id);
  }

  const std::vector<at::Tensor>& kv_caches() const {
    return torch_manager_->kv_caches();
  }

  std::optional<int> listener_port() const {
    return torch_manager_->listener_port();
  }

  bool is_listener_active() const {
    return torch_manager_->is_listener_active();
  }

  std::string transfer_address() const {
    return torch_manager_->transfer_address();
  }

  std::string listener_address() const {
    return torch_manager_->listener_address();
  }

  absl::Status PushRegisteredPlan(uint64_t uuid, const std::string& peer,
                                  const std::vector<int>& src_block_ids,
                                  const std::vector<int>& dst_block_ids,
                                  int layer_idx = -1, int parallelism = 1) {
    return torch_manager_->PushRegisteredPlan(
        uuid, peer, src_block_ids, dst_block_ids, layer_idx, parallelism);
  }

  absl::StatusOr<std::string> ReadBlockBytes(size_t layer_idx, int block_id,
                                             size_t shard_idx = 0) {
    return torch_manager_->ReadBlockBytes(layer_idx, block_id, shard_idx);
  }

  absl::Status WriteBlockBytes(size_t layer_idx, int block_id,
                               absl::string_view payload,
                               size_t shard_idx = 0) {
    return torch_manager_->WriteBlockBytes(layer_idx, block_id, payload,
                                           shard_idx);
  }

  int64_t node_id() const { return torch_manager_->node_id(); }
  int local_control_port() const {
    return torch_manager_->local_control_port();
  }
  std::optional<int> local_port() const { return torch_manager_->local_port(); }
  size_t num_layers() const { return torch_manager_->num_layers(); }
  size_t num_shards() const { return torch_manager_->num_shards(); }
  size_t slice_byte_size() const { return torch_manager_->slice_byte_size(); }
  size_t bytes_per_block() const { return torch_manager_->bytes_per_block(); }
  size_t num_block_arrays() const { return torch_manager_->num_block_arrays(); }
  size_t block_bytes(size_t block_array_idx) const {
    return torch_manager_->block_bytes(block_array_idx);
  }

  std::vector<RaidenTransferEndpoint> get_local_endpoints() const {
    return torch_manager_->get_local_endpoints();
  }

  absl::Status RegisterActivePlan(
      uint64_t uuid, const ::tpu_sync::rpc::StartTransferRequest& request,
      bool is_sender) {
    return torch_manager_->RegisterActivePlan(uuid, request, is_sender);
  }

  absl::Status UnregisterActivePlan(uint64_t uuid) {
    return torch_manager_->UnregisterActivePlan(uuid);
  }

  absl::Status RegisterRecv(uint64_t uuid, const std::string& req_id,
                            int64_t expected_block_count) {
    return torch_manager_->RegisterRecv(uuid, req_id, expected_block_count);
  }

  int64_t NotifyForRead(const std::string& req_id, uint64_t uuid,
                        const std::vector<int64_t>& block_ids) {
    return torch_manager_->NotifyForRead(req_id, uuid, block_ids);
  }

  void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::string& remote_endpoint,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt) {
    torch_manager_->StartRead(req_id, uuid, remote_endpoint, remote_block_ids,
                              local_block_ids, parallelism,
                              local_host_block_ids);
  }

  void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<std::string>& remote_endpoints,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt) {
    torch_manager_->StartRead(req_id, uuid, remote_endpoints, remote_block_ids,
                              local_block_ids, parallelism,
                              local_host_block_ids);
  }

  void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt) {
    torch_manager_->StartRead(req_id, uuid, remote_descriptors,
                              remote_block_ids, local_block_ids, parallelism,
                              local_host_block_ids);
  }

  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  CompleteReadRaw() {
    return torch_manager_->CompleteReadRaw();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt) {
    return torch_manager_->H2d(src_offsets_major_dim, dst_offsets_major_dim,
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
    return torch_manager_->D2h(src_offsets_major_dim, dst_offsets_major_dim,
                               copy_sizes_major_dim, slot_idx, layer_idx,
                               shard_idx);
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(const std::vector<int64_t>& src_offsets_major_dim = {},
                  const std::vector<int64_t>& copy_sizes_major_dim = {}) {
    return torch_manager_->D2hAutoAllocate(src_offsets_major_dim,
                                           copy_sizes_major_dim);
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1) {
    return torch_manager_->H2hWrite(peers, src_block_ids, dst_block_ids, uuid,
                                    layer_idx);
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1) {
    return torch_manager_->H2hWrite(std::move(peer), src_block_ids,
                                    dst_block_ids, uuid, layer_idx);
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids) {
    return torch_manager_->H2hRead(peers, src_block_ids);
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string peer, const std::vector<int>& src_block_ids) {
    return torch_manager_->H2hRead(std::move(peer), src_block_ids);
  }

  absl::Status RegisterPools(std::vector<kv_cache::PoolSpec> pools) {
    return torch_manager_->RegisterPools(std::move(pools));
  }

  absl::StatusOr<kv_cache::PoolBlockRef> GetPoolBlockRef(
      size_t pool_idx, size_t shard_idx, int64_t block_id) const {
    return torch_manager_->GetPoolBlockRef(pool_idx, shard_idx, block_id);
  }

  size_t num_pools() const { return torch_manager_->num_pools(); }

  bool has_explicit_pools() const {
    return torch_manager_->has_explicit_pools();
  }

  const kv_cache::PoolSpec* pool(size_t pool_idx) const {
    return torch_manager_->pool(pool_idx);
  }

  std::vector<size_t> PoolIndicesWithTag(absl::string_view tag) const {
    return torch_manager_->PoolIndicesWithTag(tag);
  }

  absl::StatusOr<uintptr_t> GetBlockHostPointerValue(size_t layer_idx,
                                                     size_t shard_idx,
                                                     int block_id) {
    return torch_manager_->GetBlockHostPointerValue(layer_idx, shard_idx,
                                                    block_id);
  }

  int64_t LayerBlockByteSize(size_t layer_idx) const {
    return torch_manager_->LayerBlockByteSize(layer_idx);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hPoolBlocks(
      size_t pool_idx, absl::Span<const int64_t> block_ids,
      std::optional<size_t> shard_idx = std::nullopt) {
    return torch_manager_->D2hPoolBlocks(pool_idx, block_ids, shard_idx);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2dPoolBlocks(
      size_t pool_idx, absl::Span<const int64_t> block_ids,
      std::optional<size_t> shard_idx = std::nullopt) {
    return torch_manager_->H2dPoolBlocks(pool_idx, block_ids, shard_idx);
  }

 private:
  void StartGrpcServer(
      int raiden_worker_port,
      std::optional<std::string> raiden_controller_address = std::nullopt,
      std::optional<std::string> worker_id = std::nullopt);

  std::unique_ptr<TorchKVCacheManager> torch_manager_;
  std::unique_ptr<tpu_raiden::controller::WorkerServiceServer>
      private_grpc_server_;
};

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_TORCH_KV_CACHE_MANAGER_H_
