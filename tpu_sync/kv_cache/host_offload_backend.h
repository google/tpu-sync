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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_HOST_OFFLOAD_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_HOST_OFFLOAD_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/worker_registry.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/kv_cache/kv_cache_store_client.h"
#include "tpu_sync/kv_cache/lru_cache.h"
#include "tpu_sync/proto/kv_cache_store_service.pb.h"

namespace tpu_raiden {

namespace controller {
class RaidenController;
}  // namespace controller

namespace kv_cache {

class BlockTracker;

class HostOffloadBackend : public KVCacheStoreBackend {
 public:
  static absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> Create(
      const BackendConfig& config,
      controller::RaidenController* absl_nonnull controller);

  ~HostOffloadBackend() override;

  void Shutdown() override;

  std::string name() const override { return "HostOffloadBackend"; }

  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override;

  bool Insert(absl::Span<const std::string> block_hashes,
              absl::Span<const RaidenBlockId> slices,
              InsertOption option = InsertOption::kBestEffortUpdate) override;

  size_t Delete(absl::Span<const std::string> block_hashes,
                const DeleteOptions& options = {}) override;

  bool Pin(absl::Span<const std::string> block_hashes) override;

  void Release(absl::Span<const std::string> block_hashes) override;

  int GetPinCount(const std::string& hash) const override;

  size_t GetCapacity() const override;

  size_t GetSize() const override;

  size_t GetAvailableSpace() const override;

  absl::StatusOr<size_t> RecoverFromLocalManifest() override;

  std::vector<std::string> GetEvictableKeys(size_t count) override;

  // --- Global Memory Pooling & RPC Methods ---
  // Loads KV cache blocks from local host DRAM (if `remote_id` is empty or
  // matches this backend's id) or from the specified peer. See KVCacheStoreBackend
  // for the argument contract.
  tsl::Future<> Load(const RaidenId& remote_id,
                     absl::Span<const std::string> block_hashes,
                     absl::Span<const int32_t> device_block_ids,
                     absl::Span<const RaidenBlockId> slices,
                     BlockTracker* absl_nonnull load_tracker) override;

  tsl::Future<> Save(absl::Span<const std::string> block_hashes,
                     absl::Span<const int64_t> src_device_block_ids,
                     absl::Span<const int32_t> dst_host_block_ids,
                     BlockTracker* absl_nonnull save_tracker) override;

  // --- WriteRemote & registry overrides ------------------------------------
  tsl::Future<> RegisterBlocksAsync(
      absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> host_block_ids) override;

  tsl::Future<> UnregisterBlocksAsync(
      absl::Span<const std::string> block_hashes) override;

  // --- Outbound remote write (SaveRemote) ----------------------------------

  // Offers `block_hashes` to `dst_raiden_id` and blocks until the ack (not
  // the bytes). `requested_deadline` is how long the destination may hold
  // its landing blocks; `hold_window` is the call's deadline. The tracker's
  // transfer status is updated when the operation completes or settles.
  absl::StatusOr<RemoteWriteAck> BeginWriteRemote(
      const RaidenId& dst_raiden_id, absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> src_host_block_ids,
      absl::Duration requested_deadline, absl::Duration hold_window,
      BlockTracker* absl_nonnull save_tracker) override;

  // Composes the KVTransferSpec from the workers currently registered with
  // the RaidenController and registers it with the global registry
  // (first-wins, idempotent; see global_registry.proto). FailedPrecondition
  // without a controller or without a global registry.
  absl::Status RegisterKVTransferSpecFromWorkers() override;

 private:
  friend class HostOffloadBackendTest;

  explicit HostOffloadBackend(
      size_t capacity, std::optional<KVCacheMetadata> metadata = std::nullopt,
      RaidenId raiden_id = {},
      controller::RaidenController* raiden_controller = nullptr,
      std::shared_ptr<global_registry::GlobalRegistryClient> registry_client =
          nullptr,
      std::string kv_pool_group = "");

  // Derives the deployment's KVTransferSpec from the given worker
  // registrations: every worker must have reported the same KV block geometry
  // (block_array_bytes, num_kv_shards), and the node ids must form the dense
  // range [0, num_workers) -- peers pair transfers on node_id, so a gap or
  // duplicate would break the pairing. Returns the composed config, or the
  // violation as an error.
  static absl::StatusOr<KVTransferSpecConfig> ComposeKVTransferSpec(
      absl::Span<const core::controller::WorkerRegistration> workers);

  // Registers the deployment's KVTransferSpec with the global registry, or
  // validates it against the already-registered one (first-wins, idempotent;
  // see global_registry.proto). Requires a global registry.
  absl::Status RegisterKVTransferSpec(const KVTransferSpecConfig& spec_config);

  absl::StatusOr<std::shared_ptr<KVCacheStoreClient>> GetKVCacheStoreClient(
      const RaidenId& remote_id);

  // Drops the cached client for `remote_id`; the next call re-resolves the
  // peer through the registry. store_clients_ never invalidates itself, so
  // without this a peer that restarts on a new port would stay unreachable.
  void InvalidateStoreClient(const RaidenId& remote_id);

  // Asks the destination what became of an accepted offer; recovery for a
  // source that lost its stream. `wait_ms > 0` asks it to hold the answer
  // until the operation is terminal. Returns the raw wire response.
  tsl::Future<proto::PollWriteRemoteResponse> PollWriteRemoteAsync(
      const RaidenId& dst_raiden_id, uint64_t operation_id,
      int64_t wait_ms = 0);

  // Pulls blocks held by the peer `remote_id` into `device_block_ids`, in two
  // hops: a Fetch RPC that lands the bytes in host blocks allocated here, then
  // a DRAM->HBM copy chained off it.
  tsl::Future<> LoadRemoteBlocks(const RaidenId& remote_id,
                                 absl::Span<const std::string> block_hashes,
                                 absl::Span<const int32_t> device_block_ids,
                                 BlockTracker* absl_nonnull load_tracker);

  // Copies blocks already resident in this node's host DRAM into
  // `device_block_ids`, as a single TransferBuffers.
  // `slices` is the caller's pre-resolved index entries.
  tsl::Future<> LoadLocalHostBlocks(absl::Span<const std::string> block_hashes,
                                    absl::Span<const int32_t> device_block_ids,
                                    absl::Span<const RaidenBlockId> slices,
                                    BlockTracker* absl_nonnull load_tracker);

  void SetMetadataEntry(absl::string_view hash, const RaidenBlockId& block)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void ClearMetadataEntry(const RaidenBlockId& block)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool InsertBestEffortUpdate(absl::Span<const std::string> block_hashes,
                              absl::Span<const RaidenBlockId> slices);
  bool InsertAndPin(absl::Span<const std::string> block_hashes,
                    absl::Span<const RaidenBlockId> slices);
  bool InsertAdmitStrict(absl::Span<const std::string> block_hashes,
                         absl::Span<const RaidenBlockId> slices);

  // INVARIANT: no thread holds `mutex_` across a registry RPC. Every call site
  // unlocks first, and nothing enforces that but their shape. The synchronous
  // Register and Unregister are Await() on a promise a gRPC callback thread
  // sets, so holding this across one waits for a path that may need it.
  mutable absl::Mutex mutex_;
  LRUCache<std::string, RaidenBlockId> lru_cache_ ABSL_GUARDED_BY(mutex_);
  std::optional<KVCacheMetadata> metadata_ ABSL_GUARDED_BY(mutex_);
  uint64_t next_metadata_seq_ ABSL_GUARDED_BY(mutex_) = 0;
  RaidenId raiden_id_ ABSL_GUARDED_BY(mutex_);
  // Set at construction, immutable afterwards; see
  // BackendConfig.kv_pool_group.
  const std::string kv_pool_group_;
  controller::RaidenController* raiden_controller_ = nullptr;

  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client_
      ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<RaidenId, std::shared_ptr<KVCacheStoreClient>,
                      RaidenIdHash>
      store_clients_ ABSL_GUARDED_BY(mutex_);

  struct Lifetime {
    absl::Mutex mu;
    bool is_alive ABSL_GUARDED_BY(mu) = true;
  };
  std::shared_ptr<Lifetime> lifetime_ = std::make_shared<Lifetime>();
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_HOST_OFFLOAD_BACKEND_H_
