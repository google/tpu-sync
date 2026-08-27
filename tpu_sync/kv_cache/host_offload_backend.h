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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/core/controller/worker_registry.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/kv_cache/kv_cache_store_client.h"
#include "tpu_sync/kv_cache/kv_cache_store_server.h"
#include "tpu_sync/kv_cache/lru_cache.h"
#include "tpu_sync/kv_cache/raiden_id.h"
#include "tpu_sync/proto/kv_cache_store_service.pb.h"

namespace tpu_raiden {

namespace controller {
class RaidenController;
}  // namespace controller

namespace kv_cache {

class HostOffloadBackend : public KVCacheStoreBackend {
 public:
  static absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> Create(
      const BackendConfig& config,
      controller::RaidenController* absl_nonnull controller);

  ~HostOffloadBackend() override;

  std::string name() const override { return "HostOffloadBackend"; }

  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override;

  std::pair<bool, BlockSliceList> Insert(
      absl::Span<const std::string> block_hashes,
      absl::Span<const RaidenBlockId> slices, bool on_host) override;

  bool InsertAndLock(absl::Span<const std::string> block_hashes,
                     absl::Span<const RaidenBlockId> slices,
                     bool on_host) override;

  size_t ReleaseAndDelete(absl::Span<const std::string> block_hashes) override;

  void Delete(absl::Span<const std::string> block_hashes,
              absl::Span<const RaidenBlockId> slices) override;

  bool Pin(absl::Span<const std::string> block_hashes) override;

  void Release(absl::Span<const std::string> block_hashes) override;

  int GetPinCount(const std::string& hash) const override;

  size_t GetCapacity() const override;

  size_t GetSize() const override;

  size_t GetAvailableSpace() const override;

  absl::StatusOr<size_t> RecoverFromLocalManifest() override;

  bool ValidateAndPinHostBlocks(absl::Span<const int> host_block_ids) override;

  std::vector<std::string> GetEvictableKeys(size_t count) override;

  std::vector<int> Evict(const std::vector<std::string>& block_hashes) override;

  std::vector<std::string> GetEvictCandidateKeys() const override;

  // --- Global Memory Pooling & RPC Methods ---
  // The peer-facing server this backend hosts, or nullptr until StartServer()
  // succeeds (construction no longer pre-creates it -- with no registry, no
  // server is started anywhere). Once started, the owning KVCacheStore adopts THIS
  // server instead of standing up a second one -- a node must never serve
  // peers from two ports. The override was lost when GlobalMemoryPoolingBackend
  // was consolidated into this class.
  KVCacheStoreServer* store_server() const override;

  // Starts this backend's own server bound to `server_address`, which must be
  // non-empty and non-wildcard. Not called from
  // Create() -- starting + publishing a server is exclusively
  // KVCacheStore::EnsureStoreServerAndRegister's job.
  absl::Status StartServer(absl::string_view server_address);

  // Loads KV cache blocks from local host DRAM (if `remote_id` is empty or
  // matches this backend's id) or from the specified peer. See KVCacheStoreBackend
  // for the argument contract.
  tsl::Future<> Load(const RaidenId& remote_id,
                     absl::Span<const std::string> block_hashes,
                     absl::Span<const int32_t> device_block_ids,
                     absl::Span<const RaidenBlockId> slices = {}) override;

  // --- Remote write (WriteRemote), the transport under KVCacheStore::Save
  // with a destination; see KVCacheStoreBackend for why each of these
  // exists rather than reusing Lookup/Insert/Delete.
  std::vector<std::string> AlreadyPresentHostResident(
      absl::Span<const std::string> block_hashes) const override;

  bool InsertAllOrNothing(absl::Span<const std::string> block_hashes,
                          absl::Span<const RaidenBlockId> slices) override;

  void RollbackInsert(absl::Span<const std::string> block_hashes,
                      absl::Span<const int32_t> host_block_ids) override;

  tsl::Future<> RegisterBlocksAsync(
      absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> host_block_ids) override;

  absl::Status RegisterBlocksSync(
      absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> host_block_ids) override;

  // --- Remote write, source side ------------------------------------------
  //
  // The two halves of the source's role in the store's Save(dst) flow. The
  // synchronous half offers blocks and comes back with the destination's
  // decision; the polling half asks
  // what became of an accepted offer. Neither owns the retry cadence or the
  // pins -- KVCacheStore does, because it is what holds the blocks.

  // What the destination decided, before any bytes have moved.
  struct RemoteWriteAck {
    // Zero when the destination answered without starting a transfer, which
    // is the case for both existence outcomes.
    uint64_t operation_id = 0;
    // Every offered hash was already there. A SUCCESS: content-addressed
    // hashes mean the peer having them is the post-condition we wanted.
    bool all_exist = false;
    // Non-empty only when the destination held a strict subset. A FAILURE --
    // the destination does not do partial writes -- and this is the list the
    // caller needs to decide what, if anything, to re-offer.
    std::vector<std::string> existing_hashes;
    absl::Duration granted_deadline;
  };

  // `requested_deadline` is how long the destination may hold its landing
  // blocks; it grants at most this and at most its own cap, and reports what
  // it actually armed. Blocking, on the caller's thread -- this is a control
  // RPC that returns as soon as the destination has decided, not when the
  // bytes have arrived.
  absl::StatusOr<RemoteWriteAck> BeginWriteRemote(
      const RaidenId& dst_raiden_id, absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> src_host_block_ids,
      absl::Duration requested_deadline);

  // Mirrors PollWriteRemoteResponse::State without depending on it, so the
  // wire enum stays an implementation detail of this file.
  enum class RemoteWriteState {
    kPending,
    kCommitted,
    kAllExist,
    kPartialExist,
    kFailed,
    // Aged out, or the destination restarted. Indistinguishable from "never
    // happened", so callers must treat it as failure.
    kUnknown,
    // The destination holds the bytes but could not publish them, so no peer
    // can find them. Not a success and not a plain failure: the transfer
    // worked. What to do about it is the source's call.
    kStoredUnregistered,
  };

  struct RemoteWriteStatus {
    RemoteWriteState state = RemoteWriteState::kPending;
    std::vector<std::string> existing_hashes;
    std::vector<std::string> unregistered_hashes;
  };

  absl::StatusOr<RemoteWriteStatus> PollWriteRemote(
      const RaidenId& dst_raiden_id, uint64_t operation_id);

  // Drops the cached client for `remote_id`, so the next call re-resolves the
  // peer through the registry instead of redialling the address it had.
  // Without this a peer that restarts on a new port stays undialable for the
  // life of THIS process, even after the registry has healed: store_clients_
  // is a cache with no invalidation of its own.
  void InvalidateStoreClient(const RaidenId& remote_id);

  // Composes the KVTransferSpec from the workers currently registered with
  // the RaidenController and registers it with the global registry
  // (first-wins, idempotent; see global_registry.proto). FailedPrecondition
  // without a controller or without a global registry.
  absl::Status RegisterKVTransferSpecFromWorkers();

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

  // Pulls blocks held by the peer `remote_id` into `device_block_ids`, in two
  // hops: a Fetch RPC that lands the bytes in host blocks allocated here, then
  // a DRAM->HBM copy chained off it.
  tsl::Future<> LoadRemoteBlocks(const RaidenId& remote_id,
                                 absl::Span<const std::string> block_hashes,
                                 absl::Span<const int32_t> device_block_ids);

  // Copies blocks already resident in this node's host DRAM into
  // `device_block_ids`, as a single TransferBuffers.
  // `slices` is the caller's pre-resolved index entries.
  tsl::Future<> LoadLocalHostBlocks(absl::Span<const std::string> block_hashes,
                                    absl::Span<const int32_t> device_block_ids,
                                    absl::Span<const RaidenBlockId> slices);

  void SetMetadataEntry(absl::string_view hash, const RaidenBlockId& block)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void ClearMetadataEntry(const RaidenBlockId& block)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

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
  std::unique_ptr<KVCacheStoreServer> server_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<RaidenId, std::shared_ptr<KVCacheStoreClient>,
                      RaidenIdHash>
      store_clients_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_HOST_OFFLOAD_BACKEND_H_
