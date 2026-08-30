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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"

namespace tpu_raiden {

namespace kv_cache {

// Forward-declared rather than included: kv_cache_store_server.h includes this
// header, so including it back would be circular.
class KVCacheStoreServer;

enum class BlockStatus {
  INIT,
  REMOTE,
  HBM,
  HOST,
  HOST_AND_HBM,
};

struct RaidenBlockId {
  RaidenId raiden_id;
  // When status is REMOTE, it represents the remote host block ID.
  // When status is HOST or HOST_AND_HBM, it represents the local host block ID.
  int host_block_id = -1;
  int device_block_id = -1;
  BlockStatus status = BlockStatus::INIT;

  RaidenBlockId() = default;
  /* implicit */ RaidenBlockId(RaidenId id, int host_id = -1,
                               BlockStatus stat = BlockStatus::INIT)
      : raiden_id(std::move(id)), host_block_id(host_id), status(stat) {}

  RaidenBlockId(RaidenId id, int host_block_id, int device_block_id,
                BlockStatus stat = BlockStatus::INIT)
      : raiden_id(std::move(id)),
        host_block_id(host_block_id),
        device_block_id(device_block_id),
        status(stat) {}

  bool operator==(const RaidenBlockId& other) const {
    return raiden_id == other.raiden_id &&
           host_block_id == other.host_block_id &&
           device_block_id == other.device_block_id && status == other.status;
  }
  bool operator!=(const RaidenBlockId& other) const {
    return !(*this == other);
  }
};

using BlockSliceList = std::vector<std::pair<std::string, RaidenBlockId>>;

// Options controlling lookup behavior across storage backends.
struct LookupOptions {
  // Controls whether the lookup is allowed to query the global registry.
  // Default is true HERE, serving the internal backend call sites; the
  // application-level KVCacheStore::Lookup(hashes, enable_global, pin_found)
  // defaults it to false instead, because the registry query is a blocking
  // RPC a caller should opt into.
  bool enable_global = true;

  // Controls whether a block cached remotely may sit between two blocks cached
  // locally in the answer. When true, the local index and the global registry
  // are consulted for every hash and the answer runs to the first hash neither
  // can resolve. When false, the answer stops at the first LOCAL miss and only
  // the hashes after it are looked for remotely, so a locally cached block can
  // never follow a remote one.
  // Default is true.
  bool enable_interleaved_lookup = true;

  // Controls whether matched local blocks in store/backends are automatically
  // pinned in memory to protect against LRU eviction.
  // Default is false.
  bool pin_found = false;
};

// Abstract interface for KV cache index storage backends.
// Implementations must be thread-safe.
class KVCacheStoreBackend {
 public:
  virtual ~KVCacheStoreBackend() = default;

  // Name identifying the backend type (e.g., "LruCacheBackend",
  // "GlobalMemoryPoolingBackend").
  virtual std::string name() const = 0;

  // Resolves cached block hashes in sequence.
  // Returns a list of matched (block_hash, RaidenBlockId) pairs up to the first
  // miss.
  virtual absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) = 0;

  // Asynchronously loads KV cache blocks to device (HBM), either from local
  // host DRAM or from the peer named by `remote_id`.
  //
  // `device_block_ids` is the destination and must name one device block per
  // hash.
  //
  // If `slices` is non-empty, the caller's pre-looked up RaidenBlockIds are
  // used directly. Note that blocks in `slices` must be already pinned
  // externally (when Load from local host), and remote loads will re-resolve
  // hashes at the peer, ignoring `slices`.
  virtual tsl::Future<> Load(const RaidenId& remote_id,
                             absl::Span<const std::string> block_hashes,
                             absl::Span<const int32_t> device_block_ids,
                             absl::Span<const RaidenBlockId> slices = {}) = 0;

  // Inserts key-block mappings into the backend. Backend-internal: the store
  // facade does not expose this form -- KVCacheStore::Insert() maps to
  // InsertAndLock below -- and its remaining callers are internal (the
  // store's pollers rewrite a completed entry through it).
  // Returns:
  //   - bool: true if all hashes were newly inserted (none already existed)
  //   - BlockSliceList: entries evicted from this backend during insertion
  virtual std::pair<bool, BlockSliceList> Insert(
      absl::Span<const std::string> block_hashes,
      absl::Span<const RaidenBlockId> slices, bool on_host) = 0;

  // Pins existing hashes and inserts & locks new hashes if space permits.
  // Performs complete rollback on failure. Returns true on full success.
  // This is the operation behind the store's public Insert().
  virtual bool InsertAndLock(absl::Span<const std::string> block_hashes,
                             absl::Span<const RaidenBlockId> slices,
                             bool on_host) = 0;

  // Reverts an InsertAndLock operation: unpins hashes, erases non-host blocks
  // whose pin count drops to 0, and restores candidate evictions.
  // Returns the number of deleted blocks.
  // Backend-internal: the store facade no longer exposes it; its remaining
  // store-level use is rolling back a multi-backend Insert when a later
  // tier refuses.
  virtual size_t ReleaseAndDelete(
      absl::Span<const std::string> block_hashes) = 0;

  // Explicitly deletes cached block entries from the backend, skipping
  // pinned ones. Backend-internal bookkeeping: the store's public API has
  // no caller-driven removal -- reclamation is the store's own eviction.
  virtual void Delete(absl::Span<const std::string> block_hashes,
                      absl::Span<const RaidenBlockId> slices) = 0;

  // Pins block hashes to protect them from LRU eviction.
  // Returns true if all hashes exist and were successfully pinned.
  // Backend-internal: application code acquires pins through the store's
  // Lookup()/Insert(); this is for the store's own extra pins (a peer's
  // read lease, a remote save's internal pin, the evict sweep).
  virtual bool Pin(absl::Span<const std::string> block_hashes) = 0;

  // Releases (unpins) previously pinned block hashes. Absent hashes are not
  // an error; unpinning them is a no-op.
  virtual void Release(absl::Span<const std::string> block_hashes) = 0;

  // Returns current pin count for a single hash (0 if absent).
  virtual int GetPinCount(const std::string& hash) const = 0;

  // Maximum capacity of entries supported by this backend.
  virtual size_t GetCapacity() const = 0;

  // Current number of active entries stored in this backend.
  virtual size_t GetSize() const = 0;

  // Remaining evictable/unpinned space in this backend.
  virtual size_t GetAvailableSpace() const = 0;

  // Reconstructs index entries from a local persistent manifest/metadata table.
  // Default implementation returns 0 (no recovery).
  virtual absl::StatusOr<size_t> RecoverFromLocalManifest() { return 0; }

  // Validates host DRAM block IDs and pins them in local memory.
  // Default implementation returns false.
  virtual bool ValidateAndPinHostBlocks(absl::Span<const int> host_block_ids) {
    return false;
  }

  // Retrieves evictable keys from the backend.
  virtual std::vector<std::string> GetEvictableKeys(size_t count) { return {}; }

  // Evicts keys from the backend and returns deallocated host block IDs.
  virtual std::vector<int> Evict(const std::vector<std::string>& block_hashes) {
    Delete(block_hashes, {});
    return {};
  }

  // Retrieves eviction candidate keys from the backend (for
  // testing/diagnostics).
  virtual std::vector<std::string> GetEvictCandidateKeys() const { return {}; }

  // --- Remote write (WriteRemote) ------------------------------------------
  //
  // These four exist because a WriteRemote handler holds only this interface,
  // and because each needs to happen under ONE acquisition of the backend's
  // lock. Composing them at the service out of Lookup/Insert/Delete would
  // check and then act with the lock dropped in between, which is exactly the
  // window a concurrent writer needs.
  //
  // The defaults are all refusals, so a backend that does not implement them
  // makes WriteRemote fail rather than half-succeed.

  // Which of `block_hashes` this backend already holds in host DRAM.
  //
  // Not Lookup(): Lookup stops at the first miss (it answers a prefix
  // question) so it cannot report a scattered subset, and it promotes LRU
  // order as a side effect. This has to see eviction CANDIDATES too -- a
  // candidate still has its host block, so treating it as absent would let a
  // duplicate through.
  virtual std::vector<std::string> AlreadyPresentHostResident(
      absl::Span<const std::string> block_hashes) const {
    return {};
  }

  // Inserts every hash or none, returning false if any part fails.
  //
  // Not Insert(): Insert rebinds a hash that is already present, which for a
  // remote write would orphan the old host block, and it reports partial
  // success, which would let the source free blocks the destination does not
  // have.
  virtual bool InsertAllOrNothing(absl::Span<const std::string> block_hashes,
                                  absl::Span<const RaidenBlockId> slices) {
    return false;
  }

  // Publishes `block_hashes` to the global registry asynchronously. Insert()
  // does not publish; a remote write awaits this before reporting COMMITTED.
  virtual tsl::Future<> RegisterBlocksAsync(
      absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> host_block_ids) {
    return tsl::Future<>(absl::UnimplementedError(
        "Backend does not implement RegisterBlocksAsync."));
  }

  // Withdraws `block_hashes` from the global registry under this store's own
  // id, asynchronously. The mirror of RegisterBlocksAsync, for a store that
  // has found the registry advertising blocks it cannot serve.
  virtual tsl::Future<> UnregisterBlocksAsync(
      absl::Span<const std::string> block_hashes) {
    return tsl::Future<>(absl::UnimplementedError(
        "Backend does not implement UnregisterBlocksAsync."));
  }

  // The peer-facing KVCacheStoreService server this backend hosts, if any.
  // Returning non-null tells the owning KVCacheStore to publish THIS server
  // rather than start a second one, so a node always serves peers from exactly
  // one port.
  virtual KVCacheStoreServer* store_server() const { return nullptr; }
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_BACKEND_H_
