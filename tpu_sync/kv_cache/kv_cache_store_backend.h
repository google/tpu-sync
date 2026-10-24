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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"

namespace tpu_raiden {

namespace kv_cache {

class BlockTracker;
class WriteRemoteCancel;

enum class BlockStatus {
  INIT,
  REMOTE,
  HBM,
  HOST,
  HOST_AND_HBM,
  SHARED_STORAGE,
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

// Controls how Lookup matches `block_hashes`.
enum class LookupMatchMode {
  // Longest contiguous prefix from index 0; stops at the first miss.
  kPrefix,
  // All matching blocks in `block_hashes`, skipping misses. When
  // `block_hashes` is empty, snapshots all matching blocks in the backend.
  kSubset,
};

// Options controlling lookup behavior across storage backends.
//
// Invalid option combinations (return `absl::InvalidArgumentError`):
// - `match_mode == LookupMatchMode::kSubset && enable_global`
// - `host_resident_only && enable_global`
// - `include_eviction_candidates && pin_found`
struct LookupOptions {
  // Whether Lookup may query the global registry on local misses.
  // Defaults to true here; KVCacheStore::Lookup(hashes, enable_global,
  // pin_found) defaults enable_global to false to avoid a blocking RPC unless
  // requested.
  bool enable_global = true;

  // When true, prefix lookup (kPrefix) may interleave local and remote hits up
  // to the first hash missing in both. When false, local lookup stops at the
  // first local miss and queries the registry for the remaining suffix.
  // Ignored in kSubset mode.
  bool enable_interleaved_lookup = true;

  // When true, atomically pins matched active local blocks under the same lock
  // acquisition (without resurrecting eviction candidates). When false, uses
  // Peek() without promoting LRU order.
  bool pin_found = false;

  // Prefix matching (kPrefix) vs. scattered subset / full snapshot (kSubset).
  LookupMatchMode match_mode = LookupMatchMode::kPrefix;

  // When true, matches only blocks physically resident in local host DRAM
  // (BlockStatus::HOST or BlockStatus::HOST_AND_HBM), excluding HBM-only
  // entries and non-DRAM tiers (unlike `enable_global = false`, which only
  // skips the remote global registry RPC).
  // TODO(jcgu): Review whether BlockStatus::HBM is still needed or filter out
  // BlockStatus::HBM entries when `enable_global = false`, so
  // `host_resident_only` and `enable_global` can be merged.
  bool host_resident_only = false;

  // When true (requires `!pin_found`), includes unpinned eviction candidates in
  // addition to active blocks.
  bool include_eviction_candidates = false;

  // Non-contiguous subset of local host-resident blocks (including eviction
  // candidates).
  static LookupOptions HostResidentSubset();

  // Snapshot of active local host-resident blocks (pass empty `block_hashes`).
  static LookupOptions SnapshotHostResident();

  // Non-contiguous subset of active local host-resident blocks, atomically
  // pinned under the same lock acquisition.
  static LookupOptions PinPresentHostResident();
};

// Options controlling KVCacheStoreBackend::Insert() behavior.
enum class InsertOption {
  // Inserts new entries without pinning and overwrites existing active entries
  // in place without changing LRU order. Unregisters displaced entries from the
  // global registry.
  kBestEffortUpdate,

  // Atomically inserts and pins all entries if unpinned space fits all new
  // keys. Pins existing active entries (updating HBM-only entries and
  // preserving HOST and HOST_AND_HBM entries), inserts new keys in reverse
  // order (making block_hashes[0] MRU), and moves displaced entries to
  // eviction candidates without unregistering them.
  kInsertAndPin,

  // Atomically inserts new entries without pinning. Fails if the batch is
  // empty, span sizes mismatch, any key is duplicated or already present
  // (including eviction candidates), or unpinned space is insufficient. Moves
  // displaced entries to eviction candidates without unregistering them.
  kAdmitStrict,
};

// Options controlling KVCacheStoreBackend::Delete.
struct DeleteOptions {
  // When true, only unpinned host-resident entries (BlockStatus::HOST or
  // BlockStatus::HOST_AND_HBM) are erased.
  bool host_resident_only = false;

  // Optional output populated with the host_block_ids of host-resident
  // entries erased by this call.
  std::vector<int>* deallocated_host_block_ids = nullptr;

  // Optional output populated with the block hashes erased by this call.
  std::vector<std::string>* deleted_hashes = nullptr;

  // Preset for evicting unpinned host-resident entries and recording their
  // deallocated host_block_ids and evicted block hashes.
  static DeleteOptions EvictHostResident(
      std::vector<int>* deallocated_host_block_ids,
      std::vector<std::string>* evicted_hashes = nullptr) {
    return DeleteOptions{
        .host_resident_only = true,
        .deallocated_host_block_ids = deallocated_host_block_ids,
        .deleted_hashes = evicted_hashes,
    };
  }
};

// Abstract interface for KV cache index storage backends.
// Implementations must be thread-safe.
class KVCacheStoreBackend {
 public:
  virtual ~KVCacheStoreBackend() = default;

  // Invalidates the backend's lifetime fence before KVCacheStore destroys its
  // controller and block trackers.
  virtual void Shutdown() {}

  // Human-readable backend name for logging and diagnostics (e.g.,
  // "HostOffloadBackend", "posix").
  virtual std::string name() const = 0;

  // Queries cached block hashes according to `options` (see LookupOptions for
  // prefix vs. subset/snapshot matching, host-residency filtering, and
  // pinning). Custom implementations must honor `options.match_mode`,
  // `options.host_resident_only`, and `options.include_eviction_candidates` or
  // return `absl::UnimplementedError`.
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
                             absl::Span<const RaidenBlockId> slices,
                             BlockTracker* absl_nonnull load_tracker) = 0;

  // Asynchronously saves KV cache blocks from device (HBM) to host DRAM.
  // The tracker's transfer status is updated when the operation completes or
  // settles.
  virtual tsl::Future<> Save(absl::Span<const std::string> block_hashes,
                             absl::Span<const int64_t> src_device_block_ids,
                             absl::Span<const int32_t> dst_host_block_ids,
                             BlockTracker* absl_nonnull save_tracker) {
    CHECK(save_tracker != nullptr);
    return tsl::Future<>(absl::UnimplementedError("Save not implemented."));
  }

  // Inserts a batch of blocks with semantics controlled by `option`.
  // Returns true if the batch succeeds, or false if the batch is rejected.
  virtual bool Insert(
      absl::Span<const std::string> block_hashes,
      absl::Span<const RaidenBlockId> slices,
      InsertOption option = InsertOption::kBestEffortUpdate) = 0;

  // Removes unpinned entries in `block_hashes` matching `options` and returns
  // the number of entries erased.
  virtual size_t Delete(absl::Span<const std::string> block_hashes,
                        const DeleteOptions& options = {}) = 0;

  // Increments the pin count of each hash in `block_hashes` to protect it
  // from LRU eviction. Returns true if all hashes are present and pinned.
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

  // Retrieves evictable keys from the backend.
  virtual std::vector<std::string> GetEvictableKeys(size_t count) { return {}; }

  // Evicts keys from the backend and returns deallocated host block IDs.
  // If `evicted_hashes` is non-null, it is populated with the block hashes that
  // were actually evicted.
  virtual std::vector<int> Evict(
      const std::vector<std::string>& block_hashes,
      std::vector<std::string>* evicted_hashes = nullptr) {
    std::vector<int> deallocated_host_block_ids;
    Delete(block_hashes, DeleteOptions::EvictHostResident(
                             &deallocated_host_block_ids, evicted_hashes));
    return deallocated_host_block_ids;
  }

  // --- Remote write (WriteRemote) & registry helpers -----------------------
  //
  // Default implementations return UnimplementedError so backends that do not
  // support remote writes fail cleanly.

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

  // --- Remote write, source side ------------------------------------------

  // What the destination decided, before any bytes have moved.
  struct RemoteWriteAck {
    // Identifies the accepted operation. Zero when the destination answered
    // from what it already holds and started no transfer.
    uint64_t operation_id = 0;
    // Every offered hash was already there. A success; nothing to transfer.
    bool all_exist = false;
    // Set when the destination held some but not all hashes. A failure; the
    // caller decides what to re-offer.
    std::vector<std::string> existing_hashes;
    // The deadline the destination armed, clamped to its own cap. The source
    // must protect the blocks at least this long.
    absl::Duration granted_deadline;
    // Ends the open call early; ~KVCacheStore uses it for abandoned offers.
    std::shared_ptr<WriteRemoteCancel> cancel;
  };

  // Offers `block_hashes` to `dst_raiden_id` and blocks until the ack (not
  // the bytes). `requested_deadline` is how long the destination may hold
  // its landing blocks; `hold_window` is the call's deadline. The tracker's
  // transfer status is updated when the operation completes or settles.
  virtual absl::StatusOr<RemoteWriteAck> BeginWriteRemote(
      const RaidenId& dst_raiden_id, absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> src_host_block_ids,
      absl::Duration requested_deadline, absl::Duration hold_window,
      BlockTracker* absl_nonnull save_tracker) {
    CHECK(save_tracker != nullptr);
    return absl::FailedPreconditionError(
        "A remote save requires a HostOffloadBackend at tier 0");
  }

  // Composes the KVTransferSpec from the workers currently registered with
  // the RaidenController and registers it with the global registry
  // (first-wins, idempotent; see global_registry.proto).
  virtual absl::Status RegisterKVTransferSpecFromWorkers() {
    return absl::FailedPreconditionError(
        "registering the KVTransferSpec requires a HostOffloadBackend at "
        "tier 0");
  }
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_BACKEND_H_
