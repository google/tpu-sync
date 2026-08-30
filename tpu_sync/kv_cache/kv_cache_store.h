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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/kv_cache/kv_cache_store_server.h"
#include "tpu_sync/kv_cache/lru_cache.h"
#include "tpu_sync/kv_cache/reshard/reshard_service.h"

namespace tpu_raiden {

namespace controller {
class RaidenController;
}  // namespace controller

namespace kv_cache {

class WriteRemoteCancel;

namespace global_registry {
class GlobalRegistryClient;
}

class StoreMonitor;

// KV Store that manages the indices and routing of prefix cache across serving
// nodes and microservice slices.
//
// THE THREE WORKFLOWS
//
// The surface exists to serve exactly three flows. Every pin is granted by one
// call and consumed by one call, so a caller never balances pins by hand:
//
//   1. Read what is cached:      Lookup() -> Load()
//        Lookup() pins each hash it returns; a successful local Load() spends
//        that pin. A REMOTE hit is not pinned and a remote Load() spends
//        nothing -- a registry-only hit never entered the local index, and a
//        peer load records nothing locally either.
//
//   2. Cache something new:      Insert() -> Save()
//        Insert() pins what it takes; a successful Save() spends it.
//
//   3. Hand a block to a peer:   Lookup() -> Save(dst)
//        The block must already be host-resident here, so Lookup() finds it
//        locally and pins it, and the successful remote Save() spends that
//        pin.
//
// Flows 1 and 2 never collide over a hash: Lookup() returns hits, Insert()
// takes the misses, so the sets are disjoint by construction and nothing is
// pinned twice.
//
// A FAILED operation does not spend the pin. Retrying, or giving up with
// Release(), is the caller's decision -- unpinning under a retry would leave
// the block evictable while the caller still believed it held it.
//
// A caller that only wants to OBSERVE what is resident, and will not go on to
// load or save it, should use the LookupOptions overload with pin_found=false
// rather than Lookup()-then-Release(): a pin/unpin round trip moves the entry
// through the pinned list and back, which reorders the LRU.
//
// Eviction is not part of any of this. The store reclaims space itself, when
// an operation needs a host block; there is no caller-driven removal.
class KVCacheStore {
 public:
  friend class KVCacheStoreTest;
  friend class RemoteWriteSourceTest;

  // NETWORK ADDRESSING (`store_server_ip`, `raiden_controller_port`)
  //
  // `store_server_ip` is the IP peers use to reach this node. It is
  // BIND-AND-ADVERTISE: both this store's KVCacheStoreService and its
  // RaidenController bind that interface, and it is the host published to the
  // global registry. It must therefore be an IP this process can bind -- not a
  // hostname, and not a NAT/service address. Ports are chosen by gRPC (the
  // store server always; the controller when `raiden_controller_port` is 0),
  // and the bound port is spliced into the advertised address. Note that the IP
  // address of the RaidenController reuses `store_server_ip`.
  //
  // `store_server_ip` is MANDATORY and must not be a wildcard: `Create()`
  // rejects an empty or wildcard value, and the raw constructors abort on it.
  // Whether this store is discoverable is decided by the registry, not by this
  // field -- with no `global_registry_address` no server is started at all.
  //
  // `expected_worker_count` > 0 makes construction block until that many
  // workers have registered with the in-process RaidenController, and fail
  // (Create() throws, the raw constructors abort the process via the
  // controller's exception) if they have not all registered within
  // RAIDEN_EXPECTED_WORKERS_TIMEOUT_S seconds (default 120). Callers whose
  // workers register only after the store exists must leave it 0.

  // Safe factory method to create KVCacheStore from a single BackendConfig.
  static absl::StatusOr<std::unique_ptr<KVCacheStore>> Create(
      const BackendConfig& config, size_t capacity = 0,
      absl::string_view global_registry_address = "", RaidenId raiden_id = {},
      int num_shards = 0, int64_t shard_size_bytes = 0,
      absl::string_view store_server_ip = "", int raiden_controller_port = 0,
      std::optional<KVCacheMetadata> metadata = std::nullopt,
      int expected_worker_count = 0);

  // Safe factory method to create KVCacheStore from a list of BackendConfigs.
  static absl::StatusOr<std::unique_ptr<KVCacheStore>> Create(
      absl::Span<const BackendConfig> backend_configs, size_t capacity = 0,
      absl::string_view global_registry_address = "", RaidenId raiden_id = {},
      int num_shards = 0, int64_t shard_size_bytes = 0,
      absl::string_view store_server_ip = "", int raiden_controller_port = 0,
      std::optional<KVCacheMetadata> metadata = std::nullopt,
      int expected_worker_count = 0);

  // Thin-store factory: hosts a RaidenController (expected_worker_count=0)
  // and a ReshardService in WorkerDelivery::Mode::kController mode within a
  // minimal KVCacheStore, eliminating the need for a separate sidecar
  // binary. Workers register their WorkerService endpoints with the
  // in-process controller; the coordinator dispatches reshard transfer
  // programs over persistent gRPC channels directly to them.
  static absl::StatusOr<std::unique_ptr<KVCacheStore>> CreateReshardStore(
      RaidenId raiden_id, absl::string_view store_server_ip,
      int raiden_controller_port = 0, int reshard_service_port = 0);

  // Flexible constructor accepting a custom root backend
  explicit KVCacheStore(std::shared_ptr<KVCacheStoreBackend> backend,
                        RaidenId raiden_id = {}, int num_shards = 0,
                        int64_t shard_size_bytes = 0,
                        absl::string_view store_server_ip = "",
                        int raiden_controller_port = 0,
                        absl::string_view global_registry_address = "",
                        int expected_worker_count = 0);

  // Constructor accepting an ordered list of backends.
  // `global_registry_address` is what this store publishes itself to. The
  // backends may each hold their own registry client; this one belongs to the
  // store, and without it the store cannot make itself discoverable.
  explicit KVCacheStore(
      std::vector<std::shared_ptr<KVCacheStoreBackend>> backends,
      RaidenId raiden_id = {}, int num_shards = 0, int64_t shard_size_bytes = 0,
      absl::string_view store_server_ip = "", int raiden_controller_port = 0,
      absl::string_view global_registry_address = "",
      int expected_worker_count = 0);

  // Links RaidenController to all backends in backends_. Returns an error iff
  // this store owns a store_server_ip and starting/publishing its server
  // failed -- store initialization requires the store to be discoverable
  // whenever the caller asked for one.
  absl::Status SetRaidenController(
      tpu_raiden::controller::RaidenController* controller);

  // If `metadata` is provided, every LRU cache entry whose data lives in
  // local host memory is mirrored into it, keeping a crash-persistent copy of
  // the LRU cache in the caller's region (see KVCacheMetadata).
  explicit KVCacheStore(size_t capacity,
                        absl::string_view global_registry_address = "",
                        RaidenId raiden_id = {}, int num_shards = 0,
                        int64_t shard_size_bytes = 0,
                        absl::string_view store_server_ip = "",
                        int raiden_controller_port = 0,
                        std::optional<KVCacheMetadata> metadata = std::nullopt,
                        int expected_worker_count = 0);

  // Test-only constructor for injecting mock controller
  explicit KVCacheStore(
      size_t capacity,
      std::unique_ptr<tpu_raiden::controller::RaidenController>
          raiden_controller,
      absl::string_view global_registry_address = "", RaidenId raiden_id = {},
      std::optional<KVCacheMetadata> metadata = std::nullopt,
      absl::string_view store_server_ip = "");

  // Create a reshard-only thin-store mode. The
  // returned store owns nothing but a bound reshard::ReshardService — no
  // offload backend, no global registry, no store gRPC server, and no
  // RaidenController submodule. It serves as a sidecar substitution for
  // the Python controller process; every full-store construction path above
  // is untouched (their validation still requires backends + num_shards>=1).
  static absl::StatusOr<std::unique_ptr<KVCacheStore>> CreateReshardSidecar(
      int reshard_port, double request_registry_ttl_s);

  // Non-null only for CreateReshardSidecar() stores (a full
  // store gains the member when constructed with co-hosted resharding).
  reshard::ReshardService* reshard_service() { return reshard_service_.get(); }

  ~KVCacheStore();

  KVCacheStore(const KVCacheStore&) = delete;
  KVCacheStore& operator=(const KVCacheStore&) = delete;

  // Authoritative KVCacheStore API implementations

  // Checks the LRU cache for cached block hashes. Returns a list of all
  // matched replica pairs (block hash and RaidenBlockId) encountered
  // in sequence prior to the first miss.
  // If enable_global is true, it will query the global registry for any
  // misses after the local lookup. Defaults to false: the registry query is a
  // blocking RPC, so a caller that only wants to know what is resident locally
  // should not pay for one by omission.
  //
  // PINS every hash it returns (pin_found = true, the default), one pin each,
  // so the answer cannot be evicted between being given and being used. The
  // operation the caller goes on to perform consumes that pin -- Load() drops
  // it on a successful local load, Save() on success. A caller that only
  // wanted to observe residency, and performs neither, should pass
  // pin_found = false rather than Release() what it was given: a pin/unpin
  // round trip moves the entry through the pinned list and back, which
  // reorders the LRU.
  //
  // Registry-only hits are not pinned: they name a block on another node, and
  // there is nothing here to hold.
  absl::StatusOr<BlockSliceList> Lookup(
      const std::vector<std::string>& block_hashes, bool enable_global = false,
      bool pin_found = true);

  // Overload accepting LookupOptions for granular control. Unlike the overload
  // above, pin_found defaults to false here, so this is what internal callers
  // and pure observers use. The struct's enable_global default is also the
  // opposite (true, not false), so a bare LookupOptions{} does query the
  // registry when one is configured.
  absl::StatusOr<BlockSliceList> Lookup(
      const std::vector<std::string>& block_hashes,
      const LookupOptions& options);

  // Caches sharded buffers into host-RAM/HBM backing store, PINNING every
  // hash it takes: existing ones are pinned in place, new ones inserted and
  // pinned. New items are inserted in reverse order so that tail blocks are
  // evicted first.
  //
  // All-or-nothing. It does NOT evict to make room: if there is not enough
  // free space for the new hashes it unpins whatever it already pinned and
  // returns false, leaving the cache exactly as it found it. Making room by
  // evicting is the store's own business, not something an insert does on the
  // caller's behalf.
  //
  // PIN CONTRACT: the pin it grants is what a subsequent Save() consumes.
  // Called with hashes Lookup() reported as misses -- the two take disjoint
  // sets, which is why no hash ends up pinned twice.
  //
  // LOCAL entries only. The LRU cache holds blocks resident on THIS node;
  // a REMOTE slice names a block on another node and has no business in it.
  // If any slice in the batch has status REMOTE the whole batch is refused
  // with InvalidArgument and nothing is inserted or pinned.
  //
  // Returns OkStatus when the whole operation succeeded, InvalidArgument for
  // a REMOTE slice in the batch, and ResourceExhausted when the cache refused
  // the batch (not enough unpinned space, or a hash could not be pinned).
  absl::Status Insert(const std::vector<std::string>& block_hashes,
                      const std::vector<RaidenBlockId>& slices, bool on_host);

  // Releases previously pinned block hashes (a.k.a. Unpin), making them
  // eligible for LRU eviction when capacity is exceeded.
  // The blocks are released in reversed order internally to ensure that the
  // last blocks (tails of sequences) are evicted first.
  //
  // LOCAL entries by construction: Insert() refuses REMOTE slices, so the
  // cache cannot hold an entry release would need to reject, and no status
  // check is performed here.
  void Release(const std::vector<std::string>& block_hashes);

  // Saves blocks, asynchronously. Source and destination depend on
  // `dst_raiden_id`:
  //
  //   absent  -- LOCAL save, device (HBM) to host (DRAM). Requires each block
  //              to have status HBM here; one already host-resident
  //              (HOST_AND_HBM) has nothing left to save and is refused.
  //   present -- REMOTE save: offer the blocks to that peer, which takes its
  //              own copy. Requires each block to be HOST-resident here
  //              already, because the bytes are pulled out of host DRAM. There
  //              is no two-hop: a block only in HBM must be saved locally
  //              first.
  //
  // Both kinds are polled with PollSaveStatus().
  //
  // PIN CONTRACT: every hash must be pinned on entry -- Lookup() grants that
  // pin for a remote save, Insert() for a local one -- and a SUCCESSFUL save
  // consumes exactly one pin per hash. The caller does not release afterwards.
  // A FAILED save does not consume it: the entry stays pinned so the caller
  // can retry or release deliberately.
  //
  // A remote save also takes a SEPARATE internal pin for as long as the
  // destination may still be reading, released when the operation goes
  // terminal or the HOLD expires. That one is not the caller's.
  //
  // A remote save requires a global registry at both ends: that is how the
  // destination is resolved, and how the blocks become reachable once they
  // land.
  //
  // IMPORTANT: a remote destination allocates landing blocks from FREE blocks
  // only and never evicts to make room, so a peer with a warm cache refuses
  // every offer with RESOURCE_EXHAUSTED. Destination-side eviction is
  // separate, unimplemented work.
  absl::Status Save(const std::vector<std::string>& block_hashes,
                    const std::optional<RaidenId>& dst_raiden_id = std::nullopt);

  // Asynchronously loads KV cache blocks to device (HBM) from local host DRAM.
  //
  // LOCAL ONLY. A hash whose entry has status REMOTE is refused with
  // InvalidArgument -- this overload never talks to a peer; the slices
  // overload below is the only peer-load path.
  //
  // `device_block_ids` is the destination and must name one device block per
  // hash.
  //
  // PIN CONTRACT: every hash must be pinned on entry -- Lookup() is what
  // normally grants that pin -- and a SUCCESSFUL load consumes exactly one pin
  // per hash. The caller does not release afterwards.
  //
  // A FAILED load does not: the entry stays pinned so the caller can retry, or
  // release it deliberately. Deciding to give up is the caller's, not this
  // store's.
  absl::Status Load(absl::Span<const std::string> block_hashes,
                    absl::Span<const int> device_block_ids);

  // Asynchronously loads KV cache blocks to device (HBM), either from local
  // host DRAM or from a peer. This is the ONLY way to load from a peer -- the
  // no-slices overload above is local-only.
  //
  // ONE CALL IS ONE SOURCE. Every block in a batch must carry the same status,
  // and remote blocks must all refer to the same peer.
  //
  // `device_block_ids` is the destination and must name one device block per
  // hash.
  //
  // `slices` carries the caller's pre-looked up RaidenBlockIds, one per hash;
  // a lookup() answer can be passed straight through. A local load uses their
  // host_block_ids directly. A remote load takes only the peer's identity
  // from them -- the hashes are re-resolved at the peer.
  //
  // PIN CONTRACT -- NOT the same as the overload above; the two sources
  // genuinely differ:
  //   local source  -- every hash must be pinned on entry, and a successful
  //                    load consumes one pin per hash.
  //   remote source -- no pin is required and none is consumed. A hash
  //                    resolved only through the registry never entered the
  //                    local index, so there is nothing here to have pinned,
  //                    and a load from a peer records nothing either.
  absl::Status Load(absl::Span<const std::string> block_hashes,
                    absl::Span<const RaidenBlockId> slices,
                    absl::Span<const int> device_block_ids);

  int GetPinCount(const std::string& hash) const;

  // Source-side ReadRemote (All-or-Nothing validate & pin block hashes at the
  // src controller): verifies ALL block_hashes exist in the LRU with status
  // HOST/HOST_AND_HBM and pins them (all-or-nothing: any miss rolls back and
  // aborts). On success returns the authoritative source host_block_ids
  // (re-derived from the LRU). NotFound => a hash is absent
  // (BLOCK_HASH_NOT_FOUND); FailedPrecondition => present but not
  // host-resident. Registered as a hook on the RaidenController so a peer's
  // ReadRemote RPC can reach this store's LRU. Public for testability.
  absl::StatusOr<std::vector<int32_t>> ValidateAndPinHostBlocks(
      absl::Span<const std::string> block_hashes);
  // Releases the pins taken by ValidateAndPinHostBlocks.
  void UnpinHostBlocks(absl::Span<const std::string> block_hashes);

  size_t capacity() const;
  std::string raiden_controller_address() const;

  // "host:port" of this store's KVCacheStoreService, as published to the
  // global registry. Empty when this store has no registry configured, since
  // that is exactly when no server is started -- not when `store_server_ip`
  // was omitted, which is now a construction error.
  std::string store_server_address() const { return store_server_address_; }

  // The store server this node serves peers from, or nullptr if it has none.
  // Owned either by this store or by whichever backend hosts it.
  KVCacheStoreServer* store_server() const { return store_server_; }

  const std::vector<std::shared_ptr<KVCacheStoreBackend>>& backends() const {
    return backends_;
  }

  const std::shared_ptr<KVCacheStoreBackend>& backend() const {
    return backends_[0];
  }

  const RaidenId& raiden_id() const { return raiden_id_; }

  tpu_raiden::controller::RaidenController* raiden_controller() const {
    return raiden_controller_.get();
  }

  // Polls the status of all active/inflight Save operations, local and remote.
  // For local saves it iterates pending futures, updates cache metadata to
  // HOST_AND_HBM on success and deallocates host blocks on failure; for remote
  // saves it asks each destination for a verdict.
  //
  // Returns {done, failed, pending, existing, unregistered}. The last two are
  // annotations on REMOTE failures, not separate outcomes -- a hash in either
  // also appears in `failed` -- and they exist because this store never
  // decides on the caller's behalf:
  //
  //   existing:     the destination already held these, so it refused the
  //                 batch. Reissuing with the remainder is the caller's call.
  //   unregistered: the destination HAS these -- the transfer succeeded -- but
  //                 could not publish them, so no peer can find them. Reported
  //                 as failed because the safe default is for the caller to
  //                 keep its own copy. A caller that only needed the peer to
  //                 have the bytes may treat it as success; one that was about
  //                 to free its copy must not.
  //
  // Both lists are empty for local saves.
  //
  // `pending` means different things on the two paths, and the difference is
  // worth knowing before waiting on it: for a local save it is bounded by a
  // DMA, for a remote one by the destination answering or the ~30s HOLD
  // window elapsing.
  //
  // Terminal results are drained on read.
  struct PollSaveStatusResult {
    std::vector<std::string> done;
    std::vector<std::string> failed;
    std::vector<std::string> pending;
    std::vector<std::string> existing;
    std::vector<std::string> unregistered;
  };
  PollSaveStatusResult PollSaveStatus();

  // Polls the status of all active/inflight Load operations.
  // Updates cache metadata upon successful H2D transfers:
  //   - Loaded from local host DRAM -> HOST_AND_HBM
  //   - Loaded from a peer          -> nothing is recorded at all.
  //
  // A peer load leaves no entry because there is nothing here to describe: no
  // local host copy is kept, so the entry could only say HBM with
  // host_block_id -1 -- which Evict cannot reclaim (it takes HOST and
  // HOST_AND_HBM only) and which nothing would ever remove. A later lookup()
  // of such a hash is therefore a miss, and the caller's own block manager is
  // what remembers it already owns the device block.
  //
  struct PollLoadStatusResult {
    std::vector<std::string> done;
    std::vector<std::string> failed;
    std::vector<std::string> pending;
  };
  PollLoadStatusResult PollLoadStatus();

  // Launches an async receiver-initiated read of REMOTE blocks from their
  // owning peers straight into local HBM. Returns as soon as the reads are
  // issued; poll with PollRemoteReadStatus().
  //
  // The caller supplies the source coordinates directly: `slices[i]` is the
  // REMOTE RaidenBlockId for `block_hashes[i]`, and only two of its fields are
  // read -- `raiden_id` (which peer owns the block) and `host_block_id` (which
  // block on that peer). A lookup() answer can be passed straight through.
  //
  // This store's LRU is not consulted and not modified. The hashes need not be
  // present locally and need not be pinned, nothing is inserted on success,
  // and nothing is left behind on failure. The bytes land ONLY in the caller's
  // device blocks; the host blocks this call allocates are pure staging and
  // are returned to the pool on both the success and the failure path. No
  // local host copy is retained, so a later local load() of the same hash is
  // still a miss.
  //
  // device_block_ids is mandatory and must match block_hashes in size, as must
  // slices; any other size is InvalidArgument.
  //
  // The device blocks are written before the source's verdict is known, so on
  // failure their contents are UNDEFINED -- treat them as scratch until the
  // read reports success.
  //
  // Compare with Load(): both bring a peer's block into local HBM. Load()
  // fetches through the store's own path and is the right call when the hash
  // may be resident locally; ReadRemote() takes a lease on the source and is
  // the right call when the caller already knows the source coordinates and
  // wants no local record of the transfer.
  //
  // Requires a global registry: it is what maps the owning peer to the
  // controller address this store acquires its read lease from. A store built
  // without one fails every read with FailedPrecondition.
  absl::Status ReadRemote(const std::vector<std::string>& block_hashes,
                          const std::vector<RaidenBlockId>& slices,
                          const std::vector<int32_t>& device_block_ids);

  // Polls status of active remote reads.
  // Returns {done_hashes, failed_hashes, pending_hashes}
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  PollRemoteReadStatus();

  // Returns the count of active remote write operations. TESTS ONLY.
  size_t InFlightRemoteWritesCountForTesting() const;


  // Rebuilds this store's LRU cache after an engine restart from the
  // crash-persistent KVCacheMetadata table in local shared memory, without
  // consulting the global registry.
  //
  // - Requires a raiden controller, an attached metadata table, and an empty
  //   LRU cache.
  // - Allocates the recorded host block IDs in the controller, then
  //   repopulates the LRU cache with unpinned HOST entries in
  //   ascending-seq order (oldest first).
  // - Entries exceeding the LRU cache capacity overflow into eviction
  //   candidates, keeping their blocks and metadata entries.
  // - The seq counter resumes past the largest recovered stamp.
  // - If two blocks claim the same hash, the entry with the largest seq is
  //   the newest binding and wins; stale ones are not recovered and are
  //   cleared from the table.
  // - On any failure the store and the table are left untouched so the
  //   caller can fall back to a cold start.
  //
  // Returns the number of recovered blocks.
  absl::StatusOr<size_t> RecoverFromLocalManifest();

 private:
  // Tag for the reshard-sidecar-only construction (no backends, no wiring).
  struct ReshardSidecarTag {};
  explicit KVCacheStore(ReshardSidecarTag);

  explicit KVCacheStore(
      std::vector<std::shared_ptr<KVCacheStoreBackend>> backends,
      RaidenId raiden_id,
      std::unique_ptr<controller::RaidenController> raiden_controller,
      absl::string_view store_server_ip = "",
      absl::string_view global_registry_address = "");

  // Registers ValidateAndPinHostBlocks/UnpinHostBlocks on raiden_controller_
  // as the source-side hooks that a peer's ReadRemote lease acquisition and
  // release run against this store's LRU (no-op if there is no controller).
  void RegisterReadRemoteHooks();

  // Evicts host blocks by their logical block hashes.
  //
  // Checks if the blocks exist in the cache and are in HOST or HOST_AND_HBM
  // status and erases them from the cache.
  // Releases the host blocks via RaidenController and unregisters them from
  // GlobalRegistry.
  //
  // Input: `block_hashes` - list of block hashes to evict.
  // Output: Number of successfully evicted host blocks.
  size_t Evict(const std::vector<std::string>& block_hashes);

  struct SaveState {
    tsl::Future<> future;
    std::vector<std::string> block_hashes;
    std::vector<int> host_block_ids;
  };

  struct LoadState {
    tsl::Future<> future;
    std::vector<std::string> block_hashes;
    std::vector<int> device_block_ids;
    // Whether the source was a peer. Decided at submit time and carried here
    // because the poller cannot re-derive it: a remote load records nothing
    // locally, so by completion there is no entry to read a status off.
    bool from_remote = false;
  };

  struct RemoteReadState {
    std::vector<std::string> block_hashes;
    // The peers this batch read from. Carried so a failed read can drop their
    // cached controller addresses -- the poller is where failure is observed,
    // and by then the grouping is gone.
    std::vector<RaidenId> src_raiden_ids;
    // The local staging blocks the bytes hop through on their way to HBM. They
    // live HERE and nowhere else: no LRU entry ever points at them, so the
    // poller returns them to the pool on both the success and the failure
    // path. The caller's device blocks are not tracked -- once the transfer is
    // terminal this store has no further interest in them.
    std::vector<int> host_block_ids;
  };

  struct FutureHash {
    size_t operator()(const tsl::Future<>& f) const {
      return reinterpret_cast<size_t>(f.async_value());
    }
  };

  struct FutureEqual {
    bool operator()(const tsl::Future<>& lhs, const tsl::Future<>& rhs) const {
      return lhs.async_value() == rhs.async_value();
    }
  };

  // Starts (if needed) the peer-facing store server, computes
  // store_server_address_, and publishes it to the global registry.
  // Idempotent: SetRaidenController may be called more than once, and Create
  // calls it again after construction. With no registry_client_, this is
  // a no-op -- no server is started anywhere. Returns an error iff a registry
  // is configured and starting or publishing the server failed.
  absl::Status EnsureStoreServerAndRegister(
      tpu_raiden::controller::RaidenController* controller);

  // Shuts down every server hosted by one of backends_, skipping
  // `already_shut` (the adopted server, when this store had one). Called from
  // the destructor body, where raiden_controller_ is still alive: a running
  // service holds it in a pointer it cannot re-seat.
  void ShutdownBackendStoreServers(KVCacheStoreServer* already_shut);

  // One bounded step of the evict sweep: checks the free-block watermarks
  // and demotes at most one batch of cold blocks to a placement target (or
  // drops the batch locally when no target will take it). Returns true when
  // the sweep should run again right away. The store monitor's thread is
  // the only caller, which is why the episode state below needs no lock.
  bool SweepOnce();

  // What became of one offered batch, polled until every block is terminal.
  struct BatchWriteResult {
    // Blocks the peer now holds; their local copies are safe to free.
    std::vector<std::string> freeable;
    // Blocks whose offer settled without success, so the sweep's caller pin
    // was not consumed and the sweep must release it itself.
    std::vector<std::string> still_pinned;
    // At least one block's transfer failed outright; its local copy stays.
    bool transfer_failed = false;
  };
  BatchWriteResult WaitForBatchWriteResult(
      const std::vector<std::string>& batch);

  mutable absl::Mutex mutex_;
  std::vector<std::shared_ptr<KVCacheStoreBackend>> backends_;
  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client_;
  RaidenId raiden_id_;
  std::unique_ptr<tpu_raiden::controller::RaidenController> raiden_controller_;

  // The store-owned reshard control plane (thin sidecar mode).
  // Never constructed by the full-store paths today.
  std::unique_ptr<reshard::ReshardService> reshard_service_;

  // The IP peers reach this node on. Never empty and never a wildcard --
  // construction rejects both.
  std::string store_server_ip_;
  // Advertised "host:port" of store_server_, empty until it is started.
  std::string store_server_address_;
  // Non-owning. Points either into owned_store_server_ or at a server owned by
  // one of backends_.
  KVCacheStoreServer* store_server_ = nullptr;
  // Set only when no backend hosts a store server and this store made its own.
  std::unique_ptr<KVCacheStoreServer> owned_store_server_;
  // True once this store published itself, so teardown knows to unpublish.
  bool registered_in_global_registry_ = false;

  // Registration identity from the tier-0 BackendConfig.
  std::string kv_pool_group_;
  int32_t evict_tier_ = 0;
  // Monitor and sweep knobs from the tier-0 BackendConfig, with zeros
  // resolved to the built-in defaults in Create.
  StoreMonitorConfig monitor_config_;
  // Pressure-episode state, touched only by SweepOnce on the monitor's
  // thread. One episode = the run of sweep steps from free blocks falling
  // below the low watermark until they recover to the high watermark (or
  // nothing more can be freed); the placement targets are fetched once per
  // episode and reused until it ends or every target has been dropped.
  bool sweep_active_ = false;
  std::vector<RaidenId> placement_targets_;
  // Constructed and started at the end of Create when enabled; stopped first
  // thing in the destructor, before anything its callbacks touch goes away.
  std::unique_ptr<StoreMonitor> store_monitor_;

  // Who asked for a remote save, and therefore who its verdict is addressed
  // to. Verdicts are drained on read, so a single shared queue lets either
  // consumer take the other's mail: an application PollSaveStatus() would
  // strand the evict sweep in WaitForBatchWriteResult, which waits for a
  // verdict that has already been thrown away, and a sweep drain would
  // silently swallow verdicts the application is waiting for.
  //
  // The owner rides on the state because it is known where the operation is
  // created and unrecoverable everywhere it is needed -- a poller thread, a
  // destructor. Keying a side table by operation_id would not do: the two
  // paths that settle inside SaveRemote itself never record an operation at
  // all, and for the sweep those are the common outcome, not the rare one.
  enum class SaveOwner { kApplication, kSweep };

  using OperationKey = uint64_t;

  struct RemoteWriteState {
    OperationKey key = 0;
    RaidenId dst_raiden_id;
    uint64_t operation_id = 0;
    std::vector<std::string> block_hashes;
    // Ends the call this offer was made on. Held for as long as this store is
    // protecting the blocks, and used when it stops: a destination left
    // holding an open call reserves its landing blocks for the rest of the
    // hold window, waiting on a peer that has gone.
    std::shared_ptr<WriteRemoteCancel> cancel;
    // When this source stops protecting the blocks and gives up, whatever the
    // destination is doing. Must outlive the deadline the destination granted,
    // or this store would unpin while the destination could still legitimately
    // commit bytes read out of blocks it has released.
    absl::Time hold_expiry;
    SaveOwner owner = SaveOwner::kApplication;
  };

  // One owner's mailbox of settled remote writes. `existing` and
  // `unregistered` ANNOTATE entries that are also in `failed`; they do not
  // replace them.
  struct RemoteWriteVerdicts {
    std::vector<std::string> done;
    std::vector<std::string> failed;
    std::vector<std::string> existing;
    std::vector<std::string> unregistered;
  };

  // Lifetime fence ensuring background completion tasks can safely check
  // if the store is still alive.
  struct Lifetime {
    absl::Mutex mu;
    KVCacheStore* store ABSL_GUARDED_BY(mu) = nullptr;
  };
  std::shared_ptr<Lifetime> lifetime_;

  std::vector<SaveState> active_saves_ ABSL_GUARDED_BY(mutex_);
  std::vector<LoadState> active_loads_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<tsl::Future<>, RemoteReadState, FutureHash, FutureEqual>
      active_remote_reads_ ABSL_GUARDED_BY(mutex_);

  absl::flat_hash_map<OperationKey, RemoteWriteState> active_remote_writes_
      ABSL_GUARDED_BY(mutex_);
  OperationKey next_op_key_ ABSL_GUARDED_BY(mutex_) = 1;

  // Addressed by owner, so the two consumers cannot take each other's mail.
  // PollSaveStatus() drains the application's; the evict sweep drains its own
  // through DrainSweepVerdicts().
  RemoteWriteVerdicts application_remote_writes_ ABSL_GUARDED_BY(mutex_);
  RemoteWriteVerdicts sweep_remote_writes_ ABSL_GUARDED_BY(mutex_);

  std::vector<std::string> done_saves_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> failed_saves_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> done_loads_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> failed_loads_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> done_remote_reads_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> failed_remote_reads_ ABSL_GUARDED_BY(mutex_);

  // In-flight saves of BOTH kinds. Local and remote saves were tracked
  // separately, which was safe only by accident: a local save requires the
  // block in HBM and a remote one requires it in host DRAM, so the two sets
  // could not overlap. One set makes that explicit and, more usefully, makes
  // "already saving" mean it whichever kind is in flight.
  absl::flat_hash_set<std::string> saving_hashes_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_set<std::string> loading_hashes_ ABSL_GUARDED_BY(mutex_);
  // Peer -> its RaidenController address, as last resolved from the global
  // registry. Read on the prefill path, so it is worth not paying a registry
  // round trip per read.
  //
  // Every entry is dropped as soon as a read against that peer FAILS, for any
  // reason. That looks over-broad -- a revoked lease or a transfer error says
  // nothing about the address -- and it is deliberate: an unnecessary
  // invalidation costs one resolve on the next read, while a missed one leaves
  // a peer that restarted on a new port unreachable for the life of this
  // process. That was the shape of the bug this cache replaced, and it is the
  // reason the old one had to go.
  //
  // Consequence worth knowing: the first read after a peer restarts still
  // fails, on the stale address, and the retry is what succeeds.
  absl::flat_hash_map<RaidenId, std::string, RaidenIdHash>
      resolved_peer_controllers_ ABSL_GUARDED_BY(mutex_);

  absl::flat_hash_set<std::string> reading_hashes_ ABSL_GUARDED_BY(mutex_);

  std::unique_ptr<std::thread> poller_thread_;
  std::atomic<bool> stop_poller_{false};

  // Blocks pinned by write-throughs still waiting on the registry. Bounded,
  // because a pinned block is invisible to eviction: a registry that connects
  // and never answers would otherwise pin the host block pool dry. Past the
  // bound the advertisement is dropped and the pins released -- invisible to
  // peers is recoverable, unevictable is not.
  //
  // The calls are not tracked; teardown's UnregisterStore purges what this
  // raiden id owns.
  struct WriteThroughState {
    absl::Mutex mutex;
    size_t in_flight_blocks ABSL_GUARDED_BY(mutex) = 0;
    size_t max_in_flight_blocks ABSL_GUARDED_BY(mutex) =
        kDefaultMaxInFlightWriteThroughBlocks;
  };
  std::shared_ptr<WriteThroughState> wt_state_ =
      std::make_shared<WriteThroughState>();
  static constexpr size_t kDefaultMaxInFlightWriteThroughBlocks = 4096;

  absl::StatusOr<std::vector<int>> AllocateBlockIds(int needed);
  void DeallocateBlockIds(absl::Span<const int> block_ids);

 public:
  // Lowers the outstanding-write-through bound so a test can reach it without
  // saving thousands of blocks.
  void SetMaxInFlightWriteThroughBlocksForTesting(size_t max_blocks) {
    absl::MutexLock lock(wt_state_->mutex);
    wt_state_->max_in_flight_blocks = max_blocks;
  }

  // Blocks currently held by write-throughs waiting on the registry.
  size_t InFlightWriteThroughBlocksForTesting() const {
    absl::MutexLock lock(wt_state_->mutex);
    return wt_state_->in_flight_blocks;
  }

 private:
  // The two halves of Save(), split by destination. Save() validates nothing
  // itself -- each half has its own residency and pin preconditions, and they
  // are genuinely different operations sharing a name and a status queue.
  absl::Status SaveLocal(const std::vector<std::string>& block_hashes);
  // `owner` is deliberately not defaulted: both call sites state it. The bug
  // this replaced was an unwritten assumption about who drains the verdict,
  // and a default is how the next caller inherits that answer without ever
  // deciding it.
  absl::Status SaveRemote(const std::vector<std::string>& block_hashes,
                          const RaidenId& dst_raiden_id, SaveOwner owner);

  // Atomically extracts an operation from active_remote_writes_ under mutex_.
  // Whoever receives the state owns settling it; concurrent callers receive
  // std::nullopt and do nothing.
  std::optional<RemoteWriteState> TakeRemoteWrite(OperationKey key);

  // Settles a taken operation by releasing internal pins and filing verdicts.
  // Must NOT be called with mutex_ held.
  void OnWriteRemoteVerdict(RemoteWriteState state, bool succeeded,
                            std::vector<std::string> existing,
                            std::vector<std::string> unregistered = {});

  // Releases this store's internal pin and clears saving_hashes_. Called once
  // per operation, whichever way it ends.
  void FinishRemoteWrite(const RemoteWriteState& state, bool succeeded,
                         std::vector<std::string> existing,
                         std::vector<std::string> unregistered = {}) {
    OnWriteRemoteVerdict(state, succeeded, std::move(existing),
                         std::move(unregistered));
  }

  // Takes the sweep's settled remote writes. Drives the pollers first for the
  // same reason PollSaveStatus() does: on the monitor's thread nothing else
  // moves an accepted offer to a verdict.
  RemoteWriteVerdicts DrainSweepVerdicts();

  void PollerLoop();
  void PollSavesInternal(std::vector<SaveState> ready_saves);
  void PollLoadsInternal(std::vector<LoadState> ready_loads);
  void PollRemoteReadsInternal(
      std::vector<std::pair<tsl::Future<>, RemoteReadState>>
          ready_remote_reads);
  void PollFuturesInternal();
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_H_
