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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_GLOBAL_REGISTRY_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_GLOBAL_REGISTRY_CLIENT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.grpc.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_sync/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

using namespace ::tpu_raiden::kv_cache::global_registry;  // NOLINT

// Deadline for a registry mutation nobody waits on. Finite so that an
// unanswered call cannot run its continuation after main() has returned, and
// so a stalled write-through gives its pins back.
inline constexpr absl::Duration kUnwaitedMutationTimeout = absl::Seconds(30);

struct Registration {
  std::string prefix_hash;
  RaidenId raiden_id;
  int32_t block_id;
  absl::Duration ttl = absl::ZeroDuration();
};

class GlobalRegistryClient {
 public:
  explicit GlobalRegistryClient(std::shared_ptr<grpc::Channel> channel);

  // Registers a batch of KV cache entries asynchronously. The returned future
  // resolves on a gRPC callback thread with OkStatus, FailedPrecondition when
  // the server rejected the batch, or Internal when the RPC itself failed.
  //
  // `timeout` bounds the call as a gRPC deadline. Beware that zero does NOT
  // mean the same thing here as it does for `Registration::ttl` and
  // `RegisterStore`'s ttl, where zero means "never expires": a `timeout` of
  // zero or less is already expired, and the returned future resolves
  // DeadlineExceeded without the RPC ever being dispatched. That case is
  // decided here rather than left to gRPC, whose behaviour for a deadline of
  // exactly "now" is a race against the round trip -- on loopback the call
  // usually wins. "No deadline" is `absl::InfiniteDuration()`, the default.
  //
  // NOT ORDERED against any other call, including one on the same block hash.
  // A publish and the withdraw that undoes it can reach the server in either
  // order, and the registry keeps whichever landed last. A withdraw that
  // overtakes its publish therefore leaves an entry naming this node for a
  // block hash it no longer holds, which HostOffloadBackend::Lookup treats as
  // unverifiable: it ends the answer there, and no TTL removes the entry. The
  // callers accept that cost -- it costs a peer one refused fetch and this
  // node the tail of one lookup -- in exchange for not serialising registry
  // traffic per block hash.
  tsl::Future<> RegisterAsync(
      const std::vector<Registration>& registrations,
      absl::Duration timeout = absl::InfiniteDuration());

  // Registers a batch of KV cache entries.
  absl::Status Register(
      const std::vector<Registration>& registrations,
      absl::Duration timeout = absl::InfiniteDuration());

  // Looks up KV cache entries for a batch of prefix hashes.
  // The lookup processes prefix hashes sequentially and stops at the first miss
  // (a hash with no active registrations). All subsequent prefix hashes in the
  // input vector are treated as misses and are omitted from the response.
  // The returned vector is aligned in order with the input `prefix_hashes` (the
  // i-th element of the returned vector corresponds to the i-th input hash).
  // The size of the returned vector will be equal to the number of sequential
  // hits before the first miss.
  absl::StatusOr<std::vector<KVBlockMetadata>> Lookup(
      const std::vector<std::string>& prefix_hashes);

  // Unregisters a batch of KV cache entries for a raiden id asynchronously.
  // See RegisterAsync for what the future resolves to, and for the meaning of
  // `timeout` (zero is "expire now", not "never").
  tsl::Future<> UnregisterAsync(
      const std::vector<std::string>& prefix_hashes,
      const RaidenId& raiden_id,
      absl::Duration timeout = absl::InfiniteDuration());

  // Unregisters a batch of KV cache entries for a raiden id.
  absl::Status Unregister(
      const std::vector<std::string>& prefix_hashes,
      const RaidenId& raiden_id,
      absl::Duration timeout = absl::InfiniteDuration());

  // A single active registration returned by PullOwned.
  struct PulledEntry {
    std::string prefix_hash;
    int32_t block_id;
    // Seconds until the registration expires as reported by the server;
    // 0 means it never expires.
    int64_t remaining_ttl_seconds;
  };

  // Pulls all active registrations owned by `raiden_id`, draining the
  // server-side stream. Intended for owner restart handling; see the
  // PullOwned RPC documentation for the consistency contract.
  absl::StatusOr<std::vector<PulledEntry>> PullOwned(const RaidenId& raiden_id);

  // Publishes where peers should reach this store. Re-registering the same
  // `raiden_id` replaces the previous coordinates.
  // `ttl` of zero (the default) means the registration never expires; see
  // StoreInfo.ttl_seconds for why stores differ from block entries here.
  // `kv_pool_group` and `evict_tier` are the store's placement attributes
  // (see StoreInfo); a store that leaves the group empty never serves as a
  // placement target.
  absl::Status RegisterStore(const RaidenId& raiden_id,
                             absl::string_view store_server_address,
                             absl::string_view controller_address = "",
                             absl::Duration ttl = absl::ZeroDuration(),
                             absl::string_view kv_pool_group = "",
                             int32_t evict_tier = 0);

  // Reports this store's mutable status and refreshes its registration TTL.
  // NotFound when the registry holds no live registration for `raiden_id`
  // (e.g. it expired, or the registry restarted): the caller must
  // RegisterStore again -- a heartbeat carries no coordinates and registers
  // nothing.
  absl::Status Heartbeat(const RaidenId& raiden_id, const StoreStatus& status);

  // Placement targets for a pressured store: live stores of `raiden_id`'s
  // registered kv_pool_group on the nearest evict tier greater than its own,
  // in the order offers should be made. Empty when no greater tier exists
  // (the caller is the bottom tier). `max_targets` of 0 (the default) means
  // the server's default cap. NotFound when `raiden_id` has no live store
  // registration: RegisterStore again, as with Heartbeat.
  absl::StatusOr<std::vector<StoreInfo>> GetPlacementTargets(
      const RaidenId& raiden_id, int32_t max_targets = 0);

  // Resolves a peer's store coordinates.
  // Returns NotFound when no live registration exists -- that is an ordinary
  // outcome, not an RPC failure.
  absl::StatusOr<StoreInfo> ResolveStore(const RaidenId& raiden_id);

  // Removes this store's registration. Returns OK when nothing was registered:
  // teardown should not fail because the entry was already gone.
  absl::Status UnregisterStore(const RaidenId& raiden_id);

  // Atomically registers `kv_pool_group`'s KVTransferSpec, or validates it
  // against the group's already-registered one; any difference is
  // InvalidArgument. Idempotent by design. `kv_pool_group` names the set of
  // raiden units sharing one KV transfer geometry, so differently-shaped
  // groups coexist in one registry.
  absl::Status RegisterKVTransferSpec(const KVTransferSpec& spec,
                                      absl::string_view kv_pool_group);

  // Reads `kv_pool_group`'s registered KVTransferSpec. NotFound when the
  // group has published none yet.
  absl::StatusOr<KVTransferSpec> GetKVTransferSpec(
      absl::string_view kv_pool_group);

 private:
  // Shared, not unique, because an in-flight async callback captures it: the
  // generated Stub holds the channel, so one capture keeps channel, transport
  // and EventEngine alive for a call that outlives this client. The cost is
  // that ~GlobalRegistryClient no longer deterministically releases the
  // channel -- the last in-flight callback does.
  std::shared_ptr<GlobalRegistryService::Stub> stub_;
};

// Cumulative hash helper.
// Calculates SHA256 hash of (parent_hash_hex + tokens_binary).
// Returns hex-encoded SHA256 string.
std::string CalculatePrefixHash(const std::vector<int64_t>& tokens,
                                absl::string_view parent_hash = "");

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_GLOBAL_REGISTRY_CLIENT_H_
