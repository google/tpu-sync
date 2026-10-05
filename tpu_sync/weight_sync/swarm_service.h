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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_SWARM_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_SWARM_SERVICE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {

// Coordinates peer-to-peer bundle pulls among the samplers of one transfer.
//
// Weights are split into bundles with contiguous indices [0, num_bundles). The
// trainer pushes every bundle to a few seed samplers; every other sampler host
// pulls each bundle from a peer host that already holds it. For each transfer
// session, SwarmService tracks which (sampler, host) holds which bundles and
// which source is currently uploading to whom, and serves two requests:
//
//  * `RegisterBundleAvailability`: a host now holds a bundle and can serve it.
//    Releases the host's pull token for that bundle, if any.
//  * `AcquireBundlePullToken`: assigns a (bundle, source host) to a requesting
//    host. Sources are limited to `max_concurrent_uploads_per_source`
//    concurrent uploads. A requester holding no token waits up to
//    `queue_timeout` for a free source; a requester that already holds a token
//    (i.e. a prefetch) never waits.
//
// Host `h` of a sampler only pulls from host `h` of other samplers.
//
// Thread-safety: all public methods are thread-safe.
class SwarmService {
 public:
  // A sampler taking part in a session.
  struct Participant {
    tpu_sync::rpc::RaidenIdProto unit;
    // Data endpoint of each host of `unit`, indexed by host index. May be
    // empty for participants that never serve as a source.
    std::vector<std::string> host_data_endpoints;
  };

  struct SessionConfig {
    int32_t max_concurrent_uploads_per_source = 1;
    // Number of times each source should serve a bundle before other bundles
    // it holds are preferred. Overridden by the
    // RAIDEN_SWARM_VARIABLE_TRANSFER_QUOTA environment variable, if set.
    int32_t target_transfers_per_bundle = 2;
    // Maximum time a requester with no active pull token waits for a source.
    absl::Duration queue_timeout = absl::Seconds(20);
  };

  // Read-only view of one (sampler, host) in a session.
  struct HostSnapshot {
    tpu_sync::rpc::RaidenIdProto unit;
    int32_t host_idx = 0;
    // Bundles this host holds, in ascending order.
    std::vector<int32_t> available_bundles;
    // Uploads this host is currently serving.
    int32_t active_uploads = 0;
    // Pull tokens granted with this host as the source, in total and per
    // bundle index.
    int64_t total_served = 0;
    absl::flat_hash_map<int32_t, int32_t> served_by_bundle;
  };

  // Read-only view of a session, for monitoring and tests.
  struct SessionSnapshot {
    // True once every host of every participant holds every bundle.
    bool complete = false;
    // True once any host has called `RegisterBundleAvailability` or
    // `AcquireBundlePullToken` for this session.
    bool has_worker_activity = false;
    // Number of `AcquireBundlePullToken` requests waiting for a source.
    int32_t queued_requests = 0;
    std::vector<HostSnapshot> hosts;
  };

  SwarmService() = default;
  SwarmService(const SwarmService&) = delete;
  SwarmService& operator=(const SwarmService&) = delete;

  // Starts tracking a transfer of |num_bundles| bundles among |participants|.
  // No-op if a session for |req_id| already exists.
  absl::Status StartSession(absl::string_view req_id, uint64_t uuid,
                            int32_t num_bundles,
                            absl::Span<const Participant> participants,
                            const SessionConfig& config);

  // Drops the session for |req_id| and fails its queued requests.
  void EndSession(absl::string_view req_id);

  // TODO(justinlu): Add an API (e.g. `ReleaseTokensForReplica`) so the
  // controller can reclaim all tokens held by, or sourced from, a sampler it
  // has detected as dead, instead of relying only on peers reporting the
  // failure (see also the token-expiry TODO on `Session::active_tokens`).

  // Handles `COMMAND_REGISTER_BUNDLE_AVAILABILITY`. If requested, also tries to
  // grant the next pull token without waiting.
  absl::StatusOr<tpu_sync::rpc::RegisterBundleAvailabilityResponse>
  RegisterBundleAvailability(
      const tpu_sync::rpc::RegisterBundleAvailabilityRequest& request);

  // Handles `COMMAND_ACQUIRE_BUNDLE_PULL_TOKEN`. May block for up to the
  // session's `queue_timeout`.
  absl::StatusOr<tpu_sync::rpc::AcquireBundlePullTokenResponse>
  AcquireBundlePullToken(
      const tpu_sync::rpc::AcquireBundlePullTokenRequest& request);

  // Blocks until the session for |req_id| is complete or ended, or until
  // |timeout| expires.
  absl::Status WaitForSessionComplete(absl::string_view req_id,
                                      absl::Duration timeout);

  // Returns a snapshot of the session for |req_id|, or nullopt if unknown.
  std::optional<SessionSnapshot> GetSessionSnapshot(
      absl::string_view req_id) const;

 private:
  // Key for per-host state: (replica_idx, host_idx).
  using HostKey = std::pair<int32_t, int32_t>;

  struct BundleHostKey {
    int32_t replica_idx = -1;
    int32_t host_idx = 0;
    int32_t bundle_index = -1;

    bool operator==(const BundleHostKey& other) const {
      return replica_idx == other.replica_idx && host_idx == other.host_idx &&
             bundle_index == other.bundle_index;
    }
    template <typename H>
    friend H AbslHashValue(H h, const BundleHostKey& k) {
      return H::combine(std::move(h), k.replica_idx, k.host_idx,
                        k.bundle_index);
    }
  };

  struct Replica {
    tpu_sync::rpc::RaidenIdProto unit;
    int32_t num_hosts = 1;
    std::vector<std::string> host_data_endpoints;
  };

  struct PendingRequest {
    int32_t dst_replica_idx = -1;
    int32_t host_idx = 0;
    tpu_sync::rpc::AcquireBundlePullTokenRequest request;
    bool fulfilled = false;
    bool cancelled = false;
    tpu_sync::rpc::AcquireBundlePullTokenResponse response;
  };

  struct Session {
    std::string req_id;
    uint64_t uuid = 0;
    int32_t num_bundles = 0;
    SessionConfig config;

    std::vector<Replica> replicas;
    // Maps replica identifiers (canonical id, job name, job replica id) to
    // indices into `replicas`.
    absl::flat_hash_map<std::string, int32_t> replica_lookup;

    absl::flat_hash_map<HostKey, absl::btree_set<int32_t>> available_bundles;
    // (bundle_index, host_idx) -> source replicas, in registration order.
    absl::flat_hash_map<std::pair<int32_t, int32_t>, std::vector<int32_t>>
        sources_by_bundle;
    absl::flat_hash_map<HostKey, int32_t> active_uploads;
    absl::flat_hash_map<HostKey, int64_t> total_served;
    absl::flat_hash_map<BundleHostKey, int32_t> served_by_bundle;
    // TODO(justinlu): A source reported as failed by any destination is
    // deprioritized for all destinations for the rest of the session, even if
    // the failure was on the destination's side. Track failures per
    // destination, or clear the mark once the source completes a transfer.
    absl::flat_hash_set<HostKey> unhealthy_sources;

    // Outstanding pull tokens: (dst_replica, host, bundle) -> source replica.
    // TODO(justinlu): Tokens never expire. If a destination dies or gives up
    // while holding a token, the source's upload slot stays occupied until the
    // session ends (with one slot per source, that source can no longer
    // serve). Record the grant time and reclaim tokens older than a
    // configurable timeout in `AcquireBundlePullToken` / `DrainQueueLocked`.
    absl::flat_hash_map<BundleHostKey, int32_t> active_tokens;
    // (dst_replica, host) -> bundles with outstanding tokens, in grant order.
    absl::flat_hash_map<HostKey, std::vector<int32_t>> active_bundles_by_dst;

    // FIFO queue of requests waiting for a free source.
    std::vector<std::shared_ptr<PendingRequest>> pending_requests;

    int32_t total_expected_host_bundles = 0;
    int32_t completed_host_bundles = 0;
    bool has_worker_activity = false;
    bool complete = false;
  };

  Session* FindSessionLocked(absl::string_view req_id, uint64_t uuid)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  static int32_t ResolveReplica(const Session& session,
                                const tpu_sync::rpc::RaidenIdProto& unit);
  static void ReleaseTokenLocked(Session* session, int32_t dst_replica_idx,
                                 int32_t host_idx, int32_t bundle_index);
  static void ReleaseTokenFromSourceLocked(Session* session,
                                           int32_t dst_replica_idx,
                                           int32_t host_idx,
                                           int32_t src_replica_idx);
  static void MarkBundleAvailableLocked(Session* session, int32_t replica_idx,
                                        int32_t host_idx, int32_t bundle_index);
  static bool TryGrantTokenLocked(
      Session* session, int32_t dst_replica_idx, int32_t host_idx,
      absl::Span<const int32_t> needed_bundles,
      tpu_sync::rpc::AcquireBundlePullTokenResponse* response);
  static void DrainQueueLocked(Session* session);

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, Session> sessions_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<uint64_t, std::string> uuid_to_req_id_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_SWARM_SERVICE_H_
