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

#include "tpu_sync/weight_sync/swarm_service.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

using ::tpu_sync::rpc::AcquireBundlePullTokenRequest;
using ::tpu_sync::rpc::AcquireBundlePullTokenResponse;
using ::tpu_sync::rpc::RaidenIdProto;
using ::tpu_sync::rpc::RegisterBundleAvailabilityRequest;
using ::tpu_sync::rpc::RegisterBundleAvailabilityResponse;

std::string CanonicalReplicaId(const RaidenIdProto& unit) {
  return absl::StrCat(unit.job_name(), ":", unit.job_replica_id(), ":",
                      unit.data_name(), ":", unit.data_replica_idx());
}

absl::Status UnknownSessionError(absl::string_view req_id, int64_t uuid) {
  return absl::NotFoundError(absl::StrCat(
      "Unknown transfer session for req_id=", req_id, " uuid=", uuid));
}

}  // namespace

absl::Status SwarmService::StartSession(
    absl::string_view req_id, uint64_t uuid, int32_t num_bundles,
    absl::Span<const Participant> participants, const SessionConfig& config) {
  if (req_id.empty()) {
    return absl::InvalidArgumentError("req_id must not be empty");
  }
  absl::MutexLock lock(mu_);
  if (sessions_.contains(req_id)) {
    return absl::OkStatus();
  }

  Session session;
  session.req_id = std::string(req_id);
  session.uuid = uuid;
  session.num_bundles = std::max<int32_t>(0, num_bundles);
  session.config = config;
  if (const char* env_q = std::getenv("RAIDEN_SWARM_VARIABLE_TRANSFER_QUOTA")) {
    int32_t q = 0;
    if (absl::SimpleAtoi(env_q, &q) && q > 0) {
      session.config.target_transfers_per_bundle = q;
    }
  }
  session.config.max_concurrent_uploads_per_source =
      std::max<int32_t>(1, session.config.max_concurrent_uploads_per_source);
  session.config.target_transfers_per_bundle =
      std::max<int32_t>(1, session.config.target_transfers_per_bundle);

  session.replicas.reserve(participants.size());
  for (const Participant& participant : participants) {
    const int32_t idx = static_cast<int32_t>(session.replicas.size());
    Replica replica;
    replica.unit = participant.unit;
    replica.num_hosts = std::max<int32_t>(
        1, static_cast<int32_t>(participant.host_data_endpoints.size()));
    replica.host_data_endpoints = participant.host_data_endpoints;

    const std::string canonical = CanonicalReplicaId(replica.unit);
    session.replica_lookup[canonical] = idx;
    session.replica_lookup[replica.unit.job_replica_id().empty()
                               ? canonical
                               : replica.unit.job_replica_id()] = idx;
    session.total_expected_host_bundles +=
        replica.num_hosts * session.num_bundles;
    session.replicas.push_back(std::move(replica));
  }
  session.complete = session.total_expected_host_bundles == 0;

  if (uuid > 0) {
    uuid_to_req_id_[uuid] = session.req_id;
  }
  sessions_.emplace(session.req_id, std::move(session));
  return absl::OkStatus();
}

void SwarmService::EndSession(absl::string_view req_id) {
  absl::MutexLock lock(mu_);
  auto it = sessions_.find(req_id);
  if (it == sessions_.end()) return;
  for (const std::shared_ptr<PendingRequest>& p : it->second.pending_requests) {
    p->cancelled = true;
  }
  if (it->second.uuid > 0) {
    uuid_to_req_id_.erase(it->second.uuid);
  }
  sessions_.erase(it);
}

SwarmService::Session* SwarmService::FindSessionLocked(absl::string_view req_id,
                                                       uint64_t uuid) {
  if (!req_id.empty()) {
    auto it = sessions_.find(req_id);
    if (it != sessions_.end()) return &it->second;
  }
  if (uuid > 0) {
    auto u_it = uuid_to_req_id_.find(uuid);
    if (u_it != uuid_to_req_id_.end()) {
      auto it = sessions_.find(u_it->second);
      if (it != sessions_.end()) return &it->second;
    }
  }
  return nullptr;
}

int32_t SwarmService::ResolveReplica(const Session& session,
                                     const RaidenIdProto& unit) {
  for (const std::string& key :
       {CanonicalReplicaId(unit), unit.job_name(), unit.job_replica_id()}) {
    if (key.empty()) continue;
    auto it = session.replica_lookup.find(key);
    if (it != session.replica_lookup.end()) return it->second;
  }
  return -1;
}

void SwarmService::ReleaseTokenLocked(Session* session, int32_t dst_replica_idx,
                                      int32_t host_idx, int32_t bundle_index) {
  auto token_it = session->active_tokens.find(BundleHostKey{
      .replica_idx = dst_replica_idx,
      .host_idx = host_idx,
      .bundle_index = bundle_index,
  });
  if (token_it == session->active_tokens.end()) return;

  int32_t& active = session->active_uploads[{token_it->second, host_idx}];
  active = std::max<int32_t>(0, active - 1);
  session->active_tokens.erase(token_it);

  auto dst_it =
      session->active_bundles_by_dst.find({dst_replica_idx, host_idx});
  if (dst_it != session->active_bundles_by_dst.end()) {
    std::vector<int32_t>& bundles = dst_it->second;
    bundles.erase(std::remove(bundles.begin(), bundles.end(), bundle_index),
                  bundles.end());
    if (bundles.empty()) session->active_bundles_by_dst.erase(dst_it);
  }
}

void SwarmService::ReleaseTokenFromSourceLocked(Session* session,
                                                int32_t dst_replica_idx,
                                                int32_t host_idx,
                                                int32_t src_replica_idx) {
  auto dst_it =
      session->active_bundles_by_dst.find({dst_replica_idx, host_idx});
  if (dst_it != session->active_bundles_by_dst.end()) {
    for (int32_t bundle_index : dst_it->second) {
      auto token_it = session->active_tokens.find(BundleHostKey{
          .replica_idx = dst_replica_idx,
          .host_idx = host_idx,
          .bundle_index = bundle_index,
      });
      if (token_it != session->active_tokens.end() &&
          token_it->second == src_replica_idx) {
        // Invalidates `dst_it`; return immediately.
        ReleaseTokenLocked(session, dst_replica_idx, host_idx, bundle_index);
        return;
      }
    }
  }
  // No matching token (e.g. it was already released); still free the
  // source's upload slot so a failed source does not stay saturated.
  int32_t& active = session->active_uploads[{src_replica_idx, host_idx}];
  active = std::max<int32_t>(0, active - 1);
}

void SwarmService::MarkBundleAvailableLocked(Session* session,
                                             int32_t replica_idx,
                                             int32_t host_idx,
                                             int32_t bundle_index) {
  if (replica_idx < 0 ||
      replica_idx >= static_cast<int32_t>(session->replicas.size())) {
    return;
  }
  const bool newly_added = session->available_bundles[{replica_idx, host_idx}]
                               .insert(bundle_index)
                               .second;

  std::vector<int32_t>& sources =
      session->sources_by_bundle[{bundle_index, host_idx}];
  if (std::find(sources.begin(), sources.end(), replica_idx) == sources.end()) {
    sources.push_back(replica_idx);
  }

  if (newly_added && bundle_index >= 0 && bundle_index < session->num_bundles) {
    session->completed_host_bundles += 1;
    if (session->completed_host_bundles >=
        session->total_expected_host_bundles) {
      session->complete = true;
      for (const std::shared_ptr<PendingRequest>& p :
           session->pending_requests) {
        p->cancelled = true;
      }
    }
  }
}

bool SwarmService::TryGrantTokenLocked(
    Session* session, int32_t dst_replica_idx, int32_t host_idx,
    absl::Span<const int32_t> needed_bundles,
    AcquireBundlePullTokenResponse* response) {
  int32_t best_bundle = -1;
  int32_t best_src = -1;
  // (unhealthy, quota_tier, pref_rank, served_by_bundle, total_served).
  using Score = std::tuple<int32_t, int32_t, int32_t, int32_t, int64_t>;
  Score best_score{std::numeric_limits<int32_t>::max(), 0, 0, 0, 0};

  for (int32_t pref_rank = 0;
       pref_rank < static_cast<int32_t>(needed_bundles.size()); ++pref_rank) {
    const int32_t bundle_index = needed_bundles[pref_rank];
    if (session->active_tokens.contains(BundleHostKey{
            .replica_idx = dst_replica_idx,
            .host_idx = host_idx,
            .bundle_index = bundle_index,
        })) {
      continue;
    }
    auto src_it = session->sources_by_bundle.find({bundle_index, host_idx});
    if (src_it == session->sources_by_bundle.end()) continue;

    for (int32_t src : src_it->second) {
      if (src == dst_replica_idx) continue;
      const HostKey src_key{src, host_idx};
      auto active_it = session->active_uploads.find(src_key);
      if (active_it != session->active_uploads.end() &&
          active_it->second >=
              session->config.max_concurrent_uploads_per_source) {
        continue;
      }
      auto served_it = session->served_by_bundle.find(BundleHostKey{
          .replica_idx = src,
          .host_idx = host_idx,
          .bundle_index = bundle_index,
      });
      const int32_t served =
          served_it != session->served_by_bundle.end() ? served_it->second : 0;
      // Tier 0: the source has started but not finished its quota for this
      //         bundle; finish it so the source can move on to the next one.
      // Tier 1: the source has not served this bundle yet.
      // Tier 2: the source has met its quota for this bundle; fallback only.
      int32_t quota_tier = 1;
      if (served >= session->config.target_transfers_per_bundle) {
        quota_tier = 2;
      } else if (served > 0) {
        quota_tier = 0;
      }
      auto total_it = session->total_served.find(src_key);
      const int64_t total =
          total_it != session->total_served.end() ? total_it->second : 0;
      const Score score{session->unhealthy_sources.contains(src_key) ? 1 : 0,
                        quota_tier, pref_rank, served, total};
      if (score < best_score) {
        best_score = score;
        best_bundle = bundle_index;
        best_src = src;
      }
    }
  }
  if (best_src < 0) return false;

  const HostKey src_key{best_src, host_idx};
  session->active_uploads[src_key] += 1;
  session->total_served[src_key] += 1;
  session->served_by_bundle[BundleHostKey{
      .replica_idx = best_src,
      .host_idx = host_idx,
      .bundle_index = best_bundle,
  }] += 1;
  session->active_tokens[BundleHostKey{
      .replica_idx = dst_replica_idx,
      .host_idx = host_idx,
      .bundle_index = best_bundle,
  }] = best_src;
  session->active_bundles_by_dst[{dst_replica_idx, host_idx}].push_back(
      best_bundle);

  const Replica& src = session->replicas[best_src];
  response->set_granted(true);
  response->set_assigned_bundle_index(best_bundle);
  *response->mutable_source_unit() = src.unit;
  if (!src.host_data_endpoints.empty()) {
    const int32_t eff_host = std::min<int32_t>(
        host_idx, static_cast<int32_t>(src.host_data_endpoints.size()) - 1);
    response->set_source_data_endpoint(src.host_data_endpoints[eff_host]);
  }
  LOG(INFO) << "[SwarmService] grant req_id=" << session->req_id
            << " bundle=" << best_bundle << " host=" << host_idx << " dst="
            << CanonicalReplicaId(session->replicas[dst_replica_idx].unit)
            << " src=" << CanonicalReplicaId(src.unit)
            << " src_served=" << session->total_served[src_key];
  return true;
}

void SwarmService::DrainQueueLocked(Session* session) {
  auto it = session->pending_requests.begin();
  while (it != session->pending_requests.end()) {
    PendingRequest& pending = **it;
    if (pending.cancelled || pending.fulfilled) {
      it = session->pending_requests.erase(it);
      continue;
    }
    AcquireBundlePullTokenResponse response;
    if (TryGrantTokenLocked(session, pending.dst_replica_idx, pending.host_idx,
                            pending.request.needed_bundle_indices(),
                            &response)) {
      pending.response = std::move(response);
      pending.fulfilled = true;
      it = session->pending_requests.erase(it);
    } else {
      ++it;
    }
  }
}

absl::StatusOr<RegisterBundleAvailabilityResponse>
SwarmService::RegisterBundleAvailability(
    const RegisterBundleAvailabilityRequest& request) {
  absl::MutexLock lock(mu_);
  Session* session = FindSessionLocked(request.req_id(),
                                       static_cast<uint64_t>(request.uuid()));
  if (session == nullptr) {
    return UnknownSessionError(request.req_id(), request.uuid());
  }
  session->has_worker_activity = true;
  const int32_t replica_idx = ResolveReplica(*session, request.unit());
  if (replica_idx < 0) {
    return absl::NotFoundError(
        absl::StrCat("Unknown unit in session ", session->req_id, ": ",
                     CanonicalReplicaId(request.unit())));
  }
  const int32_t host_idx = std::max<int32_t>(0, request.host_idx());
  LOG(INFO) << "[SwarmService] available req_id=" << session->req_id
            << " bundle=" << request.bundle_index() << " host=" << host_idx
            << " unit=" << CanonicalReplicaId(request.unit());

  ReleaseTokenLocked(session, replica_idx, host_idx, request.bundle_index());
  MarkBundleAvailableLocked(session, replica_idx, host_idx,
                            request.bundle_index());
  DrainQueueLocked(session);

  RegisterBundleAvailabilityResponse response;
  response.set_acknowledged(true);

  const absl::btree_set<int32_t>& held =
      session->available_bundles[{replica_idx, host_idx}];
  if (static_cast<int32_t>(held.size()) >= session->num_bundles ||
      (!request.request_next_bundle() &&
       request.needed_bundle_indices().empty())) {
    return response;
  }

  std::vector<int32_t> needed;
  if (!request.needed_bundle_indices().empty()) {
    for (int32_t b : request.needed_bundle_indices()) {
      if (!held.contains(b)) needed.push_back(b);
    }
  } else {
    for (int32_t b = 0; b < session->num_bundles; ++b) {
      if (!held.contains(b)) needed.push_back(b);
    }
  }
  if (!needed.empty()) {
    AcquireBundlePullTokenResponse* next = response.mutable_next_pull_token();
    if (!TryGrantTokenLocked(session, replica_idx, host_idx, needed, next)) {
      next->set_granted(false);
    }
  }
  return response;
}

absl::StatusOr<AcquireBundlePullTokenResponse>
SwarmService::AcquireBundlePullToken(
    const AcquireBundlePullTokenRequest& request) {
  absl::MutexLock lock(mu_);
  Session* session = FindSessionLocked(request.req_id(),
                                       static_cast<uint64_t>(request.uuid()));
  if (session == nullptr) {
    return UnknownSessionError(request.req_id(), request.uuid());
  }
  session->has_worker_activity = true;
  const int32_t dst_replica_idx = ResolveReplica(*session, request.dst_unit());
  if (dst_replica_idx < 0) {
    return absl::NotFoundError(
        absl::StrCat("Unknown destination unit in session ", session->req_id,
                     ": ", CanonicalReplicaId(request.dst_unit())));
  }
  const int32_t host_idx = std::max<int32_t>(0, request.host_idx());

  if (!request.failed_source_replica_id().empty()) {
    auto f_it =
        session->replica_lookup.find(request.failed_source_replica_id());
    if (f_it != session->replica_lookup.end()) {
      ReleaseTokenFromSourceLocked(session, dst_replica_idx, host_idx,
                                   f_it->second);
      DrainQueueLocked(session);
      session->unhealthy_sources.insert({f_it->second, host_idx});
    }
  }

  AcquireBundlePullTokenResponse response;
  response.set_granted(false);
  if (request.needed_bundle_indices().empty()) {
    return response;
  }

  // Only wait when this host has nothing in flight; a prefetch from a host that
  // already holds a token must not stall its in-flight pull.
  const bool has_active_token =
      session->active_bundles_by_dst.contains({dst_replica_idx, host_idx});
  const absl::Duration timeout =
      has_active_token ? absl::ZeroDuration() : session->config.queue_timeout;

  if (session->pending_requests.empty()) {
    if (TryGrantTokenLocked(session, dst_replica_idx, host_idx,
                            request.needed_bundle_indices(), &response) ||
        timeout <= absl::ZeroDuration()) {
      return response;
    }
  }

  // A newer request from the same host supersedes any older queued one.
  for (const std::shared_ptr<PendingRequest>& p : session->pending_requests) {
    if (p->dst_replica_idx == dst_replica_idx && p->host_idx == host_idx) {
      p->cancelled = true;
    }
  }
  auto pending = std::make_shared<PendingRequest>();
  pending->dst_replica_idx = dst_replica_idx;
  pending->host_idx = host_idx;
  pending->request = request;
  session->pending_requests.push_back(pending);
  DrainQueueLocked(session);

  if (!pending->fulfilled && timeout > absl::ZeroDuration()) {
    auto done = [&pending]() {
      return pending->fulfilled || pending->cancelled;
    };
    mu_.AwaitWithTimeout(absl::Condition(&done), timeout);
  }
  if (pending->fulfilled) {
    return pending->response;
  }

  // Timed out or cancelled; the session may have ended while waiting.
  session = FindSessionLocked(request.req_id(),
                              static_cast<uint64_t>(request.uuid()));
  if (session != nullptr) {
    auto rem_it = std::find(session->pending_requests.begin(),
                            session->pending_requests.end(), pending);
    if (rem_it != session->pending_requests.end()) {
      session->pending_requests.erase(rem_it);
    }
  }
  return response;
}

absl::Status SwarmService::WaitForSessionComplete(absl::string_view req_id,
                                                  absl::Duration timeout) {
  absl::MutexLock lock(mu_);
  auto done = [this, req_id]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = sessions_.find(req_id);
    return it == sessions_.end() || it->second.complete;
  };
  if (!mu_.AwaitWithTimeout(absl::Condition(&done), timeout)) {
    return absl::DeadlineExceededError(absl::StrCat(
        "Timed out waiting for all samplers to receive every bundle for ",
        req_id));
  }
  return absl::OkStatus();
}

std::optional<SwarmService::SessionSnapshot> SwarmService::GetSessionSnapshot(
    absl::string_view req_id) const {
  absl::MutexLock lock(mu_);
  auto it = sessions_.find(req_id);
  if (it == sessions_.end()) return std::nullopt;
  const Session& session = it->second;

  SessionSnapshot snapshot;
  snapshot.complete = session.complete;
  snapshot.has_worker_activity = session.has_worker_activity;
  snapshot.queued_requests =
      static_cast<int32_t>(session.pending_requests.size());
  for (int32_t r = 0; r < static_cast<int32_t>(session.replicas.size()); ++r) {
    const Replica& replica = session.replicas[r];
    for (int32_t h = 0; h < replica.num_hosts; ++h) {
      HostSnapshot host;
      host.unit = replica.unit;
      host.host_idx = h;
      const HostKey key{r, h};
      if (auto a_it = session.available_bundles.find(key);
          a_it != session.available_bundles.end()) {
        host.available_bundles.assign(a_it->second.begin(), a_it->second.end());
      }
      if (auto u_it = session.active_uploads.find(key);
          u_it != session.active_uploads.end()) {
        host.active_uploads = u_it->second;
      }
      if (auto t_it = session.total_served.find(key);
          t_it != session.total_served.end()) {
        host.total_served = t_it->second;
      }
      snapshot.hosts.push_back(std::move(host));
    }
  }
  for (const auto& [key, served] : session.served_by_bundle) {
    const Replica& replica = session.replicas[key.replica_idx];
    if (key.host_idx >= replica.num_hosts) continue;
    // Hosts are laid out replica-major in `snapshot.hosts`.
    int32_t offset = 0;
    for (int32_t r = 0; r < key.replica_idx; ++r) {
      offset += session.replicas[r].num_hosts;
    }
    snapshot.hosts[offset + key.host_idx].served_by_bundle[key.bundle_index] =
        served;
  }
  return snapshot;
}

}  // namespace weight_sync
}  // namespace tpu_raiden
