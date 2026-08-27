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

#include "tpu_sync/kv_cache/global_registry/global_registry_server.h"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_sync/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

using namespace ::tpu_raiden::kv_cache::global_registry;  // NOLINT

namespace {

RaidenId FromProto(const ::tpu_sync::rpc::RaidenIdProto& proto) {
  return RaidenId{
      .job_name = proto.job_name(),
      .job_replica_id = proto.job_replica_id(),
      .data_name = proto.data_name(),
      .data_replica_idx = proto.data_replica_idx(),
  };
}

void ToProto(const RaidenId& id, ::tpu_sync::rpc::RaidenIdProto* proto) {
  proto->set_job_name(id.job_name);
  proto->set_job_replica_id(id.job_replica_id);
  proto->set_data_name(id.data_name);
  proto->set_data_replica_idx(id.data_replica_idx);
}

}  // namespace

GlobalRegistryServiceImpl::GlobalRegistryServiceImpl(
    absl::Duration default_ttl, absl::Duration cleanup_interval,
    int64_t pull_owned_batch_size)
    : default_ttl_(default_ttl),
      cleanup_interval_(cleanup_interval),
      pull_owned_batch_size_(pull_owned_batch_size > 0
                                 ? pull_owned_batch_size
                                 : kDefaultPullOwnedBatchSize) {
  StartCleanupThread();
}

GlobalRegistryServiceImpl::~GlobalRegistryServiceImpl() { StopCleanupThread(); }

void GlobalRegistryServiceImpl::StartCleanupThread() {
  if (cleanup_interval_ > absl::ZeroDuration()) {
    cleanup_thread_ =
        std::thread(&GlobalRegistryServiceImpl::CleanupLoop, this);
  }
}

void GlobalRegistryServiceImpl::StopCleanupThread() {
  shutdown_ = true;
  if (cleanup_thread_.joinable()) {
    cleanup_thread_.join();
  }
}

void GlobalRegistryServiceImpl::CleanupLoop() {
  while (!shutdown_) {
    // Sleep in small increments to respond to shutdown quickly
    absl::Time start = absl::Now();
    while (absl::Now() - start < cleanup_interval_ && !shutdown_) {
      absl::SleepFor(absl::Milliseconds(100));
    }
    if (shutdown_) break;
    CleanupExpiredEntries();
  }
}

grpc::Status GlobalRegistryServiceImpl::Register(grpc::ServerContext* context,
                                                 const RegisterRequest* request,
                                                 RegisterResponse* response) {
  absl::MutexLock lock(mutex_);
  if (context != nullptr && context->IsCancelled()) {
    return grpc::Status(grpc::StatusCode::CANCELLED, "Call was cancelled");
  }

  // 1. Fail-fast validation
  for (const auto& entry : request->entries()) {
    if (entry.prefix_hash().empty()) {
      response->set_success(false);
      response->set_error_message("prefix_hash cannot be empty");
      return grpc::Status::OK;
    }
    if (!entry.has_metadata() || !entry.metadata().has_raiden_id() ||
        entry.metadata().raiden_id().job_name().empty()) {
      response->set_success(false);
      response->set_error_message(
          "metadata.raiden_id.job_name cannot be empty");
      return grpc::Status::OK;
    }
  }

  // 2. Apply updates
  absl::Time now = absl::Now();
  for (const auto& entry : request->entries()) {
    const std::string& prefix_hash = entry.prefix_hash();
    const KVBlockMetadata& meta = entry.metadata();
    RaidenId raiden_id = FromProto(meta.raiden_id());

    absl::Duration ttl = entry.ttl_seconds() > 0
                             ? absl::Seconds(entry.ttl_seconds())
                             : default_ttl_;
    absl::Time expire_time = now + ttl;
    int32_t block_id = meta.block_id();

    auto& entries = registry_[prefix_hash];
    bool found = false;
    for (auto& existing : entries) {
      if (existing.raiden_id == raiden_id) {
        existing.block_id = block_id;
        existing.expire_time = expire_time;
        found = true;
        break;
      }
    }
    if (!found) {
      entries.push_back({raiden_id, block_id, expire_time});
      // Updated together with registry_ under the same critical section; both
      // are in-memory only, so a crash discards them together. For replica
      // support, treat owner_index_ as derived data.
      owner_index_[raiden_id].insert(prefix_hash);
    }
  }

  response->set_success(true);
  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::Lookup(grpc::ServerContext* context,
                                               const LookupRequest* request,
                                               LookupResponse* response) {
  const RaidenId caller = FromProto(request->client_raiden_id());
  const bool filter_caller = !caller.empty();

  absl::MutexLock lock(mutex_);
  absl::Time now = absl::Now();

  for (const std::string& hash : request->prefix_hashes()) {
    auto it = registry_.find(hash);
    if (it == registry_.end()) {
      break;
    }

    auto& entries = it->second;
    for (const auto& entry : entries) {
      if (entry.expire_time <= now ||
          (filter_caller && entry.raiden_id == caller)) {
        EraseFromOwnerIndex(entry.raiden_id, hash);
      }
    }
    entries.erase(
        std::remove_if(entries.begin(), entries.end(),
                       [now, filter_caller, &caller](const RegistryEntry& entry) {
                         return entry.expire_time <= now ||
                                (filter_caller && entry.raiden_id == caller);
                       }),
        entries.end());

    if (entries.empty()) {
      registry_.erase(it);
      round_robin_indices_.erase(hash);
      break;
    }

    // Round-robin selection
    size_t& idx = round_robin_indices_[hash];
    idx = idx % entries.size();
    const auto& picked = entries[idx];
    idx++;  // Increment for next lookup

    auto* meta = response->add_results();
    ToProto(picked.raiden_id, meta->mutable_raiden_id());
    meta->set_block_id(picked.block_id);
  }

  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::Unregister(
    grpc::ServerContext* context, const UnregisterRequest* request,
    UnregisterResponse* response) {
  absl::MutexLock lock(mutex_);
  if (context != nullptr && context->IsCancelled()) {
    return grpc::Status(grpc::StatusCode::CANCELLED, "Call was cancelled");
  }

  if (!request->has_raiden_id() || request->raiden_id().job_name().empty()) {
    response->set_success(false);
    response->set_error_message("raiden_id cannot be empty");
    return grpc::Status::OK;
  }
  RaidenId raiden_id = FromProto(request->raiden_id());

  bool overall_success = true;
  std::vector<std::string> errors;

  for (const std::string& hash : request->prefix_hashes()) {
    if (hash.empty()) continue;

    auto it = registry_.find(hash);
    if (it == registry_.end()) {
      // Already gone, safe to ignore for eviction.
      continue;
    }

    auto& entries = it->second;
    auto orig_size = entries.size();
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [&raiden_id](const RegistryEntry& entry) {
                                   return entry.raiden_id == raiden_id;
                                 }),
                  entries.end());

    if (entries.size() == orig_size) {
      overall_success = false;
      errors.push_back(absl::StrCat(absl::BytesToHexString(hash),
                                    ": raiden_id mismatch or not found"));
    } else {
      EraseFromOwnerIndex(raiden_id, hash);
    }

    if (entries.empty()) {
      registry_.erase(it);
      round_robin_indices_.erase(hash);
    }
  }

  response->set_success(overall_success);
  if (!errors.empty()) {
    std::string error_msg;
    for (const auto& err : errors) {
      if (!error_msg.empty()) error_msg += "; ";
      error_msg += err;
    }
    response->set_error_message(error_msg);
  }

  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::PullOwned(
    grpc::ServerContext* context, const PullOwnedRequest* request,
    grpc::ServerWriter<PullOwnedResponse>* writer) {
  if (!request->has_raiden_id() || request->raiden_id().job_name().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "raiden_id cannot be empty");
  }
  RaidenId raiden_id = FromProto(request->raiden_id());

  // Snapshot the owner's hash set in one short critical section. Every batch
  // below is re-validated against `registry_`, so entries removed after this
  // snapshot are simply omitted from the stream.
  std::vector<std::string> hashes;
  {
    absl::MutexLock lock(mutex_);
    auto it = owner_index_.find(raiden_id);
    if (it == owner_index_.end()) {
      return grpc::Status::OK;
    }
    hashes.assign(it->second.begin(), it->second.end());
  }

  const size_t batch_size = static_cast<size_t>(pull_owned_batch_size_);
  for (size_t start = 0; start < hashes.size(); start += batch_size) {
    const size_t end = std::min(hashes.size(), start + batch_size);
    PullOwnedResponse response;
    {
      absl::MutexLock lock(mutex_);
      absl::Time now = absl::Now();
      for (size_t i = start; i < end; ++i) {
        auto it = registry_.find(hashes[i]);
        if (it == registry_.end()) continue;
        for (const RegistryEntry& entry : it->second) {
          if (!(entry.raiden_id == raiden_id)) continue;
          // Only reachable for this owner's entry, and Register keeps at most
          // one entry per (hash, owner): an expired match means there is
          // nothing to emit for this hash.
          if (entry.expire_time <= now) break;
          PullOwnedEntry* out = response.add_entries();
          out->set_prefix_hash(hashes[i]);
          out->set_block_id(entry.block_id);
          if (entry.expire_time == absl::InfiniteFuture()) {
            out->set_remaining_ttl_seconds(0);
          } else {
            absl::Duration remaining = entry.expire_time - now;
            int64_t seconds = absl::ToInt64Seconds(remaining);
            if (absl::Seconds(seconds) < remaining) ++seconds;
            out->set_remaining_ttl_seconds(std::max<int64_t>(seconds, 1));
          }
          break;  // Register keeps at most one entry per (hash, owner).
        }
      }
    }
    // Write outside the lock: a slow client must not stall the registry.
    if (response.entries_size() > 0 && !writer->Write(response)) {
      return grpc::Status(grpc::StatusCode::CANCELLED,
                          "PullOwned stream closed by client");
    }
  }
  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::RegisterStore(
    grpc::ServerContext* context, const RegisterStoreRequest* request,
    RegisterStoreResponse* response) {
  if (!request->has_store() || !request->store().has_raiden_id() ||
      request->store().raiden_id().job_name().empty()) {
    response->set_success(false);
    response->set_error_message("store.raiden_id.job_name cannot be empty");
    return grpc::Status::OK;
  }
  if (request->store().store_server_address().empty()) {
    response->set_success(false);
    response->set_error_message("store.store_server_address cannot be empty");
    return grpc::Status::OK;
  }

  const StoreInfo& store = request->store();
  // Non-positive ttl means "never expires" for stores -- see StoreInfo.
  const absl::Duration ttl = store.ttl_seconds() > 0
                                 ? absl::Seconds(store.ttl_seconds())
                                 : absl::InfiniteDuration();
  // Adding InfiniteDuration saturates to InfiniteFuture.
  const absl::Time expire_time = absl::Now() + ttl;

  absl::MutexLock lock(mutex_);
  // Assignment, not insertion: re-registering the same RaidenId replaces the
  // previous coordinates, which is how a restarted store heals its own entry.
  store_registry_[FromProto(store.raiden_id())] = StoreRecord{
      .store_server_address = store.store_server_address(),
      .controller_address = store.controller_address(),
      .expire_time = expire_time,
      .ttl = ttl,
      .kv_pool_group = store.kv_pool_group(),
      .evict_tier = store.evict_tier(),
  };

  response->set_success(true);
  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::ResolveStore(
    grpc::ServerContext* context, const ResolveStoreRequest* request,
    ResolveStoreResponse* response) {
  if (!request->has_raiden_id() || request->raiden_id().job_name().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "raiden_id cannot be empty");
  }

  const RaidenId raiden_id = FromProto(request->raiden_id());

  absl::MutexLock lock(mutex_);
  auto it = store_registry_.find(raiden_id);
  // An expired record is a miss, and is left for the cleanup thread to reap.
  if (it == store_registry_.end() || it->second.expire_time <= absl::Now()) {
    response->set_found(false);
    return grpc::Status::OK;
  }

  response->set_found(true);
  StoreInfo* out = response->mutable_store();
  ToProto(raiden_id, out->mutable_raiden_id());
  out->set_store_server_address(it->second.store_server_address);
  out->set_controller_address(it->second.controller_address);
  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::UnregisterStore(
    grpc::ServerContext* context, const UnregisterStoreRequest* request,
    UnregisterStoreResponse* response) {
  if (!request->has_raiden_id() || request->raiden_id().job_name().empty()) {
    response->set_success(false);
    response->set_error_message("raiden_id cannot be empty");
    return grpc::Status::OK;
  }

  absl::MutexLock lock(mutex_);
  const RaidenId raiden_id = FromProto(request->raiden_id());
  const size_t erased = store_registry_.erase(raiden_id);
  if (erased > 0) {
    // The address just became undialable, so the block entries this store
    // owns are unreadable; they leave with the registration. An owner that
    // never registered a store is not affected: erased == 0 skips the purge.
    PurgeOwnedEntries(raiden_id);
  }
  response->set_success(erased > 0);
  if (erased == 0) {
    response->set_error_message("no store registered for that raiden_id");
  }
  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::Heartbeat(
    grpc::ServerContext* context, const HeartbeatRequest* request,
    HeartbeatResponse* response) {
  if (!request->has_raiden_id() || request->raiden_id().job_name().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "raiden_id cannot be empty");
  }

  absl::MutexLock lock(mutex_);
  auto it = store_registry_.find(FromProto(request->raiden_id()));
  // An expired record counts as already deleted, even before the cleanup
  // thread removes it: the heartbeat must not extend it, the store has to
  // re-register with its full coordinates.
  if (it == store_registry_.end() || it->second.expire_time <= absl::Now()) {
    response->set_registered(false);
    return grpc::Status::OK;
  }

  it->second.status = request->status();
  if (it->second.ttl != absl::InfiniteDuration()) {
    it->second.expire_time = absl::Now() + it->second.ttl;
  }
  response->set_registered(true);
  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::GetPlacementTargets(
    grpc::ServerContext* context, const GetPlacementTargetsRequest* request,
    GetPlacementTargetsResponse* response) {
  if (!request->has_raiden_id() || request->raiden_id().job_name().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "raiden_id cannot be empty");
  }
  const int32_t max_targets = request->max_targets() > 0
                                  ? request->max_targets()
                                  : kDefaultMaxPlacementTargets;

  const absl::Time now = absl::Now();
  absl::MutexLock lock(mutex_);

  // The caller's own registration supplies its group and tier. Same rule as
  // Heartbeat: an expired record counts as already deleted.
  auto caller_it = store_registry_.find(FromProto(request->raiden_id()));
  if (caller_it == store_registry_.end() ||
      caller_it->second.expire_time <= now) {
    return grpc::Status(
        grpc::StatusCode::NOT_FOUND,
        "no live store registration for the caller; RegisterStore again");
  }
  if (caller_it->second.kv_pool_group.empty()) {
    return grpc::Status(
        grpc::StatusCode::FAILED_PRECONDITION,
        "caller registered without a kv_pool_group, so it has no peers to "
        "place onto");
  }
  const std::string& group = caller_it->second.kv_pool_group;
  const int32_t caller_tier = caller_it->second.evict_tier;

  // One scan collects the group's live greater-tier members and the nearest
  // such tier; only that tier's members stay. No per-group index: a linear
  // scan over the whole roster (hundreds of entries, read once per sweep
  // period per pressured store) is acceptable today. If it stops being so,
  // maintain a kv_pool_group -> raiden_id map here and scan only the
  // caller's group.
  struct Candidate {
    const RaidenId* raiden_id;
    const StoreRecord* record;
  };
  std::vector<Candidate> candidates;
  int32_t target_tier = std::numeric_limits<int32_t>::max();
  for (const auto& [raiden_id, record] : store_registry_) {
    if (record.kv_pool_group == group && record.evict_tier > caller_tier &&
        record.expire_time > now) {
      target_tier = std::min(target_tier, record.evict_tier);
      candidates.push_back({&raiden_id, &record});
    }
  }
  std::erase_if(candidates, [target_tier](const Candidate& c) {
    return c.record->evict_tier != target_tier;
  });

  // Descending reported free capacity; stores reporting none sort last.
  // Spreading load across destinations currently comes from kv_pool_group
  // partitioning alone: each group has its own top-K, so no single global top-K
  // is handed to every caller. If one group's callers still converge too hard
  // on its fullest-free stores, replace this sort with a draw weighted by free
  // capacity (e.g. Efraimidis-Spirakis sampling without replacement).
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.record->status.free_blocks() >
                     b.record->status.free_blocks();
            });

  for (size_t i = 0;
       i < candidates.size() && response->targets_size() < max_targets; ++i) {
    StoreInfo* out = response->add_targets();
    ToProto(*candidates[i].raiden_id, out->mutable_raiden_id());
    out->set_store_server_address(candidates[i].record->store_server_address);
    out->set_controller_address(candidates[i].record->controller_address);
  }
  return grpc::Status::OK;
}

namespace {

// Returns an empty string when `spec` is structurally valid, else the reason
// it is not.
std::string SpecValidationError(const KVTransferSpec& spec) {
  if (spec.block_arrays().empty()) {
    return "spec must declare at least one block array";
  }
  for (int i = 0; i < spec.block_arrays_size(); ++i) {
    if (spec.block_arrays(i).block_bytes() <= 0) {
      return absl::StrCat("block_arrays[", i, "].block_bytes must be positive");
    }
  }
  if (spec.num_kv_shards() <= 0) {
    return "num_kv_shards must be positive";
  }
  if (spec.num_workers() <= 0) {
    return "num_workers must be positive";
  }
  return "";
}

// Returns an empty string when the two specs are identical, else a
// description of the first difference, named from `incoming`'s perspective.
std::string SpecDifference(const KVTransferSpec& registered,
                           const KVTransferSpec& incoming) {
  if (incoming.block_arrays_size() != registered.block_arrays_size()) {
    return absl::StrCat("block_arrays has ", incoming.block_arrays_size(),
                        " entries, registered spec has ",
                        registered.block_arrays_size());
  }
  for (int i = 0; i < incoming.block_arrays_size(); ++i) {
    if (incoming.block_arrays(i).block_bytes() !=
        registered.block_arrays(i).block_bytes()) {
      return absl::StrCat("block_arrays[", i, "].block_bytes is ",
                          incoming.block_arrays(i).block_bytes(),
                          ", registered spec has ",
                          registered.block_arrays(i).block_bytes());
    }
  }
  if (incoming.num_kv_shards() != registered.num_kv_shards()) {
    return absl::StrCat("num_kv_shards is ", incoming.num_kv_shards(),
                        ", registered spec has ", registered.num_kv_shards());
  }
  if (incoming.num_workers() != registered.num_workers()) {
    return absl::StrCat("num_workers is ", incoming.num_workers(),
                        ", registered spec has ", registered.num_workers());
  }
  return "";
}

}  // namespace

grpc::Status GlobalRegistryServiceImpl::RegisterKVTransferSpec(
    grpc::ServerContext* context, const RegisterKVTransferSpecRequest* request,
    RegisterKVTransferSpecResponse* response) {
  if (!request->has_spec()) {
    response->set_success(false);
    response->set_error_message("spec cannot be empty");
    return grpc::Status::OK;
  }
  if (request->kv_pool_group().empty()) {
    response->set_success(false);
    response->set_error_message("kv_pool_group cannot be empty");
    return grpc::Status::OK;
  }
  const std::string invalid = SpecValidationError(request->spec());
  if (!invalid.empty()) {
    response->set_success(false);
    response->set_error_message(invalid);
    return grpc::Status::OK;
  }

  absl::MutexLock lock(mutex_);
  auto [it, inserted] =
      transfer_specs_.try_emplace(request->kv_pool_group(), request->spec());
  if (inserted) {
    response->set_success(true);
    return grpc::Status::OK;
  }
  const std::string diff = SpecDifference(it->second, request->spec());
  if (diff.empty()) {
    // Identical republish: the idempotent restart path.
    response->set_success(true);
    return grpc::Status::OK;
  }
  response->set_success(false);
  response->set_error_message(
      absl::StrCat("KVTransferSpec mismatch in kv_pool_group '",
                   request->kv_pool_group(), "': ", diff));
  *response->mutable_registered_spec() = it->second;
  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::GetKVTransferSpec(
    grpc::ServerContext* context, const GetKVTransferSpecRequest* request,
    GetKVTransferSpecResponse* response) {
  if (request->kv_pool_group().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "kv_pool_group cannot be empty");
  }
  absl::MutexLock lock(mutex_);
  auto it = transfer_specs_.find(request->kv_pool_group());
  if (it == transfer_specs_.end()) {
    response->set_found(false);
    return grpc::Status::OK;
  }
  response->set_found(true);
  *response->mutable_spec() = it->second;
  return grpc::Status::OK;
}

size_t GlobalRegistryServiceImpl::GetOwnerIndexSizeForTest(
    const RaidenId& raiden_id) const {
  absl::MutexLock lock(mutex_);
  auto it = owner_index_.find(raiden_id);
  return it == owner_index_.end() ? 0 : it->second.size();
}

void GlobalRegistryServiceImpl::EraseFromOwnerIndex(const RaidenId& raiden_id,
                                                    const std::string& hash) {
  auto it = owner_index_.find(raiden_id);
  if (it == owner_index_.end()) {
    return;
  }
  it->second.erase(hash);
  if (it->second.empty()) {
    owner_index_.erase(it);
  }
}

void GlobalRegistryServiceImpl::PurgeOwnedEntries(const RaidenId& raiden_id) {
  auto owner_it = owner_index_.find(raiden_id);
  if (owner_it == owner_index_.end()) {
    return;
  }
  for (const std::string& hash : owner_it->second) {
    auto it = registry_.find(hash);
    if (it == registry_.end()) {
      continue;
    }
    auto& entries = it->second;
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [&raiden_id](const RegistryEntry& entry) {
                                   return entry.raiden_id == raiden_id;
                                 }),
                  entries.end());
    if (entries.empty()) {
      registry_.erase(it);
      round_robin_indices_.erase(hash);
    }
  }
  owner_index_.erase(owner_it);
}

void GlobalRegistryServiceImpl::CleanupExpiredEntries() {
  absl::MutexLock lock(mutex_);
  absl::Time now = absl::Now();

  // Expired store registrations go first, taking every block entry the store
  // owns with them: a store that stopped heartbeating past its TTL is
  // presumed dead, and its blocks are unreadable without an address.
  std::vector<RaidenId> stores_to_remove;
  for (const auto& [raiden_id, record] : store_registry_) {
    if (record.expire_time <= now) {
      stores_to_remove.push_back(raiden_id);
    }
  }
  for (const auto& raiden_id : stores_to_remove) {
    PurgeOwnedEntries(raiden_id);
    store_registry_.erase(raiden_id);
  }

  std::vector<std::string> keys_to_remove;

  for (auto& [hash, entries] : registry_) {
    for (const auto& entry : entries) {
      if (entry.expire_time <= now) {
        EraseFromOwnerIndex(entry.raiden_id, hash);
      }
    }
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [now](const RegistryEntry& entry) {
                                   return entry.expire_time <= now;
                                 }),
                  entries.end());

    if (entries.empty()) {
      keys_to_remove.push_back(hash);
    }
  }

  for (const auto& key : keys_to_remove) {
    registry_.erase(key);
    round_robin_indices_.erase(key);
  }
}

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden
