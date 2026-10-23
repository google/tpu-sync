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

#include "tpu_sync/kv_cache/host_offload_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/buffer.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/kv_cache/block_tracker.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/kv_cache/kv_cache_store_client.h"
#include "tpu_sync/kv_cache/lru_cache.h"
#include "tpu_sync/proto/kv_cache_store_service.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {

std::vector<::tpu_sync::proto::RaidenWorkerEndpointsProto>
BuildLocalWorkerEndpoints(controller::RaidenController* ctrl) {
  std::vector<::tpu_sync::proto::RaidenWorkerEndpointsProto> result;
  if (ctrl->worker_registry() == nullptr) {
    return result;
  }
  const auto& registered_workers = ctrl->worker_registry()->GetRegisteredWorkers();
  result.reserve(registered_workers.size());
  for (const auto& reg : registered_workers) {
    ::tpu_sync::proto::RaidenWorkerEndpointsProto group;
    group.set_node_id(reg.node_id);
    group.set_worker_id(reg.worker_id);
    group.mutable_endpoints()->Reserve(reg.raiden_transfer_endpoints.size());
    for (const auto& ep : reg.raiden_transfer_endpoints) {
      *group.add_endpoints() = ep.ToProto();
    }
    result.push_back(std::move(group));
  }
  return result;
}

// True for the codes an RPC fails with when the channel itself is the
// problem, as opposed to the peer's handler answering with an error.
bool IsTransportError(const absl::Status& status) {
  return absl::IsUnavailable(status) || absl::IsDeadlineExceeded(status);
}

bool IsHostResidentStatus(BlockStatus status) {
  return status == BlockStatus::HOST || status == BlockStatus::HOST_AND_HBM;
}

}  // namespace

HostOffloadBackend::HostOffloadBackend(
    size_t capacity, std::optional<KVCacheMetadata> metadata,
    RaidenId raiden_id, controller::RaidenController* raiden_controller,
    std::shared_ptr<global_registry::GlobalRegistryClient> registry_client,
    std::string kv_pool_group)
    : lru_cache_(capacity),
      metadata_(std::move(metadata)),
      raiden_id_(std::move(raiden_id)),
      kv_pool_group_(std::move(kv_pool_group)),
      raiden_controller_(raiden_controller),
      registry_client_(std::move(registry_client)) {}

HostOffloadBackend::~HostOffloadBackend() { Shutdown(); }

void HostOffloadBackend::Shutdown() {
  // Completions hold lifetime_->mu in shared (ReaderMutexLock) mode so
  // concurrent Save/Load continuations do not serialize against each other;
  // taking the exclusive WriterMutexLock here waits for all in-flight
  // completions to finish before KVCacheStore destroys its borrowed objects.
  absl::WriterMutexLock lock(lifetime_->mu);
  lifetime_->is_alive = false;
}

absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> HostOffloadBackend::Create(
    const BackendConfig& config,
    controller::RaidenController* absl_nonnull controller) {
  if (config.capacity == 0) {
    return absl::InvalidArgumentError(
        "HostOffloadBackend requires capacity > 0");
  }
  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client;
  if (!config.global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(config.global_registry_address,
                                       grpc::InsecureChannelCredentials());
    registry_client = std::make_shared<
        ::tpu_raiden::kv_cache::global_registry::GlobalRegistryClient>(channel);
  }
  auto backend = std::shared_ptr<HostOffloadBackend>(new HostOffloadBackend(
      config.capacity, config.metadata, config.raiden_id, controller,
      std::move(registry_client), config.kv_pool_group));
  if (config.kv_transfer_spec.has_value()) {
    ABSL_RETURN_IF_ERROR(
        backend->RegisterKVTransferSpec(*config.kv_transfer_spec));
  }
  return backend;
}

absl::StatusOr<BlockSliceList> HostOffloadBackend::Lookup(
    absl::Span<const std::string> block_hashes, const LookupOptions& options) {
  if (options.match_mode == LookupMatchMode::kSubset && options.enable_global) {
    return absl::InvalidArgumentError(
        "LookupMatchMode::kSubset is not supported with enable_global=true");
  }
  if (options.host_resident_only && options.enable_global) {
    return absl::InvalidArgumentError(
        "host_resident_only is not supported with enable_global=true");
  }
  if (options.include_eviction_candidates && options.pin_found) {
    return absl::InvalidArgumentError(
        "include_eviction_candidates is not supported with pin_found=true");
  }

  // Fast Path A: full snapshot of the local index.
  if (options.match_mode == LookupMatchMode::kSubset && block_hashes.empty()) {
    absl::MutexLock lock(mutex_);
    BlockSliceList results;
    results.reserve(lru_cache_.size());
    for (const auto& [hash, _] : lru_cache_.map()) {
      if (!options.include_eviction_candidates && !lru_cache_.Contains(hash)) {
        continue;
      }
      const RaidenBlockId* block_id_ptr =
          options.include_eviction_candidates
              ? lru_cache_.PeekIncludingCandidates(hash)
              : lru_cache_.Peek(hash);
      if (block_id_ptr == nullptr) {
        continue;
      }
      if (options.host_resident_only &&
          !IsHostResidentStatus(block_id_ptr->status)) {
        continue;
      }
      if (options.pin_found) {
        lru_cache_.Pin(hash);
      }
      results.emplace_back(hash, *block_id_ptr);
    }
    return results;
  }

  // Fast Path B: non-contiguous subset over block_hashes.
  if (options.match_mode == LookupMatchMode::kSubset) {
    absl::MutexLock lock(mutex_);
    BlockSliceList results;
    results.reserve(block_hashes.size());
    for (const std::string& hash : block_hashes) {
      const RaidenBlockId* block_id_ptr =
          options.include_eviction_candidates
              ? lru_cache_.PeekIncludingCandidates(hash)
              : lru_cache_.Peek(hash);
      if (block_id_ptr == nullptr) {
        continue;
      }
      if (options.host_resident_only &&
          !IsHostResidentStatus(block_id_ptr->status)) {
        continue;
      }
      if (options.pin_found) {
        lru_cache_.Pin(hash);
      }
      results.emplace_back(hash, *block_id_ptr);
    }
    return results;
  }

  // Fast Path C: local-only contiguous prefix over block_hashes.
  if (!options.enable_global) {
    if (block_hashes.empty()) {
      return BlockSliceList{};
    }
    absl::MutexLock lock(mutex_);
    BlockSliceList results;
    results.reserve(block_hashes.size());
    for (const std::string& hash : block_hashes) {
      const RaidenBlockId* block_id_ptr =
          options.include_eviction_candidates
              ? lru_cache_.PeekIncludingCandidates(hash)
              : lru_cache_.Peek(hash);
      if (block_id_ptr == nullptr) {
        break;
      }
      if (options.host_resident_only &&
          !IsHostResidentStatus(block_id_ptr->status)) {
        break;
      }
      if (options.pin_found) {
        lru_cache_.Pin(hash);
      }
      results.emplace_back(hash, *block_id_ptr);
    }
    return results;
  }

  if (block_hashes.empty()) {
    return BlockSliceList{};
  }

  // Track local hits as bits rather than copying RaidenBlockIds or holding
  // pointers across the unlocked registry RPC; Phase 3 re-reads under mutex_.
  std::vector<bool> local_hit(block_hashes.size(), false);
  std::vector<std::string> missing_hashes;

  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;

  // Phase 1: sweep the local index and pin active hits up front so they cannot
  // be evicted during the blocking registry RPC (tail hits past the first miss
  // are unpinned in Phase 4).
  {
    absl::MutexLock lock(mutex_);
    client = registry_client_;
    local_id = raiden_id_;
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      const auto& hash = block_hashes[i];
      // Use Peek + Pin rather than GetAndPin so pinning never resurrects an
      // eviction candidate.
      const RaidenBlockId* existing =
          options.include_eviction_candidates
              ? lru_cache_.PeekIncludingCandidates(hash)
              : lru_cache_.Peek(hash);
      if (existing != nullptr) {
        if (options.pin_found) {
          lru_cache_.Pin(hash);
        }
        local_hit[i] = true;
        continue;
      }
      if (!options.enable_interleaved_lookup) {
        missing_hashes.assign(block_hashes.begin() + i, block_hashes.end());
        break;
      }
      missing_hashes.push_back(hash);
    }
  }

  // Phase 2: query the global registry for local misses. RPC failures are
  // logged and treated as remote misses.
  std::vector<global_registry::KVBlockMetadata> remote_hits;
  if (!missing_hashes.empty() && options.enable_global && client != nullptr) {
    auto lookup_remote = [&]() -> absl::Status {
      ABSL_ASSIGN_OR_RETURN(
          remote_hits, client->Lookup(missing_hashes, local_id),
          _.LogWarning() << "Global registry lookup failed: ");
      return absl::OkStatus();
    };
    (void)lookup_remote();
  }

  // Phases 3 & 4: assemble the contiguous prefix from local and remote hits
  // under mutex_, then release any Phase 1 pins past the first unresolved miss.
  BlockSliceList results;
  results.reserve(block_hashes.size());
  {
    absl::MutexLock lock(mutex_);
    size_t remote_idx = 0;
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      if (local_hit[i]) {
        // Re-verify unpinned local hits in case they were evicted during
        // Phase 2.
        const RaidenBlockId* existing =
            options.include_eviction_candidates
                ? lru_cache_.PeekIncludingCandidates(block_hashes[i])
                : lru_cache_.Peek(block_hashes[i]);
        if (existing == nullptr) {
          break;
        }
        results.emplace_back(block_hashes[i], *existing);
        continue;
      }
      if (remote_idx >= remote_hits.size()) {
        break;
      }
      const auto& metadata = remote_hits[remote_idx++];
      const auto& proto_id = metadata.raiden_id();
      RaidenId remote_id{
          .job_name = proto_id.job_name(),
          .job_replica_id = proto_id.job_replica_id(),
          .data_name = proto_id.data_name(),
          .data_replica_idx = proto_id.data_replica_idx(),
      };
      if (remote_id == local_id) {
        // Ignore stale self-entries returned by the registry.
        break;
      }
      results.emplace_back(
          block_hashes[i],
          RaidenBlockId(remote_id, metadata.block_id(), BlockStatus::REMOTE));
    }

    // Phase 4: unpin local hits that fell past the returned prefix.
    if (options.pin_found) {
      for (size_t i = results.size(); i < block_hashes.size(); ++i) {
        if (local_hit[i]) {
          lru_cache_.Unpin(block_hashes[i]);
        }
      }
    }
  }

  return results;
}

bool HostOffloadBackend::Insert(absl::Span<const std::string> block_hashes,
                                absl::Span<const RaidenBlockId> slices,
                                InsertOption option) {
  switch (option) {
    case InsertOption::kBestEffortUpdate:
      return InsertBestEffortUpdate(block_hashes, slices);
    case InsertOption::kInsertAndPin:
      return InsertAndPin(block_hashes, slices);
    case InsertOption::kAdmitStrict:
      return InsertAdmitStrict(block_hashes, slices);
  }
  return false;
}

bool HostOffloadBackend::InsertBestEffortUpdate(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockId> slices) {
  std::vector<std::string> evicted_hashes;
  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;

  {
    absl::MutexLock lock(mutex_);
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      const std::string& hash = block_hashes[i];
      if (lru_cache_.Contains(hash)) {
        if (i < slices.size()) {
          if (RaidenBlockId* existing = lru_cache_.PeekMutable(hash)) {
            *existing = slices[i];
            SetMetadataEntry(hash, slices[i]);
          }
        }
        continue;
      }
      if (metadata_.has_value()) {
        if (const RaidenBlockId* stale =
                lru_cache_.PeekIncludingCandidates(hash)) {
          ClearMetadataEntry(*stale);
        }
      }
      std::optional<std::pair<std::string, RaidenBlockId>> evicted;
      if (i < slices.size()) {
        evicted = lru_cache_.Put(hash, slices[i]);
        SetMetadataEntry(hash, slices[i]);
      } else {
        evicted = lru_cache_.Put(hash, RaidenBlockId());
      }
      if (evicted.has_value()) {
        evicted_hashes.push_back(std::move(evicted->first));
      }
    }
    client = registry_client_;
    local_id = raiden_id_;
  }

  // Unregister entries displaced from the active LRU cache.
  if (client != nullptr && !evicted_hashes.empty()) {
    client
        ->UnregisterAsync(evicted_hashes, local_id,
                          global_registry::kUnwaitedMutationTimeout)
        .OnReady([](absl::Status status) {
          if (!status.ok()) {
            LOG(WARNING)
                << "Global registry unregister for evicted blocks failed: "
                << status.message();
          }
        });
  }

  return true;
}

bool HostOffloadBackend::InsertAndPin(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockId> slices) {
  absl::MutexLock lock(mutex_);

  std::vector<size_t> existing_indices;
  std::vector<size_t> new_indices;
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (lru_cache_.Contains(block_hashes[i])) {
      existing_indices.push_back(i);
    } else {
      new_indices.push_back(i);
    }
  }

  for (size_t idx = 0; idx < existing_indices.size(); ++idx) {
    size_t i = existing_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[existing_indices[j]]);
      }
      return false;
    }
    // Overwrite HBM-only entries with the new slice; preserve HOST and
    // HOST_AND_HBM entries.
    if (i < slices.size()) {
      RaidenBlockId* existing = lru_cache_.PeekMutable(block_hashes[i]);
      if (existing != nullptr && existing->status == BlockStatus::HBM) {
        ClearMetadataEntry(*existing);
        *existing = slices[i];
        SetMetadataEntry(block_hashes[i], slices[i]);
      }
    }
  }

  if (lru_cache_.available_space() < new_indices.size()) {
    for (size_t i : existing_indices) {
      lru_cache_.Unpin(block_hashes[i]);
    }
    return false;
  }

  // Insert in reverse order so block_hashes[0] becomes MRU.
  for (auto it = new_indices.rbegin(); it != new_indices.rend(); ++it) {
    size_t i = *it;
    const std::string& hash = block_hashes[i];
    if (metadata_.has_value()) {
      if (const RaidenBlockId* stale =
              lru_cache_.PeekIncludingCandidates(hash)) {
        ClearMetadataEntry(*stale);
      }
    }
    if (i < slices.size()) {
      lru_cache_.Put(hash, slices[i]);
      SetMetadataEntry(hash, slices[i]);
    } else {
      lru_cache_.Put(hash, RaidenBlockId());
    }
  }

  for (size_t idx = 0; idx < new_indices.size(); ++idx) {
    size_t i = new_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[new_indices[j]]);
      }
      for (size_t j : existing_indices) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      for (size_t j : new_indices) {
        if (const RaidenBlockId* val =
                lru_cache_.PeekIncludingCandidates(block_hashes[j])) {
          ClearMetadataEntry(*val);
        }
        lru_cache_.Erase(block_hashes[j]);
      }
      return false;
    }
  }

  return true;
}

bool HostOffloadBackend::InsertAdmitStrict(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockId> slices) {
  if (block_hashes.empty() || block_hashes.size() != slices.size()) {
    return false;
  }
  absl::MutexLock lock(mutex_);

  // Reject duplicate keys (in batch, active, or eviction candidates) or
  // insufficient unpinned space.
  absl::flat_hash_set<absl::string_view> seen_in_batch;
  seen_in_batch.reserve(block_hashes.size());
  for (const auto& hash : block_hashes) {
    if (!seen_in_batch.insert(hash).second ||
        lru_cache_.PeekIncludingCandidates(hash) != nullptr) {
      return false;
    }
  }
  if (block_hashes.size() > lru_cache_.available_space()) {
    return false;
  }

  // Insert all keys and roll back the batch if any insertion fails.
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    const std::string& hash = block_hashes[i];
    std::optional<std::pair<std::string, RaidenBlockId>> evicted =
        lru_cache_.Put(hash, slices[i]);
    if (lru_cache_.Peek(hash) == nullptr) {
      LOG(ERROR) << "Insert(kAdmitStrict): LRUCache::Put inserted nothing "
                    "for a validated hash; rolling the batch back";
      for (size_t j = 0; j < i; ++j) {
        ClearMetadataEntry(slices[j]);
        lru_cache_.Erase(block_hashes[j]);
      }
      return false;
    }
    SetMetadataEntry(hash, slices[i]);
    // Displaced entries move to eviction candidates and keep their host
    // blocks.
    (void)evicted;
  }
  return true;
}

size_t HostOffloadBackend::Delete(absl::Span<const std::string> block_hashes,
                                  const DeleteOptions& options) {
  if (options.deallocated_host_block_ids != nullptr) {
    options.deallocated_host_block_ids->clear();
  }
  if (options.deleted_hashes != nullptr) {
    options.deleted_hashes->clear();
  }

  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;
  std::vector<std::string> hashes_to_unregister;
  size_t erased_count = 0;

  {
    absl::MutexLock lock(mutex_);
    for (const std::string& hash : block_hashes) {
      if (lru_cache_.GetPinCount(hash) > 0) {
        if (!options.host_resident_only) {
          LOG(WARNING)
              << "Delete skipped pinned block hash (release it first): "
              << absl::BytesToHexString(hash);
        }
        continue;
      }
      const RaidenBlockId* block = lru_cache_.PeekIncludingCandidates(hash);
      if (block == nullptr) {
        continue;
      }
      const bool is_host_resident = IsHostResidentStatus(block->status);
      if (options.host_resident_only && !is_host_resident) {
        continue;
      }
      if (is_host_resident) {
        if (options.deallocated_host_block_ids != nullptr) {
          options.deallocated_host_block_ids->push_back(block->host_block_id);
        }
        hashes_to_unregister.push_back(hash);
      }
      ClearMetadataEntry(*block);
      lru_cache_.Erase(hash);
      if (options.deleted_hashes != nullptr) {
        options.deleted_hashes->push_back(hash);
      }
      ++erased_count;
    }
    client = registry_client_;
    local_id = raiden_id_;
  }

  if (client != nullptr && !hashes_to_unregister.empty()) {
    client
        ->UnregisterAsync(hashes_to_unregister, local_id,
                          global_registry::kUnwaitedMutationTimeout)
        .OnReady([](absl::Status status) {
          if (!status.ok()) {
            LOG(WARNING) << "Global registry unregister failed: "
                         << status.message();
          }
        });
  }

  return erased_count;
}

bool HostOffloadBackend::Pin(absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < i; ++j) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      return false;
    }
  }
  return true;
}

void HostOffloadBackend::Release(absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  // Reverse order so that block_hashes[0] ends up nearest the MRU end: Unpin
  // splices an entry to the front of the active list, and the first hash of a
  // prefix chain is the one most worth keeping.
  for (auto it = block_hashes.rbegin(); it != block_hashes.rend(); ++it) {
    lru_cache_.Unpin(*it);
  }
}

int HostOffloadBackend::GetPinCount(const std::string& hash) const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetPinCount(hash);
}

size_t HostOffloadBackend::GetCapacity() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.capacity();
}

size_t HostOffloadBackend::GetSize() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.size();
}

size_t HostOffloadBackend::GetAvailableSpace() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.available_space();
}

absl::StatusOr<size_t> HostOffloadBackend::RecoverFromLocalManifest() {
  absl::MutexLock lock(mutex_);
  if (!metadata_.has_value()) {
    return absl::FailedPreconditionError(
        "KVCacheMetadata is required for crash recovery");
  }
  std::vector<KVCacheMetadata::Entry> entries = metadata_->ValidEntries();
  if (entries.empty()) {
    return 0;
  }

  absl::flat_hash_map<absl::string_view, const KVCacheMetadata::Entry*> newest;
  newest.reserve(entries.size());
  for (const KVCacheMetadata::Entry& entry : entries) {
    auto [it, inserted] = newest.try_emplace(entry.hash, &entry);
    if (!inserted && entry.seq > it->second->seq) {
      it->second = &entry;
    }
  }

  std::vector<const KVCacheMetadata::Entry*> recoverable;
  recoverable.reserve(newest.size());
  for (const auto& [hash, entry] : newest) {
    recoverable.push_back(entry);
  }
  std::sort(recoverable.begin(), recoverable.end(),
            [](const KVCacheMetadata::Entry* a,
               const KVCacheMetadata::Entry* b) { return a->seq < b->seq; });

  std::vector<int> block_ids;
  block_ids.reserve(recoverable.size());
  for (const KVCacheMetadata::Entry* entry : recoverable) {
    block_ids.push_back(entry->block_id);
  }
  ABSL_RETURN_IF_ERROR(raiden_controller_->AllocateTargetBlockIds(block_ids));

  uint64_t max_seq = 0;
  for (const KVCacheMetadata::Entry* entry : recoverable) {
    lru_cache_.Put(entry->hash, RaidenBlockId(raiden_id_, entry->block_id,
                                              BlockStatus::HOST));
    max_seq = std::max(max_seq, entry->seq);
  }
  next_metadata_seq_ = max_seq + 1;

  for (const KVCacheMetadata::Entry& entry : entries) {
    if (newest.at(entry.hash)->block_id != entry.block_id) {
      absl::Status status = metadata_->Clear(entry.block_id);
      if (!status.ok()) {
        LOG(WARNING) << "Failed to clear the stale metadata entry for block "
                     << entry.block_id << ": " << status.message();
      }
    }
  }

  return recoverable.size();
}

std::vector<std::string> HostOffloadBackend::GetEvictableKeys(size_t count) {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetEvictableKeys(count);
}

std::vector<int> HostOffloadBackend::Evict(
    const std::vector<std::string>& block_hashes,
    std::vector<std::string>* evicted_hashes) {
  std::vector<int> host_ids_to_deallocate;
  Delete(block_hashes, DeleteOptions::EvictHostResident(&host_ids_to_deallocate,
                                                        evicted_hashes));
  return host_ids_to_deallocate;
}

absl::StatusOr<std::shared_ptr<KVCacheStoreClient>>
HostOffloadBackend::GetKVCacheStoreClient(const RaidenId& remote_id) {
  std::shared_ptr<global_registry::GlobalRegistryClient> registry;
  {
    absl::MutexLock lock(mutex_);
    auto it = store_clients_.find(remote_id);
    if (it != store_clients_.end()) {
      return it->second;
    }
    registry = registry_client_;
  }

  // The peer's KVCacheStoreService address comes from the global registry,
  // which is also what told us this peer owns the blocks. Note this is NOT the
  // controller address: the two services listen on separate ports, and
  // ResolvePeerController answers for the wrong one.
  if (registry == nullptr) {
    return absl::FailedPreconditionError(
        "No global registry client; cannot resolve peer store address");
  }

  ABSL_ASSIGN_OR_RETURN(global_registry::StoreInfo store_info,
                        registry->ResolveStore(remote_id));
  if (store_info.store_server_address().empty()) {
    return absl::NotFoundError(
        "Peer is registered but published an empty store server address");
  }

  auto channel = grpc::CreateChannel(store_info.store_server_address(),
                                     grpc::InsecureChannelCredentials());
  auto client = std::make_shared<KVCacheStoreClient>(channel);
  {
    absl::MutexLock lock(mutex_);
    store_clients_[remote_id] = client;
  }
  return client;
}

void HostOffloadBackend::InvalidateStoreClient(const RaidenId& remote_id) {
  absl::MutexLock lock(mutex_);
  store_clients_.erase(remote_id);
}

absl::StatusOr<HostOffloadBackend::RemoteWriteAck>
HostOffloadBackend::BeginWriteRemote(
    const RaidenId& dst_raiden_id, absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> src_host_block_ids,
    absl::Duration requested_deadline, absl::Duration hold_window,
    BlockTracker* absl_nonnull save_tracker) {
  CHECK(save_tracker != nullptr);
  if (block_hashes.empty()) {
    return absl::InvalidArgumentError("WriteRemote requires at least one hash");
  }
  if (block_hashes.size() != src_host_block_ids.size()) {
    return absl::InvalidArgumentError(
        "src_host_block_ids must have one entry per block hash");
  }

  // Resolves the peer through the global registry. A missing registry
  // client fails here; there is no separate precondition check.
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<KVCacheStoreClient> client,
                        GetKVCacheStoreClient(dst_raiden_id));

  auto hold_expiry = std::make_shared<absl::Time>(absl::Now() + hold_window);

  // The reactor is deliberately not given the tracker. A verdict published
  // from the stream's callback would reach a waiting sweep before the pins
  // below are dropped, and the sweep's eviction skips pinned blocks -- it
  // would free nothing and give up the episode. Every verdict is recorded
  // here instead, after the release.
  auto call = client->WriteRemote(
      raiden_controller_->unit(), block_hashes, src_host_block_ids,
      BuildLocalWorkerEndpoints(raiden_controller_),
      absl::ToInt64Milliseconds(requested_deadline), hold_window,
      /*tracker=*/nullptr);
  auto response = call.ack.Await();
  if (!response.ok()) {
    // On a transport error the peer may have restarted on a new port; drop
    // the cached client so the next attempt re-resolves it. An application
    // error (e.g. RESOURCE_EXHAUSTED) means the peer is alive, so the
    // channel is kept.
    if (IsTransportError(response.status())) {
      InvalidateStoreClient(dst_raiden_id);
    }
    return response.status();
  }

  RemoteWriteAck ack;
  ack.cancel = std::move(call.cancel);
  ack.operation_id = response->operation_id();
  ack.granted_deadline = absl::Milliseconds(response->granted_deadline_ms());
  if (ack.granted_deadline >= hold_window) {
    *hold_expiry = std::max(*hold_expiry, absl::Now() + ack.granted_deadline);
  }
  switch (response->exist_state()) {
    case ::tpu_raiden::kv_cache::proto::WRITE_ALL_EXIST:
      ack.all_exist = true;
      // Release before the verdict, as every settle path does: a waiter woken
      // by the verdict must find the blocks already unpinned.
      Release(block_hashes);
      Release(block_hashes);
      save_tracker->MarkDone(block_hashes);
      return ack;
    case ::tpu_raiden::kv_cache::proto::WRITE_PARTIAL_EXIST:
      ack.existing_hashes.assign(response->existing_hashes().begin(),
                                 response->existing_hashes().end());
      // Only the transfer's hold; a failed offer keeps the caller's pin for
      // the caller to drop.
      Release(block_hashes);
      save_tracker->MarkFailedWithExisting(block_hashes, ack.existing_hashes);
      return ack;
    default:
      break;
  }
  if (ack.operation_id == 0) {
    // Neither an existence answer nor an accepted operation: a protocol
    // error, not a silent no-op.
    return absl::InternalError(
        "Destination accepted the offer but returned no operation id.");
  }
  std::vector<std::string> hashes(block_hashes.begin(), block_hashes.end());
  call.result.OnReady([this, lifetime = lifetime_, dst_raiden_id, save_tracker,
                       hashes = std::move(hashes), hold_expiry,
                       op_id = ack.operation_id](
                          absl::StatusOr<proto::WriteRemoteResult> result_or) {
    // Register OnReady after releasing lifetime->mu in case poll_future is
    // already resolved.
    tsl::Future<proto::PollWriteRemoteResponse> poll_future;
    {
      absl::ReaderMutexLock lock(lifetime->mu);
      if (!lifetime->is_alive) {
        return;
      }
      if (!result_or.ok() &&
          result_or.status().code() == absl::StatusCode::kCancelled) {
        return;
      }
      // Release, then publish the verdict -- never the other way round. A
      // waiter is woken by the verdict, and the evict sweep's eviction skips
      // pinned blocks, so a verdict that outran the release would leave the
      // sweep freeing nothing and abandoning its episode.
      if (result_or.ok()) {
        const auto& result = *result_or;
        const bool succeeded =
            (result.state() == proto::PollWriteRemoteResponse::COMMITTED ||
             result.state() == proto::PollWriteRemoteResponse::ALL_EXIST);
        Release(hashes);
        if (succeeded) {
          // A successful save also consumes the caller's pin.
          Release(hashes);
        }
        switch (result.state()) {
          case proto::PollWriteRemoteResponse::COMMITTED:
          case proto::PollWriteRemoteResponse::ALL_EXIST:
            save_tracker->MarkDone(hashes);
            break;
          case proto::PollWriteRemoteResponse::PARTIAL_EXIST:
            save_tracker->MarkFailedWithExisting(
                hashes,
                std::vector<std::string>(result.existing_hashes().begin(),
                                         result.existing_hashes().end()));
            break;
          case proto::PollWriteRemoteResponse::STORED_UNREGISTERED:
            save_tracker->MarkFailedWithUnregistered(
                hashes,
                std::vector<std::string>(result.unregistered_hashes().begin(),
                                         result.unregistered_hashes().end()));
            break;
          default:
            save_tracker->MarkFailed(hashes);
            break;
        }
        return;
      }

      const absl::Duration remaining_hold = *hold_expiry - absl::Now();
      if (op_id == 0 || remaining_hold <= absl::ZeroDuration()) {
        // No operation to chase, or the destination's hold has lapsed.
        Release(hashes);
        save_tracker->MarkFailed(hashes);
        return;
      }
      // Started while holding the lock, so `this` cannot be destroyed under us.
      poll_future = PollWriteRemoteAsync(
          dst_raiden_id, op_id, absl::ToInt64Milliseconds(remaining_hold));
    }

    poll_future.OnReady(
        [this, lifetime, save_tracker,
         hashes](absl::StatusOr<proto::PollWriteRemoteResponse> resp) {
          absl::ReaderMutexLock lock(lifetime->mu);
          if (!lifetime->is_alive) {
            return;
          }
          const bool poll_succeeded =
              resp.ok() &&
              (resp->state() == proto::PollWriteRemoteResponse::COMMITTED ||
               resp->state() == proto::PollWriteRemoteResponse::ALL_EXIST);
          Release(hashes);
          if (poll_succeeded) {
            Release(hashes);
          }
          if (!resp.ok()) {
            save_tracker->MarkFailed(hashes);
            return;
          }
          switch (resp->state()) {
            case proto::PollWriteRemoteResponse::COMMITTED:
            case proto::PollWriteRemoteResponse::ALL_EXIST:
              save_tracker->MarkDone(hashes);
              break;
            case proto::PollWriteRemoteResponse::PARTIAL_EXIST:
              save_tracker->MarkFailedWithExisting(
                  hashes,
                  std::vector<std::string>(resp->existing_hashes().begin(),
                                           resp->existing_hashes().end()));
              break;
            case proto::PollWriteRemoteResponse::STORED_UNREGISTERED:
              save_tracker->MarkFailedWithUnregistered(
                  hashes,
                  std::vector<std::string>(resp->unregistered_hashes().begin(),
                                           resp->unregistered_hashes().end()));
              break;
            default:
              save_tracker->MarkFailed(hashes);
              break;
          }
        });
  });
  return ack;
}

tsl::Future<proto::PollWriteRemoteResponse>
HostOffloadBackend::PollWriteRemoteAsync(const RaidenId& dst_raiden_id,
                                        uint64_t operation_id,
                                        int64_t wait_ms) {
  auto client = GetKVCacheStoreClient(dst_raiden_id);
  if (!client.ok()) {
    return tsl::Future<proto::PollWriteRemoteResponse>(client.status());
  }
  return (*client)->PollWriteRemote(operation_id, wait_ms);
}

tsl::Future<> HostOffloadBackend::RegisterBlocksAsync(
    absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> host_block_ids) {
  if (block_hashes.size() != host_block_ids.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(absl::StrCat(
        "Mismatched block_hashes count (", block_hashes.size(),
        ") vs host_block_ids count (", host_block_ids.size(), ").")));
  }
  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;
  {
    absl::MutexLock lock(mutex_);
    client = registry_client_;
    local_id = raiden_id_;
  }
  if (client == nullptr) {
    // No registry configured, so there is nothing to advertise. Success.
    return tsl::Future<>(absl::OkStatus());
  }

  std::vector<global_registry::Registration> registrations;
  registrations.reserve(block_hashes.size());
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    registrations.push_back({
        .prefix_hash = std::string(block_hashes[i]),
        .raiden_id = local_id,
        .block_id = host_block_ids[i],
    });
  }
  return client->RegisterAsync(registrations);
}

tsl::Future<> HostOffloadBackend::UnregisterBlocksAsync(
    absl::Span<const std::string> block_hashes) {
  if (block_hashes.empty()) {
    return tsl::Future<>(absl::OkStatus());
  }
  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;
  {
    absl::MutexLock lock(mutex_);
    client = registry_client_;
    local_id = raiden_id_;
  }
  if (client == nullptr) {
    // Nothing was ever advertised, so there is nothing to take back. Success.
    return tsl::Future<>(absl::OkStatus());
  }
  // Scoped to this store's own id, so a mistake here can only reach entries
  // this store published.
  return client->UnregisterAsync(
      std::vector<std::string>(block_hashes.begin(), block_hashes.end()),
      local_id, global_registry::kUnwaitedMutationTimeout);
}

tsl::Future<> HostOffloadBackend::Load(
    const RaidenId& remote_id, absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> device_block_ids,
    absl::Span<const RaidenBlockId> slices,
    BlockTracker* absl_nonnull load_tracker) {
  CHECK(load_tracker != nullptr);
  if (block_hashes.empty()) {
    return tsl::Future<>(absl::OkStatus());
  }

  // Every hash needs a destination device block.
  if (device_block_ids.size() != block_hashes.size()) {
    load_tracker->MarkFailed(block_hashes);
    return tsl::Future<>(absl::InvalidArgumentError(absl::StrCat(
        "Mismatched device_block_ids count (", device_block_ids.size(),
        ") vs block_hashes count (", block_hashes.size(), ").")));
  }

  bool is_remote = false;
  {
    absl::MutexLock lock(mutex_);
    is_remote = !remote_id.empty() && remote_id != raiden_id_;
  }

  if (is_remote) {
    return LoadRemoteBlocks(remote_id, block_hashes, device_block_ids,
                            load_tracker);
  }
  return LoadLocalHostBlocks(block_hashes, device_block_ids, slices,
                             load_tracker);
}

tsl::Future<> HostOffloadBackend::LoadRemoteBlocks(
    const RaidenId& remote_id, absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> device_block_ids,
    BlockTracker* absl_nonnull load_tracker) {
  CHECK(load_tracker != nullptr);
  auto client = GetKVCacheStoreClient(remote_id);
  if (!client.ok()) {
    load_tracker->MarkFailed(block_hashes);
    return tsl::Future<>(client.status());
  }
  std::shared_ptr<KVCacheStoreClient> client_ptr = *std::move(client);

  auto host_blocks = raiden_controller_->AllocateBlockIds(block_hashes.size());
  if (!host_blocks.ok()) {
    load_tracker->MarkFailed(block_hashes);
    return tsl::Future<>(host_blocks.status());
  }
  std::vector<int32_t> dst_host_block_ids(host_blocks->begin(),
                                          host_blocks->end());

  auto [load_promise, load_future] = tsl::MakePromise<>();

  ::tpu_sync::rpc::RaidenIdProto client_raiden_id = raiden_controller_->unit();
  std::vector<::tpu_sync::proto::RaidenWorkerEndpointsProto>
      client_worker_endpoints = BuildLocalWorkerEndpoints(raiden_controller_);
  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> fetch_future =
      client_ptr->Fetch(block_hashes, device_block_ids, dst_host_block_ids,
                        client_raiden_id, client_worker_endpoints);

  fetch_future.OnReady(
      [this, lifetime = lifetime_, remote_id, dst_host_block_ids,
       hashes =
           std::vector<std::string>(block_hashes.begin(), block_hashes.end()),
       dev_ids_vec = std::vector<int32_t>(device_block_ids.begin(),
                                          device_block_ids.end()),
       load_tracker, load_promise = std::move(load_promise)](
          const absl::StatusOr<::tpu_raiden::kv_cache::proto::FetchResponse>&
              fetch_response) mutable {
        tsl::Future<> h2d_future;
        {
          absl::ReaderMutexLock lock(lifetime->mu);
          if (!lifetime->is_alive) {
            load_promise.Set(absl::CancelledError("Backend destroyed"));
            return;
          }
          if (!fetch_response.ok()) {
            (void)raiden_controller_->DeallocateBlockIds(dst_host_block_ids);
            // The peer may have restarted on a new port; drop the cached client
            // so the next attempt re-resolves instead of redialling a dead one.
            InvalidateStoreClient(remote_id);
            load_tracker->MarkFailed(hashes);
            load_promise.Set(fetch_response.status());
            return;
          }

          const auto& response = *fetch_response;
          if (!response.failed_block_hashes().empty()) {
            (void)raiden_controller_->DeallocateBlockIds(dst_host_block_ids);
            std::string err_msg = response.error_message().empty()
                                      ? "Fetch RPC returned failed blocks"
                                      : response.error_message();
            load_tracker->MarkFailed(hashes);
            load_promise.Set(absl::InternalError(err_msg));
            return;
          }

          std::vector<Buffer> src_buffers;
          src_buffers.reserve(dst_host_block_ids.size());
          for (int id : dst_host_block_ids) {
            src_buffers.emplace_back(id, std::vector<BufferShard>{},
                                     std::nullopt,
                                     ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
          }

          std::vector<Buffer> dst_buffers;
          dst_buffers.reserve(dev_ids_vec.size());
          for (int id : dev_ids_vec) {
            dst_buffers.emplace_back(id, std::vector<BufferShard>{},
                                     std::nullopt,
                                     ::tpu_sync::rpc::MEMORY_TYPE_HBM);
          }

          h2d_future =
              raiden_controller_->TransferBuffers(src_buffers, dst_buffers);
        }

        h2d_future.OnReady([this, lifetime, dst_host_block_ids,
                            hashes = std::move(hashes), load_tracker,
                            load_promise = std::move(load_promise)](
                               absl::Status status) mutable {
          absl::ReaderMutexLock lock(lifetime->mu);
          if (!lifetime->is_alive) {
            load_promise.Set(absl::CancelledError("Backend destroyed"));
            return;
          }
          (void)raiden_controller_->DeallocateBlockIds(dst_host_block_ids);
          if (status.ok()) {
            load_tracker->MarkDone(hashes);
          } else {
            load_tracker->MarkFailed(hashes);
          }
          load_promise.Set(status);
        });
      });

  return load_future;
}

tsl::Future<> HostOffloadBackend::LoadLocalHostBlocks(
    absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> device_block_ids,
    absl::Span<const RaidenBlockId> slices,
    BlockTracker* absl_nonnull load_tracker) {
  CHECK(load_tracker != nullptr);
  std::vector<int64_t> src_host_block_ids;
  src_host_block_ids.reserve(block_hashes.size());
  if (!slices.empty()) {
    if (slices.size() != block_hashes.size()) {
      load_tracker->MarkFailed(block_hashes);
      return tsl::Future<>(absl::InvalidArgumentError(
          "Mismatched slices count vs block_hashes count."));
    }
    for (size_t i = 0; i < slices.size(); ++i) {
      if (slices[i].host_block_id == -1) {
        load_tracker->MarkFailed(block_hashes);
        return tsl::Future<>(absl::FailedPreconditionError(
            absl::StrCat("Block host_block_id is -1: ", block_hashes[i])));
      }
      src_host_block_ids.push_back(slices[i].host_block_id);
    }
  } else {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : block_hashes) {
      const RaidenBlockId* entry = lru_cache_.Peek(hash);
      if (entry == nullptr) {
        load_tracker->MarkFailed(block_hashes);
        return tsl::Future<>(absl::NotFoundError(
            absl::StrCat("Block hash not found in host backend: ", hash)));
      }
      if (entry->status != BlockStatus::HOST &&
          entry->status != BlockStatus::HOST_AND_HBM) {
        load_tracker->MarkFailed(block_hashes);
        return tsl::Future<>(absl::FailedPreconditionError(
            absl::StrCat("Block is not on host: ", hash)));
      }
      if (entry->host_block_id == -1) {
        load_tracker->MarkFailed(block_hashes);
        return tsl::Future<>(absl::FailedPreconditionError(
            absl::StrCat("Block host_block_id is -1: ", hash)));
      }
      src_host_block_ids.push_back(entry->host_block_id);
    }
  }

  if (raiden_controller_ == nullptr) {
    load_tracker->MarkFailed(block_hashes);
    return tsl::Future<>(
        absl::FailedPreconditionError("RaidenController is not initialized"));
  }

  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_host_block_ids.size());
  for (int64_t id : src_host_block_ids) {
    src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  }

  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(device_block_ids.size());
  for (int id : device_block_ids) {
    dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  }

  tsl::Future<> h2d_future =
      raiden_controller_->TransferBuffers(src_buffers, dst_buffers);

  auto [promise, future] = tsl::MakePromise<>();
  h2d_future.OnReady([this, lifetime = lifetime_,
                      hashes = std::vector<std::string>(block_hashes.begin(),
                                                        block_hashes.end()),
                      dev_ids = std::vector<int32_t>(device_block_ids.begin(),
                                                     device_block_ids.end()),
                      load_tracker, promise = std::move(promise)](
                         absl::Status status) mutable {
    absl::ReaderMutexLock lifetime_lock(lifetime->mu);
    if (!lifetime->is_alive) {
      promise.Set(absl::CancelledError("Backend destroyed"));
      return;
    }
    if (status.ok()) {
      std::vector<std::string> update_hashes;
      std::vector<RaidenBlockId> update_slices;
      update_hashes.reserve(hashes.size());
      update_slices.reserve(hashes.size());
      {
        absl::MutexLock lock(mutex_);
        for (size_t i = 0; i < hashes.size(); ++i) {
          const std::string& hash = hashes[i];
          if (const RaidenBlockId* existing = lru_cache_.Peek(hash)) {
            RaidenBlockId block = *existing;
            block.device_block_id = dev_ids[i];
            block.status = BlockStatus::HOST_AND_HBM;
            update_hashes.push_back(hash);
            update_slices.push_back(block);
          }
        }
      }
      if (!update_hashes.empty()) {
        Insert(update_hashes, update_slices);
        Release(update_hashes);
      }
      load_tracker->MarkDone(hashes);
      promise.Set(absl::OkStatus());
    } else {
      load_tracker->MarkFailed(hashes);
      promise.Set(status);
    }
  });
  return future;
}

tsl::Future<> HostOffloadBackend::Save(
    absl::Span<const std::string> block_hashes,
    absl::Span<const int64_t> src_device_block_ids,
    absl::Span<const int32_t> dst_host_block_ids,
    BlockTracker* absl_nonnull save_tracker) {
  CHECK(save_tracker != nullptr);
  if (block_hashes.empty()) {
    return tsl::Future<>(absl::OkStatus());
  }
  if (src_device_block_ids.size() != block_hashes.size() ||
      dst_host_block_ids.size() != block_hashes.size()) {
    save_tracker->MarkFailed(block_hashes);
    return tsl::Future<>(absl::InvalidArgumentError(
        "Mismatched block_hashes, src_device_block_ids, or dst_host_block_ids "
        "count."));
  }
  if (raiden_controller_ == nullptr) {
    save_tracker->MarkFailed(block_hashes);
    return tsl::Future<>(
        absl::FailedPreconditionError("RaidenController is not initialized"));
  }

  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_device_block_ids.size());
  for (int64_t id : src_device_block_ids) {
    src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  }
  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(dst_host_block_ids.size());
  for (int32_t id : dst_host_block_ids) {
    dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  }

  tsl::Future<> transfer_future = raiden_controller_->TransferBuffers(
      src_buffers, dst_buffers, /*staging_host_buffers=*/{},
      /*copy_sizes=*/{});

  auto [save_promise, save_future] = tsl::MakePromise<>();
  transfer_future.OnReady(
      [this, lifetime = lifetime_,
       hashes =
           std::vector<std::string>(block_hashes.begin(), block_hashes.end()),
       src_device_ids = std::vector<int64_t>(src_device_block_ids.begin(),
                                             src_device_block_ids.end()),
       dst_host_ids = std::vector<int32_t>(dst_host_block_ids.begin(),
                                           dst_host_block_ids.end()),
       save_tracker,
       save_promise = std::move(save_promise)](absl::Status status) mutable {
        absl::ReaderMutexLock lifetime_lock(lifetime->mu);
        if (!lifetime->is_alive) {
          save_promise.Set(absl::CancelledError("Backend destroyed"));
          return;
        }
        if (status.ok()) {
          std::vector<std::string> update_hashes;
          std::vector<RaidenBlockId> update_slices;
          update_hashes.reserve(hashes.size());
          update_slices.reserve(hashes.size());
          {
            absl::MutexLock lock(mutex_);
            for (size_t i = 0; i < hashes.size(); ++i) {
              const std::string& hash = hashes[i];
              const RaidenBlockId* existing = lru_cache_.Peek(hash);
              RaidenBlockId block =
                  existing != nullptr
                      ? *existing
                      : RaidenBlockId(raiden_id_, dst_host_ids[i],
                                      src_device_ids[i],
                                      BlockStatus::HOST_AND_HBM);
              block.host_block_id = dst_host_ids[i];
              block.device_block_id = src_device_ids[i];
              block.status = BlockStatus::HOST_AND_HBM;
              update_hashes.push_back(hash);
              update_slices.push_back(block);
            }
          }
          if (!update_hashes.empty()) {
            Insert(update_hashes, update_slices);
          }
          save_tracker->MarkDone(hashes);
          save_promise.Set(absl::OkStatus());
        } else {
          if (raiden_controller_ != nullptr) {
            (void)raiden_controller_->DeallocateBlockIds(dst_host_ids);
          }
          save_tracker->MarkFailed(hashes);
          save_promise.Set(status);
        }
      });

  return save_future;
}

absl::StatusOr<KVTransferSpecConfig> HostOffloadBackend::ComposeKVTransferSpec(
    absl::Span<const core::controller::WorkerRegistration> workers) {
  if (workers.empty()) {
    return absl::FailedPreconditionError(
        "no registered workers to derive a KVTransferSpec from");
  }
  const core::controller::WorkerRegistration& reference = workers[0];
  std::vector<int64_t> node_ids;
  node_ids.reserve(workers.size());
  for (const core::controller::WorkerRegistration& worker : workers) {
    if (worker.block_array_bytes.empty() || worker.num_kv_shards <= 0) {
      return absl::FailedPreconditionError(absl::StrCat(
          "worker ", worker.worker_id,
          " registered without its KV block geometry (block_array_bytes, "
          "num_kv_shards); every worker must report it for the deployment's "
          "KVTransferSpec"));
    }
    if (worker.block_array_bytes != reference.block_array_bytes ||
        worker.num_kv_shards != reference.num_kv_shards) {
      return absl::InvalidArgumentError(absl::StrCat(
          "workers ", reference.worker_id, " and ", worker.worker_id,
          " registered different KV block geometry; one KVTransferSpec must "
          "describe every worker of the deployment"));
    }
    node_ids.push_back(worker.node_id);
  }
  std::sort(node_ids.begin(), node_ids.end());
  for (size_t i = 0; i < node_ids.size(); ++i) {
    if (node_ids[i] != static_cast<int64_t>(i)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "worker node ids must form the dense range [0, ", workers.size(),
          "); got [", absl::StrJoin(node_ids, ", "), "]"));
    }
  }
  KVTransferSpecConfig spec;
  spec.block_array_bytes = reference.block_array_bytes;
  spec.num_kv_shards = reference.num_kv_shards;
  spec.num_workers = static_cast<int>(workers.size());
  return spec;
}

absl::Status HostOffloadBackend::RegisterKVTransferSpecFromWorkers() {
  if (raiden_controller_->worker_registry() == nullptr) {
    return absl::FailedPreconditionError(
        "HostOffloadBackend has no RaidenController; there are no worker "
        "registrations to derive a KVTransferSpec from.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const KVTransferSpecConfig spec,
      ComposeKVTransferSpec(
          raiden_controller_->worker_registry()->GetRegisteredWorkers()));
  return RegisterKVTransferSpec(spec);
}

absl::Status HostOffloadBackend::RegisterKVTransferSpec(
    const KVTransferSpecConfig& spec_config) {
  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client;
  std::string kv_pool_group = kv_pool_group_;
  {
    absl::MutexLock lock(mutex_);
    registry_client = registry_client_;
    if (kv_pool_group.empty()) {
      kv_pool_group = raiden_id_.job_name;
    }
  }
  if (registry_client == nullptr) {
    return absl::FailedPreconditionError(
        "HostOffloadBackend has no global registry configured; cannot "
        "register a KVTransferSpec.");
  }
  if (kv_pool_group.empty()) {
    return absl::FailedPreconditionError(
        "no KV pool group to register the KVTransferSpec under: set "
        "BackendConfig.kv_pool_group or a raiden_id with a job_name.");
  }
  global_registry::KVTransferSpec spec;
  for (uint64_t bytes : spec_config.block_array_bytes) {
    spec.add_block_arrays()->set_block_bytes(static_cast<int64_t>(bytes));
  }
  spec.set_num_kv_shards(static_cast<int32_t>(spec_config.num_kv_shards));
  spec.set_num_workers(static_cast<int32_t>(spec_config.num_workers));
  return registry_client->RegisterKVTransferSpec(spec, kv_pool_group);
}

void HostOffloadBackend::SetMetadataEntry(absl::string_view hash,
                                          const RaidenBlockId& block) {
  if (!metadata_.has_value()) {
    return;
  }
  if (block.status != BlockStatus::HOST &&
      block.status != BlockStatus::HOST_AND_HBM) {
    return;
  }
  absl::Status status =
      metadata_->Set(block.host_block_id, hash, next_metadata_seq_++);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to set the metadata entry for block "
                 << block.host_block_id << ": " << status.message();
  }
}

void HostOffloadBackend::ClearMetadataEntry(const RaidenBlockId& block) {
  if (!metadata_.has_value()) {
    return;
  }
  if (block.status != BlockStatus::HOST &&
      block.status != BlockStatus::HOST_AND_HBM) {
    return;
  }
  absl::Status status = metadata_->Clear(block.host_block_id);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to clear the metadata entry for block "
                 << block.host_block_id << ": " << status.message();
  }
}

}  // namespace kv_cache
}  // namespace tpu_raiden

REGISTER_KV_CACHE_STORE_BACKEND(
    "HostOffloadBackend",
    static_cast<absl::StatusOr<
        std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend>> (*)(
        const ::tpu_raiden::kv_cache::BackendConfig&,
        ::tpu_raiden::controller::RaidenController*)>(
        ::tpu_raiden::kv_cache::HostOffloadBackend::Create));
