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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/buffer.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/status_macros.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/kv_cache/kv_cache_store_client.h"
#include "tpu_sync/kv_cache/kv_cache_store_server.h"
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
      auto* ep_proto = group.add_endpoints();
      ep_proto->set_endpoint(ep.endpoint);
      ep_proto->mutable_shards()->Add(ep.shards.begin(), ep.shards.end());
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

HostOffloadBackend::~HostOffloadBackend() {
  if (server_) {
    server_->Shutdown();
  }
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
  // Server start + publish is exclusively
  // KVCacheStore::EnsureStoreServerAndRegister's job -- starting one here
  // would race it and could bind before the owning store has decided whether
  // a registry exists at all.
  auto backend = std::shared_ptr<HostOffloadBackend>(new HostOffloadBackend(
      config.capacity, config.metadata, config.raiden_id, controller,
      std::move(registry_client), config.kv_pool_group));
  if (config.kv_transfer_spec.has_value()) {
    RETURN_IF_ERROR(backend->RegisterKVTransferSpec(*config.kv_transfer_spec));
  }
  return backend;
}

absl::Status HostOffloadBackend::StartServer(absl::string_view server_address) {
  if (server_address.empty() || server_address == "[::]" ||
      server_address == "::" || server_address == "0.0.0.0" ||
      server_address == "0:0:0:0:0:0:0:0") {
    return absl::InvalidArgumentError(
        "server_address must be a non-empty, non-wildcard host; it is not "
        "derived from the controller address.");
  }

  absl::MutexLock lock(mutex_);
  if (!server_) {
    server_ = KVCacheStoreServer::Create();
  }

  return server_->StartServer(this, raiden_controller_, server_address);
}

absl::StatusOr<BlockSliceList> HostOffloadBackend::Lookup(
    absl::Span<const std::string> block_hashes, const LookupOptions& options) {
  if (block_hashes.empty()) {
    return BlockSliceList{};
  }

  // Which hashes the local sweep matched -- one bit each, not the entries
  // themselves. Neither the matched values nor pointers to them survive the
  // registry RPC below: the RPC runs with mutex_ released, and copying a
  // RaidenBlockId means copying the three strings inside its RaidenId, while a
  // pointer into an LRU node dangles the moment a concurrent Delete or Evict
  // erases that node. The assembly phase re-reads the index under the lock
  // instead, which is both cheaper than the copies and the only way to tell
  // that an entry is still there.
  //
  // The bits are still needed: they say which pins this call took, and which
  // hashes the registry was asked about.
  std::vector<bool> local_hit(block_hashes.size(), false);
  std::vector<std::string> missing_hashes;

  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;

  // Phase 1: sweep the local index for EVERY hash, and pin each hit as it is
  // found. Pinning up front is what makes the answer trustworthy: the registry
  // RPC below is a blocking round trip, and without the pins another thread
  // could evict a block between matching it here and returning it. Hits that
  // turn out to sit past the end of the answer are unpinned again in phase 4.
  {
    absl::MutexLock lock(mutex_);
    client = registry_client_;
    local_id = raiden_id_;
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      const auto& hash = block_hashes[i];
      // Peek in BOTH cases, so pin_found decides only whether a hit is pinned,
      // never whether it is a hit. GetAndPin would also resurrect an eviction
      // candidate -- it splices the node out of evict_candidate_list_ -- which
      // would make a lookup silently un-queue a block the store had already
      // decided to reclaim, and make the space accounting behind it wrong.
      // Candidates are invisible to lookup; pinning must not change that.
      const RaidenBlockId* existing = lru_cache_.Peek(hash);
      if (existing != nullptr) {
        if (options.pin_found) {
          // Not a candidate (Peek just proved it), so this only moves the node
          // from the active LRU list to the pinned list.
          lru_cache_.Pin(hash);
        }
        local_hit[i] = true;
        continue;
      }
      if (!options.enable_interleaved_lookup) {
        // Without interleaving the answer cannot continue past a local miss,
        // so everything from here on is the registry's to answer.
        missing_hashes.assign(block_hashes.begin() + i, block_hashes.end());
        break;
      }
      missing_hashes.push_back(hash);
    }
  }

  // Phase 2: ask the registry about the holes only. A failure is not an error
  // for the caller -- it just means nothing fills the holes, so the answer ends
  // at the first one.
  std::vector<global_registry::KVBlockMetadata> remote_hits;
  if (!missing_hashes.empty() && options.enable_global && client != nullptr) {
    auto global_res_or = client->Lookup(missing_hashes, local_id);
    if (global_res_or.ok()) {
      remote_hits = std::move(global_res_or).value();
    } else {
      LOG(WARNING) << "Global registry lookup failed: "
                   << global_res_or.status().message();
    }
  }

  // Phases 3 and 4, under one lock: walk the request in order taking each hash
  // from whichever source has it, then hand back the pins that fell past the
  // end of the answer.
  BlockSliceList results;
  results.reserve(block_hashes.size());
  {
    absl::MutexLock lock(mutex_);

    // The registry answers positionally against missing_hashes and truncates at
    // its own first miss, so its results line up with the holes in the order
    // they were found.
    size_t remote_idx = 0;
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      if (local_hit[i]) {
        // Re-read rather than trusting the sweep: a hit that was not pinned
        // (pin_found unset) can have been evicted while the registry RPC was in
        // flight, and reporting a block this node no longer holds is worse than
        // a short answer. A pinned hit is still here by construction, and this
        // costs one index lookup either way.
        const RaidenBlockId* existing = lru_cache_.Peek(block_hashes[i]);
        if (existing == nullptr) {
          break;  // Gone underneath us: the answer ends here.
        }
        results.emplace_back(block_hashes[i], *existing);
        continue;
      }
      if (remote_idx >= remote_hits.size()) {
        break;  // Neither source has this hash: the answer ends here.
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
        // Unverified HOST hit from a registry that did not filter the caller.
        break;
      }
      results.emplace_back(
          block_hashes[i], RaidenBlockId(remote_id, metadata.block_id(), BlockStatus::REMOTE));
    }

    // Give back the pins taken in phase 1 for hits that fell past the end of
    // the answer. The caller never learns about those blocks, so it would never
    // release them itself. Unpin is a no-op for a hash that is no longer there.
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

std::pair<bool, BlockSliceList> HostOffloadBackend::Insert(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockId> slices, bool /*on_host*/) {
  BlockSliceList evicted_entries;
  bool all_inserted = true;
  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;

  {
    absl::MutexLock lock(mutex_);
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      const std::string& hash = block_hashes[i];
      if (lru_cache_.Contains(hash)) {
        all_inserted = false;
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
        evicted_entries.push_back(std::move(*evicted));
      }
    }
    client = registry_client_;
    local_id = raiden_id_;
  }

  // Withdraw only: a completed save publishes, this does not. What was evicted
  // to make room is unreadable from now on, and nothing else retires it.
  if (client != nullptr && !evicted_entries.empty()) {
    std::vector<std::string> evicted_hashes;
    evicted_hashes.reserve(evicted_entries.size());
    for (const auto& [hash, block_id] : evicted_entries) {
      evicted_hashes.push_back(hash);
    }
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

  return std::make_pair(all_inserted, std::move(evicted_entries));
}

bool HostOffloadBackend::InsertAndLock(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockId> slices, bool /*on_host*/) {
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
    // An entry that is only in HBM describes a device block and nothing else:
    // no host bytes were ever written for it, so there is nothing here worth
    // keeping. Its device block belongs to whoever is inserting now, and the
    // one recorded earlier may since have been handed back and refilled with
    // unrelated data -- a save driven from that stale id would copy the wrong
    // bytes. Take the caller's binding, which is the only current one.
    //
    // HOST and HOST_AND_HBM entries are left alone: those DO hold host bytes,
    // and the caller is re-offering a block the store already has.
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

  for (auto it = new_indices.rbegin(); it != new_indices.rend(); ++it) {
    size_t i = *it;
    const std::string& hash = block_hashes[i];
    if (metadata_.has_value()) {
      if (const RaidenBlockId* stale =
              lru_cache_.PeekIncludingCandidates(hash)) {
        ClearMetadataEntry(*stale);
      }
    }
    // Put's displaced entry is discarded: a displaced entry becomes an
    // eviction candidate that still holds its host block, and GetEvictableKeys
    // finds it there. Nothing puts a displaced entry back.
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

size_t HostOffloadBackend::ReleaseAndDelete(
    absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  size_t deleted_blocks = 0;

  for (auto it = block_hashes.rbegin(); it != block_hashes.rend(); ++it) {
    lru_cache_.Unpin(*it);
  }

  for (const std::string& hash : block_hashes) {
    auto* val = lru_cache_.Peek(hash);
    if (val != nullptr && lru_cache_.GetPinCount(hash) == 0 &&
        val->status != BlockStatus::HOST &&
        val->status != BlockStatus::HOST_AND_HBM) {
      lru_cache_.Erase(hash);
      deleted_blocks++;
    }
  }

  return deleted_blocks;
}

void HostOffloadBackend::Delete(absl::Span<const std::string> block_hashes,
                                absl::Span<const RaidenBlockId> /*slices*/) {
  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;
  std::vector<std::string> deleted_hashes;

  {
    absl::MutexLock lock(mutex_);
    for (const std::string& hash : block_hashes) {
      if (lru_cache_.GetPinCount(hash) > 0) {
        LOG(WARNING) << "Delete skipped pinned block hash (release it first): "
                     << absl::BytesToHexString(hash);
        continue;
      }
      if (const RaidenBlockId* val = lru_cache_.PeekIncludingCandidates(hash)) {
        ClearMetadataEntry(*val);
      }
      lru_cache_.Erase(hash);
      deleted_hashes.push_back(hash);
    }
    client = registry_client_;
    local_id = raiden_id_;
  }

  if (client != nullptr && !deleted_hashes.empty()) {
    client
        ->UnregisterAsync(deleted_hashes, local_id,
                          global_registry::kUnwaitedMutationTimeout)
        .OnReady([](absl::Status status) {
          if (!status.ok()) {
            LOG(WARNING) << "Global registry unregister failed: "
                         << status.message();
          }
        });
  }
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
  absl::Status allocate_status =
      raiden_controller_->AllocateTargetBlockIds(block_ids);
  if (!allocate_status.ok()) {
    return allocate_status;
  }

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

bool HostOffloadBackend::ValidateAndPinHostBlocks(
    absl::Span<const int> host_block_ids) {
  absl::MutexLock lock(mutex_);
  std::vector<std::string> keys_to_pin;
  keys_to_pin.reserve(host_block_ids.size());
  for (int host_id : host_block_ids) {
    bool found = false;
    for (const auto& [key, it] : lru_cache_.map()) {
      if (it->value.host_block_id == host_id &&
          (it->value.status == BlockStatus::HOST ||
           it->value.status == BlockStatus::HOST_AND_HBM)) {
        keys_to_pin.push_back(key);
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  for (size_t i = 0; i < keys_to_pin.size(); ++i) {
    if (!lru_cache_.Pin(keys_to_pin[i])) {
      for (size_t j = 0; j < i; ++j) {
        lru_cache_.Unpin(keys_to_pin[j]);
      }
      return false;
    }
  }
  return true;
}

std::vector<std::string> HostOffloadBackend::GetEvictableKeys(size_t count) {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetEvictableKeys(count);
}

std::vector<int> HostOffloadBackend::Evict(
    const std::vector<std::string>& block_hashes) {
  std::vector<int> host_ids_to_deallocate;
  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;
  std::vector<std::string> evicted_hashes;

  {
    absl::MutexLock lock(mutex_);
    for (const std::string& hash : block_hashes) {
      const RaidenBlockId* block = lru_cache_.PeekIncludingCandidates(hash);
      if (block != nullptr && lru_cache_.GetPinCount(hash) == 0 &&
          (block->status == BlockStatus::HOST ||
           block->status == BlockStatus::HOST_AND_HBM)) {
        host_ids_to_deallocate.push_back(block->host_block_id);
        ClearMetadataEntry(*block);
        lru_cache_.Erase(hash);
        evicted_hashes.push_back(hash);
      }
    }
    client = registry_client_;
    local_id = raiden_id_;
  }

  if (client != nullptr && !evicted_hashes.empty()) {
    client
        ->UnregisterAsync(evicted_hashes, local_id,
                          global_registry::kUnwaitedMutationTimeout)
        .OnReady([](absl::Status status) {
          if (!status.ok()) {
            LOG(WARNING) << "Global registry unregister failed: "
                         << status.message();
          }
        });
  }

  return host_ids_to_deallocate;
}

std::vector<std::string> HostOffloadBackend::GetEvictCandidateKeys() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetEvictCandidateKeys();
}

KVCacheStoreServer* HostOffloadBackend::store_server() const {
  absl::MutexLock lock(mutex_);
  return server_.get();
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

  ASSIGN_OR_RETURN(global_registry::StoreInfo store_info,
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
    absl::Duration requested_deadline,
    absl::Duration hold_window,
    WriteRemoteVerdictCallback on_verdict) {
  if (block_hashes.empty()) {
    return absl::InvalidArgumentError("WriteRemote requires at least one hash");
  }
  if (block_hashes.size() != src_host_block_ids.size()) {
    return absl::InvalidArgumentError(
        "src_host_block_ids must have one entry per block hash");
  }

  // This resolves the peer through the global registry, and its own
  // FailedPrecondition ("No global registry client") IS the registry
  // precondition for the whole feature -- there is no separate check.
  ASSIGN_OR_RETURN(std::shared_ptr<KVCacheStoreClient> client,
                   GetKVCacheStoreClient(dst_raiden_id));

  auto call = client->WriteRemote(raiden_controller_->unit(), block_hashes,
                                  src_host_block_ids,
                                  BuildLocalWorkerEndpoints(raiden_controller_),
                                  absl::ToInt64Milliseconds(requested_deadline),
                                  hold_window, std::move(on_verdict));
  auto response_or = call.ack.Await();
  if (!response_or.ok()) {
    // On a transport error the peer may have restarted on a new port; drop
    // the store client so the next attempt re-resolves instead of
    // redialling a dead one. An application answer -- e.g. the
    // RESOURCE_EXHAUSTED refusing a batch the peer cannot fit -- proves the
    // peer is alive on this channel, and refusals cluster exactly when a
    // reconnect is most wasteful: under memory pressure.
    if (IsTransportError(response_or.status())) {
      InvalidateStoreClient(dst_raiden_id);
    }
    return response_or.status();
  }

  RemoteWriteAck ack;
  ack.cancel = std::move(call.cancel);
  ack.operation_id = response_or->operation_id();
  ack.granted_deadline = absl::Milliseconds(response_or->granted_deadline_ms());
  switch (response_or->exist_state()) {
    case ::tpu_raiden::kv_cache::proto::WRITE_ALL_EXIST:
      ack.all_exist = true;
      return ack;
    case ::tpu_raiden::kv_cache::proto::WRITE_PARTIAL_EXIST:
      ack.existing_hashes.assign(response_or->existing_hashes().begin(),
                                 response_or->existing_hashes().end());
      return ack;
    default:
      break;
  }
  if (ack.operation_id == 0) {
    // Neither an existence answer nor an accepted operation: a
    // default-initialised reply, which is a protocol error rather than a
    // silent no-op.
    return absl::InternalError(
        "Destination accepted the offer but returned no operation id.");
  }
  return ack;
}

tsl::Future<proto::PollWriteRemoteResponse>
HostOffloadBackend::PollWriteRemoteAsync(const RaidenId& dst_raiden_id,
                                        uint64_t operation_id,
                                        int64_t wait_ms) {
  auto client_or = GetKVCacheStoreClient(dst_raiden_id);
  if (!client_or.ok()) {
    return tsl::Future<proto::PollWriteRemoteResponse>(client_or.status());
  }
  return (*client_or)->PollWriteRemote(operation_id, wait_ms);
}

std::vector<std::string> HostOffloadBackend::AlreadyPresentHostResident(
    absl::Span<const std::string> block_hashes) const {
  absl::MutexLock lock(mutex_);
  std::vector<std::string> present;
  for (const auto& hash : block_hashes) {
    // PeekIncludingCandidates, not Peek: an eviction candidate still holds its
    // host block, so calling it absent would let a duplicate insert through.
    // Peek also promotes LRU order, which a question has no business doing.
    const RaidenBlockId* entry = lru_cache_.PeekIncludingCandidates(hash);
    if (entry != nullptr && (entry->status == BlockStatus::HOST ||
                             entry->status == BlockStatus::HOST_AND_HBM)) {
      present.push_back(hash);
    }
  }
  return present;
}

bool HostOffloadBackend::InsertAllOrNothing(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockId> slices) {
  if (block_hashes.empty() || block_hashes.size() != slices.size()) {
    return false;
  }
  absl::MutexLock lock(mutex_);

  // Phase 1: validate the WHOLE batch before touching anything. Three things
  // this has to get right that a plain Insert does not:
  //
  //   * PeekIncludingCandidates, not Contains. Contains reports false for an
  //     eviction candidate, but a candidate is still HOST with its block
  //     still allocated, so Contains would let a duplicate through.
  //   * available_space(), not capacity(). available_space() subtracts the
  //     pinned entries; capacity() ignores them, so a cache full of pinned
  //     entries looks roomy.
  //   * Do not trust the precheck alone. Put has silent do-nothing paths --
  //     with everything pinned, its internal evict finds no victim and it
  //     inserts nothing -- which would commit some hashes, drop others, and
  //     report success for all.
  for (const auto& hash : block_hashes) {
    if (lru_cache_.PeekIncludingCandidates(hash) != nullptr) {
      return false;
    }
  }
  if (block_hashes.size() > lru_cache_.available_space()) {
    return false;
  }

  // Phase 2: insert. Phase 1 rejected duplicates, so no Put can rebind an
  // existing hash in place and orphan its old host block. The check below is
  // what keeps that true if Phase 1 ever changes.
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    const std::string& hash = block_hashes[i];
    std::optional<std::pair<std::string, RaidenBlockId>> evicted =
        lru_cache_.Put(hash, slices[i]);
    if (lru_cache_.Peek(hash) == nullptr) {
      LOG(ERROR) << "InsertAllOrNothing: LRUCache::Put inserted nothing for a "
                    "validated hash; rolling the batch back";
      for (size_t j = 0; j < i; ++j) {
        lru_cache_.Erase(block_hashes[j]);
      }
      return false;
    }
    SetMetadataEntry(hash, slices[i]);
    // An eviction here only demotes an entry to the candidate list; its host
    // block stays allocated, so there is nothing to free.
    (void)evicted;
  }
  return true;
}

void HostOffloadBackend::RollbackInsert(
    absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> host_block_ids) {
  {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : block_hashes) {
      if (const RaidenBlockId* entry =
              lru_cache_.PeekIncludingCandidates(hash)) {
        ClearMetadataEntry(*entry);
      }
      lru_cache_.Erase(hash);
    }
  }
  // Erasing an entry does NOT return its block to the pool. Without this a
  // failed registration leaks every landing block it touched.
  if (!host_block_ids.empty()) {
    (void)raiden_controller_->DeallocateBlockIds(
        std::vector<int>(host_block_ids.begin(), host_block_ids.end()));
  }
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
    // No registry configured, so there is nothing to advertise and no way for
    // these blocks to end up unreachable-but-believed-reachable. Success.
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
    absl::Span<const RaidenBlockId> slices) {
  if (block_hashes.empty()) {
    return tsl::Future<>(absl::OkStatus());
  }

  // Every hash needs a destination device block.
  if (device_block_ids.size() != block_hashes.size()) {
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
    return LoadRemoteBlocks(remote_id, block_hashes, device_block_ids);
  }
  return LoadLocalHostBlocks(block_hashes, device_block_ids, slices);
}

tsl::Future<> HostOffloadBackend::LoadRemoteBlocks(
    const RaidenId& remote_id, absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> device_block_ids) {
  auto client_or = GetKVCacheStoreClient(remote_id);
  if (!client_or.ok()) {
    return tsl::Future<>(client_or.status());
  }
  std::shared_ptr<KVCacheStoreClient> client = std::move(client_or.value());

  auto host_blocks_or =
      raiden_controller_->AllocateBlockIds(block_hashes.size());
  if (!host_blocks_or.ok()) {
    return tsl::Future<>(host_blocks_or.status());
  }
  std::vector<int32_t> dst_host_block_ids(host_blocks_or.value().begin(),
                                          host_blocks_or.value().end());

  auto [load_promise, load_future] = tsl::MakePromise<>();

  ::tpu_sync::rpc::RaidenIdProto client_raiden_id = raiden_controller_->unit();
  std::vector<::tpu_sync::proto::RaidenWorkerEndpointsProto>
      client_worker_endpoints = BuildLocalWorkerEndpoints(raiden_controller_);
  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> fetch_future =
      client->Fetch(block_hashes, device_block_ids, dst_host_block_ids,
                    client_raiden_id, client_worker_endpoints);

  fetch_future.OnReady(
      [this, remote_id, dst_host_block_ids,
       dev_ids_vec = std::vector<int32_t>(device_block_ids.begin(),
                                          device_block_ids.end()),
       load_promise = std::move(load_promise)](
          const absl::StatusOr<::tpu_raiden::kv_cache::proto::FetchResponse>&
              response_or) mutable {
        if (!response_or.ok()) {
          (void)raiden_controller_->DeallocateBlockIds(dst_host_block_ids);
          // The peer may have restarted on a new port; drop the cached client
          // so the next attempt re-resolves instead of redialling a dead one.
          InvalidateStoreClient(remote_id);
          load_promise.Set(response_or.status());
          return;
        }

        const auto& response = response_or.value();
        if (!response.failed_block_hashes().empty()) {
          (void)raiden_controller_->DeallocateBlockIds(dst_host_block_ids);
          std::string err_msg = response.error_message().empty()
                                    ? "Fetch RPC returned failed blocks"
                                    : response.error_message();
          load_promise.Set(absl::InternalError(err_msg));
          return;
        }

        std::vector<Buffer> src_buffers;
        src_buffers.reserve(dst_host_block_ids.size());
        for (int id : dst_host_block_ids) {
          src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                                   ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
        }

        std::vector<Buffer> dst_buffers;
        dst_buffers.reserve(dev_ids_vec.size());
        for (int id : dev_ids_vec) {
          dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                                   ::tpu_sync::rpc::MEMORY_TYPE_HBM);
        }

        tsl::Future<> h2d_future =
            raiden_controller_->TransferBuffers(src_buffers, dst_buffers);

        h2d_future.OnReady(
            [this, dst_host_block_ids, load_promise = std::move(load_promise)](
                absl::Status status) mutable {
              (void)raiden_controller_->DeallocateBlockIds(dst_host_block_ids);
              load_promise.Set(status);
            });
      });

  return load_future;
}

tsl::Future<> HostOffloadBackend::LoadLocalHostBlocks(
    absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> device_block_ids,
    absl::Span<const RaidenBlockId> slices) {
  std::vector<int64_t> src_host_block_ids;
  src_host_block_ids.reserve(block_hashes.size());
  if (!slices.empty()) {
    if (slices.size() != block_hashes.size()) {
      return tsl::Future<>(absl::InvalidArgumentError(
          "Mismatched slices count vs block_hashes count."));
    }
    for (size_t i = 0; i < slices.size(); ++i) {
      if (slices[i].host_block_id == -1) {
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
        return tsl::Future<>(absl::NotFoundError(
            absl::StrCat("Block hash not found in host backend: ", hash)));
      }
      if (entry->status != BlockStatus::HOST &&
          entry->status != BlockStatus::HOST_AND_HBM) {
        return tsl::Future<>(absl::FailedPreconditionError(
            absl::StrCat("Block is not on host: ", hash)));
      }
      if (entry->host_block_id == -1) {
        return tsl::Future<>(absl::FailedPreconditionError(
            absl::StrCat("Block host_block_id is -1: ", hash)));
      }
      src_host_block_ids.push_back(entry->host_block_id);
    }
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

  return raiden_controller_->TransferBuffers(src_buffers, dst_buffers);
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
  ASSIGN_OR_RETURN(
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
