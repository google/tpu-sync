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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_STORAGE_LOOKUP_HELPER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_STORAGE_LOOKUP_HELPER_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

// Coordinator-side lookup logic shared by the storage-tier
// KVCacheStoreBackend implementations.
//
// A backend passes two callables with each call: one expands a block hash
// into the storage keys that must all exist for the block to be available,
// and one probes a batch of keys on storage. The helper does the rest:
// LookupOptions handling, batched probing, the storage_metadata_check_total
// counter, and the SHARED_STORAGE results. A backend that keeps a cache of
// confirmed keys passes two more callables, so that Lookup uses its cache.
//
// Thread-safe. The helper holds no mutable state.
class StorageLookupHelper {
 public:
  // Returns the storage keys ("shard keys") that must all exist for
  // `block_hash` to be available.
  using KeyExpander = absl::FunctionRef<absl::StatusOr<std::vector<BlockKey>>(
      const std::string& block_hash)>;

  // Synchronously probes storage for `keys`. Returns one entry per key, true
  // iff the key exists. A key whose probe failed is reported as absent.
  using ExistenceProber =
      absl::FunctionRef<std::vector<bool>(absl::Span<const BlockKey> keys)>;

  // Reads the backend's cache of confirmed keys. shard_keys[i] holds the
  // shard keys of the i-th block. Returns fresh[i][j] = true iff
  // shard_keys[i][j] is cached. Lookup does not probe a cached key.
  using CachedFreshFn = absl::FunctionRef<std::vector<std::vector<bool>>(
      absl::Span<const std::vector<BlockKey>> shard_keys)>;

  // Records in the backend's cache the keys that a probe confirmed.
  using CacheInsertFn =
      absl::FunctionRef<void(absl::Span<const BlockKey* const> keys)>;

  // `data_name` is the RaidenId::data_name of every block Lookup returns.
  // `dp_rank` labels the storage_metadata_check_total counter.
  // `lookup_batch_size` is the number of keys per probe and must be > 0.
  StorageLookupHelper(std::string data_name, std::string dp_rank,
                      size_t lookup_batch_size);

  // Returns one key per (pp_rank, pcp_rank, tp_rank) worker of `mapper`'s
  // topology, mapper.shards_per_block() keys in total, pp_rank-major, then
  // pcp_rank. An undeclared axis spans a single shard and is left unset in
  // the mapping options.
  static absl::StatusOr<std::vector<BlockKey>> ExpandShardKeys(
      const BlockKeyMapper& mapper, const std::string& block_hash);

  // Probes `keys` through backend.BatchExistsAsync and waits for the answer.
  // Returns one entry per key; a short or failed answer is reported as
  // absent.
  static std::vector<bool> ProbeExists(KVBackend& backend,
                                       absl::Span<const BlockKey> keys);

  // Returns InvalidArgumentError for the option combinations that
  // LookupOptions documents as invalid.
  static absl::Status ValidateLookupOptions(const LookupOptions& options);

  // Queries `block_hashes` on storage: the longest available prefix in
  // kPrefix mode, or every available block in kSubset mode. A block is
  // available when all of its shard keys exist. Returns empty when
  // `options.host_resident_only` is set, because the storage tier is not
  // host-DRAM-resident. Every shard key is probed on storage.
  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes, const LookupOptions& options,
      KeyExpander expand_keys, ExistenceProber probe_exists) const;

  // Same as above, for a backend with a cache of confirmed keys. A key that
  // `cached_fresh` reports is not probed, and the keys that a probe confirms
  // are passed to `cache_insert`. See the definition for the lookup phases.
  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes, const LookupOptions& options,
      KeyExpander expand_keys, ExistenceProber probe_exists,
      CachedFreshFn cached_fresh, CacheInsertFn cache_insert) const;

 private:
  RaidenBlockId MakeSharedStorageBlock() const;

  const std::string data_name_;
  const std::string dp_rank_;
  const size_t lookup_batch_size_;
};

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_STORAGE_LOOKUP_HELPER_H_
