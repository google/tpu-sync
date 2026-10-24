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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_POSIX_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_POSIX_BACKEND_H_

#include <fcntl.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/lru_cache.h"

namespace tpu_raiden {
namespace kv_cache {

class BlockTracker;

namespace backends {
namespace storage {

// Canonical identifier for the POSIX storage tier.
inline constexpr absl::string_view kPosixBackendName = "posix";

inline constexpr size_t kDefaultLookupBatchSize = 32;

// Prefix-directory widths, in hex characters, over the encoded block hash.
// 3 + 2 yields 16^5 == 1,048,576 leaf directories per topology namespace.
inline constexpr size_t kHashDirL1Width = 3;
inline constexpr size_t kHashDirL2Width = 2;
// Hex encoding doubles the length, and the ".bin" suffix costs 4 more
// characters: 2*125 + 4 == 254 <= NAME_MAX (255). Temp files are named
// independently of the hash (see MakeTempPath), so this is the only
// length budget a block hash has to fit.
inline constexpr size_t kMaxBlockHashBytes = 125;

// Returns a unique path for a temporary file in the same directory as
// `final_path`: "<dir>/.tmp_<pid>_<nonce>_<seq>". <nonce> is a 64-bit value
// drawn once per process from OS entropy, so names are unique across hosts
// and PID namespaces that share one filesystem (e.g. pods writing to one
// Lustre directory), where pid and a counter alone repeat. <pid> still
// distinguishes a forked child, which inherits the nonce; <seq> distinguishes
// successive calls within a process. The name does not embed the final
// file's basename, so its length (< 60 characters) is independent of the key.
std::string MakeTempPath(absl::string_view final_path);

// Typed, validated view of a POSIX backend's configuration. Every POSIX knob
// lives here; `FromProperties` is the single parse-and-validate point.
struct PosixBackendOptions {
  std::string root_dir = "/tmp/raiden_storage";
  std::string model_name = "unknown";
  // Storage topology; see ParallelismConfig. -1 = axis not declared.
  int tp_size = kAxisUndeclared;
  int tp_rank = kAxisUndeclared;
  int pcp_size = kAxisUndeclared;
  int pcp_rank = kAxisUndeclared;
  int pp_size = kAxisUndeclared;
  int pp_rank = kAxisUndeclared;
  size_t capacity_bytes = 0;
  size_t lookup_batch_size = kDefaultLookupBatchSize;

  // Worker threads in this backend instance's NumaThreadPool. Every async
  // POSIX file operation runs on it -- WriteAsync, ReadAsync and
  // BatchExistsAsync share one pool and one queue, one task per block -- so
  // this is also the ceiling on concurrent pwrite/pread/stat syscalls issued
  // by this backend. Not per-NUMA-node, not per-file-descriptor.
  int storage_io_thread_pool_size = 16;

  // When true, the backend opens block files with O_DIRECT to bypass the
  // Linux page cache. Defaults to false if omitted. Probed once at
  // construction; if the filesystem under `root_dir` does not accept O_DIRECT,
  // construction fails. Every buffer pointer, buffer length and file offset
  // must then be a multiple of DirectIOAlignment(); otherwise the operation
  // fails with InvalidArgument. There is no buffered fallback.
  bool direct_io = false;

  // Capacity of the coordinator-side PosixKVCacheStoreBackend metadata cache,
  // which remembers storage files confirmed to exist so repeated lookups of a
  // hot prefix skip the existence syscall. Counted in shard entries (one per
  // file). 0 disables the cache (default); kUnboundedMetadataCache (-1) means
  // unbounded. Least recently used entries are dropped past the cap.
  // Recommended for large topologies; see the lookup cost in PosixPathMapper.
  int64_t metadata_cache_max_entries = 0;
  // Idle TTL, in seconds, of a cached existence entry: it expires once it has
  // gone this long without a cache hit or a storage re-confirmation. Stale
  // entries for deleted files are also dropped when a recall from this tier
  // fails. kUnboundedMetadataCache (-1) means entries never expire.
  int64_t metadata_cache_ttl_secs = 60;

  // Parses and validates `properties`. Returns InvalidArgumentError for an
  // unparseable value, a negative thread-pool size, or a topology rejected by
  // ValidateWorkerParallelism (e.g. tp_size 0, a rank out of range, or a rank
  // given without its axis size).
  static absl::StatusOr<PosixBackendOptions> FromProperties(
      const absl::flat_hash_map<std::string, std::string>& properties);
};

// Sentinel for metadata_cache_max_entries / metadata_cache_ttl_secs.
inline constexpr int64_t kUnboundedMetadataCache = -1;

// Runtime configuration of PosixKVCacheStoreBackend's existence cache.
struct MetadataCacheOptions {
  // Shard entries. 0 disables the cache; std::numeric_limits<size_t>::max()
  // means unbounded.
  size_t max_entries = 0;
  // Idle TTL. absl::InfiniteDuration() means entries never expire.
  absl::Duration ttl = absl::Seconds(60);

  bool enabled() const { return max_entries > 0; }

  // Converts the validated POSIX options into runtime cache options.
  static MetadataCacheOptions FromPosixOptions(
      const PosixBackendOptions& options);
};

// Alignment required for O_DIRECT buffers, lengths and file offsets: the
// runtime page size (4 KiB on x86_64, commonly 16 KiB or 64 KiB on arm64).
// Queried once and cached.
size_t DirectIOAlignment();

// True when every slice's address and length are multiples of
// DirectIOAlignment().
bool SlicesAreDirectIOAligned(absl::Span<const HostBufferDescriptor> slices);

// PosixKVBackend implements KVBackend for POSIX filesystems (e.g., Lustre,
// local disk).
class PosixKVBackend : public KVBackend {
 public:
  // `properties` carries the topology the caller declares (`tp_size`/`tp_rank`,
  // `pcp_size`/`pcp_rank`, `pp_size`/`pp_rank`; none declared is valid), which
  // PosixBackendOptions validates. An invalid configuration is a LOG(FATAL).
  PosixKVBackend(std::string name,
                 absl::flat_hash_map<std::string, std::string> properties);

  std::string name() const override { return name_; }

  const PosixBackendOptions& options() const { return options_; }

  bool is_direct_io_supported() const { return direct_io_supported_; }

  static bool ProbeDirectIO(absl::string_view dir);

  void WriteAsync(const BlockKey& key,
                  absl::Span<const HostBufferDescriptor> slices,
                  size_t total_bytes,
                  std::function<void(absl::Status)> callback) override;

  void ReadAsync(const BlockKey& key,
                 absl::Span<const HostBufferDescriptor> slices,
                 size_t total_bytes,
                 std::function<void(absl::Status)> callback) override;

  void BatchExistsAsync(
      absl::Span<const BlockKey> keys,
      std::function<void(std::vector<absl::StatusOr<bool>>)> callback) override;

 private:
  absl::StatusOr<bool> Exists(const BlockKey& key);

  std::string name_;
  PosixBackendOptions options_;
  bool direct_io_supported_ = false;
  // MUST be the last-declared member: destruction runs in reverse declaration
  // order, and ~NumaThreadPool joins its workers after draining the queue, so
  // declaring it last is what guarantees every in-flight task finishes before
  // any member it might touch is destroyed. The scheduled lambdas capture raw
  // `this` and rely on this.
  std::unique_ptr<NumaThreadPool> thread_pool_;
};

// Replaces every character of `model_name` outside [A-Za-z0-9._-] with '_',
// so a HuggingFace-style ID (e.g. "meta-llama/Llama-3.1-70B") stays one path
// component. Returns "unknown" for an empty name.
std::string SanitizeModelName(absl::string_view model_name);

// Returns the `<topology_dir>` path component for `parallelism`: one
// `<axis><size>_r<rank>` segment per declared axis, joined with '_', outermost
// first in the fixed order pp, pcp, tp (e.g. `pp2_r1_pcp2_r0_tp4_r3`). A
// declared size-1 axis still contributes a segment (`pp1_r0`). Returns "" when
// no axis is declared. Does not validate `parallelism`.
std::string FormatTopologyDir(const ParallelismConfig& parallelism);

// PosixPathMapper implements the filesystem path resolution policy.
// Maps block hash identifiers and the worker's parallelism coordinates to a
// rank-partitioned hierarchical directory layout over the HEX-ENCODED block
// hash (see BlockKeyMapper::MapKey):
//   `<root_dir>/<model_name>[/<topology_dir>]/<l1>/<l2>/<hash_hex>.bin`
// `<topology_dir>` joins with '_' one `<axis><size>_r<rank>` segment per
// declared axis, outermost first in the fixed order pp, pcp, tp (vLLM's rank
// layout; e.g. `pp2_r1_pcp2_r0_tp4_r3`, `pcp8_r3_tp1_r0`, `tp8_r5`). With no
// axis declared the directory level is omitted. See FormatTopologyDir.
//
// A declared size-1 axis still adds a segment, so `tp8_r3` and `pp1_r0_tp8_r3`
// are different directories. Every engine sharing a `root_dir`, and the store,
// must therefore declare the same axes with the same sizes, size-1 axes
// included. Keeping the segment leaves data written as `tp1_r0/` reachable by
// a worker that declares `tp_size=1, tp_rank=0`.
//
// Lookup cost: the coordinator probes shards_per_block() files per block
// (pp x pcp x tp, e.g. 64 at PCP=8, TP=8). With the metadata cache off (the
// default), every hit costs that many existence syscalls per block, so set
// PosixBackendOptions::metadata_cache_max_entries for large topologies.
class PosixPathMapper : public BlockKeyMapper {
 public:
  static absl::string_view GetParentDir(absl::string_view path) {
    size_t last_slash = path.find_last_of('/');
    if (last_slash == absl::string_view::npos) return "";
    return path.substr(0, last_slash);
  }

  // `parallelism` is the worker's default topology; each axis whose size is
  // kAxisUndeclared is left out of the path.
  PosixPathMapper(absl::string_view root_dir, absl::string_view model_name,
                  const ParallelismConfig& parallelism);

  // Each `options.parallelism` field other than -1 (kAxisUndeclared) overrides
  // the configured value of that field, including invalid values. Returns
  // InvalidArgumentError if the resulting topology fails
  // ValidateWorkerParallelism.
  absl::StatusOr<BlockKey> MapKey(
      const std::string& block_hash,
      const KeyMappingOptions& options = {}) const override;
  int tp_size() const override { return parallelism_.tp_size; }
  int pcp_size() const override { return parallelism_.pcp_size; }
  int pp_size() const override { return parallelism_.pp_size; }

 private:
  std::string root_dir_;
  std::string model_name_;
  ParallelismConfig parallelism_;
};

// PosixKVCacheStoreBackend probes persistent storage (e.g., Lustre, POSIX).
//
// Optional metadata cache: when enabled, Lookup remembers which shard files it
// has confirmed on storage and answers later lookups of them without a storage
// probe. See Lookup for the block-to-shard-key mapping and cache contents.
class PosixKVCacheStoreBackend : public KVCacheStoreBackend {
 public:
  PosixKVCacheStoreBackend(std::shared_ptr<KVBackend> storage_backend,
                           std::string name = std::string(kPosixBackendName),
                           size_t capacity_bytes = 0,
                           size_t lookup_batch_size = kDefaultLookupBatchSize,
                           MetadataCacheOptions cache_options = {});

  std::string name() const override { return name_; }

  size_t lookup_batch_size() const { return lookup_batch_size_; }

  const MetadataCacheOptions& metadata_cache_options() const {
    return cache_options_;
  }

  // Number of entries currently held by the metadata cache (0 if disabled).
  size_t metadata_cache_size() const;

  // Queries `block_hashes` on storage (longest prefix in kPrefix mode, or all
  // available blocks in kSubset mode; returns empty when host_resident_only is
  // set).
  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override;

  tsl::Future<> Load(const RaidenId& remote_id,
                     absl::Span<const std::string> block_hashes,
                     absl::Span<const int32_t> device_block_ids,
                     absl::Span<const RaidenBlockId> slices,
                     BlockTracker* absl_nonnull load_tracker) override {
    return tsl::Future<>(absl::OkStatus());
  }

  // No-op on the storage tier (blocks are written via async offload transfers).
  // Returns false for kAdmitStrict, which applies only to host-RAM admission.
  bool Insert(absl::Span<const std::string> block_hashes,
              absl::Span<const RaidenBlockId> slices,
              InsertOption option = InsertOption::kBestEffortUpdate) override {
    return option != InsertOption::kAdmitStrict;
  }

  // Drops cached shard keys of `block_hashes` so the next Lookup re-probes
  // storage. Returns the number of block hashes whose cached keys were erased.
  // No-op (returns 0) when `options.host_resident_only` is true or the cache
  // is disabled.
  size_t Delete(absl::Span<const std::string> block_hashes,
                const DeleteOptions& options = {}) override;
  bool Pin(absl::Span<const std::string> block_hashes) override { return true; }
  void Release(absl::Span<const std::string> block_hashes) override {}
  int GetPinCount(const std::string& hash) const override { return 0; }
  size_t GetCapacity() const override { return capacity_bytes_; }
  size_t GetSize() const override { return 0; }
  size_t GetAvailableSpace() const override { return capacity_bytes_; }

  std::shared_ptr<KVBackend> storage_backend() const {
    return storage_backend_;
  }

 private:
  struct ExistenceEntry {
    // Last cache hit or storage confirmation of the shard file. Anchor of the
    // idle TTL.
    absl::Time last_used;
  };

  // Returns the storage keys ("shard keys") that must all exist for ONE block,
  // `block_hash`, to be available: one key per (pp_rank, pcp_rank, tp_rank)
  // worker, mapper()->shards_per_block() keys in total, pp_rank-major, then
  // pcp_rank.
  absl::StatusOr<std::vector<BlockKey>> MapShardKeys(
      const std::string& block_hash) const;

  // Indexing: i = position of the block in the Lookup request
  // (block_hashes[i]); j = position of the shard key within
  // MapShardKeys(block_hashes[i]).
  // Returns fresh[i][j] = true iff shard_keys[i][j] has a fresh cache entry.
  // Expired entries are erased and reported as absent.
  std::vector<std::vector<bool>> CachedFresh(
      absl::Span<const std::vector<BlockKey>> shard_keys);

  // Records storage-confirmed shard keys. A key that is already cached keeps
  // its original timestamp.
  void CacheInsert(absl::Span<const BlockKey* const> keys);

  // Synchronously probes storage for `keys`. Returns one entry per key; a
  // short or failed answer is reported as absent from that point on.
  std::vector<bool> ProbeExists(absl::Span<const BlockKey> keys);

  RaidenBlockId MakeSharedStorageBlock() const;

  std::shared_ptr<KVBackend> storage_backend_;
  std::string name_ = std::string(kPosixBackendName);
  size_t capacity_bytes_ = 0;
  size_t lookup_batch_size_ = 32;
  const MetadataCacheOptions cache_options_;

  // Never held across storage I/O or key mapping.
  mutable absl::Mutex cache_mu_;
  LRUCache<std::string, ExistenceEntry> cache_ ABSL_GUARDED_BY(cache_mu_);
};

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_POSIX_BACKEND_H_
