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

#include "tpu_sync/kv_cache/backends/storage/storage_lookup_helper.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/statusor.h"  // IWYU pragma: keep
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

constexpr char kUnmappableHash[] = "unmappable";

// Maps a block hash to "<hash>/pp<rank>/pcp<rank>/tp<rank>". The rank of an
// undeclared axis prints as -1.
class FakeMapper : public BlockKeyMapper {
 public:
  explicit FakeMapper(ParallelismConfig parallelism)
      : parallelism_(parallelism) {}

  absl::StatusOr<BlockKey> MapKey(
      const std::string& block_hash,
      const KeyMappingOptions& options = {}) const override {
    if (block_hash == kUnmappableHash) {
      return absl::InvalidArgumentError("hash cannot be mapped");
    }
    const ParallelismConfig& p = options.parallelism;
    return BlockKey{
        .block_hash = block_hash,
        .resolved_key = absl::StrCat(block_hash, "/pp", p.pp_rank, "/pcp",
                                     p.pcp_rank, "/tp", p.tp_rank)};
  }
  int tp_size() const override { return parallelism_.tp_size; }
  int pcp_size() const override { return parallelism_.pcp_size; }
  int pp_size() const override { return parallelism_.pp_size; }

 private:
  ParallelismConfig parallelism_;
};

// In-memory storage with a StorageLookupHelper wired to it. It also keeps a
// set of cached keys, which only LookupWithCache reads and writes.
class FakeStorage {
 public:
  explicit FakeStorage(ParallelismConfig parallelism,
                       size_t lookup_batch_size = 32)
      : mapper_(parallelism), helper_("fake_store", "7", lookup_batch_size) {}

  // Stores the shard of every rank of `block_hash`.
  void StoreBlock(const std::string& block_hash) {
    for (const BlockKey& key : ShardKeys(block_hash)) {
      stored_.insert(key.resolved_key);
    }
  }
  void StoreKey(const std::string& resolved_key) {
    stored_.insert(resolved_key);
  }
  void EvictKey(const std::string& resolved_key) {
    stored_.erase(resolved_key);
  }

  std::vector<BlockKey> ShardKeys(const std::string& block_hash) const {
    return *StorageLookupHelper::ExpandShardKeys(mapper_, block_hash);
  }

  // Lookup of a backend without a cache.
  absl::StatusOr<BlockSliceList> Lookup(std::vector<std::string> block_hashes,
                                        const LookupOptions& options) {
    return helper_.Lookup(
        block_hashes, options,
        [this](const std::string& block_hash) { return Expand(block_hash); },
        [this](absl::Span<const BlockKey> keys) { return Probe(keys); });
  }

  // Lookup of a backend that caches confirmed keys in cached_keys().
  absl::StatusOr<BlockSliceList> LookupWithCache(
      std::vector<std::string> block_hashes, const LookupOptions& options) {
    return helper_.Lookup(
        block_hashes, options,
        [this](const std::string& block_hash) { return Expand(block_hash); },
        [this](absl::Span<const BlockKey> keys) { return Probe(keys); },
        [this](absl::Span<const std::vector<BlockKey>> shard_keys) {
          std::vector<std::vector<bool>> fresh;
          for (const std::vector<BlockKey>& keys : shard_keys) {
            std::vector<bool>& row = fresh.emplace_back();
            for (const BlockKey& key : keys) {
              row.push_back(cached_.contains(key.resolved_key));
            }
          }
          return fresh;
        },
        [this](absl::Span<const BlockKey* const> keys) {
          for (const BlockKey* key : keys) cached_.insert(key->resolved_key);
        });
  }

  // Block hashes of a Lookup answer.
  static std::vector<std::string> Hashes(
      const absl::StatusOr<BlockSliceList>& found) {
    EXPECT_TRUE(found.ok()) << found.status();
    std::vector<std::string> hashes;
    if (!found.ok()) return hashes;
    for (const auto& [hash, block] : *found) hashes.push_back(hash);
    return hashes;
  }

  std::set<std::string>& cached_keys() { return cached_; }
  // Number of keys in each probe call, in call order.
  const std::vector<size_t>& probe_sizes() const { return probe_sizes_; }
  void ResetProbes() { probe_sizes_.clear(); }

 private:
  absl::StatusOr<std::vector<BlockKey>> Expand(const std::string& block_hash) {
    return StorageLookupHelper::ExpandShardKeys(mapper_, block_hash);
  }

  std::vector<bool> Probe(absl::Span<const BlockKey> keys) {
    probe_sizes_.push_back(keys.size());
    std::vector<bool> exists;
    exists.reserve(keys.size());
    for (const BlockKey& key : keys) {
      exists.push_back(stored_.contains(key.resolved_key));
    }
    return exists;
  }

  FakeMapper mapper_;
  StorageLookupHelper helper_;
  std::set<std::string> stored_;
  std::set<std::string> cached_;
  std::vector<size_t> probe_sizes_;
};

LookupOptions PrefixOptions() {
  return LookupOptions{.enable_global = false,
                       .match_mode = LookupMatchMode::kPrefix};
}

LookupOptions SubsetOptions() {
  return LookupOptions{.enable_global = false,
                       .match_mode = LookupMatchMode::kSubset};
}

BlockKey KeyOf(const std::string& resolved_key) {
  return BlockKey{.block_hash = resolved_key, .resolved_key = resolved_key};
}

// ===========================================================================
// Key expansion and probing
// ===========================================================================

TEST(StorageLookupHelperTest, ExpandShardKeysCoversEveryRankPpMajor) {
  const FakeMapper mapper({.tp_size = 2, .pp_size = 2});

  TF_ASSERT_OK_AND_ASSIGN(std::vector<BlockKey> keys,
                       StorageLookupHelper::ExpandShardKeys(mapper, "h"));

  std::vector<std::string> resolved_keys;
  for (const BlockKey& key : keys) {
    EXPECT_EQ(key.block_hash, "h");
    resolved_keys.push_back(key.resolved_key);
  }
  // The undeclared pcp axis spans one shard and keeps rank -1.
  EXPECT_THAT(resolved_keys, ElementsAre("h/pp0/pcp-1/tp0", "h/pp0/pcp-1/tp1",
                                         "h/pp1/pcp-1/tp0", "h/pp1/pcp-1/tp1"));
}

TEST(StorageLookupHelperTest, ExpandShardKeysReturnsMapperError) {
  const FakeMapper mapper({.tp_size = 2});
  EXPECT_THAT(StorageLookupHelper::ExpandShardKeys(mapper, kUnmappableHash),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// Answers BatchExistsAsync with a fixed list, whatever the keys are.
class FixedAnswerBackend : public KVBackend {
 public:
  explicit FixedAnswerBackend(std::vector<absl::StatusOr<bool>> answers)
      : answers_(std::move(answers)) {}

  std::string name() const override { return "fixed_answer"; }
  void WriteAsync(const BlockKey& key,
                  absl::Span<const HostBufferDescriptor> slices,
                  size_t total_bytes,
                  std::function<void(absl::Status)> callback) override {
    callback(absl::UnimplementedError("WriteAsync"));
  }
  void ReadAsync(const BlockKey& key,
                 absl::Span<const HostBufferDescriptor> slices,
                 size_t total_bytes,
                 std::function<void(absl::Status)> callback) override {
    callback(absl::UnimplementedError("ReadAsync"));
  }
  void BatchExistsAsync(absl::Span<const BlockKey> keys,
                        std::function<void(std::vector<absl::StatusOr<bool>>)>
                            callback) override {
    callback(answers_);
  }

 private:
  std::vector<absl::StatusOr<bool>> answers_;
};

TEST(StorageLookupHelperTest, ProbeExistsReportsFailedProbesAsAbsent) {
  FixedAnswerBackend backend({true, false, absl::UnavailableError("down")});
  const std::vector<BlockKey> keys = {KeyOf("a"), KeyOf("b"), KeyOf("c")};
  EXPECT_THAT(StorageLookupHelper::ProbeExists(backend, keys),
              ElementsAre(true, false, false));
}

TEST(StorageLookupHelperTest, ProbeExistsReportsShortAnswerAsAbsent) {
  FixedAnswerBackend backend({true});
  const std::vector<BlockKey> keys = {KeyOf("a"), KeyOf("b")};
  EXPECT_THAT(StorageLookupHelper::ProbeExists(backend, keys),
              ElementsAre(false, false));
}

// ===========================================================================
// Lookup without a cache
// ===========================================================================

TEST(StorageLookupHelperTest, LookupRejectsInvalidOptionCombinations) {
  FakeStorage storage({.tp_size = 1});
  storage.StoreBlock("a");

  EXPECT_THAT(
      storage.Lookup({"a"}, {.enable_global = true,
                             .match_mode = LookupMatchMode::kSubset}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("kSubset")));
  EXPECT_THAT(storage.Lookup(
                  {"a"}, {.enable_global = true, .host_resident_only = true}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("host_resident_only")));
  EXPECT_THAT(storage.Lookup({"a"}, {.enable_global = false,
                                     .pin_found = true,
                                     .include_eviction_candidates = true}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("include_eviction_candidates")));
  EXPECT_THAT(storage.probe_sizes(), IsEmpty());
}

TEST(StorageLookupHelperTest, LookupReturnsEmptyWithoutProbing) {
  FakeStorage storage({.tp_size = 1});
  storage.StoreBlock("a");

  LookupOptions host_only = PrefixOptions();
  host_only.host_resident_only = true;
  EXPECT_THAT(FakeStorage::Hashes(storage.Lookup({"a"}, host_only)), IsEmpty());
  EXPECT_THAT(FakeStorage::Hashes(storage.Lookup({}, PrefixOptions())),
              IsEmpty());
  EXPECT_THAT(storage.probe_sizes(), IsEmpty());
}

TEST(StorageLookupHelperTest, LookupReturnsSharedStorageBlocks) {
  FakeStorage storage({.tp_size = 1});
  storage.StoreBlock("a");

  TF_ASSERT_OK_AND_ASSIGN(BlockSliceList found,
                       storage.Lookup({"a"}, PrefixOptions()));

  ASSERT_EQ(found.size(), 1);
  EXPECT_EQ(found[0].first, "a");
  EXPECT_EQ(found[0].second.status, BlockStatus::SHARED_STORAGE);
  EXPECT_EQ(found[0].second.raiden_id.job_replica_id, "shared");
  EXPECT_EQ(found[0].second.raiden_id.data_name, "fake_store");
}

TEST(StorageLookupHelperTest, BlockNeedsTheShardOfEveryRank) {
  FakeStorage storage({.tp_size = 2});
  storage.StoreBlock("a");
  storage.StoreKey("b/pp-1/pcp-1/tp0");  // Rank 1 has not written "b".
  storage.StoreBlock("c");

  EXPECT_THAT(
      FakeStorage::Hashes(storage.Lookup({"a", "b", "c"}, PrefixOptions())),
      ElementsAre("a"));
  EXPECT_THAT(
      FakeStorage::Hashes(storage.Lookup({"a", "b", "c"}, SubsetOptions())),
      ElementsAre("a", "c"));
}

TEST(StorageLookupHelperTest, PrefixStopsProbingAfterChunkWithAbsentKey) {
  FakeStorage storage({.tp_size = 1}, /*lookup_batch_size=*/2);
  storage.StoreBlock("a");
  storage.StoreBlock("c");
  storage.StoreBlock("d");

  // "b" is absent in the first chunk {a, b}, so {c, d} is never probed.
  EXPECT_THAT(FakeStorage::Hashes(
                  storage.Lookup({"a", "b", "c", "d"}, PrefixOptions())),
              ElementsAre("a"));
  EXPECT_THAT(storage.probe_sizes(), ElementsAre(2));
}

TEST(StorageLookupHelperTest, SubsetProbesEveryChunk) {
  FakeStorage storage({.tp_size = 1}, /*lookup_batch_size=*/2);
  storage.StoreBlock("a");
  storage.StoreBlock("c");
  storage.StoreBlock("d");

  EXPECT_THAT(FakeStorage::Hashes(
                  storage.Lookup({"a", "b", "c", "d", "e"}, SubsetOptions())),
              ElementsAre("a", "c", "d"));
  EXPECT_THAT(storage.probe_sizes(), ElementsAre(2, 2, 1));
}

TEST(StorageLookupHelperTest, UnmappableHashEndsPrefixAndIsSkippedInSubset) {
  FakeStorage storage({.tp_size = 1});
  storage.StoreBlock("a");
  storage.StoreBlock("c");

  EXPECT_THAT(FakeStorage::Hashes(
                  storage.Lookup({"a", kUnmappableHash, "c"}, PrefixOptions())),
              ElementsAre("a"));
  EXPECT_THAT(FakeStorage::Hashes(
                  storage.Lookup({"a", kUnmappableHash, "c"}, SubsetOptions())),
              ElementsAre("a", "c"));
  EXPECT_THAT(FakeStorage::Hashes(
                  storage.Lookup({kUnmappableHash, "a"}, PrefixOptions())),
              IsEmpty());
}

TEST(StorageLookupHelperTest, EveryLookupProbesStorageAgain) {
  FakeStorage storage({.tp_size = 2});
  storage.StoreBlock("a");
  EXPECT_THAT(FakeStorage::Hashes(storage.Lookup({"a"}, PrefixOptions())),
              ElementsAre("a"));

  // Storage evicts one rank's shard out of band. The next Lookup sees it.
  storage.EvictKey("a/pp-1/pcp-1/tp1");
  EXPECT_THAT(FakeStorage::Hashes(storage.Lookup({"a"}, PrefixOptions())),
              IsEmpty());
  EXPECT_THAT(storage.probe_sizes(), ElementsAre(2, 2));
}

TEST(StorageLookupHelperTest, ShortProbeAnswerIsTreatedAsAbsent) {
  const StorageLookupHelper helper("fake_store", "0", /*lookup_batch_size=*/8);
  const FakeMapper mapper({.tp_size = 1});

  TF_ASSERT_OK_AND_ASSIGN(
      BlockSliceList found,
      helper.Lookup(
          {"a", "b"}, SubsetOptions(),
          [&mapper](const std::string& block_hash) {
            return StorageLookupHelper::ExpandShardKeys(mapper, block_hash);
          },
          [](absl::Span<const BlockKey> keys) {
            return std::vector<bool>{true};  // One answer for two keys.
          }));

  EXPECT_THAT(found, IsEmpty());
}

// ===========================================================================
// Lookup with the backend's cache of confirmed keys
// ===========================================================================

TEST(StorageLookupHelperTest, ConfirmedKeysAreCachedAndNotProbedAgain) {
  FakeStorage storage({.tp_size = 2});
  storage.StoreBlock("a");
  storage.StoreBlock("b");

  EXPECT_THAT(
      FakeStorage::Hashes(storage.LookupWithCache({"a", "b"}, PrefixOptions())),
      ElementsAre("a", "b"));
  EXPECT_THAT(storage.probe_sizes(), ElementsAre(4));
  EXPECT_THAT(storage.cached_keys(),
              UnorderedElementsAre("a/pp-1/pcp-1/tp0", "a/pp-1/pcp-1/tp1",
                                   "b/pp-1/pcp-1/tp0", "b/pp-1/pcp-1/tp1"));

  storage.ResetProbes();
  EXPECT_THAT(
      FakeStorage::Hashes(storage.LookupWithCache({"a", "b"}, PrefixOptions())),
      ElementsAre("a", "b"));
  EXPECT_THAT(storage.probe_sizes(), IsEmpty());
}

TEST(StorageLookupHelperTest, ConfirmedShardsOfAnUnavailableBlockAreCached) {
  FakeStorage storage({.tp_size = 2});
  storage.StoreKey("a/pp-1/pcp-1/tp0");

  EXPECT_THAT(
      FakeStorage::Hashes(storage.LookupWithCache({"a"}, PrefixOptions())),
      IsEmpty());
  EXPECT_THAT(storage.cached_keys(), ElementsAre("a/pp-1/pcp-1/tp0"));

  // Once rank 1 writes its shard, only that shard is probed.
  storage.StoreKey("a/pp-1/pcp-1/tp1");
  storage.ResetProbes();
  EXPECT_THAT(
      FakeStorage::Hashes(storage.LookupWithCache({"a"}, PrefixOptions())),
      ElementsAre("a"));
  EXPECT_THAT(storage.probe_sizes(), ElementsAre(1));
}

TEST(StorageLookupHelperTest, CachedKeyHidesAnEvictionUntilItLeavesTheCache) {
  FakeStorage storage({.tp_size = 2});
  storage.StoreBlock("a");
  EXPECT_THAT(
      FakeStorage::Hashes(storage.LookupWithCache({"a"}, PrefixOptions())),
      ElementsAre("a"));

  // Storage evicts one rank's shard out of band. The cache still holds it.
  storage.EvictKey("a/pp-1/pcp-1/tp1");
  EXPECT_THAT(
      FakeStorage::Hashes(storage.LookupWithCache({"a"}, PrefixOptions())),
      ElementsAre("a"));

  // Once the backend drops the key from its cache, Lookup probes it again.
  storage.cached_keys().erase("a/pp-1/pcp-1/tp1");
  EXPECT_THAT(
      FakeStorage::Hashes(storage.LookupWithCache({"a"}, PrefixOptions())),
      IsEmpty());
}

// ===========================================================================
// storage_metadata_check_total
// ===========================================================================

auto MetadataCheckLabels(absl::string_view dp_rank, absl::string_view result) {
  return testing::ElementsAre(
      testing::FieldsAre(telemetry::metric_labels::kDpRank,
                         std::string(dp_rank)),
      testing::FieldsAre(telemetry::metric_labels::kResult,
                         std::string(result)));
}

// Expects `times` increments by 1 of the counter with the given result label.
void ExpectMetadataCheck(telemetry::MockMetricsBackend& backend,
                         absl::string_view result, int times) {
  EXPECT_CALL(backend, IncrementCounter(
                           telemetry::metric_names::kStorageMetadataCheckTotal,
                           MetadataCheckLabels("7", result), 1))
      .Times(times);
}

TEST(StorageLookupHelperTest, CountsCacheHitsStatHitsAndStatMisses) {
  FakeStorage storage({.tp_size = 1});
  storage.StoreBlock("a");
  storage.StoreBlock("b");
  ASSERT_THAT(
      FakeStorage::Hashes(storage.LookupWithCache({"a"}, PrefixOptions())),
      ElementsAre("a"));

  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  ExpectMetadataCheck(*mock_backend, telemetry::metric_labels::kResultCacheHit,
                      1);
  ExpectMetadataCheck(*mock_backend, telemetry::metric_labels::kResultStatHit,
                      1);
  ExpectMetadataCheck(*mock_backend, telemetry::metric_labels::kResultStatMiss,
                      1);
  telemetry::ScopedMetricsBackendReset scoped_reset(std::move(mock_backend));

  // "a" is cached, "b" is on storage, "c" is absent.
  EXPECT_THAT(FakeStorage::Hashes(
                  storage.LookupWithCache({"a", "b", "c"}, SubsetOptions())),
              ElementsAre("a", "b"));
}

TEST(StorageLookupHelperTest, CountsNoCacheHitWithoutACache) {
  FakeStorage storage({.tp_size = 1});
  storage.StoreBlock("a");
  ASSERT_THAT(FakeStorage::Hashes(storage.Lookup({"a"}, PrefixOptions())),
              ElementsAre("a"));

  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  ExpectMetadataCheck(*mock_backend, telemetry::metric_labels::kResultCacheHit,
                      0);
  ExpectMetadataCheck(*mock_backend, telemetry::metric_labels::kResultStatHit,
                      1);
  ExpectMetadataCheck(*mock_backend, telemetry::metric_labels::kResultStatMiss,
                      1);
  telemetry::ScopedMetricsBackendReset scoped_reset(std::move(mock_backend));

  // "a" was found before, but it is probed again. "c" is absent.
  EXPECT_THAT(FakeStorage::Hashes(storage.Lookup({"a", "c"}, SubsetOptions())),
              ElementsAre("a"));
}

}  // namespace
}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden
