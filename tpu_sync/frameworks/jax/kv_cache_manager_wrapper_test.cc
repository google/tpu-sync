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

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/buffer.h"
#include "tpu_sync/core/controller/controller_service.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/frameworks/jax/kv_cache_manager.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace jax {
namespace {

struct StartReadCall {
  std::string req_id;
  uint64_t uuid;
  std::string remote_endpoint;
  std::vector<int64_t> remote_block_ids;
  std::vector<int64_t> local_block_ids;
  std::optional<std::vector<int64_t>> local_host_block_ids;
};

class MockSubManager : public KVCacheManagerWithTransfer {
 public:
  MockSubManager()
      : KVCacheManagerWithTransfer({}, std::nullopt, std::nullopt, false, 1,
                                   nullptr) {}

  std::vector<std::string> sending, recving, failed;
  std::string started_ep;
  std::vector<int64_t> started_remote_blocks, started_local_blocks;
  std::vector<StartReadCall> start_read_calls;
  std::string lease_endpoint;
  std::vector<uint64_t> lease_uuids;
  bool lease_cancelled = false;

  int d2h_calls = 0;
  int h2d_calls = 0;
  std::vector<int64_t> last_d2h_src_offsets;
  std::vector<int64_t> last_d2h_dst_offsets;
  std::vector<int64_t> last_d2h_copy_sizes;
  std::vector<int64_t> last_h2d_src_offsets;
  std::vector<int64_t> last_h2d_dst_offsets;
  std::vector<int64_t> last_h2d_copy_sizes;

  void StartRead(const std::string& req_id, uint64_t uuid,
                 const std::string& remote_endpoint,
                 const std::vector<int64_t>& remote_block_ids,
                 const std::vector<int64_t>& local_block_ids,
                 int parallelism = 1,
                 std::optional<std::vector<int64_t>> local_host_block_ids =
                     std::nullopt) override {
    started_ep = remote_endpoint;
    started_remote_blocks = remote_block_ids;
    started_local_blocks = local_block_ids;
    start_read_calls.push_back({req_id, uuid, remote_endpoint, remote_block_ids,
                                local_block_ids, local_host_block_ids});
  }

  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  CompleteReadRaw() override {
    auto res = std::make_tuple(sending, recving, failed);
    sending.clear();
    recving.clear();
    failed.clear();
    return res;
  }

  std::vector<int32_t> RenewRemoteLeases(
      const std::string& remote_endpoint,
      const std::vector<uint64_t>& uuids) override {
    lease_endpoint = remote_endpoint;
    lease_uuids = uuids;
    lease_cancelled = false;
    return std::vector<int32_t>(uuids.size(), 1);
  }

  std::vector<int32_t> CancelRemoteLeases(
      const std::string& remote_endpoint,
      const std::vector<uint64_t>& uuids) override {
    lease_endpoint = remote_endpoint;
    lease_uuids = uuids;
    lease_cancelled = true;
    return std::vector<int32_t>(uuids.size(), 1);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt) override {
    d2h_calls++;
    last_d2h_src_offsets = src_offsets_major_dim;
    last_d2h_dst_offsets = dst_offsets_major_dim;
    last_d2h_copy_sizes = copy_sizes_major_dim;
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt) override {
    h2d_calls++;
    last_h2d_src_offsets = src_offsets_major_dim;
    last_h2d_dst_offsets = dst_offsets_major_dim;
    last_h2d_copy_sizes = copy_sizes_major_dim;
    return raiden::PjRtCopyFuture();
  }

  // Records the single remote endpoint the wrapper's shard-matching selected
  // for this sub-manager (the wrapper narrows the full remote_descriptors list
  // down to one endpoint per sub-manager before calling the string overload).
  int h2h_write_calls = 0;
  int h2h_read_calls = 0;
  std::string last_h2h_write_peer;
  std::string last_h2h_read_peer;
  std::vector<int> last_h2h_write_src_blocks;
  std::vector<int> last_h2h_write_dst_blocks;
  std::vector<int> last_h2h_read_src_blocks;

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1) override {
    h2h_write_calls++;
    last_h2h_write_peer = std::move(peer);
    last_h2h_write_src_blocks = src_block_ids;
    last_h2h_write_dst_blocks = dst_block_ids;
    return std::make_pair(std::vector<int>(), raiden::PjRtCopyFuture());
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string peer, const std::vector<int>& src_block_ids) override {
    h2h_read_calls++;
    last_h2h_read_peer = std::move(peer);
    last_h2h_read_src_blocks = src_block_ids;
    return std::make_pair(std::vector<int>(), raiden::PjRtCopyFuture());
  }
};

TEST(KVCacheManagerWrapperTest,
     VotingConsensusPromotesDoneRecvingOnlyWhenAllAgree) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));

  // 1. Only sub0 finishes recving for req_123
  ptr0->recving.push_back("req_123");
  auto [s1, r1, f1] = mgr.CompleteReadRaw();
  EXPECT_TRUE(s1.empty());
  EXPECT_TRUE(r1.empty());
  EXPECT_TRUE(f1.empty());

  // 2. Now sub1 finishes recving for req_123
  ptr1->recving.push_back("req_123");
  auto [s2, r2, f2] = mgr.CompleteReadRaw();
  EXPECT_TRUE(s2.empty());
  EXPECT_EQ(r2, std::vector<std::string>{"req_123"});
  EXPECT_TRUE(f2.empty());
}

TEST(KVCacheManagerWrapperTest, LeaseOperationsUseExactlyOneControlClient) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();
  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));
  KVCacheManager manager(std::move(subs));

  EXPECT_EQ(manager.RenewRemoteLeases("10.0.0.1:45000", {1, 2}),
            std::vector<int32_t>({1, 1}));
  EXPECT_EQ(ptr0->lease_endpoint, "10.0.0.1:45000");
  EXPECT_EQ(ptr0->lease_uuids, std::vector<uint64_t>({1, 2}));
  EXPECT_TRUE(ptr1->lease_endpoint.empty());

  EXPECT_EQ(manager.CancelRemoteLeases("10.0.0.1:45000", {3}),
            std::vector<int32_t>({1}));
  EXPECT_TRUE(ptr0->lease_cancelled);
  EXPECT_EQ(ptr0->lease_uuids, std::vector<uint64_t>({3}));
  EXPECT_TRUE(ptr1->lease_endpoint.empty());
}

TEST(KVCacheManagerWrapperTest,
     AnyFailureImmediatelyTriggersFailedRecvingAndSuppressesFutureAcks) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));

  // 1. sub0 reports failure for req_fail
  ptr0->failed.push_back("req_fail");
  auto [s1, r1, f1] = mgr.CompleteReadRaw();
  EXPECT_TRUE(s1.empty());
  EXPECT_TRUE(r1.empty());
  EXPECT_EQ(f1, std::vector<std::string>{"req_fail"});

  // 2. Later sub1 finishes recving for req_fail (should be suppressed)
  ptr1->recving.push_back("req_fail");
  auto [s2, r2, f2] = mgr.CompleteReadRaw();
  EXPECT_TRUE(s2.empty());
  EXPECT_TRUE(r2.empty());
  EXPECT_TRUE(f2.empty());
}

TEST(KVCacheManagerWrapperTest,
     VotingConsensusPromotesDoneSendingOnlyWhenAllAgree) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));

  ptr0->sending.push_back("req_send");
  EXPECT_TRUE(std::get<0>(mgr.CompleteReadRaw()).empty());

  ptr1->sending.push_back("req_send");
  EXPECT_EQ(std::get<0>(mgr.CompleteReadRaw()),
            std::vector<std::string>{"req_send"});
}

TEST(KVCacheManagerWrapperTest,
     StartReadMatchesRemoteEndpointsByShardIntersection) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));

  // Local sub0 holds shards [4, 5, 6, 7], sub1 holds [0, 1, 2, 3]
  mgr.SetSubmanagerShardsForTesting({{4, 5, 6, 7}, {0, 1, 2, 3}});

  std::vector<RaidenTransferEndpoint> remote_descs = {
      {"10.0.0.1:45000", {0, 1, 2, 3}}, {"10.0.0.2:45000", {4, 5, 6, 7}}};

  std::vector<int64_t> remote_blocks = {10, 20};
  std::vector<int64_t> local_blocks = {30, 40};

  mgr.StartRead("req_test", 999, remote_descs, remote_blocks, local_blocks);

  EXPECT_EQ(ptr0->started_ep, "10.0.0.2:45000");
  EXPECT_EQ(ptr0->started_remote_blocks, remote_blocks);
  EXPECT_EQ(ptr0->started_local_blocks, local_blocks);

  EXPECT_EQ(ptr1->started_ep, "10.0.0.1:45000");
  EXPECT_EQ(ptr1->started_remote_blocks, remote_blocks);
  EXPECT_EQ(ptr1->started_local_blocks, local_blocks);
}

// H2hWrite is the sub-manager transfer the controller-orchestrated ReadRemote
// path drives (via WorkerServiceImpl). Each local sub-manager must be matched
// to the remote (destination peer) endpoint whose advertised shards intersect
// its own global shards -- not by index or list position.
TEST(KVCacheManagerWrapperTest,
     H2hWriteMatchesRemoteEndpointsByShardIntersection) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));

  // Local sub0 holds shards [4, 5, 6, 7], sub1 holds [0, 1, 2, 3].
  mgr.SetSubmanagerShardsForTesting({{4, 5, 6, 7}, {0, 1, 2, 3}});

  // Remote peer endpoints tagged with their global shards, listed in the
  // OPPOSITE order to the local sub-managers to prove matching is by shard, not
  // position.
  std::vector<RaidenTransferEndpoint> remote_descs = {
      {"10.0.0.1:45000", {0, 1, 2, 3}}, {"10.0.0.2:45000", {4, 5, 6, 7}}};

  std::vector<int> src_blocks = {10, 20};
  std::vector<int> dst_blocks = {30, 40};

  auto status = mgr.H2hWrite(remote_descs, src_blocks, dst_blocks, /*uuid=*/7,
                             /*layer_idx=*/0);
  ASSERT_TRUE(status.ok()) << status.status().message();

  // sub0 ([4-7]) -> the [4-7] endpoint; sub1 ([0-3]) -> the [0-3] endpoint.
  EXPECT_EQ(ptr0->h2h_write_calls, 1);
  EXPECT_EQ(ptr0->last_h2h_write_peer, "10.0.0.2:45000");
  EXPECT_EQ(ptr0->last_h2h_write_src_blocks, src_blocks);
  EXPECT_EQ(ptr0->last_h2h_write_dst_blocks, dst_blocks);

  EXPECT_EQ(ptr1->h2h_write_calls, 1);
  EXPECT_EQ(ptr1->last_h2h_write_peer, "10.0.0.1:45000");
  EXPECT_EQ(ptr1->last_h2h_write_src_blocks, src_blocks);
  EXPECT_EQ(ptr1->last_h2h_write_dst_blocks, dst_blocks);
}

TEST(KVCacheManagerWrapperTest,
     H2hReadMatchesRemoteEndpointsByShardIntersection) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));

  // Local sub0 holds shards [4, 5, 6, 7], sub1 holds [0, 1, 2, 3].
  mgr.SetSubmanagerShardsForTesting({{4, 5, 6, 7}, {0, 1, 2, 3}});

  std::vector<RaidenTransferEndpoint> remote_descs = {
      {"10.0.0.1:45000", {0, 1, 2, 3}}, {"10.0.0.2:45000", {4, 5, 6, 7}}};

  std::vector<int> src_blocks = {11, 22};

  auto status = mgr.H2hRead(remote_descs, src_blocks);
  ASSERT_TRUE(status.ok()) << status.status().message();

  EXPECT_EQ(ptr0->h2h_read_calls, 1);
  EXPECT_EQ(ptr0->last_h2h_read_peer, "10.0.0.2:45000");
  EXPECT_EQ(ptr0->last_h2h_read_src_blocks, src_blocks);

  EXPECT_EQ(ptr1->h2h_read_calls, 1);
  EXPECT_EQ(ptr1->last_h2h_read_peer, "10.0.0.1:45000");
  EXPECT_EQ(ptr1->last_h2h_read_src_blocks, src_blocks);
}

// When endpoints carry no shard info (e.g. registrations with empty shard
// lists), shard intersection never matches and every sub-manager falls back to
// the first remote endpoint. This documents the fallback the empty-shard unit
// tests implicitly rely on.
TEST(KVCacheManagerWrapperTest, H2hWriteEmptyShardsFallsBackToFirstEndpoint) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));
  mgr.SetSubmanagerShardsForTesting({{4, 5, 6, 7}, {0, 1, 2, 3}});

  // Endpoints with empty shard lists -> no intersection possible.
  std::vector<RaidenTransferEndpoint> remote_descs = {{"10.0.0.1:45000", {}},
                                                      {"10.0.0.2:45000", {}}};

  auto status = mgr.H2hWrite(remote_descs, /*src_block_ids=*/{10},
                             /*dst_block_ids=*/{30});
  ASSERT_TRUE(status.ok()) << status.status().message();

  EXPECT_EQ(ptr0->last_h2h_write_peer, "10.0.0.1:45000");
  EXPECT_EQ(ptr1->last_h2h_write_peer, "10.0.0.1:45000");
}

// The common non-NUMA ReadRemote transfer: a single sub-manager H2hWrite must
// forward the source/destination block ids and the matched endpoint through to
// that sub-manager.
TEST(KVCacheManagerWrapperTest,
     H2hWriteSingleSubManagerForwardsBlocksAndEndpoint) {
  auto sub0 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  KVCacheManager mgr(std::move(subs));
  mgr.SetSubmanagerShardsForTesting({{0, 1, 2, 3}});

  std::vector<RaidenTransferEndpoint> remote_descs = {
      {"10.0.0.5:41000", {0, 1, 2, 3}}};
  std::vector<int> src_blocks = {5, 6};
  std::vector<int> dst_blocks = {7, 8};

  auto status = mgr.H2hWrite(remote_descs, src_blocks, dst_blocks, /*uuid=*/3,
                             /*layer_idx=*/0);
  ASSERT_TRUE(status.ok()) << status.status().message();
  EXPECT_EQ(ptr0->h2h_write_calls, 1);
  EXPECT_EQ(ptr0->last_h2h_write_peer, "10.0.0.5:41000");
  EXPECT_EQ(ptr0->last_h2h_write_src_blocks, src_blocks);
  EXPECT_EQ(ptr0->last_h2h_write_dst_blocks, dst_blocks);
}

TEST(KVCacheManagerWrapperTest,
     H2hReadSingleSubManagerForwardsBlocksAndEndpoint) {
  auto sub0 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  KVCacheManager mgr(std::move(subs));
  mgr.SetSubmanagerShardsForTesting({{0, 1, 2, 3}});

  std::vector<RaidenTransferEndpoint> remote_descs = {
      {"10.0.0.5:41000", {0, 1, 2, 3}}};
  std::vector<int> src_blocks = {11, 12};

  auto status = mgr.H2hRead(remote_descs, src_blocks);
  ASSERT_TRUE(status.ok()) << status.status().message();
  EXPECT_EQ(ptr0->h2h_read_calls, 1);
  EXPECT_EQ(ptr0->last_h2h_read_peer, "10.0.0.5:41000");
  EXPECT_EQ(ptr0->last_h2h_read_src_blocks, src_blocks);
}

TEST(KVCacheManagerWrapperTest, StartReadUnifiedMultiEndpointMultiSubManager) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));
  // Local sub0 holds shards [0, 1], sub1 holds [2, 3]
  mgr.SetSubmanagerShardsForTesting({{0, 1}, {2, 3}});

  std::vector<RaidenTransferEndpoint> remote_descs = {
      {"10.0.0.1:45000", {0, 1}},
      {"10.0.0.2:45000", {0, 1}},
      {"10.0.0.3:45000", {2, 3}},
      {"10.0.0.4:45000", {2, 3}}};

  std::vector<int64_t> remote_blocks = {10, 20, 30, 40, 50};
  std::vector<int64_t> local_blocks = {100, 200, 300, 400, 500};
  std::vector<int64_t> host_blocks = {1000, 2000, 3000, 4000, 5000};
  uint64_t uuid = 1000;

  mgr.StartRead("req_unified", uuid, remote_descs, remote_blocks, local_blocks,
                1, host_blocks);

  // In unified architecture, full block lists are passed to each submanager
  // with original req_id (no _ep0/_ep1 slicing)
  ASSERT_EQ(ptr0->start_read_calls.size(), 1);
  const auto& call0 = ptr0->start_read_calls[0];
  EXPECT_EQ(call0.req_id, "req_unified");
  EXPECT_EQ(call0.uuid, uuid);
  EXPECT_EQ(call0.remote_block_ids, remote_blocks);
  EXPECT_EQ(call0.local_block_ids, local_blocks);
  ASSERT_TRUE(call0.local_host_block_ids.has_value());
  EXPECT_EQ(call0.local_host_block_ids.value(), host_blocks);

  ASSERT_EQ(ptr1->start_read_calls.size(), 1);
  const auto& call1 = ptr1->start_read_calls[0];
  EXPECT_EQ(call1.req_id, "req_unified");
  EXPECT_EQ(call1.uuid, uuid);
  EXPECT_EQ(call1.remote_block_ids, remote_blocks);
  EXPECT_EQ(call1.local_block_ids, local_blocks);
  ASSERT_TRUE(call1.local_host_block_ids.has_value());
  EXPECT_EQ(call1.local_host_block_ids.value(), host_blocks);

  // Completion Tracking Aggregation:
  // 1. sub0 completes first -> parent req_unified should NOT be returned yet
  // (needs sub1 too)
  ptr0->recving.push_back("req_unified");
  auto [s1, r1, f1] = mgr.CompleteReadRaw();
  EXPECT_TRUE(r1.empty());

  // 2. sub1 completes -> parent req_unified SHOULD be returned in done_recving
  ptr1->recving.push_back("req_unified");
  auto [s2, r2, f2] = mgr.CompleteReadRaw();
  EXPECT_EQ(r2, std::vector<std::string>{"req_unified"});
}

TEST(KVCacheManagerWrapperTest, StartReadInputVectorValidation) {
  auto sub0 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));

  KVCacheManager mgr(std::move(subs));

  std::vector<RaidenTransferEndpoint> remote_descs = {{"10.0.0.1:45000", {0}}};
  std::vector<int64_t> remote_blocks = {10, 20};
  std::vector<int64_t> local_blocks = {100};  // Size mismatch!

  mgr.StartRead("req_mismatch", 100, remote_descs, remote_blocks, local_blocks);
  EXPECT_TRUE(ptr0->start_read_calls.empty());
}

TEST(KVCacheManagerWrapperTest, GrpcServerOptionalAndOffByDefault) {
  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs1;
  subs1.push_back(std::make_unique<MockSubManager>());
  KVCacheManager mgr_default(std::move(subs1));
  EXPECT_EQ(mgr_default.GetRaidenWorkerPort(), 0);

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs2;
  subs2.push_back(std::make_unique<MockSubManager>());
  KVCacheManager mgr_explicit_off(std::move(subs2), /*raiden_worker_port=*/0,
                                  /*raiden_controller_address=*/std::nullopt);
  EXPECT_EQ(mgr_explicit_off.GetRaidenWorkerPort(), 0);

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs3;
  subs3.push_back(std::make_unique<MockSubManager>());
  KVCacheManager mgr_started(std::move(subs3), /*raiden_worker_port=*/0,
                             /*raiden_controller_address=*/"localhost:12345");
  EXPECT_GT(mgr_started.GetRaidenWorkerPort(), 0);
}

TEST(KVCacheManagerWrapperTest, RaidenControllerTransferBuffersIntegration) {
  auto sub0 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));

  KVCacheManager mgr(std::move(subs), /*raiden_worker_port=*/0,
                     /*raiden_controller_address=*/"localhost:12345");
  int port = mgr.GetRaidenWorkerPort();
  ASSERT_GT(port, 0);

  ::tpu_sync::rpc::RaidenIdProto unit;
  unit.set_job_name("test_job");
  unit.set_job_replica_id("0");
  unit.set_data_name("test_data");

  std::string server_address = absl::StrCat("localhost:", port);

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(
          unit, std::vector<std::string>{server_address}, /*num_blocks=*/5,
          /*num_shards=*/1, /*shard_size_bytes=*/512));

  std::vector<int64_t> src_offsets = {10, 30};
  std::vector<int64_t> dst_offsets = {20, 40};
  std::vector<int64_t> copy_sizes = {1, 2};

  Buffer src_d2h_1(10, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer src_d2h_2(30, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer dst_d2h_1(20, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  Buffer dst_d2h_2(40, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);

  auto status_d2h =
      controller
          ->TransferBuffers("worker_0", {src_d2h_1, src_d2h_2},
                            {dst_d2h_1, dst_d2h_2}, /*staging_host_buffers=*/{},
                            copy_sizes)
          .Await();
  ASSERT_TRUE(status_d2h.ok());
  EXPECT_EQ(ptr0->d2h_calls, 1);
  EXPECT_EQ(ptr0->h2d_calls, 0);
  EXPECT_EQ(ptr0->last_d2h_src_offsets, src_offsets);
  EXPECT_EQ(ptr0->last_d2h_dst_offsets, dst_offsets);
  EXPECT_EQ(ptr0->last_d2h_copy_sizes, copy_sizes);

  Buffer src_h2d_1(10, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  Buffer src_h2d_2(30, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  Buffer dst_h2d_1(20, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  Buffer dst_h2d_2(40, {}, std::nullopt, ::tpu_sync::rpc::MEMORY_TYPE_HBM);

  auto status_h2d =
      controller
          ->TransferBuffers("worker_0", {src_h2d_1, src_h2d_2},
                            {dst_h2d_1, dst_h2d_2}, /*staging_host_buffers=*/{},
                            copy_sizes)
          .Await();
  ASSERT_TRUE(status_h2d.ok());
  EXPECT_EQ(ptr0->d2h_calls, 1);
  EXPECT_EQ(ptr0->h2d_calls, 1);
  EXPECT_EQ(ptr0->last_h2d_src_offsets, src_offsets);
  EXPECT_EQ(ptr0->last_h2d_dst_offsets, dst_offsets);
  EXPECT_EQ(ptr0->last_h2d_copy_sizes, copy_sizes);
}

TEST(KVCacheManagerWrapperTest, WorkerSelfRegistrationWithControllerSuccess) {
  auto test_server = core::controller::CreateTestControllerServer();
  ASSERT_NE(test_server, nullptr);

  std::string raiden_controller_address = test_server->server_address;

  auto sub0 = std::make_unique<MockSubManager>();
  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));

  KVCacheManager mgr(std::move(subs), /*raiden_worker_port=*/0,
                     raiden_controller_address, "test_worker_node");

  auto workers =
      test_server->service->worker_registry()->GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 1);
  EXPECT_EQ(workers[0].worker_id, "test_worker_node");
  EXPECT_NE(workers[0].raiden_worker_endpoint.find(
                std::to_string(mgr.GetRaidenWorkerPort())),
            std::string::npos);
}

}  // namespace
}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden
