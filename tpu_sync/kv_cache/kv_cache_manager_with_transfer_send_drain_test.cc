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

// A pull-serve send that expires or fails while its device-to-host copies
// are still running, exercised without a device: the copies complete when
// the test says so.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <ios>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/future.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/transfer_receive_session.h"
#include "tpu_sync/kv_cache/transfer_send_session.h"
#include "tpu_sync/kv_cache/transfer_session.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::Optional;
using ::testing::UnorderedElementsAre;

constexpr int64_t kSlots = 2;
constexpr double kTimeoutS = 0.05;

// A producer whose device-to-host copies complete when the test says so.
class TestManager : public KVCacheManagerWithTransfer {
 public:
  explicit TestManager(size_t num_layers, double timeout_s = kTimeoutS)
      : KVCacheManagerWithTransfer(std::make_unique<TestBase>(num_layers, this),
                                   /*node_id=*/0,
                                   /*local_control_port=*/-1, /*max_blocks=*/1,
                                   /*num_slots=*/kSlots, timeout_s) {}

  ~TestManager() override {
    std::vector<std::function<void(absl::Status)>> pending;
    {
      absl::MutexLock lock(h2h_mu_);
      pending.swap(h2h_callbacks_);
    }
    for (auto& cb : pending) {
      if (cb) cb(absl::CancelledError("TestManager shutting down"));
    }
  }

  // Serves a pull for `uuid` the way ProcessPullStream does once the
  // consumer is acknowledged: the push runs on this thread and returns
  // with its copies issued.
  void ServePull(uint64_t uuid) {
    std::shared_ptr<TransferSendSession> session;
    {
      absl::MutexLock lock(mu_);
      session = send_sessions_.at(uuid);
      served_sends_[uuid] = session;
    }
    session->StartPush({"127.0.0.1:1"}, /*src_block_ids=*/{0},
                       /*dst_block_ids=*/{0});
  }

  // Serves a pull for `uuid` on its own thread, the way HandlePullStream
  // runs StartPush on a detached pull worker. With bounded layer staging the
  // push blocks in AcquireStagingSlot until pushes free host layer buffers,
  // so it cannot run on the test thread. Join the returned thread.
  std::thread ServePullAsync(uint64_t uuid) {
    return std::thread([this, uuid]() { ServePull(uuid); });
  }

  // Fails the send for `uuid` the way a consumer cancellation or the
  // manager's shutdown does.
  void CancelSend(uint64_t uuid) {
    std::shared_ptr<TransferSendSession> session;
    {
      absl::MutexLock lock(mu_);
      session = send_sessions_.at(uuid);
    }
    session->Finish(absl::CancelledError("send cancelled"));
  }

  using KVCacheManagerWithTransfer::HandleAck;
  absl::StatusOr<PullStreamResponseSpec> ProcessPull(
      uint64_t uuid, std::vector<int64_t> blocks) {
    return HandlePullStream(
        PullStreamRequestSpec{.uuid = uuid,
                              .consumer_data_port = 1,
                              .consumer_ips = {"127.0.0.1"},
                              .src_block_ids = std::move(blocks),
                              .dst_block_ids = {0}},
        "");
  }
  int h2h_calls() {
    absl::MutexLock lock(h2h_mu_);
    return h2h_calls_;
  }
  size_t active_h2h_count() {
    absl::MutexLock lock(h2h_mu_);
    return active_h2h_uuids_.size();
  }
  bool WaitForH2hCalls(int expected,
                       absl::Duration timeout = absl::Seconds(5)) {
    absl::MutexLock lock(h2h_mu_);
    auto done = [&] {
      h2h_mu_.AssertReaderHeld();
      return h2h_calls_ >= expected;
    };
    return h2h_mu_.AwaitWithTimeout(absl::Condition(&done), timeout);
  }
  bool WaitForCopies(size_t expected,
                     absl::Duration timeout = absl::Seconds(5)) {
    absl::MutexLock lock(copies_mu_);
    auto done = [&] {
      copies_mu_.AssertReaderHeld();
      return copies_.size() >= expected;
    };
    return copies_mu_.AwaitWithTimeout(absl::Condition(&done), timeout);
  }
  size_t copies_issued() {
    absl::MutexLock lock(copies_mu_);
    return copies_.size();
  }

  // Completes the `index`-th copy issued.
  void FinishCopy(size_t index, absl::Status status) {
    absl::MutexLock lock(copies_mu_);
    copies_.at(index).Set(std::move(status));
  }

  size_t free_slots() { return staging_allocator_->num_free_slots(); }
  StagingBlockAllocator* staging_allocator() {
    return staging_allocator_.get();
  }

  // White-box construction for the send retirement state machine. Tests that
  // need registration and dispatch use NotifyForRead and ServePull instead.
  std::shared_ptr<TransferSendSession> AddSyntheticSend(
      const std::string& req_id, uint64_t uuid, int in_flight) {
    CHECK_GT(NotifyForRead(req_id, uuid, {0}), 0);
    std::shared_ptr<TransferSendSession> session;
    {
      absl::MutexLock lock(mu_);
      session = send_sessions_.at(uuid);
    }
    if (in_flight > 0) {
      ServePull(uuid);
    }
    return session;
  }

  void Decide(const std::shared_ptr<TransferSendSession>& session,
              bool failed) {
    session->Finish(failed ? absl::InternalError("send failed")
                           : absl::OkStatus());
  }

  void End(const std::shared_ptr<TransferSendSession>& session) {
    session->EndSendOp();
  }

  bool has_send(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = send_sessions_.find(uuid);
    return it != send_sessions_.end() && !it->second->Done();
  }

  bool send_draining(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = send_sessions_.find(uuid);
    return it != send_sessions_.end() && it->second->IsDraining();
  }

  // Host layer buffer the `index`-th copy issued writes to; -1 when the copy
  // named none.
  int64_t copy_destination(size_t index) {
    absl::MutexLock lock(copies_mu_);
    return copy_destinations_.at(index);
  }

  bool WaitForCopiesIssued(size_t expected, absl::Duration timeout) {
    absl::MutexLock lock(copies_mu_);
    auto cond = [this, expected]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(copies_mu_) {
      return copies_.size() >= expected;
    };
    return copies_mu_.AwaitWithTimeout(absl::Condition(&cond), timeout);
  }

  void SetManualH2h(bool manual) {
    absl::MutexLock lock(h2h_mu_);
    manual_h2h_ = manual;
  }

  size_t h2h_issued() {
    absl::MutexLock lock(h2h_mu_);
    return h2h_callbacks_.size();
  }

  bool WaitForH2hIssued(size_t expected, absl::Duration timeout) {
    absl::MutexLock lock(h2h_mu_);
    auto cond = [this, expected]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(h2h_mu_) {
      return h2h_callbacks_.size() >= expected;
    };
    return h2h_mu_.AwaitWithTimeout(absl::Condition(&cond), timeout);
  }

  uint8_t* h2h_chunk_ptr(size_t index) {
    absl::MutexLock lock(h2h_mu_);
    return h2h_chunk_ptrs_.at(index);
  }

  void FinishH2h(size_t index, absl::Status status) {
    std::function<void(absl::Status)> cb;
    {
      absl::MutexLock lock(h2h_mu_);
      cb = std::move(h2h_callbacks_.at(index));
    }
    cb(std::move(status));
  }

  const StagingLayerManager& staging_layer_manager(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = send_sessions_.find(uuid);
    if (it != send_sessions_.end()) {
      served_sends_[uuid] = it->second;
      return it->second->staging_layer_manager();
    }
    return served_sends_.at(uuid)->staging_layer_manager();
  }

  absl::Status AwaitSendDone(uint64_t uuid) {
    std::shared_ptr<TransferSendSession> session;
    {
      absl::MutexLock lock(mu_);
      auto it = send_sessions_.find(uuid);
      session =
          it != send_sessions_.end() ? it->second : served_sends_.at(uuid);
    }
    return session->AwaitForDone();
  }

  bool WaitForSendDraining(uint64_t uuid, absl::Duration timeout) {
    const absl::Time deadline = absl::Now() + timeout;
    while (!send_draining(uuid) && absl::Now() < deadline) {
      absl::SleepFor(absl::Milliseconds(1));
    }
    return send_draining(uuid);
  }

 private:
  class TestBase : public kv_cache::KVCacheManagerBase {
   public:
    TestBase(size_t num_layers, TestManager* owner)
        : kv_cache::KVCacheManagerBase(num_layers, /*num_shards=*/1,
                                       std::vector<size_t>(num_layers, 128),
                                       /*local_port=*/std::nullopt,
                                       /*host_blocks_to_allocate=*/kSlots,
                                       /*parallelism=*/1, nullptr),
          owner_(owner) {}

    absl::StatusOr<raiden::PjRtCopyFuture> D2hSyncDispatch(
        const std::vector<int64_t>& src_offsets_major_dim,
        const std::vector<int64_t>& dst_offsets_major_dim,
        const std::vector<int64_t>& copy_sizes_major_dim,
        std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
        std::optional<size_t> host_layer_idx,
        std::optional<size_t> shard_idx) override {
      auto [promise, future] = xla::MakePromise<>();
      absl::MutexLock lock(owner_->copies_mu_);
      owner_->copies_.push_back(std::move(promise));
      owner_->copy_destinations_.push_back(
          host_layer_idx.has_value() ? static_cast<int64_t>(*host_layer_idx)
                                     : -1);
      return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
    }

    void H2hWriteDirectAsync(
        const std::vector<std::string>& peers,
        const std::vector<int>& src_block_ids,
        const std::vector<int>& dst_block_ids, uint64_t uuid, int layer_idx,
        std::function<void(absl::StatusOr<std::vector<int>>)> on_complete)
        override {
      {
        absl::MutexLock lock(owner_->h2h_mu_);
        if (owner_->manual_h2h_) {
          ++owner_->h2h_calls_;
          owner_->active_h2h_uuids_.insert(uuid);
          const size_t layer = static_cast<size_t>(layer_idx);
          std::vector<int64_t> block_ids(src_block_ids.begin(),
                                         src_block_ids.end());
          std::vector<transport::BlockChunk> chunks = GetBlockChunks(
              layer, /*shard_idx=*/0, block_ids, block_bytes(layer), uuid);
          owner_->h2h_chunk_ptrs_.push_back(chunks.empty() ? nullptr
                                                           : chunks[0].ptr);
          owner_->h2h_callbacks_.push_back(
              [on_complete = std::move(on_complete), src_block_ids, uuid,
               owner = owner_](absl::Status status) mutable {
                {
                  absl::MutexLock lock(owner->h2h_mu_);
                  owner->active_h2h_uuids_.erase(uuid);
                }
                if (!status.ok()) {
                  on_complete(std::move(status));
                  return;
                }
                on_complete(src_block_ids);
              });
          return;
        }
      }
      kv_cache::KVCacheManagerBase::H2hWriteDirectAsync(
          peers, src_block_ids, dst_block_ids, uuid, layer_idx,
          std::move(on_complete));
    }

   private:
    TestManager* owner_;
  };

  absl::Mutex copies_mu_;
  std::vector<xla::Promise<>> copies_ ABSL_GUARDED_BY(copies_mu_);
  std::vector<int64_t> copy_destinations_ ABSL_GUARDED_BY(copies_mu_);
  absl::Mutex h2h_mu_;
  bool manual_h2h_ ABSL_GUARDED_BY(h2h_mu_) = false;
  int h2h_calls_ ABSL_GUARDED_BY(h2h_mu_) = 0;
  absl::flat_hash_set<uint64_t> active_h2h_uuids_ ABSL_GUARDED_BY(h2h_mu_);
  std::vector<uint8_t*> h2h_chunk_ptrs_ ABSL_GUARDED_BY(h2h_mu_);
  std::vector<std::function<void(absl::Status)>> h2h_callbacks_
      ABSL_GUARDED_BY(h2h_mu_);
  absl::flat_hash_map<uint64_t, std::shared_ptr<TransferSendSession>>
      served_sends_ ABSL_GUARDED_BY(mu_);
};

// A consumer whose host-to-device copies complete when the test says so.
class RecvTestManager : public KVCacheManagerWithTransfer {
 public:
  explicit RecvTestManager(size_t num_layers, double timeout_s = 5.0)
      : KVCacheManagerWithTransfer(std::make_unique<RecvBase>(num_layers, this),
                                   /*node_id=*/0,
                                   /*local_control_port=*/-1, /*max_blocks=*/1,
                                   /*num_slots=*/kSlots, timeout_s) {}

  void AddRecv(const std::string& req_id, uint64_t uuid,
               int32_t blocks_per_layer = 1,
               std::optional<std::chrono::steady_clock::time_point> deadline =
                   std::nullopt) {
    ::tpu_sync::rpc::StartTransferRequest plan;
    plan.set_uuid(uuid);
    plan.set_use_block_chunks(true);
    auto* entry = (*plan.mutable_shard_push_schedules())[0].add_entries();
    entry->set_dst_shard_idx(0);
    entry->set_src_block_id(0);
    entry->set_dst_block_id(0);
    entry->set_size_bytes(base()->block_bytes(0));
    entry->set_count(blocks_per_layer);
    CHECK_OK(RegisterActivePlan(uuid, plan, /*is_sender=*/false));
    absl::MutexLock lock(mu_);
    active_recv_sessions_[uuid] = *TransferReceiveSession::Create(
        base(), staging_allocator_.get(), uuid, req_id, blocks_per_layer,
        deadline.value_or(DeadlineFromNow()), /*acquire_staging=*/true);
  }

  absl::Status ReceiveLayer(size_t layer, uint64_t uuid) {
    if (base()->num_host_layers() < base()->num_layers()) {
      std::vector<transport::BlockChunk> chunks =
          base()->GetBlockChunks(layer, /*shard_idx=*/0, /*block_ids=*/{0},
                                 base()->block_bytes(layer), uuid);
      if (chunks.empty()) {
        return absl::CancelledError("GetBlockChunks returned no chunks");
      }
    }
    return base()->OnLayerReceived(layer, uuid);
  }

  absl::Status ReceiveBlocks(const std::vector<int>& blocks, uint64_t uuid) {
    std::vector<int> all_shards(base()->num_shards());
    std::iota(all_shards.begin(), all_shards.end(), 0);
    return OnBlockShardsReceived(blocks, all_shards, uuid);
  }

  void FinishCopy(size_t index, absl::Status status) {
    absl::MutexLock lock(copies_mu_);
    copies_.at(index).Set(std::move(status));
  }

  size_t copies_issued() {
    absl::MutexLock lock(copies_mu_);
    return copies_.size();
  }

  // Host layer buffer the `index`-th copy issued reads from; -1 when the copy
  // named none.
  int64_t copy_source(size_t index) {
    absl::MutexLock lock(copies_mu_);
    return copy_sources_.at(index);
  }

  size_t free_slots() { return staging_allocator_->num_free_slots(); }
  StagingBlockAllocator* staging_allocator() {
    return staging_allocator_.get();
  }

  const StagingLayerManager& staging_layer_manager(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    return active_recv_sessions_.at(uuid)->staging_layer_manager();
  }

  bool has_recv(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = active_recv_sessions_.find(uuid);
    return it != active_recv_sessions_.end() && !it->second->Done();
  }

  void FailRecv(uint64_t uuid, absl::Status status) {
    std::shared_ptr<TransferReceiveSession> session;
    {
      absl::MutexLock lock(mu_);
      auto it = active_recv_sessions_.find(uuid);
      if (it == active_recv_sessions_.end()) return;
      session = it->second;
    }
    session->Finish(status);
  }

  void BlockH2dDispatch() { block_dispatch_.store(true); }

  bool WaitForH2dDispatch(absl::Duration timeout) {
    return dispatch_entered_.WaitForNotificationWithTimeout(timeout);
  }

  void ReleaseH2dDispatch() {
    if (!release_dispatch_.HasBeenNotified()) release_dispatch_.Notify();
  }

 private:
  class RecvBase : public kv_cache::KVCacheManagerBase {
   public:
    RecvBase(size_t num_layers, RecvTestManager* owner)
        : kv_cache::KVCacheManagerBase(num_layers, /*num_shards=*/1,
                                       std::vector<size_t>(num_layers, 128),
                                       /*local_port=*/std::nullopt,
                                       /*host_blocks_to_allocate=*/kSlots,
                                       /*parallelism=*/1, nullptr),
          owner_(owner) {}

    absl::StatusOr<raiden::PjRtCopyFuture> H2dSyncDispatch(
        const std::vector<int64_t>& src_offsets_major_dim,
        const std::vector<int64_t>& dst_offsets_major_dim,
        const std::vector<int64_t>& copy_sizes_major_dim,
        std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
        std::optional<size_t> host_layer_idx,
        std::optional<size_t> shard_idx) override {
      if (owner_->block_dispatch_.load()) {
        owner_->dispatch_entered_.Notify();
        owner_->release_dispatch_.WaitForNotification();
      }
      auto [promise, future] = xla::MakePromise<>();
      absl::MutexLock lock(owner_->copies_mu_);
      owner_->copies_.push_back(std::move(promise));
      owner_->copy_sources_.push_back(
          host_layer_idx.has_value() ? static_cast<int64_t>(*host_layer_idx)
                                     : -1);
      return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
    }

   private:
    RecvTestManager* owner_;
  };

  absl::Mutex copies_mu_;
  std::vector<xla::Promise<>> copies_;
  std::vector<int64_t> copy_sources_;
  std::atomic<bool> block_dispatch_{false};
  absl::Notification dispatch_entered_;
  absl::Notification release_dispatch_;
};

// Sets environment variable |name| to |value| until destroyed. The staging
// geometry is read once when a manager is constructed, so the guard only has
// to outlive construction.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    setenv(name, value, /*overwrite=*/1);
  }
  ~ScopedEnv() { unsetenv(name_); }

 private:
  const char* name_;
};

class DispatchReleaseGuard {
 public:
  explicit DispatchReleaseGuard(RecvTestManager* manager) : manager_(manager) {}
  ~DispatchReleaseGuard() { Release(); }

  void Release() {
    if (manager_ == nullptr) return;
    manager_->ReleaseH2dDispatch();
    manager_ = nullptr;
  }

 private:
  RecvTestManager* manager_;
};

using Reports = std::tuple<std::vector<std::string>, std::vector<std::string>,
                           std::vector<std::string>>;

const std::vector<std::string>& DoneSending(const Reports& r) {
  return std::get<0>(r);
}
const std::vector<std::string>& DoneReceiving(const Reports& r) {
  return std::get<1>(r);
}
const std::vector<std::string>& FailedRecving(const Reports& r) {
  return std::get<2>(r);
}

int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

TEST(SendDrainTest, ExpiredSendKeepsItsStagingUntilTheCopyEnds) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/7, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);
  producer.ServePull(7);
  ASSERT_EQ(producer.copies_issued(), 1);
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  // The deadline passes while the copy runs: the send is not reported and
  // its slot stays out of the pool.
  Reports during = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  // The copy lands after the deadline: the send is reported failed, not
  // done, and only now hands its slot back.
  producer.FinishCopy(0, absl::OkStatus());
  Reports after = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), IsEmpty());
  EXPECT_THAT(FailedRecving(after), Contains("req"));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendDrainTest, FailedLayerWaitsForTheOtherLayersCopies) {
  TestManager producer(/*num_layers=*/2);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/8, {0}), 0);
  producer.ServePull(8);
  ASSERT_EQ(producer.copies_issued(), 2);

  // Layer 0's copy fails while layer 1's still runs: the failure and the
  // slot are held back.
  producer.FinishCopy(0, absl::InternalError("copy failed"));
  Reports during = producer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  producer.FinishCopy(1, absl::OkStatus());
  Reports after = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), IsEmpty());
  EXPECT_THAT(FailedRecving(after), Contains("req"));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendDrainTest, SendNobodyPulledFailsAtItsDeadline) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/9, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);
  Reports swept = producer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(swept), Contains("req"));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, DuplicateRegistrationCannotReplaceLiveOffer) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("first", /*uuid=*/10, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);
  EXPECT_THROW(producer.NotifyForRead("replacement", /*uuid=*/10, {0}),
               std::runtime_error);
  EXPECT_TRUE(producer.has_send(10));

  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), IsEmpty());
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), ElementsAre("first"));
  Reports repeated = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(repeated), IsEmpty());
  EXPECT_THAT(DoneReceiving(repeated), IsEmpty());
  EXPECT_THAT(FailedRecving(repeated), IsEmpty());
}

TEST(SendLifecycleTest, DuplicateBlocksAreRejectedAtRegistration) {
  TestManager producer(/*num_layers=*/1);

  // Returning 0 would be indistinguishable from "nothing to send"; the caller
  // must receive an exception identifying the duplicate block and its indices.
  std::string error_message;
  try {
    producer.NotifyForRead("req", /*uuid=*/15, {5, 9, 5});
    FAIL() << "a duplicate block must be rejected";
  } catch (const std::invalid_argument& e) {
    error_message = e.what();
  }
  EXPECT_THAT(error_message, HasSubstr("duplicate block 5"));
  EXPECT_THAT(error_message, HasSubstr("req_id=req, uuid=15"));
  EXPECT_THAT(error_message, HasSubstr("positions 0 and 2 of 3 blocks"));
  EXPECT_FALSE(producer.has_send(15));
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), IsEmpty());
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
}

TEST(SendLifecycleTest, SuccessfulSendWithoutWorkSettlesImmediately) {
  TestManager producer(/*num_layers=*/1);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/10, /*in_flight=*/0);

  producer.Decide(session, /*failed=*/false);
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), Contains("req"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
  EXPECT_FALSE(producer.has_send(10));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, FailedSendWithoutWorkSettlesImmediately) {
  TestManager producer(/*num_layers=*/1);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/11, /*in_flight=*/0);

  producer.Decide(session, /*failed=*/true);
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(producer.has_send(11));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, SuccessfulSendWaitsForEveryOperation) {
  TestManager producer(/*num_layers=*/2);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/12, /*in_flight=*/2);

  producer.Decide(session, /*failed=*/false);
  producer.FinishCopy(0, absl::OkStatus());
  Reports during = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_TRUE(producer.has_send(12));
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  producer.FinishCopy(1, absl::OkStatus());
  Reports after = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), Contains("req"));
  EXPECT_THAT(FailedRecving(after), IsEmpty());
  EXPECT_FALSE(producer.has_send(12));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, FailureCannotOverrideAnEarlierSuccess) {
  TestManager producer(/*num_layers=*/1);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/13, /*in_flight=*/1);

  producer.Decide(session, /*failed=*/false);
  producer.Decide(session, /*failed=*/true);
  producer.FinishCopy(0, absl::OkStatus());
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), Contains("req"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
  EXPECT_FALSE(producer.has_send(13));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, SuccessCannotOverrideAnEarlierFailure) {
  TestManager producer(/*num_layers=*/1);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/14, /*in_flight=*/1);

  producer.Decide(session, /*failed=*/true);
  producer.Decide(session, /*failed=*/false);
  producer.FinishCopy(0, absl::OkStatus());
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(producer.has_send(14));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, DetailsReportWhenTheSendCompleted) {
  TestManager producer(/*num_layers=*/1);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/16, /*in_flight=*/1);
  producer.Decide(session, /*failed=*/false);

  const int64_t before = SteadyNowNs();
  producer.FinishCopy(0, absl::OkStatus());
  const int64_t completed = SteadyNowNs();
  // A late poll must not move the reported completion time.
  absl::SleepFor(absl::Milliseconds(20));
  CompleteReadResult completions = producer.CompleteReadWithDetails();

  EXPECT_THAT(completions.sent, ElementsAre("req"));
  EXPECT_THAT(completions.details["req"].completed_ns,
              Optional(AllOf(Ge(before), Le(completed))));
  EXPECT_THAT(producer.CompleteReadWithDetails().details, IsEmpty());
}

TEST(SendLifecycleTest, DetailsSayAnUnpulledSendTimedOutAwaitingItsPull) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/17, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("Transfer failed: req_id=req "
                                 "code=DEADLINE_EXCEEDED phase=await_pull")));
  log.StartCapturingLogs();

  CompleteReadResult completions = producer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "DEADLINE_EXCEEDED");
  EXPECT_EQ(details.reason, "Send session timed out");
  EXPECT_EQ(details.phase, "await_pull");
  EXPECT_EQ(details.peer, std::nullopt);
  EXPECT_THAT(details.elapsed_ns, Optional(Ge(0)));
}

TEST(SendDrainTest, DetailsSayAnExpiredSendWasStillCopying) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/18, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);
  producer.ServePull(18);
  ASSERT_THAT(producer.CompleteReadWithDetails().failed, IsEmpty());

  producer.FinishCopy(0, absl::OkStatus());
  CompleteReadResult completions = producer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "DEADLINE_EXCEEDED");
  EXPECT_EQ(details.phase, "d2h");
  EXPECT_EQ(details.peer, "127.0.0.1:1");
}

TEST(SendDrainTest, DetailsCarryTheErrorOfAFailedCopy) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/19, {0}), 0);
  producer.ServePull(19);
  producer.FinishCopy(0, absl::InternalError("copy failed"));

  CompleteReadResult completions = producer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "INTERNAL");
  EXPECT_EQ(details.reason, "copy failed");
  EXPECT_EQ(details.phase, "d2h");
}

TEST(SendDrainTest, DetailsSayAFailedPushFailedInPushWhileCopiesRun) {
  GetFaultInjector().Reset();
  absl::Cleanup reset_faults = [] { GetFaultInjector().Reset(); };
  FaultInjectionRule failed_push;
  failed_push.hook = std::string(hooks::kRaidenManagerBaseH2hWrite);
  failed_push.probability = 1.0;
  ABSL_ASSERT_OK(GetFaultInjector().Install({failed_push}));
  TestManager producer(/*num_layers=*/2, /*timeout_s=*/5.0);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/27, {0}), 0);
  producer.ServePull(27);
  ASSERT_EQ(producer.copies_issued(), 2);

  // Layer 0's push fails on a push worker while layer 1 is still copying.
  producer.FinishCopy(0, absl::OkStatus());
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (!producer.send_draining(27) && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  ASSERT_TRUE(producer.send_draining(27));
  producer.FinishCopy(1, absl::OkStatus());
  CompleteReadResult completions = producer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "INTERNAL");
  EXPECT_THAT(details.reason, Optional(HasSubstr("injected fault")));
  EXPECT_EQ(details.phase, "push");
}

TEST(RecvLifecycleTest, NetworkCompletionWaitsForH2d) {
  RecvTestManager consumer(/*num_layers=*/1);
  consumer.AddRecv("req", /*uuid=*/20);
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/20).ok());
  ASSERT_EQ(consumer.copies_issued(), 1);
  ASSERT_TRUE(consumer.ReceiveBlocks({0}, /*uuid=*/20).ok());

  Reports during = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(during), IsEmpty());
  EXPECT_TRUE(consumer.has_recv(20));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FinishCopy(0, absl::OkStatus());
  Reports after = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(after), Contains("req"));
  EXPECT_THAT(FailedRecving(after), IsEmpty());
  EXPECT_FALSE(consumer.has_recv(20));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest, DetailsReportWhenTheLastH2dLanded) {
  RecvTestManager consumer(/*num_layers=*/1);
  consumer.AddRecv("req", /*uuid=*/26);
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/26).ok());
  ASSERT_TRUE(consumer.ReceiveBlocks({0}, /*uuid=*/26).ok());

  const int64_t before = SteadyNowNs();
  consumer.FinishCopy(0, absl::OkStatus());
  const int64_t completed = SteadyNowNs();
  absl::SleepFor(absl::Milliseconds(20));
  CompleteReadResult completions = consumer.CompleteReadWithDetails();

  EXPECT_THAT(completions.received, ElementsAre("req"));
  EXPECT_THAT(completions.details["req"].completed_ns,
              Optional(AllOf(Ge(before), Le(completed))));
}

TEST(RecvLifecycleTest, DetailsSayAReceiveTimedOutWaitingForData) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/kTimeoutS);
  consumer.AddRecv(
      "req", /*uuid=*/28, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() - std::chrono::milliseconds(1));

  CompleteReadResult completions = consumer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "DEADLINE_EXCEEDED");
  EXPECT_EQ(details.reason, "Receive session timed out");
  EXPECT_EQ(details.phase, "receive");
}

TEST(RecvLifecycleTest, DetailsSayAFailedUploadFailedInH2dWhileDataArrives) {
  RecvTestManager consumer(/*num_layers=*/1);
  consumer.AddRecv("req", /*uuid=*/29);
  ABSL_ASSERT_OK(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/29));
  consumer.FinishCopy(0, absl::InternalError("copy failed"));

  CompleteReadResult completions = consumer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "INTERNAL");
  EXPECT_EQ(details.reason, "copy failed");
  EXPECT_EQ(details.phase, "h2d");
  EXPECT_EQ(details.peer, std::nullopt);
}

TEST(RecvLifecycleTest, InvalidReadShapeDoesNotLeakStaging) {
  RecvTestManager consumer(/*num_layers=*/1);
  const size_t free_before = consumer.free_slots();

  EXPECT_THROW(consumer.StartRead("req", /*uuid=*/25,
                                  /*remote_endpoint=*/"unused:1",
                                  /*remote_block_ids=*/{0, 1},
                                  /*local_block_ids=*/{0}),
               std::invalid_argument);
  EXPECT_EQ(consumer.free_slots(), free_before);
  EXPECT_FALSE(consumer.has_recv(25));
}

TEST(RecvLifecycleTest, LateBlockAccountingAfterRetirementIsANoOp) {
  RecvTestManager consumer(/*num_layers=*/1);
  consumer.AddRecv("req", /*uuid=*/21);
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/21).ok());
  consumer.FinishCopy(0, absl::OkStatus());

  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), Contains("req"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
  EXPECT_FALSE(consumer.has_recv(21));
  EXPECT_EQ(consumer.free_slots(), kSlots);

  // BlockTransport calls this immediately after OnLayerReceived. A fast H2D
  // callback may already have retired the receive, so the late accounting is
  // deliberately harmless.
  EXPECT_TRUE(consumer.ReceiveBlocks({0}, /*uuid=*/21).ok());
  Reports after_late_accounting = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after_late_accounting), IsEmpty());
  EXPECT_THAT(DoneReceiving(after_late_accounting), IsEmpty());
  EXPECT_THAT(FailedRecving(after_late_accounting), IsEmpty());
  EXPECT_FALSE(consumer.has_recv(21));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest, OutOfOrderLayersSettleAfterEveryH2d) {
  RecvTestManager consumer(/*num_layers=*/2);
  consumer.AddRecv("req", /*uuid=*/22);

  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/1, /*uuid=*/22).ok());
  ASSERT_TRUE(consumer.ReceiveBlocks({0}, /*uuid=*/22).ok());
  consumer.FinishCopy(0, absl::OkStatus());
  Reports after_one = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(after_one), IsEmpty());
  EXPECT_TRUE(consumer.has_recv(22));

  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/22).ok());
  ASSERT_TRUE(consumer.ReceiveBlocks({0}, /*uuid=*/22).ok());
  consumer.FinishCopy(1, absl::OkStatus());
  Reports after_both = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(after_both), Contains("req"));
  EXPECT_THAT(FailedRecving(after_both), IsEmpty());
  EXPECT_FALSE(consumer.has_recv(22));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest, SingleFailedH2dReportsFailureAndReturnsStaging) {
  RecvTestManager consumer(/*num_layers=*/1);
  consumer.AddRecv("req", /*uuid=*/23);
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/23).ok());
  consumer.FinishCopy(0, absl::InternalError("copy failed"));

  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(consumer.has_recv(23));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest, ReceiveWithoutTrafficFailsAtItsDeadline) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/kTimeoutS);
  consumer.AddRecv(
      "req", /*uuid=*/24, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() - std::chrono::milliseconds(1));

  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(consumer.has_recv(24));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvDrainTest, ExpiredReceiveKeepsStagingUntilH2dEnds) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/kTimeoutS);
  consumer.AddRecv(
      "req", /*uuid=*/25, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/25).ok());
  ASSERT_EQ(consumer.copies_issued(), 1);

  absl::SleepFor(absl::Milliseconds(15));
  Reports during = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(DoneReceiving(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_TRUE(consumer.has_recv(25));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FinishCopy(0, absl::OkStatus());
  Reports after = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), IsEmpty());
  EXPECT_THAT(DoneReceiving(after), IsEmpty());
  EXPECT_THAT(FailedRecving(after), ElementsAre("req"));
  EXPECT_FALSE(consumer.has_recv(25));
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports repeated = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(repeated), IsEmpty());
  EXPECT_THAT(DoneReceiving(repeated), IsEmpty());
  EXPECT_THAT(FailedRecving(repeated), IsEmpty());
}

TEST(RecvDrainTest, DuplicateUuidIsRejectedUntilExpiredReceiveDrains) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/kTimeoutS);
  consumer.AddRecv(
      "old", /*uuid=*/28, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/28).ok());
  ASSERT_EQ(consumer.copies_issued(), 1);

  absl::SleepFor(absl::Milliseconds(15));
  Reports during = consumer.CompleteReadRaw();
  ASSERT_THAT(DoneSending(during), IsEmpty());
  ASSERT_THAT(DoneReceiving(during), IsEmpty());
  ASSERT_THAT(FailedRecving(during), IsEmpty());
  ASSERT_TRUE(consumer.has_recv(28));

  // Reusing a UUID while callbacks can still arrive would let old traffic
  // mutate the replacement. The public registration API must keep the old
  // session authoritative until its issued work has drained.
  EXPECT_FALSE(consumer
                   .RegisterRecv(/*uuid=*/28, "retry",
                                 /*expected_block_count=*/1)
                   .ok());
  EXPECT_TRUE(consumer.has_recv(28));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FinishCopy(0, absl::OkStatus());
  Reports retired = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(retired), IsEmpty());
  EXPECT_THAT(DoneReceiving(retired), IsEmpty());
  EXPECT_THAT(FailedRecving(retired), ElementsAre("old"));
  EXPECT_FALSE(consumer.has_recv(28));
  EXPECT_EQ(consumer.free_slots(), kSlots);

  EXPECT_TRUE(consumer
                  .RegisterRecv(/*uuid=*/28, "retry",
                                /*expected_block_count=*/1,
                                std::chrono::steady_clock::now() -
                                    std::chrono::milliseconds(1))
                  .ok());
  Reports retry = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(retry), IsEmpty());
  EXPECT_THAT(DoneReceiving(retry), IsEmpty());
  EXPECT_THAT(FailedRecving(retry), ElementsAre("retry"));
  EXPECT_FALSE(consumer.has_recv(28));

  Reports repeated = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(repeated), IsEmpty());
  EXPECT_THAT(DoneReceiving(repeated), IsEmpty());
  EXPECT_THAT(FailedRecving(repeated), IsEmpty());
}

TEST(RecvDrainTest, FailedLayerWaitsForOtherH2dCopies) {
  RecvTestManager consumer(/*num_layers=*/2);
  consumer.AddRecv("req", /*uuid=*/26);
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/26).ok());
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/1, /*uuid=*/26).ok());
  ASSERT_EQ(consumer.copies_issued(), 2);

  consumer.FinishCopy(0, absl::InternalError("copy failed"));
  Reports during = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(DoneReceiving(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_TRUE(consumer.has_recv(26));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FinishCopy(1, absl::OkStatus());
  Reports after = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), IsEmpty());
  EXPECT_THAT(DoneReceiving(after), IsEmpty());
  EXPECT_THAT(FailedRecving(after), ElementsAre("req"));
  EXPECT_FALSE(consumer.has_recv(26));
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports repeated = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(repeated), IsEmpty());
  EXPECT_THAT(DoneReceiving(repeated), IsEmpty());
  EXPECT_THAT(FailedRecving(repeated), IsEmpty());
}

TEST(RecvDrainTest, TimeoutDuringH2dDispatchKeepsStaging) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/kTimeoutS);
  consumer.AddRecv(
      "req", /*uuid=*/27, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
  consumer.BlockH2dDispatch();
  auto receive = std::async(std::launch::async, [&consumer] {
    return consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/27);
  });
  DispatchReleaseGuard release_dispatch(&consumer);
  ASSERT_TRUE(consumer.WaitForH2dDispatch(absl::Seconds(5)));

  absl::SleepFor(absl::Milliseconds(15));
  Reports during = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(DoneReceiving(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_TRUE(consumer.has_recv(27));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  release_dispatch.Release();
  ASSERT_EQ(receive.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  ASSERT_TRUE(receive.get().ok());
  ASSERT_EQ(consumer.copies_issued(), 1);
  consumer.FinishCopy(0, absl::OkStatus());
  Reports after = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), IsEmpty());
  EXPECT_THAT(DoneReceiving(after), IsEmpty());
  EXPECT_THAT(FailedRecving(after), ElementsAre("req"));
  EXPECT_FALSE(consumer.has_recv(27));
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports repeated = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(repeated), IsEmpty());
  EXPECT_THAT(DoneReceiving(repeated), IsEmpty());
  EXPECT_THAT(FailedRecving(repeated), IsEmpty());
}

TEST(SendDrainTest, WaitForPendingWorkWaitsForActiveSendSession) {
  TestManager producer(/*num_layers=*/1, /*timeout_s=*/5.0);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/77, /*in_flight=*/1);
  producer.Decide(session, /*failed=*/false);
  ASSERT_EQ(producer.copies_issued(), 1);

  absl::Notification wait_finished;
  absl::Status wait_status;
  std::thread waiter([&]() {
    wait_status = producer.WaitForPendingWork();
    wait_finished.Notify();
  });

  EXPECT_FALSE(
      wait_finished.WaitForNotificationWithTimeout(absl::Milliseconds(150)));

  producer.FinishCopy(0, absl::OkStatus());
  EXPECT_TRUE(wait_finished.WaitForNotificationWithTimeout(absl::Seconds(2)));
  EXPECT_THAT(wait_status, ::absl_testing::IsOk());
  waiter.join();
}

TEST(RecvLifecycleTest,
     SessionRemainsValidAcrossAsyncH2dCallbacksAfterCallerDropsPtr) {
  RecvTestManager consumer(/*num_layers=*/1);
  std::weak_ptr<TransferReceiveSession> weak_session;
  {
    std::shared_ptr<TransferReceiveSession> session =
        *TransferReceiveSession::Create(
            consumer.base(), consumer.staging_allocator(),
            /*uuid=*/88, "req_lifetime",
            /*total_blocks=*/1,
            std::chrono::steady_clock::now() + std::chrono::seconds(10),
            /*acquire_staging=*/true);
    weak_session = session;
    EXPECT_TRUE(session->HasStaging());
    EXPECT_EQ(consumer.free_slots(), kSlots - 1);
    ASSERT_THAT(session->ExecuteLayerH2d(consumer, /*layer_idx=*/0),
                ::absl_testing::IsOk());
    ASSERT_EQ(consumer.copies_issued(), 1);
  }
  // Caller dropped its std::shared_ptr; the pending H2D callback retains the
  // session and its staging slot until the copy completes.
  EXPECT_FALSE(weak_session.expired());
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FinishCopy(0, absl::OkStatus());
  EXPECT_TRUE(weak_session.expired());
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest,
     IncomingPushLeasePinsStagingDuringWriteAndRejectsWhenDraining) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/10.0);
  consumer.AddRecv("req_ingress_lease", /*uuid=*/91, /*blocks_per_layer=*/1);
  ASSERT_EQ(consumer.free_slots(), kSlots - 1);

  EXPECT_THAT(consumer.base()->BeginIncomingPush(/*uuid=*/999),
              ::absl_testing::StatusIs(absl::StatusCode::kNotFound));

  ASSERT_THAT(consumer.base()->BeginIncomingPush(/*uuid=*/91),
              ::absl_testing::IsOk());

  // Failing/timing out mid-write marks the session draining and keeps staging
  // pinned until EndIncomingPush finishes.
  consumer.FailRecv(/*uuid=*/91, absl::InternalError("simulated failure"));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  EXPECT_THAT(consumer.base()->BeginIncomingPush(/*uuid=*/91),
              ::absl_testing::StatusIs(absl::StatusCode::kCancelled));

  EXPECT_THAT(consumer.base()->EndIncomingPush(/*uuid=*/91),
              ::absl_testing::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest,
     IncomingPushLeaseSpansLayerH2dAndBlockAccountingBeforeReleasing) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/10.0);
  consumer.AddRecv("req_handoff", /*uuid=*/92, /*blocks_per_layer=*/1);
  ASSERT_EQ(consumer.free_slots(), kSlots - 1);

  // Simulate BlockTransport::HandleIncomingPush holding the incoming push lease
  // across OnLayerReceived, synchronous H2D completion, and
  // OnBlockShardsReceived.
  ASSERT_THAT(consumer.base()->BeginIncomingPush(/*uuid=*/92),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/92),
              ::absl_testing::IsOk());
  consumer.FinishCopy(0, absl::OkStatus());
  // Even though the H2D copy finished, staging must remain pinned until
  // OnBlockShardsReceived and EndIncomingPush complete.
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  ASSERT_THAT(consumer.ReceiveBlocks({0}, /*uuid=*/92), ::absl_testing::IsOk());
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  EXPECT_THAT(consumer.base()->EndIncomingPush(/*uuid=*/92),
              ::absl_testing::IsOk());
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), Contains("req_handoff"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
}

TEST(
    RecvLifecycleTest,
    FailedIncomingPushImmediatelyFailsSessionAndReleasesStagingBeforeDeadline) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/120.0);
  consumer.AddRecv("req_push_died", /*uuid=*/93, /*blocks_per_layer=*/1);
  ASSERT_EQ(consumer.free_slots(), kSlots - 1);

  ABSL_ASSERT_OK(consumer.base()->BeginIncomingPush(/*uuid=*/93));

  // When an incoming push fails (e.g. socket read timeout or peer disconnect),
  // EndIncomingPush receives the non-OK push status and immediately fails the
  // session, releasing the staging slot without waiting for the 120s deadline.
  EXPECT_THAT(
      consumer.base()->EndIncomingPush(
          /*uuid=*/93, absl::DeadlineExceededError("Socket read timed out")),
      ::absl_testing::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(consumer.free_slots(), kSlots);

  // Subsequent incoming pushes for the same uuid are immediately rejected.
  EXPECT_THAT(consumer.base()->BeginIncomingPush(/*uuid=*/93),
              ::absl_testing::StatusIs(absl::StatusCode::kCancelled));

  // CompleteReadRaw immediately reports the request in failed_recving and
  // retires the session before the 120s deadline.
  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), ElementsAre("req_push_died"));
  EXPECT_FALSE(consumer.has_recv(93));
}

TEST(RecvLifecycleTest,
     NonSessionTransfersSucceedOverBlockTransportWhileStaleUuidIsRejected) {
  constexpr size_t kSliceBytes = 128;
  KVCacheManagerWithTransfer sender(
      /*num_layers=*/1, /*num_shards=*/1, kSliceBytes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/4,
      /*parallelism=*/1, /*node_id=*/0, /*local_control_port=*/-1,
      /*max_blocks=*/1, /*num_slots=*/1, /*timeout_s=*/10.0);
  KVCacheManagerWithTransfer receiver(
      /*num_layers=*/1, /*num_shards=*/1, kSliceBytes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/4,
      /*parallelism=*/1, /*node_id=*/1, /*local_control_port=*/-1,
      /*max_blocks=*/1, /*num_slots=*/1, /*timeout_s=*/10.0);

  const std::string peer =
      absl::StrCat("127.0.0.1:", *receiver.base()->local_port());

  // 1. Raw H2H push with dynamic block allocation (op = 1, uuid = 0).
  std::memset(sender.base()->GetBlockHostPointer(0, 0, 0), 0x5A, kSliceBytes);
  absl::StatusOr<std::vector<int>> dyn_ids =
      sender.base()->H2hWriteDirect(peer, /*src_block_ids=*/{0});
  ASSERT_THAT(dyn_ids, ::absl_testing::IsOk());
  ASSERT_EQ(dyn_ids->size(), 1);
  EXPECT_EQ(receiver.base()->GetBlockHostPointer(0, 0, (*dyn_ids)[0])[0], 0x5A);

  // 2. Raw H2H push with explicit destination block (op = 6, uuid = 0).
  std::memset(sender.base()->GetBlockHostPointer(0, 0, 0), 0xA5, kSliceBytes);
  absl::StatusOr<std::vector<int>> exp_ids = sender.base()->H2hWriteDirect(
      peer, /*src_block_ids=*/{0}, /*dst_block_ids=*/{2}, /*uuid=*/0);
  ASSERT_THAT(exp_ids, ::absl_testing::IsOk());
  EXPECT_EQ(receiver.base()->GetBlockHostPointer(0, 0, 2)[0], 0xA5);

  // 3. Host-only (MEMORY_TYPE_DRAM) active plan push (op = 6, uuid = 42,
  //    no TransferReceiveSession in active_recv_sessions_).
  ::tpu_sync::rpc::StartTransferRequest dram_plan;
  dram_plan.set_uuid(42);
  dram_plan.set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  dram_plan.set_use_block_chunks(true);
  auto* entry = (*dram_plan.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer(peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_block_id(0);
  entry->set_dst_block_id(3);
  entry->set_size_bytes(kSliceBytes);
  entry->set_count(1);
  ASSERT_THAT(receiver.RegisterActivePlan(42, dram_plan, /*is_sender=*/false),
              ::absl_testing::IsOk());

  std::memset(sender.base()->GetBlockHostPointer(0, 0, 0), 0x3C, kSliceBytes);
  absl::StatusOr<std::vector<int>> plan_ids = sender.base()->H2hWriteDirect(
      peer, /*src_block_ids=*/{0}, /*dst_block_ids=*/{3}, /*uuid=*/42);
  ASSERT_THAT(plan_ids, ::absl_testing::IsOk());
  EXPECT_EQ(receiver.base()->GetBlockHostPointer(0, 0, 3)[0], 0x3C);

  // 4. Once the DRAM plan is unregistered, pushes for uuid = 42 are rejected
  //    before writing to host memory, preserving the existing bytes at block 3.
  ASSERT_THAT(receiver.UnregisterActivePlan(42), ::absl_testing::IsOk());
  std::memset(sender.base()->GetBlockHostPointer(0, 0, 0), 0xFF, kSliceBytes);
  absl::StatusOr<std::vector<int>> rejected = sender.base()->H2hWriteDirect(
      peer, /*src_block_ids=*/{0}, /*dst_block_ids=*/{3}, /*uuid=*/42);
  EXPECT_FALSE(rejected.ok());
  EXPECT_EQ(receiver.base()->GetBlockHostPointer(0, 0, 3)[0], 0x3C);
}

// StagingLayerManager: TPU_RAIDEN_NUM_STAGING_LAYERS=4 leaves four host layer
// buffers for a 64-layer cache. Layers lease them in arrival order, a fifth
// layer waits for a release, and a released buffer goes to the waiting layer
// whichever layer freed it.
TEST(RecvLifecycleTest, StagingLayerManagerCyclesHostLayerBuffers) {
  constexpr size_t kNumLayers = 64;
  constexpr size_t kNumStagingLayers = 4;
  const size_t full_dram_bytes =
      RecvTestManager(kNumLayers).base()->GetAllocatedHostDramBytes();
  ASSERT_GT(full_dram_bytes, 0);

  ScopedEnv staging_layers("TPU_RAIDEN_NUM_STAGING_LAYERS", "4");
  RecvTestManager consumer(kNumLayers, /*timeout_s=*/10.0);
  kv_cache::KVCacheManagerBase* base = consumer.base();
  EXPECT_EQ(base->num_host_layers(), kNumStagingLayers);
  EXPECT_EQ(base->GetAllocatedHostDramBytes(),
            full_dram_bytes / (kNumLayers / kNumStagingLayers));
  EXPECT_NE(base->GetHostPointer(kNumStagingLayers - 1, 0), nullptr);
  EXPECT_EQ(base->GetHostPointer(kNumStagingLayers, 0), nullptr);

  consumer.AddRecv("req_staging_layers", /*uuid=*/104);
  const StagingLayerManager& staging_manager =
      consumer.staging_layer_manager(104);
  EXPECT_EQ(staging_manager.num_free(), kNumStagingLayers);

  // The transport addresses layer 0's bytes before the layer completes: the
  // chunk lands in the buffer leased to layer 0 on that first touch.
  const size_t block_bytes = base->block_bytes(0);
  std::vector<transport::BlockChunk> chunks = base->GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, /*block_ids=*/{0}, block_bytes,
      /*uuid=*/104);
  ASSERT_EQ(chunks.size(), 1);
  ASSERT_THAT(staging_manager.Lookup(0), Optional(0u));
  EXPECT_EQ(chunks[0].ptr, base->GetHostPointer(0, 0));

  // Layers 0..3 lease the four buffers in arrival order and upload from them.
  for (size_t layer = 0; layer < kNumStagingLayers; ++layer) {
    ASSERT_THAT(consumer.ReceiveLayer(layer, /*uuid=*/104),
                ::absl_testing::IsOk());
    EXPECT_THAT(staging_manager.Lookup(layer), Optional(layer));
    EXPECT_EQ(consumer.copy_source(layer), static_cast<int64_t>(layer));
  }
  EXPECT_EQ(consumer.copies_issued(), kNumStagingLayers);
  EXPECT_EQ(staging_manager.num_free(), 0);

  // Layer 4 finds every buffer leased and waits.
  absl::Notification layer4_started;
  absl::Notification layer4_done;
  absl::Status layer4_status;
  std::thread layer4_thread([&] {
    layer4_started.Notify();
    layer4_status = consumer.ReceiveLayer(/*layer=*/4, /*uuid=*/104);
    layer4_done.Notify();
  });
  layer4_started.WaitForNotification();
  EXPECT_FALSE(
      layer4_done.WaitForNotificationWithTimeout(absl::Milliseconds(50)));
  EXPECT_EQ(consumer.copies_issued(), kNumStagingLayers);

  // Completing layer 1's upload frees its buffer, which layer 4 takes.
  consumer.FinishCopy(1, absl::OkStatus());
  ASSERT_TRUE(layer4_done.WaitForNotificationWithTimeout(absl::Seconds(5)));
  layer4_thread.join();
  EXPECT_THAT(layer4_status, ::absl_testing::IsOk());
  EXPECT_EQ(staging_manager.Lookup(1), std::nullopt);
  EXPECT_THAT(staging_manager.Lookup(4), Optional(1u));
  EXPECT_EQ(consumer.copy_source(4), 1);

  // Stream the remaining layers through the manager: each takes the buffer of
  // the upload completed just before it.
  for (size_t layer = 5; layer < kNumLayers; ++layer) {
    const size_t finished_layer = layer == 5 ? 0 : layer - 4;
    const size_t freed_slot = *staging_manager.Lookup(finished_layer);
    consumer.FinishCopy(finished_layer, absl::OkStatus());
    ASSERT_THAT(consumer.ReceiveLayer(layer, /*uuid=*/104),
                ::absl_testing::IsOk());
    EXPECT_THAT(staging_manager.Lookup(layer), Optional(freed_slot));
    EXPECT_EQ(consumer.copy_source(layer), static_cast<int64_t>(freed_slot));
  }
  for (size_t layer = kNumLayers - kNumStagingLayers; layer < kNumLayers;
       ++layer) {
    consumer.FinishCopy(layer, absl::OkStatus());
  }
  EXPECT_EQ(staging_manager.num_free(), kNumStagingLayers);

  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), Contains("req_staging_layers"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
  EXPECT_FALSE(consumer.has_recv(104));
}

TEST(RecvLifecycleTest,
     StagingLayerManagerReleasesAllResourcesAcrossFailureModes) {
  constexpr size_t kNumLayers = 16;
  constexpr size_t kNumStagingLayers = 4;
  ScopedEnv staging_layers("TPU_RAIDEN_NUM_STAGING_LAYERS", "4");
  RecvTestManager consumer(kNumLayers, /*timeout_s=*/10.0);
  kv_cache::KVCacheManagerBase* base = consumer.base();
  const size_t block_bytes = base->block_bytes(0);

  // 1. H2D failure while a 5th layer is blocked in
  //    StagingLayerManager::Acquire: failing layer 0 shuts down the manager
  //    before releasing slot 0 so layer 4 wakes with cancellation instead of
  //    issuing another H2D copy, and the host staging allocation stays pinned
  //    until layers 1..3 finish.
  consumer.AddRecv("req_h2d_fail_with_waiter", /*uuid=*/201);
  const StagingLayerManager& manager1 = consumer.staging_layer_manager(201);
  // Calling OnLayerReceived before GetBlockChunks leased a buffer is rejected.
  EXPECT_THAT(base->OnLayerReceived(/*layer_idx=*/0, /*uuid=*/201),
              ::absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
  for (size_t layer = 0; layer < kNumStagingLayers; ++layer) {
    ASSERT_THAT(consumer.ReceiveLayer(layer, /*uuid=*/201),
                ::absl_testing::IsOk());
  }
  EXPECT_EQ(consumer.copies_issued(), kNumStagingLayers);
  EXPECT_EQ(manager1.num_free(), 0);
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  absl::Notification layer4_started;
  absl::Notification layer4_done;
  absl::Status layer4_status;
  std::thread layer4_thread([&] {
    layer4_started.Notify();
    layer4_status = consumer.ReceiveLayer(/*layer=*/4, /*uuid=*/201);
    layer4_done.Notify();
  });
  layer4_started.WaitForNotification();
  EXPECT_FALSE(
      layer4_done.WaitForNotificationWithTimeout(absl::Milliseconds(50)));

  consumer.FinishCopy(0, absl::InternalError("H2D error on layer 0"));
  ASSERT_TRUE(layer4_done.WaitForNotificationWithTimeout(absl::Seconds(5)));
  layer4_thread.join();
  EXPECT_THAT(layer4_status,
              ::absl_testing::StatusIs(absl::StatusCode::kCancelled));
  // Layer 4 was cancelled in GetBlockChunks and did not start a 5th H2D copy.
  EXPECT_EQ(consumer.copies_issued(), kNumStagingLayers);
  // Layers 1..3 are still in flight, so the staging block remains pinned.
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  for (size_t layer = 1; layer < kNumStagingLayers; ++layer) {
    consumer.FinishCopy(layer, absl::OkStatus());
  }
  EXPECT_EQ(manager1.num_free(), kNumStagingLayers);
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports reports1 = consumer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(reports1), Contains("req_h2d_fail_with_waiter"));
  EXPECT_FALSE(consumer.has_recv(201));

  // 2. Network push failure after GetBlockChunks leased a staging slot (before
  //    OnLayerReceived is ever reached): EndIncomingPush fails the session and
  //    releases both the layer lease and the staging block.
  consumer.AddRecv("req_push_abort_after_chunk", /*uuid=*/202);
  const StagingLayerManager& manager2 = consumer.staging_layer_manager(202);
  ASSERT_THAT(base->BeginIncomingPush(/*uuid=*/202), ::absl_testing::IsOk());
  std::vector<transport::BlockChunk> chunks = base->GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, /*block_ids=*/{0}, block_bytes,
      /*uuid=*/202);
  ASSERT_EQ(chunks.size(), 1);
  EXPECT_EQ(manager2.num_free(), kNumStagingLayers - 1);
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FailRecv(/*uuid=*/202, absl::UnavailableError("socket closed"));
  // Still pinned while the incoming push is in flight.
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);
  EXPECT_THAT(base->EndIncomingPush(/*uuid=*/202),
              ::absl_testing::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(manager2.num_free(), kNumStagingLayers);
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports reports2 = consumer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(reports2), Contains("req_push_abort_after_chunk"));
  EXPECT_FALSE(consumer.has_recv(202));

  // 3. StagingLayerManager::Acquire deadline exceeded while all 4 buffers are
  //    held by in-flight H2D copies: the timeout fails the session, keeps
  //    staging pinned until the in-flight copies complete, and then releases
  //    all slots.
  consumer.AddRecv(
      "req_staging_timeout", /*uuid=*/203, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() + std::chrono::milliseconds(80));
  const StagingLayerManager& manager3 = consumer.staging_layer_manager(203);
  const size_t base_copies = consumer.copies_issued();
  for (size_t layer = 0; layer < kNumStagingLayers; ++layer) {
    ASSERT_THAT(consumer.ReceiveLayer(layer, /*uuid=*/203),
                ::absl_testing::IsOk());
  }
  EXPECT_EQ(manager3.num_free(), 0);
  EXPECT_THAT(consumer.ReceiveLayer(/*layer=*/4, /*uuid=*/203),
              ::absl_testing::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  for (size_t i = 0; i < kNumStagingLayers; ++i) {
    consumer.FinishCopy(base_copies + i, absl::OkStatus());
  }
  EXPECT_EQ(manager3.num_free(), kNumStagingLayers);
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports reports3 = consumer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(reports3), Contains("req_staging_timeout"));
  EXPECT_FALSE(consumer.has_recv(203));
}

// Sender-side StagingLayerManager: with TPU_RAIDEN_NUM_STAGING_LAYERS=4 on a
// 64-layer cache, StartPush leases the four host layer buffers for layers 0..3
// and waits before issuing layer 4's D2H copy. Completing a layer's H2H push
// returns its host layer buffer to the manager so the next waiting layer can
// D2H into it and H2H out of it.
TEST(SendLifecycleTest, StagingLayerManagerCyclesHostLayerBuffers) {
  constexpr size_t kNumLayers = 64;
  constexpr size_t kNumStagingLayers = 4;
  const size_t full_dram_bytes =
      TestManager(kNumLayers).base()->GetAllocatedHostDramBytes();
  ASSERT_GT(full_dram_bytes, 0);

  ScopedEnv staging_layers("TPU_RAIDEN_NUM_STAGING_LAYERS", "4");
  TestManager producer(kNumLayers, /*timeout_s=*/10.0);
  producer.SetManualH2h(true);
  kv_cache::KVCacheManagerBase* base = producer.base();
  EXPECT_EQ(base->num_host_layers(), kNumStagingLayers);
  EXPECT_EQ(base->GetAllocatedHostDramBytes(),
            full_dram_bytes / (kNumLayers / kNumStagingLayers));
  EXPECT_NE(base->GetHostPointer(kNumStagingLayers - 1, 0), nullptr);
  EXPECT_EQ(base->GetHostPointer(kNumStagingLayers, 0), nullptr);

  ASSERT_GT(producer.NotifyForRead("req_send_staging_layers", /*uuid=*/304,
                                   /*block_ids=*/{0}),
            0);
  const StagingLayerManager& staging_manager =
      producer.staging_layer_manager(304);
  EXPECT_EQ(staging_manager.num_free(), kNumStagingLayers);

  std::thread pull_thread = producer.ServePullAsync(304);
  absl::Cleanup join_pull = [&pull_thread]() { pull_thread.join(); };

  // Layers 0..3 lease the four buffers in order and issue D2H into them;
  // layer 4 waits on the pull worker thread because all 4 buffers are leased.
  ASSERT_TRUE(
      producer.WaitForCopiesIssued(kNumStagingLayers, absl::Seconds(5)));
  for (size_t layer = 0; layer < kNumStagingLayers; ++layer) {
    EXPECT_THAT(staging_manager.Lookup(layer), Optional(layer));
    EXPECT_EQ(producer.copy_destination(layer), static_cast<int64_t>(layer));
  }
  EXPECT_EQ(staging_manager.num_free(), 0);
  EXPECT_EQ(producer.copies_issued(), kNumStagingLayers);

  // Completing D2H for layers 0..3 starts their H2H pushes (whose
  // GetBlockChunks calls resolve to their leased host layer buffers), while
  // keeping all 4 buffers leased until H2H completes.
  for (size_t layer = 0; layer < kNumStagingLayers; ++layer) {
    producer.FinishCopy(layer, absl::OkStatus());
  }
  ASSERT_TRUE(producer.WaitForH2hIssued(kNumStagingLayers, absl::Seconds(5)));
  EXPECT_EQ(staging_manager.num_free(), 0);
  EXPECT_EQ(producer.copies_issued(), kNumStagingLayers);
  for (size_t layer = 0; layer < kNumStagingLayers; ++layer) {
    EXPECT_NE(producer.h2h_chunk_ptr(layer), nullptr);
    EXPECT_GE(producer.h2h_chunk_ptr(layer), base->GetHostPointer(layer, 0));
    EXPECT_LT(producer.h2h_chunk_ptr(layer),
              base->GetHostPointer(layer, 0) + base->GetHostSize(layer, 0));
  }

  // Completing layer 1's H2H push frees buffer 1, which layer 4 acquires for
  // its D2H copy and subsequent H2H push.
  producer.FinishH2h(1, absl::OkStatus());
  ASSERT_TRUE(producer.WaitForCopiesIssued(5, absl::Seconds(5)));
  EXPECT_EQ(staging_manager.Lookup(1), std::nullopt);
  EXPECT_THAT(staging_manager.Lookup(4), Optional(1u));
  EXPECT_EQ(producer.copy_destination(4), 1);
  producer.FinishCopy(4, absl::OkStatus());
  ASSERT_TRUE(producer.WaitForH2hIssued(5, absl::Seconds(5)));
  EXPECT_GE(producer.h2h_chunk_ptr(4), base->GetHostPointer(1, 0));
  EXPECT_LT(producer.h2h_chunk_ptr(4),
            base->GetHostPointer(1, 0) + base->GetHostSize(1, 0));

  // Stream the remaining layers through the manager: each takes the buffer of
  // the H2H push completed just before it.
  for (size_t layer = 5; layer < kNumLayers; ++layer) {
    const size_t finished_layer = layer == 5 ? 0 : layer - 4;
    const size_t freed_slot = *staging_manager.Lookup(finished_layer);
    producer.FinishH2h(finished_layer, absl::OkStatus());
    ASSERT_TRUE(producer.WaitForCopiesIssued(layer + 1, absl::Seconds(5)));
    EXPECT_THAT(staging_manager.Lookup(layer), Optional(freed_slot));
    EXPECT_EQ(producer.copy_destination(layer),
              static_cast<int64_t>(freed_slot));
    producer.FinishCopy(layer, absl::OkStatus());
    ASSERT_TRUE(producer.WaitForH2hIssued(layer + 1, absl::Seconds(5)));
    EXPECT_GE(producer.h2h_chunk_ptr(layer),
              base->GetHostPointer(freed_slot, 0));
    EXPECT_LT(
        producer.h2h_chunk_ptr(layer),
        base->GetHostPointer(freed_slot, 0) + base->GetHostSize(freed_slot, 0));
  }

  for (size_t layer = kNumLayers - kNumStagingLayers; layer < kNumLayers;
       ++layer) {
    producer.FinishH2h(layer, absl::OkStatus());
  }
  ABSL_EXPECT_OK(producer.AwaitSendDone(304));
  EXPECT_EQ(staging_manager.num_free(), kNumStagingLayers);
  EXPECT_EQ(producer.free_slots(), kSlots);

  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), Contains("req_send_staging_layers"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
  EXPECT_FALSE(producer.has_send(304));
}

TEST(SendLifecycleTest,
     StagingLayerManagerReleasesAllResourcesAcrossFailureModes) {
  constexpr size_t kNumLayers = 16;
  constexpr size_t kNumStagingLayers = 4;
  ScopedEnv staging_layers("TPU_RAIDEN_NUM_STAGING_LAYERS", "4");
  TestManager producer(kNumLayers, /*timeout_s=*/10.0);
  producer.SetManualH2h(true);
  kv_cache::KVCacheManagerBase* base = producer.base();
  const size_t block_bytes = base->block_bytes(0);

  // 1. D2H failure while layer 4 is blocked in StagingLayerManager::Acquire:
  //    failing layer 0 shuts down the manager before releasing slot 0 so
  //    the pull worker thread wakes with cancellation instead of issuing a
  //    5th D2H copy, and the host staging allocation stays pinned until
  //    layers 1..3 finish.
  ASSERT_GT(producer.NotifyForRead("req_d2h_fail_with_waiter", /*uuid=*/401,
                                   /*block_ids=*/{0}),
            0);
  const StagingLayerManager& manager1 = producer.staging_layer_manager(401);
  // Calling GetBlockChunks before StartPush leased a buffer returns empty.
  EXPECT_THAT(base->GetBlockChunks(/*layer_idx=*/0, /*shard_idx=*/0,
                                   /*block_ids=*/{0}, block_bytes,
                                   /*uuid=*/401),
              IsEmpty());

  std::thread pull_thread1 = producer.ServePullAsync(401);
  ASSERT_TRUE(
      producer.WaitForCopiesIssued(kNumStagingLayers, absl::Seconds(5)));
  EXPECT_EQ(producer.copies_issued(), kNumStagingLayers);
  EXPECT_EQ(manager1.num_free(), 0);
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  producer.FinishCopy(0, absl::InternalError("D2H error on layer 0"));
  // Layer 4 is cancelled in AcquireStagingSlot and does not start a 5th D2H.
  // Layers 1..3 are still in flight, so the staging block remains pinned.
  pull_thread1.join();
  EXPECT_EQ(producer.copies_issued(), kNumStagingLayers);
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  for (size_t layer = 1; layer < kNumStagingLayers; ++layer) {
    producer.FinishCopy(layer, absl::OkStatus());
  }
  EXPECT_THAT(producer.AwaitSendDone(401),
              ::absl_testing::StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(producer.copies_issued(), kNumStagingLayers);
  EXPECT_EQ(producer.h2h_issued(), 0);
  EXPECT_EQ(manager1.num_free(), kNumStagingLayers);
  EXPECT_EQ(producer.free_slots(), kSlots);
  Reports reports1 = producer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(reports1), Contains("req_d2h_fail_with_waiter"));
  EXPECT_FALSE(producer.has_send(401));

  // 2. H2H failure while layer 4 is blocked in StagingLayerManager::Acquire and
  //    layers 1..3 H2H pushes are in flight: failing layer 0's H2H shuts down
  //    the manager, wakes the pull worker thread without issuing a 5th D2H
  //    copy, and keeps staging pinned until layers 1..3 H2H complete.
  ASSERT_GT(producer.NotifyForRead("req_h2h_fail_with_waiter", /*uuid=*/402,
                                   /*block_ids=*/{0}),
            0);
  const StagingLayerManager& manager2 = producer.staging_layer_manager(402);
  const size_t base_copies2 = producer.copies_issued();
  std::thread pull_thread2 = producer.ServePullAsync(402);
  ASSERT_TRUE(producer.WaitForCopiesIssued(base_copies2 + kNumStagingLayers,
                                           absl::Seconds(5)));
  EXPECT_EQ(producer.copies_issued(), base_copies2 + kNumStagingLayers);
  for (size_t i = 0; i < kNumStagingLayers; ++i) {
    producer.FinishCopy(base_copies2 + i, absl::OkStatus());
  }
  ASSERT_TRUE(producer.WaitForH2hIssued(kNumStagingLayers, absl::Seconds(5)));
  EXPECT_EQ(manager2.num_free(), 0);
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  producer.FinishH2h(0, absl::UnavailableError("H2H error on layer 0"));
  pull_thread2.join();
  EXPECT_EQ(producer.copies_issued(), base_copies2 + kNumStagingLayers);
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  for (size_t i = 1; i < kNumStagingLayers; ++i) {
    producer.FinishH2h(i, absl::OkStatus());
  }
  EXPECT_THAT(producer.AwaitSendDone(402),
              ::absl_testing::StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_EQ(producer.copies_issued(), base_copies2 + kNumStagingLayers);
  EXPECT_EQ(manager2.num_free(), kNumStagingLayers);
  EXPECT_EQ(producer.free_slots(), kSlots);
  Reports reports2 = producer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(reports2), Contains("req_h2h_fail_with_waiter"));
  EXPECT_FALSE(producer.has_send(402));

  // 3. StagingLayerManager::Acquire deadline exceeded while all 4 buffers are
  //    held by in-flight D2H copies: the timeout fails the session, keeps
  //    staging pinned until the in-flight copies complete, and then releases
  //    all slots.
  ASSERT_GT(
      producer.NotifyForRead(
          "req_send_staging_timeout", /*uuid=*/403, /*block_ids=*/{0},
          std::chrono::steady_clock::now() + std::chrono::milliseconds(80)),
      0);
  const StagingLayerManager& manager3 = producer.staging_layer_manager(403);
  const size_t base_copies3 = producer.copies_issued();
  std::thread pull_thread3 = producer.ServePullAsync(403);
  ASSERT_TRUE(producer.WaitForCopiesIssued(base_copies3 + kNumStagingLayers,
                                           absl::Seconds(5)));
  EXPECT_EQ(producer.copies_issued(), base_copies3 + kNumStagingLayers);
  ASSERT_TRUE(producer.WaitForSendDraining(403, absl::Seconds(5)));
  pull_thread3.join();
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  for (size_t i = 0; i < kNumStagingLayers; ++i) {
    producer.FinishCopy(base_copies3 + i, absl::OkStatus());
  }
  EXPECT_THAT(producer.AwaitSendDone(403),
              ::absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_EQ(manager3.num_free(), kNumStagingLayers);
  EXPECT_EQ(producer.free_slots(), kSlots);
  CompleteReadResult completions3 = producer.CompleteReadWithDetails();
  EXPECT_THAT(completions3.failed, Contains("req_send_staging_timeout"));
  // The send timed out waiting for a host layer buffer, not in a copy.
  EXPECT_EQ(completions3.details["req_send_staging_timeout"].phase, "staging");
  EXPECT_FALSE(producer.has_send(403));

  // 4. Cancellation (a consumer cancel or the manager's shutdown) while layer
  //    4 is blocked in StagingLayerManager::Acquire: the pull worker wakes
  //    without issuing a 5th D2H copy, and staging stays pinned until the
  //    issued copies complete.
  ASSERT_GT(producer.NotifyForRead("req_send_cancel_with_waiter", /*uuid=*/404,
                                   /*block_ids=*/{0}),
            0);
  const StagingLayerManager& manager4 = producer.staging_layer_manager(404);
  const size_t base_copies4 = producer.copies_issued();
  std::thread pull_thread4 = producer.ServePullAsync(404);
  ASSERT_TRUE(producer.WaitForCopiesIssued(base_copies4 + kNumStagingLayers,
                                           absl::Seconds(5)));
  EXPECT_EQ(manager4.num_free(), 0);

  producer.CancelSend(404);
  pull_thread4.join();
  EXPECT_EQ(producer.copies_issued(), base_copies4 + kNumStagingLayers);
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  for (size_t i = 0; i < kNumStagingLayers; ++i) {
    producer.FinishCopy(base_copies4 + i, absl::OkStatus());
  }
  EXPECT_THAT(producer.AwaitSendDone(404),
              ::absl_testing::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(manager4.num_free(), kNumStagingLayers);
  EXPECT_EQ(producer.free_slots(), kSlots);
  EXPECT_FALSE(producer.has_send(404));
}

TEST(SendDrainTest,
     DynamicProducerSubPullsEnforceOneChunkInH2hAndInOrderH2hEntry) {
  TestManager producer(/*num_layers=*/1);
  producer.SetManualH2h(true);
  producer.SetStagingWindowConfig(1, 2);
  ASSERT_GT(producer.NotifyForRead("req_h2h_order", 1000, {10, 20}), 0);
  // Sub-plan 1 arrives before sub-plan 0.
  ASSERT_TRUE(producer.ProcessPull(SubPlanUuid(1000, 1), {20}).ok());
  ASSERT_TRUE(producer.ProcessPull(1000, {10}).ok());
  ASSERT_TRUE(producer.WaitForCopies(2));
  EXPECT_EQ(producer.copies_issued(), 2);
  producer.FinishCopy(0, absl::OkStatus());
  producer.FinishCopy(1, absl::OkStatus());
  ASSERT_TRUE(producer.WaitForH2hCalls(1));
  EXPECT_EQ(producer.h2h_calls(), 1);
  EXPECT_EQ(producer.active_h2h_count(), 1);
  producer.FinishH2h(0, absl::OkStatus());
  ASSERT_TRUE(producer.WaitForH2hCalls(2));
  EXPECT_EQ(producer.h2h_calls(), 2);
  EXPECT_EQ(producer.active_h2h_count(), 1);
  Reports during = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_THROW(producer.NotifyForRead("req_h2h_order", 1000, {10, 20}),
               std::runtime_error);
  producer.FinishH2h(1, absl::OkStatus());
  producer.WaitForPendingWork().IgnoreError();
  Reports after = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), ElementsAre("req_h2h_order"));
  EXPECT_THAT(FailedRecving(after), IsEmpty());
  EXPECT_EQ(producer.base()->host_block_manager()->num_allocated_blocks(), 0u);
}

TEST(SendDrainTest,
     GatedSendSessionCancellationTimeoutAndShutdownDoNotDeadlock) {
  {
    TestManager producer(/*num_layers=*/1);
    producer.SetManualH2h(true);
    ASSERT_GT(producer.NotifyForRead("req_fail_abort", 2000, {10, 20}), 0);
    ASSERT_TRUE(producer.ProcessPull(SubPlanUuid(2000, 1), {20}).ok());
    ASSERT_TRUE(producer.WaitForCopies(1));
    ASSERT_TRUE(producer.ProcessPull(2000, {10}).ok());
    ASSERT_TRUE(producer.WaitForCopies(2));
    producer.FinishCopy(0, absl::OkStatus());
    producer.FinishCopy(1, absl::InternalError("d2h failed"));
    Reports reports = producer.CompleteReadRaw();
    EXPECT_THAT(DoneSending(reports), IsEmpty());
    EXPECT_THAT(FailedRecving(reports), ElementsAre("req_fail_abort"));
    EXPECT_EQ(producer.free_slots(), kSlots);
  }
  {
    auto producer = std::make_unique<TestManager>(/*num_layers=*/1);
    producer->SetManualH2h(true);
    ASSERT_GT(producer->NotifyForRead("req_shutdown", 3000, {10, 20}), 0);
    ASSERT_TRUE(producer->ProcessPull(SubPlanUuid(3000, 1), {20}).ok());
    ASSERT_TRUE(producer->WaitForCopies(1));
    ASSERT_TRUE(producer->ProcessPull(3000, {10}).ok());
    ASSERT_TRUE(producer->WaitForCopies(2));
    producer->FinishCopy(0, absl::OkStatus());
    producer->FinishCopy(1, absl::OkStatus());
    ASSERT_TRUE(producer->WaitForH2hCalls(1));
    producer.reset();
  }
  {
    TestManager producer(/*num_layers=*/1);
    producer.SetManualH2h(true);
    producer.SetStagingWindowConfig(1, 2);
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    ASSERT_GT(producer.NotifyForRead("req_timeout", 4000, {10, 20}, deadline),
              0);
    ASSERT_TRUE(producer.ProcessPull(SubPlanUuid(4000, 1), {20}).ok());
    ASSERT_TRUE(producer.WaitForCopies(1));
    producer.FinishCopy(0, absl::OkStatus());
    absl::SleepFor(absl::Milliseconds(60));
    CompleteReadResult result = producer.CompleteReadWithDetails();
    EXPECT_THAT(result.failed, ElementsAre("req_timeout"));
    EXPECT_EQ(producer.base()->host_block_manager()->num_allocated_blocks(),
              0u);
  }
}

TEST(SendLifecycleTest,
     PendingAckOnSubPlanAcknowledgesTransferInNotifyForRead) {
  TestManager producer(/*num_layers=*/1);
  EXPECT_TRUE(producer.HandleAck(SubPlanUuid(4000, 1)).ok());
  EXPECT_EQ(producer.NotifyForRead("req_pending_sub_ack", 4000, {10, 20}), 0);
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), ElementsAre("req_pending_sub_ack"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
  EXPECT_EQ(producer.free_slots(), kSlots);
}

struct SimulatedBase : public kv_cache::KVCacheManagerBase {
  SimulatedBase(size_t l, size_t sz, int b)
      : kv_cache::KVCacheManagerBase(l, 1, sz, std::nullopt, b, 1, nullptr),
        device_data(l, std::vector<uint8_t>(64 * sz)) {
    for (size_t i = 0; i < l; ++i) {
      for (size_t j = 0; j < 64 * sz; ++j) device_data[i][j] = i * 37 + j * 7;
    }
  }
  void SetHoldH2d(bool hold) {
    absl::MutexLock l(h2d_mu);
    hold_h2d = hold;
  }
  void ReleaseH2d(absl::Status s = absl::OkStatus(), bool all = true) {
    absl::MutexLock l(h2d_mu);
    while (!held_promises.empty()) {
      held_promises.front().Set(s);
      held_promises.erase(held_promises.begin());
      if (!all) break;
    }
  }
  bool WaitForHeldH2d(size_t n) {
    absl::MutexLock l(h2d_mu);
    auto c = [&] {
      h2d_mu.AssertReaderHeld();
      return held_promises.size() >= n;
    };
    return h2d_mu.AwaitWithTimeout(absl::Condition(&c), absl::Seconds(5));
  }
  using V = const std::vector<int64_t>&;
  using O = std::optional<size_t>;
  using H = raiden::BufferHolders;
  using F = raiden::PjRtCopyFuture;
  absl::StatusOr<F> D2hSyncDispatch(V src, V dst, V sz, std::optional<int64_t>,
                                    O l, O host_l, O s) override {
    return Copy(false, l.value_or(0), host_l.value_or(l.value_or(0)),
                s.value_or(0), src, dst, sz);
  }
  absl::StatusOr<F> H2dSyncDispatch(V src, V dst, V sz, std::optional<int64_t>,
                                    O l, O host_l, O s) override {
    return Copy(true, l.value_or(0), host_l.value_or(l.value_or(0)),
                s.value_or(0), src, dst, sz);
  }
  F Copy(bool h2d, size_t l, size_t host_l, size_t s, V src, V dst, V sz) {
    if (auto* m = host_block_manager()) {
      size_t cur = m->num_allocated_blocks(), prev = peak.load();
      while (cur > prev && !peak.compare_exchange_weak(prev, cur)) {
      }
    }
    uint8_t *host = GetHostPointer(host_l, s), *dev = device_data[l].data();
    for (size_t i = 0; i < src.size(); ++i) {
      std::memcpy((h2d ? dev : host) + dst[i] * slice_byte_size_,
                  (h2d ? host : dev) + src[i] * slice_byte_size_,
                  sz[i] * slice_byte_size_);
    }
    absl::MutexLock lock(h2d_mu);
    if (!h2d || !hold_h2d) return F(H{});
    auto [p, f] = xla::MakePromise<>();
    held_promises.push_back(std::move(p));
    return F(std::move(f), H{});
  }
  std::vector<std::vector<uint8_t>> device_data;
  absl::Mutex h2d_mu;
  bool hold_h2d = false;
  std::vector<xla::Promise<>> held_promises;
  std::atomic<size_t> peak = 0;
};

class SimulatedDeviceManager : public KVCacheManagerWithTransfer {
 public:
  SimulatedDeviceManager(size_t l, size_t sz, int b)
      : KVCacheManagerWithTransfer(std::make_unique<SimulatedBase>(l, sz, b), 0,
                                   0, 8, 2, 10.0) {}
  ~SimulatedDeviceManager() override {
    if (sim()) sim()->ReleaseH2d();
  }
  SimulatedBase* sim() { return static_cast<SimulatedBase*>(base()); }
  std::string endpoint() const {
    return absl::StrCat("127.0.0.1:", local_control_port());
  }
  size_t allocated() const {
    return base()->host_block_manager()->num_allocated_blocks();
  }
};

void WaitUntil(std::function<bool()> check) {
  for (const absl::Time end = absl::Now() + absl::Seconds(5);
       !check() && absl::Now() < end;) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(check());
}

TEST(BoundedWindowTest,
     BoundedWindowStartReadTransfers4xHostCapacityWithDoubleBuffering) {
  SimulatedDeviceManager p(2, 128, 8), c(2, 128, 4);
  p.SetStagingWindowConfig(2, 2);
  c.SetStagingWindowConfig(2, 2);
  const std::vector<int64_t> b = {0, 1, 2, 3, 4, 5, 6, 7};
  p.NotifyForRead("req_4x", 10001, b);
  c.StartRead("req_4x", 10001, p.endpoint(), b, b);

  std::vector<std::string> c_recv, p_sent;
  WaitUntil([&] {
    auto [s, r, f] = c.CompleteReadRaw();
    EXPECT_THAT(f, IsEmpty());
    c_recv.insert(c_recv.end(), r.begin(), r.end());
    auto [ps, pr, pf] = p.CompleteReadRaw();
    EXPECT_THAT(pf, IsEmpty());
    p_sent.insert(p_sent.end(), ps.begin(), ps.end());
    return !c_recv.empty() && !p_sent.empty();
  });

  EXPECT_THAT(c_recv, ElementsAre("req_4x"));
  EXPECT_THAT(p_sent, ElementsAre("req_4x"));
  EXPECT_EQ(c.sim()->device_data, p.sim()->device_data);
  EXPECT_LE(c.sim()->peak.load(), 4u);
  EXPECT_GE(c.sim()->peak.load(), 2u);
  EXPECT_EQ(c.allocated(), 0u);
  EXPECT_EQ(p.allocated(), 0u);

  const std::vector<int64_t> b1 = {0, 1};
  p.NotifyForRead("req_1x", 10002, b1);
  c.StartRead("req_1x", 10002, p.endpoint(), b1, b1);
  ABSL_ASSERT_OK(c.WaitForPendingWork());
  CompleteReadResult r1 = c.CompleteReadWithDetails();
  EXPECT_THAT(r1.received, ElementsAre("req_1x"));
  EXPECT_THAT(r1.failed, IsEmpty());
  p.WaitForPendingWork().IgnoreError();
  EXPECT_EQ(c.allocated(), 0u);
  EXPECT_EQ(p.allocated(), 0u);
}

TEST(
    BoundedWindowTest,
    BoundedWindowStartReadQueuesOnStagingContentionAndResumesWithoutBlockingMu) {
  SimulatedDeviceManager p(2, 128, 4), c(2, 128, 2);
  p.SetStagingWindowConfig(1, 2);
  c.SetStagingWindowConfig(1, 2);
  c.sim()->SetHoldH2d(true);
  p.NotifyForRead("req_contention_1", 20001, {0, 1});
  c.StartRead("req_contention_1", 20001, p.endpoint(), {0, 1}, {0, 1});

  ASSERT_TRUE(c.sim()->WaitForHeldH2d(2));
  EXPECT_EQ(c.allocated(), 2u);

  p.NotifyForRead("req_contention_2", 20002, {2, 3});
  const absl::Time t0 = absl::Now();
  c.StartRead("req_contention_2", 20002, p.endpoint(), {2, 3}, {2, 3});
  EXPECT_LT(absl::Now() - t0, absl::Milliseconds(500));
  EXPECT_FALSE(c.IsShuttingDown());
  EXPECT_EQ(c.allocated(), 2u);

  c.sim()->SetHoldH2d(false);
  c.sim()->ReleaseH2d();

  std::vector<std::string> c_recv, p_sent;
  WaitUntil([&] {
    auto [cs, cr, cf] = c.CompleteReadRaw();
    EXPECT_THAT(cf, IsEmpty());
    c_recv.insert(c_recv.end(), cr.begin(), cr.end());
    auto [ps, pr, pf] = p.CompleteReadRaw();
    EXPECT_THAT(pf, IsEmpty());
    p_sent.insert(p_sent.end(), ps.begin(), ps.end());
    return c_recv.size() >= 2 && p_sent.size() >= 2;
  });

  EXPECT_THAT(c_recv,
              UnorderedElementsAre("req_contention_1", "req_contention_2"));
  EXPECT_THAT(p_sent,
              UnorderedElementsAre("req_contention_1", "req_contention_2"));
  EXPECT_EQ(c.sim()->device_data, p.sim()->device_data);
  EXPECT_EQ(c.allocated(), 0u);
  EXPECT_EQ(p.allocated(), 0u);
}

TEST(BoundedWindowTest,
     SubPlanChainAbortsUndispatchedAndCancelsInFlightSiblingOnTimeout) {
  SimulatedDeviceManager p(1, 128, 4), c(1, 128, 4);
  p.SetStagingWindowConfig(1, 2);
  c.SetStagingWindowConfig(1, 2);
  c.sim()->SetHoldH2d(true);
  p.NotifyForRead("req_chain_abort", 30001, {0, 1, 2, 3});
  c.StartRead("req_chain_abort", 30001, p.endpoint(), {0, 1, 2, 3},
              {0, 1, 2, 3});

  ASSERT_TRUE(c.sim()->WaitForHeldH2d(2));

  // Fail sub-plan 0's H2D copy to trigger RetireSettledRecvSessionLocked ->
  // AbortRecvChainLocked (aborting sub-plans 2, 3) and
  // CancelAndUnregisterRecvSiblings (cancelling sub-plan 1).
  c.sim()->ReleaseH2d(absl::InternalError("injected h2d error"), /*all=*/false);
  c.sim()->SetHoldH2d(false);
  c.sim()->ReleaseH2d();

  std::vector<std::string> failed;
  WaitUntil([&] {
    auto [s, r, f] = c.CompleteReadRaw();
    EXPECT_THAT(r, IsEmpty());
    failed.insert(failed.end(), f.begin(), f.end());
    return !failed.empty();
  });
  EXPECT_THAT(failed, ElementsAre("req_chain_abort"));

  auto after = c.CompleteReadRaw();
  EXPECT_THAT(std::get<1>(after), IsEmpty());
  EXPECT_THAT(std::get<2>(after), IsEmpty());

  p.WaitForPendingWork().IgnoreError();
  EXPECT_EQ(c.allocated(), 0u);
  EXPECT_EQ(p.allocated(), 0u);
}

TEST(BoundedWindowTest,
     ThreeStagePipelineOverlapsD2hH2hH2dWithStrictlyOneChunkInH2h) {
  SimulatedDeviceManager p(2, 128, 4), c(2, 128, 4);
  p.SetStagingWindowConfig(2, 2);
  c.SetStagingWindowConfig(2, 2);
  c.sim()->SetHoldH2d(true);
  const std::vector<int64_t> b = {0, 1, 2, 3, 4, 5, 6, 7};
  p.NotifyForRead("req_overlap_pipe", 50001, b);
  c.StartRead("req_overlap_pipe", 50001, p.endpoint(), b, b);

  // Sub-plan 0 enters H2D (Stage 3) and is held. Sub-plan 1 completes D2H
  // (Stage 1) + cross-host H2H (Stage 2) and also enters Consumer H2D (Stage
  // 3). Both sub-plans across 2 layers produce 4 held H2D promises.
  ASSERT_TRUE(c.sim()->WaitForHeldH2d(4));
  EXPECT_EQ(c.allocated(), 4u);
  EXPECT_LE(p.sim()->peak.load(), 4u);

  // Releasing both layers of sub-plan 0 frees D-slot 0, so sub-plan 2
  // dispatches and enters H2D across both layers.
  c.sim()->ReleaseH2d(absl::OkStatus(), /*all=*/false);
  c.sim()->ReleaseH2d(absl::OkStatus(), /*all=*/false);

  // With sub-plan 1 still held and sub-plan 2 entering H2D across both layers,
  // held promises reaches 4 again.
  ASSERT_TRUE(c.sim()->WaitForHeldH2d(4));

  // Release both layers of sub-plan 1 to allow sub-plan 3 to progress into
  // D-slot 1.
  c.sim()->ReleaseH2d(absl::OkStatus(), /*all=*/false);
  c.sim()->ReleaseH2d(absl::OkStatus(), /*all=*/false);

  // With sub-plan 2 still held and sub-plan 3 entering H2D across both layers,
  // held promises reaches 4 again.
  ASSERT_TRUE(c.sim()->WaitForHeldH2d(4));

  // Release remaining copies and unhold future H2D dispatches.
  c.sim()->SetHoldH2d(false);
  c.sim()->ReleaseH2d();

  std::vector<std::string> c_recv, p_sent;
  WaitUntil([&] {
    auto [s, r, f] = c.CompleteReadRaw();
    EXPECT_THAT(f, IsEmpty());
    c_recv.insert(c_recv.end(), r.begin(), r.end());
    auto [ps, pr, pf] = p.CompleteReadRaw();
    EXPECT_THAT(pf, IsEmpty());
    p_sent.insert(p_sent.end(), ps.begin(), ps.end());
    return !c_recv.empty() && !p_sent.empty();
  });

  EXPECT_THAT(c_recv, ElementsAre("req_overlap_pipe"));
  EXPECT_THAT(p_sent, ElementsAre("req_overlap_pipe"));
  EXPECT_EQ(c.sim()->device_data, p.sim()->device_data);
  EXPECT_LE(p.sim()->peak.load(), 4u);
  EXPECT_LE(c.sim()->peak.load(), 4u);
  EXPECT_EQ(c.allocated(), 0u);
  EXPECT_EQ(p.allocated(), 0u);
}

TEST(BoundedWindowTest,
     MidChainErrorAndDeadlineExpiryAbortSiblingsAndCleanUpMaps) {
  SimulatedDeviceManager p(2, 128, 4), c(2, 128, 4);
  p.SetStagingWindowConfig(2, 2);
  c.SetStagingWindowConfig(2, 2);
  c.sim()->SetHoldH2d(true);
  const std::vector<int64_t> b = {0, 1, 2, 3, 4, 5, 6, 7};
  const auto p_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  p.NotifyForRead("req_mid_abort", 60001, b, p_deadline);
  c.StartRead("req_mid_abort", 60001, p.endpoint(), b, b);

  // Wait for initial sub-plans (0 and 1) to enter Consumer H2D across both
  // layers (4 promises total) so sub-plan 1 is an active in-flight sibling.
  ASSERT_TRUE(c.sim()->WaitForHeldH2d(4));

  // Inject a failure on sub-plan 0 to abort active and queued siblings.
  c.sim()->ReleaseH2d(absl::InternalError("injected mid-chain failure"),
                      /*all=*/false);
  c.sim()->SetHoldH2d(false);
  c.sim()->ReleaseH2d();

  std::vector<std::string> failed;
  WaitUntil([&] {
    auto [s, r, f] = c.CompleteReadRaw();
    EXPECT_THAT(r, IsEmpty());
    failed.insert(failed.end(), f.begin(), f.end());
    return !failed.empty();
  });
  EXPECT_THAT(failed, ElementsAre("req_mid_abort"));

  // Sibling completion must not emit duplicate failure notifications.
  auto after = c.CompleteReadRaw();
  EXPECT_THAT(std::get<1>(after), IsEmpty());
  EXPECT_THAT(std::get<2>(after), IsEmpty());

  p.WaitForPendingWork().IgnoreError();
  (void)p.CompleteReadRaw();
  EXPECT_EQ(c.allocated(), 0u);
  EXPECT_EQ(p.allocated(), 0u);

  // Exercise deadline expiry on a bounded-window transfer while H2D is held.
  c.sim()->SetHoldH2d(true);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
  p.NotifyForRead("req_deadline_abort", 60002, b, deadline);
  c.StartRead("req_deadline_abort", 60002, p.endpoint(), b, b,
              /*parallelism=*/1, /*local_host_block_ids=*/std::nullopt,
              deadline);
  ASSERT_TRUE(c.sim()->WaitForHeldH2d(4));
  absl::SleepFor(absl::Milliseconds(100));
  c.sim()->SetHoldH2d(false);
  c.sim()->ReleaseH2d();

  std::vector<std::string> deadline_failed;
  WaitUntil([&] {
    auto [s, r, f] = c.CompleteReadRaw();
    EXPECT_THAT(r, IsEmpty());
    deadline_failed.insert(deadline_failed.end(), f.begin(), f.end());
    return !deadline_failed.empty();
  });
  EXPECT_THAT(deadline_failed, ElementsAre("req_deadline_abort"));

  p.WaitForPendingWork().IgnoreError();
  (void)p.CompleteReadRaw();
  EXPECT_EQ(c.allocated(), 0u);
  EXPECT_EQ(p.allocated(), 0u);

  // Verify internal registration/chain maps are completely cleaned up by
  // immediately re-running a successful transfer with the same uuid and req_id.
  p.NotifyForRead("req_mid_abort", 60001, b);
  c.StartRead("req_mid_abort", 60001, p.endpoint(), b, b);

  std::vector<std::string> c_recv, p_sent;
  WaitUntil([&] {
    auto [s, r, f] = c.CompleteReadRaw();
    EXPECT_THAT(f, IsEmpty());
    c_recv.insert(c_recv.end(), r.begin(), r.end());
    auto [ps, pr, pf] = p.CompleteReadRaw();
    EXPECT_THAT(pf, IsEmpty());
    p_sent.insert(p_sent.end(), ps.begin(), ps.end());
    return !c_recv.empty() && !p_sent.empty();
  });

  EXPECT_THAT(c_recv, ElementsAre("req_mid_abort"));
  EXPECT_THAT(p_sent, ElementsAre("req_mid_abort"));
  EXPECT_EQ(c.sim()->device_data, p.sim()->device_data);
  EXPECT_EQ(c.allocated(), 0u);
  EXPECT_EQ(p.allocated(), 0u);
}

TEST(BoundedWindowTest,
     BoundedWindowTransferEmitsSubTransferPerfettoStageSlices) {
  std::string temp_dir = testing::TempDir();
  std::string trace_file =
      absl::StrCat(temp_dir, "/raiden_subtransfer_test_",
                   absl::ToUnixMicros(absl::Now()), ".pftrace");

  StopPerfettoTraceToFile();
  ASSERT_TRUE(StartPerfettoTraceToFile(trace_file));
  SetTraceBackendForTesting(TraceBackend::kPerfetto);

  {
    SimulatedDeviceManager p(2, 128, 4), c(2, 128, 4);
    p.SetStagingWindowConfig(2, 2);
    c.SetStagingWindowConfig(2, 2);
    const std::vector<int64_t> b = {0, 1, 2, 3};
    p.NotifyForRead("req_trace_stages", 70001, b);
    c.StartRead("req_trace_stages", 70001, p.endpoint(), b, b);

    std::vector<std::string> c_recv, p_sent;
    WaitUntil([&] {
      auto [s, r, f] = c.CompleteReadRaw();
      EXPECT_THAT(f, IsEmpty());
      c_recv.insert(c_recv.end(), r.begin(), r.end());
      auto [ps, pr, pf] = p.CompleteReadRaw();
      EXPECT_THAT(pf, IsEmpty());
      p_sent.insert(p_sent.end(), ps.begin(), ps.end());
      return !c_recv.empty() && !p_sent.empty();
    });

    EXPECT_THAT(c_recv, ElementsAre("req_trace_stages"));
    EXPECT_THAT(p_sent, ElementsAre("req_trace_stages"));
    EXPECT_EQ(c.sim()->device_data, p.sim()->device_data);
    EXPECT_EQ(c.allocated(), 0u);
    EXPECT_EQ(p.allocated(), 0u);
  }

  SetTraceBackendForTesting(std::nullopt);
  ASSERT_TRUE(StopPerfettoTraceToFile());

  std::ifstream in(trace_file, std::ios::in | std::ios::binary);
  ASSERT_TRUE(in.is_open());
  std::string file_content((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
  in.close();

  EXPECT_FALSE(file_content.empty());
  absl::string_view view(file_content);
  EXPECT_TRUE(absl::StrContains(view, "SubTransferD2h"));
  EXPECT_TRUE(absl::StrContains(view, "SubTransferH2hGateWait"));
  EXPECT_TRUE(absl::StrContains(view, "SubTransferH2hPush"));
  EXPECT_TRUE(absl::StrContains(view, "SubTransferH2d"));
  EXPECT_TRUE(absl::StrContains(view, "sub_plan_index="));
}

}  // namespace
}  // namespace tpu_raiden
