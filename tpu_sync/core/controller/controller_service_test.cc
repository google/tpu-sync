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

#include "tpu_sync/core/controller/controller_service.h"

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/proto/controller_service.grpc.pb.h"
#include "tpu_sync/proto/controller_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {

namespace {

// The unqualified EXPECT_OK/ASSERT_OK spellings are gated behind
// ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS. Use the explicitly
// qualified macros.
using ::testing::ElementsAre;

class RaidenControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = CreateTestControllerServer();
  }

  std::unique_ptr<TestControllerServer> test_server_;
};

TEST_F(RaidenControllerTest, RegisterWorkerSuccessfully) {
  std::string transfer_addr = "10.0.0.1:8000";
  absl::Status status = test_server_->client->RegisterWorker(
      "worker_0", "10.0.0.1:9000",
      {::tpu_raiden::RaidenTransferEndpoint{transfer_addr, {}}});
  ABSL_EXPECT_OK(status);

  auto workers =
      test_server_->service->worker_registry()->GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 1);
  EXPECT_EQ(workers[0].worker_id, "worker_0");
  EXPECT_EQ(workers[0].raiden_worker_endpoint, "10.0.0.1:9000");
  ASSERT_EQ(workers[0].raiden_transfer_endpoints.size(), 1);
  EXPECT_EQ(workers[0].raiden_transfer_endpoints[0].endpoint, transfer_addr);
}

TEST_F(RaidenControllerTest, RegisterWorkerAliasSnakeCase) {
  std::string transfer_addr = "10.0.0.2:8000";
  absl::Status status = test_server_->client->register_worker(
      "worker_1", "10.0.0.2:9000",
      {::tpu_raiden::RaidenTransferEndpoint{transfer_addr, {}}});
  ABSL_EXPECT_OK(status);

  auto worker_or =
      test_server_->service->worker_registry()->GetWorker("worker_1");
  ABSL_ASSERT_OK(worker_or);
  EXPECT_EQ(worker_or->worker_id, "worker_1");
  EXPECT_EQ(worker_or->raiden_worker_endpoint, "10.0.0.2:9000");
  ASSERT_EQ(worker_or->raiden_transfer_endpoints.size(), 1);
  EXPECT_EQ(worker_or->raiden_transfer_endpoints[0].endpoint, transfer_addr);
}

TEST_F(RaidenControllerTest, ConstructWithEndpointString) {
  RaidenControllerClient client(test_server_->server_address);
  absl::Status status = client.RegisterWorker(
      "worker_endpoint_ctor", "10.0.0.1:9000",
      {::tpu_raiden::RaidenTransferEndpoint{"10.0.0.1:8000", {}}});
  ABSL_EXPECT_OK(status);

  auto workers =
      test_server_->service->worker_registry()->GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 1);
  EXPECT_EQ(workers[0].worker_id, "worker_endpoint_ctor");
}

TEST_F(RaidenControllerTest, RegisterWorkerEmptyIdFails) {
  absl::Status status = test_server_->client->RegisterWorker(
      "", "10.0.0.1:9000",
      {::tpu_raiden::RaidenTransferEndpoint{"10.0.0.1:8000", {}}});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}

TEST_F(RaidenControllerTest, RegisterWorkerNoAddressesFails) {
  absl::Status status =
      test_server_->client->RegisterWorker("worker_no_addrs", "", {});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}


// ===========================================================================
// Read-lease protocol (source side)
// ===========================================================================

// A stand-in for the source KVCacheStore: records pin counts per hash so tests
// can assert the all-or-nothing property and the never-underflow property
// directly, and lets a test declare which hashes are missing or not
// host-resident so the verify hook's error codes are exercised for real.
class FakePinStore {
 public:
  void SetHostResident(const std::vector<std::string>& hashes) {
    absl::MutexLock lock(&mu_);
    for (const auto& h : hashes) host_resident_.insert(h);
  }
  void SetPresentButNotHostResident(const std::string& hash) {
    absl::MutexLock lock(&mu_);
    present_wrong_status_.insert(hash);
  }
  // ids are deterministic so a test can assert the response carried the
  // SOURCE's ids rather than echoing the destination's advisory ones.
  int32_t IdFor(const std::string& hash) const {
    return static_cast<int32_t>(1000 + hash.size() * 10 + hash.back());
  }
  int PinCount(const std::string& hash) const {
    absl::MutexLock lock(&mu_);
    auto it = pins_.find(hash);
    return it == pins_.end() ? 0 : it->second;
  }
  int total_unpin_calls() const {
    absl::MutexLock lock(&mu_);
    return total_unpin_calls_;
  }

  absl::StatusOr<std::vector<int32_t>> ValidateAndPin(
      absl::Span<const std::string> hashes) {
    absl::MutexLock lock(&mu_);
    // All-or-nothing: validate everything BEFORE pinning anything.
    for (const auto& h : hashes) {
      if (present_wrong_status_.contains(h)) {
        return absl::FailedPreconditionError("not host resident: " + h);
      }
      if (!host_resident_.contains(h)) {
        return absl::NotFoundError("no such hash: " + h);
      }
    }
    std::vector<int32_t> ids;
    ids.reserve(hashes.size());
    for (const auto& h : hashes) {
      pins_[h]++;
      ids.push_back(IdFor(h));
    }
    return ids;
  }

  void Unpin(absl::Span<const std::string> hashes) {
    absl::MutexLock lock(&mu_);
    total_unpin_calls_++;
    for (const auto& h : hashes) {
      auto it = pins_.find(h);
      // A pin count that would go negative is the bug this whole test suite
      // exists to catch, so make it loud rather than silently clamping.
      ASSERT_TRUE(it != pins_.end() && it->second > 0)
          << "pin underflow on hash " << h;
      it->second--;
    }
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_set<std::string> host_resident_;
  absl::flat_hash_set<std::string> present_wrong_status_;
  absl::flat_hash_map<std::string, int> pins_;
  int total_unpin_calls_ = 0;
};

class ReadLeaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = CreateTestControllerServer();
    store_ = std::make_unique<FakePinStore>();
    store_->SetHostResident({"hash_a", "hash_b", "hash_c"});
    test_server_->service->SetReadRemoteHooks(
        [this](absl::Span<const std::string> h) {
          return store_->ValidateAndPin(h);
        },
        [this](absl::Span<const std::string> h) { store_->Unpin(h); });
    stub_ = ::tpu_sync::proto::RaidenControllerService::NewStub(
        test_server_->channel);
  }

  void TearDown() override {
    // Hooks must be detached before the store dies -- the sweeper and the
    // destructor both invoke the unpin hook. (Null-safe: a test may already
    // have torn the service down itself.)
    if (test_server_ && test_server_->service) {
      test_server_->service->ClearReadRemoteHooks();
    }
  }

  grpc::Status Acquire(const std::vector<std::string>& hashes,
                       ::tpu_sync::proto::AcquireReadLeaseResponse* resp,
                       int64_t ttl_ms = 0,
                       const std::string& requester = "peer_a") {
    ::tpu_sync::proto::AcquireReadLeaseRequest req;
    for (const auto& h : hashes) req.add_block_hashes(h);
    req.set_ttl_ms(ttl_ms);
    req.mutable_requester_raiden_id()->set_job_name(requester);
    grpc::ClientContext ctx;
    return stub_->AcquireReadLease(&ctx, req, resp);
  }

  ::tpu_sync::proto::LeaseVerdict Release(uint64_t lease_id) {
    ::tpu_sync::proto::ReleaseReadLeaseRequest req;
    req.set_lease_id(lease_id);
    ::tpu_sync::proto::ReleaseReadLeaseResponse resp;
    grpc::ClientContext ctx;
    EXPECT_TRUE(stub_->ReleaseReadLease(&ctx, req, &resp).ok());
    return resp.verdict();
  }

  ::tpu_sync::proto::RenewReadLeaseResponse Renew(uint64_t lease_id,
                                                  int64_t extend_ms = 0) {
    ::tpu_sync::proto::RenewReadLeaseRequest req;
    req.set_lease_id(lease_id);
    req.set_extend_ms(extend_ms);
    ::tpu_sync::proto::RenewReadLeaseResponse resp;
    grpc::ClientContext ctx;
    EXPECT_TRUE(stub_->RenewReadLease(&ctx, req, &resp).ok());
    return resp;
  }

  std::unique_ptr<TestControllerServer> test_server_;
  std::unique_ptr<FakePinStore> store_;
  std::unique_ptr<::tpu_sync::proto::RaidenControllerService::Stub> stub_;
};

TEST_F(ReadLeaseTest, AcquireValidatesAndPinsAllOrNothing) {
  ::tpu_sync::proto::AcquireReadLeaseResponse resp;
  grpc::Status s = Acquire({"hash_a", "hash_missing", "hash_b"}, &resp);
  EXPECT_EQ(s.error_code(), grpc::StatusCode::NOT_FOUND);
  // Nothing pinned -- not even the hashes that were individually fine.
  EXPECT_EQ(store_->PinCount("hash_a"), 0);
  EXPECT_EQ(store_->PinCount("hash_b"), 0);
  EXPECT_EQ(test_server_->service->LeaseCountForTest(), 0u);
}

TEST_F(ReadLeaseTest, AcquireWrongStatusReturnsFailedPrecondition) {
  store_->SetPresentButNotHostResident("hash_c");
  ::tpu_sync::proto::AcquireReadLeaseResponse resp;
  grpc::Status s = Acquire({"hash_a", "hash_c"}, &resp);
  // The verify hook's distinctions must survive the RPC boundary: a hash that
  // is present but not host-resident is a different problem from an absent one.
  EXPECT_EQ(s.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(store_->PinCount("hash_a"), 0);
}

TEST_F(ReadLeaseTest, AcquireEmptyHashesRejected) {
  ::tpu_sync::proto::AcquireReadLeaseResponse resp;
  grpc::Status s = Acquire({}, &resp);
  EXPECT_EQ(s.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(test_server_->service->LeaseCountForTest(), 0u);
}

TEST_F(ReadLeaseTest, AcquireReturnsAuthoritativeIdsAndEndpoints) {
  ASSERT_TRUE(test_server_
                  ->client
                  ->RegisterWorker("worker_0", "10.0.0.1:9000",
                                   {::tpu_raiden::RaidenTransferEndpoint{
                                        "10.0.0.1:8000", {0, 1}},
                                    ::tpu_raiden::RaidenTransferEndpoint{
                                        "10.0.0.1:8001", {2, 3}}},
                                   /*node_id=*/7)
                  .ok());

  ::tpu_sync::proto::AcquireReadLeaseResponse resp;
  ASSERT_TRUE(Acquire({"hash_a", "hash_b"}, &resp).ok());
  EXPECT_NE(resp.lease_id(), 0u) << "0 is reserved as never-granted";

  ASSERT_EQ(resp.src_host_block_ids_size(), 2);
  EXPECT_EQ(resp.src_host_block_ids(0), store_->IdFor("hash_a"));
  EXPECT_EQ(resp.src_host_block_ids(1), store_->IdFor("hash_b"));

  ASSERT_EQ(resp.src_worker_endpoints_size(), 1);
  const auto& group = resp.src_worker_endpoints(0);
  EXPECT_EQ(group.node_id(), 7);
  EXPECT_EQ(group.worker_id(), "worker_0");
  ASSERT_EQ(group.endpoints_size(), 2);
  EXPECT_EQ(group.endpoints(0).endpoint(), "10.0.0.1:8000");
  EXPECT_THAT(group.endpoints(0).shards(), ElementsAre(0, 1));
  EXPECT_EQ(group.endpoints(1).endpoint(), "10.0.0.1:8001");

  EXPECT_EQ(store_->PinCount("hash_a"), 1);
  EXPECT_EQ(store_->PinCount("hash_b"), 1);
}

TEST_F(ReadLeaseTest, AcquireClampsTtl) {
  ::tpu_sync::proto::AcquireReadLeaseResponse defaulted;
  ASSERT_TRUE(Acquire({"hash_a"}, &defaulted, /*ttl_ms=*/0).ok());
  EXPECT_EQ(defaulted.granted_ttl_ms(),
            absl::ToInt64Milliseconds(
                RaidenControllerServiceImpl::DefaultLeaseTtl()));

  ::tpu_sync::proto::AcquireReadLeaseResponse too_big;
  ASSERT_TRUE(Acquire({"hash_b"}, &too_big, /*ttl_ms=*/999999999).ok());
  EXPECT_EQ(
      too_big.granted_ttl_ms(),
      absl::ToInt64Milliseconds(RaidenControllerServiceImpl::MaxLeaseTtl()));

  ::tpu_sync::proto::AcquireReadLeaseResponse too_small;
  ASSERT_TRUE(Acquire({"hash_c"}, &too_small, /*ttl_ms=*/1).ok());
  EXPECT_EQ(
      too_small.granted_ttl_ms(),
      absl::ToInt64Milliseconds(RaidenControllerServiceImpl::MinLeaseTtl()));
}

TEST_F(ReadLeaseTest, LeaseExpiryUnpinsAndMarksRevoked) {
  ::tpu_sync::proto::AcquireReadLeaseResponse resp;
  ASSERT_TRUE(Acquire({"hash_a"}, &resp).ok());
  EXPECT_EQ(store_->PinCount("hash_a"), 1);

  ASSERT_TRUE(test_server_->service->ForceExpireForTest(resp.lease_id()));
  EXPECT_EQ(store_->PinCount("hash_a"), 0) << "expiry must reclaim the pins";

  // The sweeper MARKS, it does not forget: a late release must learn REVOKED,
  // not UNKNOWN, so expiry stays distinguishable from a source restart.
  EXPECT_EQ(Release(resp.lease_id()), ::tpu_sync::proto::LEASE_REVOKED);
  EXPECT_EQ(store_->PinCount("hash_a"), 0) << "and must not unpin twice";
}

TEST_F(ReadLeaseTest, ConcurrentLeasesSameHashesIndependent) {
  ::tpu_sync::proto::AcquireReadLeaseResponse a, b;
  ASSERT_TRUE(Acquire({"hash_a"}, &a).ok());
  ASSERT_TRUE(Acquire({"hash_a"}, &b).ok());
  EXPECT_NE(a.lease_id(), b.lease_id());
  EXPECT_EQ(store_->PinCount("hash_a"), 2) << "pins are a refcount";

  EXPECT_EQ(Release(a.lease_id()), ::tpu_sync::proto::LEASE_HELD);
  // B's protection must survive A's release -- this is the property that
  // breaks the moment leases are deduped by hash set.
  EXPECT_EQ(store_->PinCount("hash_a"), 1);
  EXPECT_EQ(Renew(b.lease_id()).verdict(), ::tpu_sync::proto::LEASE_HELD);

  EXPECT_EQ(Release(b.lease_id()), ::tpu_sync::proto::LEASE_HELD);
  EXPECT_EQ(store_->PinCount("hash_a"), 0);
}

TEST_F(ReadLeaseTest, ReleaseIsIdempotent) {
  ::tpu_sync::proto::AcquireReadLeaseResponse resp;
  ASSERT_TRUE(Acquire({"hash_a"}, &resp).ok());
  EXPECT_EQ(Release(resp.lease_id()), ::tpu_sync::proto::LEASE_HELD);
  EXPECT_EQ(Release(resp.lease_id()), ::tpu_sync::proto::LEASE_HELD);
  EXPECT_EQ(Release(resp.lease_id()), ::tpu_sync::proto::LEASE_HELD);
  EXPECT_EQ(store_->PinCount("hash_a"), 0);
  EXPECT_EQ(store_->total_unpin_calls(), 1) << "exactly one unpin per lease";
}

TEST_F(ReadLeaseTest, ReleaseUnknownLeaseReturnsUnknown) {
  EXPECT_EQ(Release(0xdeadbeefcafe), ::tpu_sync::proto::LEASE_UNKNOWN);
  EXPECT_EQ(Release(0), ::tpu_sync::proto::LEASE_UNKNOWN)
      << "0 is never granted";
  EXPECT_EQ(store_->total_unpin_calls(), 0);
}

TEST_F(ReadLeaseTest, RenewExtendsLiveLeaseOnly) {
  ::tpu_sync::proto::AcquireReadLeaseResponse resp;
  ASSERT_TRUE(Acquire({"hash_a"}, &resp, /*ttl_ms=*/40000).ok());
  ::tpu_sync::proto::RenewReadLeaseResponse renewed =
      Renew(resp.lease_id(), 60000);
  EXPECT_EQ(renewed.verdict(), ::tpu_sync::proto::LEASE_HELD);
  EXPECT_EQ(renewed.new_ttl_ms(), 60000);
  EXPECT_EQ(store_->PinCount("hash_a"), 1);

  ASSERT_TRUE(test_server_->service->ForceExpireForTest(resp.lease_id()));
  // Never resurrect a revoked lease: the pins are already gone, so extending
  // the record would promise protection that no longer exists.
  EXPECT_EQ(Renew(resp.lease_id()).verdict(), ::tpu_sync::proto::LEASE_REVOKED);
  EXPECT_EQ(store_->PinCount("hash_a"), 0);
}

TEST_F(ReadLeaseTest, RenewUnknownLeaseReturnsUnknown) {
  EXPECT_EQ(Renew(12345).verdict(), ::tpu_sync::proto::LEASE_UNKNOWN);
}

TEST_F(ReadLeaseTest, SweeperReleaseRaceUnpinsOnce) {
  // Expiry racing release: whoever transitions the lease out of "live" first
  // owns the single unpin. Run many rounds to give the race a chance.
  for (int i = 0; i < 50; ++i) {
    ::tpu_sync::proto::AcquireReadLeaseResponse resp;
    ASSERT_TRUE(Acquire({"hash_a"}, &resp).ok());
    absl::Notification go;
    std::thread expirer([&] {
      go.WaitForNotification();
      test_server_->service->ForceExpireForTest(resp.lease_id());
    });
    std::thread releaser([&] {
      go.WaitForNotification();
      Release(resp.lease_id());
    });
    go.Notify();
    expirer.join();
    releaser.join();
    EXPECT_EQ(store_->PinCount("hash_a"), 0) << "round " << i;
  }
}

TEST_F(ReadLeaseTest, ConcurrentAcquireReleaseKeepsPinsExact) {
  constexpr int kThreads = 8;
  constexpr int kRounds = 20;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (int r = 0; r < kRounds; ++r) {
        ::tpu_sync::proto::AcquireReadLeaseResponse resp;
        if (Acquire({"hash_a", "hash_b"}, &resp).ok()) {
          Release(resp.lease_id());
        }
      }
    });
  }
  for (auto& t : threads) t.join();
  EXPECT_EQ(store_->PinCount("hash_a"), 0);
  EXPECT_EQ(store_->PinCount("hash_b"), 0);
}

TEST_F(ReadLeaseTest, DestructorUnpinsLiveLeases) {
  ::tpu_sync::proto::AcquireReadLeaseResponse a, b;
  ASSERT_TRUE(Acquire({"hash_a"}, &a).ok());
  ASSERT_TRUE(Acquire({"hash_b", "hash_c"}, &b).ok());
  EXPECT_EQ(store_->PinCount("hash_a"), 1);
  EXPECT_EQ(store_->PinCount("hash_c"), 1);

  test_server_->server->Shutdown();
  test_server_->server.reset();
  test_server_->service.reset();  // runs the destructor, joins the sweeper

  EXPECT_EQ(store_->PinCount("hash_a"), 0);
  EXPECT_EQ(store_->PinCount("hash_b"), 0);
  EXPECT_EQ(store_->PinCount("hash_c"), 0);
  test_server_.reset();
}

TEST_F(ReadLeaseTest, ClearedHooksSurviveExpiry) {
  ::tpu_sync::proto::AcquireReadLeaseResponse resp;
  ASSERT_TRUE(Acquire({"hash_a"}, &resp).ok());
  // Simulates store shutdown ordering: once the hooks are detached, neither
  // the sweeper nor the destructor may reach into the dead store.
  test_server_->service->ClearReadRemoteHooks();
  // The lease still transitions (the record is bookkeeping the service owns),
  // but no callback reaches the store.
  EXPECT_TRUE(test_server_->service->ForceExpireForTest(resp.lease_id()));
  EXPECT_EQ(store_->total_unpin_calls(), 0);
  EXPECT_EQ(store_->PinCount("hash_a"), 1) << "pin leaks by design once the "
                                              "hooks are gone; the store is "
                                              "being torn down anyway";
}

TEST_F(ReadLeaseTest, PerPeerLeasedBlockMetric) {
  ::tpu_sync::proto::AcquireReadLeaseResponse a, b;
  ASSERT_TRUE(Acquire({"hash_a", "hash_b"}, &a, 0, "peer_a").ok());
  ASSERT_TRUE(Acquire({"hash_c"}, &b, 0, "peer_b").ok());
  EXPECT_EQ(test_server_->service->LeasedBlocksForPeerForTest("peer_a///0"), 2);
  EXPECT_EQ(test_server_->service->LeasedBlocksForPeerForTest("peer_b///0"), 1);
  // The count falls as reads complete -- which is why a quota is deferred
  // rather than required: steady-state exposure is self-limiting.
  Release(a.lease_id());
  EXPECT_EQ(test_server_->service->LeasedBlocksForPeerForTest("peer_a///0"), 0);
}

}  // namespace
}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
