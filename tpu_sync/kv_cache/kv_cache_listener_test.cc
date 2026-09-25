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

#include "tpu_sync/kv_cache/kv_cache_listener.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::tpu_sync::rpc::ControlRequest;
using ::tpu_sync::rpc::ControlResponse;
using ::tpu_sync::rpc::StartTransferRequest;

enum class Route {
  kNone,
  kPoolReshardPush,
  kPoolReshardRegisterRecv,
  kRegisterActivePlan,
};

class RoutingKVCacheManager final : public KVCacheManagerBase {
 public:
  explicit RoutingKVCacheManager(
      std::optional<int> host_blocks_to_allocate = std::nullopt)
      : KVCacheManagerBase(/*num_layers=*/1, /*num_shards=*/1,
                           /*slice_byte_size=*/64,
                           /*local_port=*/std::nullopt,
                           host_blocks_to_allocate) {}

  absl::Status PoolReshardPush(const StartTransferRequest& request,
                               absl::Span<const int64_t> src_block_ids,
                               int parallelism) override {
    request_ = request;
    local_block_ids_.assign(src_block_ids.begin(), src_block_ids.end());
    recorded_parallelism_ = parallelism;
    route_.store(Route::kPoolReshardPush, std::memory_order_release);
    return absl::OkStatus();
  }

  absl::Status PoolReshardRegisterRecv(
      const StartTransferRequest& request,
      absl::Span<const int64_t> chip_block_ids) override {
    request_ = request;
    local_block_ids_.assign(chip_block_ids.begin(), chip_block_ids.end());
    route_.store(Route::kPoolReshardRegisterRecv, std::memory_order_release);
    return absl::OkStatus();
  }

  absl::Status RegisterActivePlan(uint64_t uuid,
                                  const StartTransferRequest& request,
                                  bool is_sender) override {
    request_ = request;
    registered_uuid_ = uuid;
    registered_is_sender_ = is_sender;
    route_.store(Route::kRegisterActivePlan, std::memory_order_release);
    return absl::OkStatus();
  }

  Route route() const { return route_.load(std::memory_order_acquire); }
  const StartTransferRequest& request() const { return request_; }
  const std::vector<int64_t>& local_block_ids() const {
    return local_block_ids_;
  }
  int parallelism() const { return recorded_parallelism_; }
  uint64_t registered_uuid() const { return registered_uuid_; }
  bool registered_is_sender() const { return registered_is_sender_; }

 private:
  std::atomic<Route> route_{Route::kNone};
  StartTransferRequest request_;
  std::vector<int64_t> local_block_ids_;
  int recorded_parallelism_ = 0;
  uint64_t registered_uuid_ = 0;
  bool registered_is_sender_ = true;
};

bool WriteAll(int fd, const void* data, size_t size) {
  const char* bytes = static_cast<const char*>(data);
  size_t written = 0;
  while (written < size) {
    const ssize_t result = write(fd, bytes + written, size - written);
    if (result <= 0) return false;
    written += static_cast<size_t>(result);
  }
  return true;
}

bool ReadAll(int fd, void* data, size_t size) {
  char* bytes = static_cast<char*>(data);
  size_t read_bytes = 0;
  while (read_bytes < size) {
    const ssize_t result = read(fd, bytes + read_bytes, size - read_bytes);
    if (result <= 0) return false;
    read_bytes += static_cast<size_t>(result);
  }
  return true;
}

ControlResponse SendRequest(int listener_port, const ControlRequest& request) {
  int sock = socket(AF_INET6, SOCK_STREAM, 0);
  EXPECT_GE(sock, 0);
  if (sock < 0) return {};

  sockaddr_in6 address{};
  address.sin6_family = AF_INET6;
  address.sin6_port = htons(listener_port);
  address.sin6_addr = in6addr_loopback;
  EXPECT_EQ(
      connect(sock, reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);

  std::string payload;
  EXPECT_TRUE(request.SerializeToString(&payload));
  const uint32_t payload_size = htonl(payload.size());
  EXPECT_TRUE(WriteAll(sock, &payload_size, sizeof(payload_size)));
  EXPECT_TRUE(WriteAll(sock, payload.data(), payload.size()));

  uint32_t response_size = 0;
  EXPECT_TRUE(ReadAll(sock, &response_size, sizeof(response_size)));
  response_size = ntohl(response_size);
  std::string response_bytes(response_size, '\0');
  EXPECT_TRUE(ReadAll(sock, response_bytes.data(), response_bytes.size()));
  close(sock);

  ControlResponse response;
  EXPECT_TRUE(response.ParseFromString(response_bytes));
  return response;
}

StartTransferRequest* AddPoolPlan(ControlRequest* request, bool is_sender) {
  request->set_command(ControlRequest::COMMAND_START_TRANSFER);
  StartTransferRequest* plan = request->mutable_start_transfer_request();
  plan->set_uuid(1234);
  plan->set_req_id("request-1");
  plan->set_is_sender(is_sender);
  plan->add_transfer_pool_indices(0);
  auto* group = plan->add_pool_groups();
  group->add_pool_indices(0);
  group->set_expected_pushes(2);
  return plan;
}

TEST(KVCacheListenerTest, PoolSenderRoutesToPoolReshardPush) {
  RoutingKVCacheManager manager;
  KVCacheListener listener(&manager, /*listener_port=*/0);

  ControlRequest request;
  StartTransferRequest* plan = AddPoolPlan(&request, /*is_sender=*/true);
  plan->set_parallelism(3);
  // The sender's local block list derives from its schedule entries (no
  // scalar mirror on the wire).
  auto* schedule = &(*plan->mutable_shard_push_schedules())[0];
  schedule->add_entries()->set_src_block_id(9);
  schedule->add_entries()->set_src_block_id(7);
  schedule->add_entries()->set_src_block_id(9);

  ControlResponse response = SendRequest(listener.listener_port(), request);

  EXPECT_TRUE(response.success()) << response.message();
  EXPECT_EQ(manager.route(), Route::kPoolReshardPush);
  EXPECT_EQ(manager.request().uuid(), 1234);
  EXPECT_EQ(manager.local_block_ids(), (std::vector<int64_t>{7, 9}));
  EXPECT_EQ(manager.parallelism(), 3);
}

TEST(KVCacheListenerTest, PoolReceiverRoutesToPoolReshardRegisterRecv) {
  RoutingKVCacheManager manager;
  KVCacheListener listener(&manager, /*listener_port=*/0);

  ControlRequest request;
  StartTransferRequest* plan = AddPoolPlan(&request, /*is_sender=*/false);
  // The receiver's local block list is the groups' concatenation (no
  // scalar mirror on the wire).
  plan->mutable_pool_groups(0)->add_dst_device_block_ids(5);
  plan->mutable_pool_groups(0)->add_dst_device_block_ids(3);
  auto* source_zero = &(*plan->mutable_shard_push_schedules())[0];
  source_zero->add_entries()->set_dst_block_id(99);

  ControlResponse response = SendRequest(listener.listener_port(), request);

  EXPECT_TRUE(response.success()) << response.message();
  EXPECT_EQ(manager.route(), Route::kPoolReshardRegisterRecv);
  EXPECT_EQ(manager.request().uuid(), 1234);
  EXPECT_EQ(manager.local_block_ids(), (std::vector<int64_t>{5, 3}));
}

TEST(KVCacheListenerTest, PoolReceiverArmReplyCarriesHostBaseAddrs) {
  RoutingKVCacheManager manager(/*host_blocks_to_allocate=*/1);
  KVCacheListener listener(&manager, /*listener_port=*/0);
  absl::StatusOr<std::vector<uint64_t>> expected = manager.PoolHostBaseAddrs(0);
  ASSERT_TRUE(expected.ok()) << expected.status();
  ASSERT_EQ(expected->size(), 1u);

  ControlRequest request;
  StartTransferRequest* plan = AddPoolPlan(&request, /*is_sender=*/false);
  plan->add_transfer_pool_indices(7);  // Unknown pool: left out of the reply.
  plan->mutable_pool_groups(0)->add_dst_device_block_ids(5);

  ControlResponse response = SendRequest(listener.listener_port(), request);

  EXPECT_TRUE(response.success()) << response.message();
  ASSERT_TRUE(response.receiver_pool_addrs().contains(0));
  const auto& addrs = response.receiver_pool_addrs().at(0).host_base_addrs();
  EXPECT_EQ(std::vector<uint64_t>(addrs.begin(), addrs.end()), *expected);
  EXPECT_FALSE(response.receiver_pool_addrs().contains(7));
}

TEST(KVCacheListenerTest, PartiallyPopulatedPoolPlanStillRoutesToPoolPath) {
  // The fork predicate is fail-closed: a plan carrying only one of the pool
  // fields must reach the pool executor (whose validation rejects it), never
  // silently fall back to the legacy path.
  RoutingKVCacheManager manager;
  KVCacheListener listener(&manager, /*listener_port=*/0);

  ControlRequest request;
  request.set_command(ControlRequest::COMMAND_START_TRANSFER);
  StartTransferRequest* plan = request.mutable_start_transfer_request();
  plan->set_uuid(9012);
  plan->set_is_sender(false);
  plan->add_pool_groups()->set_expected_pushes(4);  // no transfer_pool_indices

  ControlResponse response = SendRequest(listener.listener_port(), request);

  EXPECT_TRUE(response.success()) << response.message();
  EXPECT_EQ(manager.route(), Route::kPoolReshardRegisterRecv);
}

TEST(KVCacheListenerTest, PoollessReceiverPreservesRegisterActivePlan) {
  RoutingKVCacheManager manager;
  KVCacheListener listener(&manager, /*listener_port=*/0);

  ControlRequest request;
  request.set_command(ControlRequest::COMMAND_START_TRANSFER);
  StartTransferRequest* plan = request.mutable_start_transfer_request();
  plan->set_uuid(5678);
  plan->set_is_sender(false);
  // pool_dtype_tags predates the pool executor fields and must not by itself
  // opt a legacy command into PoolReshardRegisterRecv.
  plan->add_pool_dtype_tags("legacy-dtype");

  ControlResponse response = SendRequest(listener.listener_port(), request);

  EXPECT_TRUE(response.success()) << response.message();
  EXPECT_EQ(manager.route(), Route::kRegisterActivePlan);
  EXPECT_EQ(manager.registered_uuid(), 5678);
  EXPECT_FALSE(manager.registered_is_sender());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
