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

#include "tpu_sync/kv_cache/reshard/reshard_service.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/reshard/declaration_types.h"
#include "tpu_sync/kv_cache/reshard/framed_rpc.h"
#include "tpu_sync/kv_cache/reshard/request_block_registry.h"
#include "tpu_sync/rpc/controller_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

using ::testing::HasSubstr;

RaidenId Unit(int rank) {
  RaidenId id;
  id.job_name = "prefill";
  id.job_replica_id = absl::StrCat(rank);
  id.data_name = "kv_cache";
  id.data_replica_idx = 0;
  return id;
}

RaidenId DstUnit() {
  RaidenId id;
  id.job_name = "decode";
  id.job_replica_id = "0";
  id.data_name = "kv_cache";
  id.data_replica_idx = 0;
  return id;
}

// One pool, one region covering [0, live) of each block.
tpu_sync::rpc::PoolSpecProto MakePool(const std::string& tag, int64_t live,
                                      int64_t stride, int64_t num_blocks) {
  tpu_sync::rpc::PoolSpecProto pool;
  pool.set_tag(tag);
  pool.set_storage_index(0);
  pool.set_base_offset_bytes(0);
  pool.set_block_stride_bytes(stride);
  pool.set_num_blocks(num_blocks);
  pool.set_dtype_tag("uint8");
  auto* region = pool.add_regions();
  region->set_name("live");
  region->set_offset_bytes(0);
  region->set_stride_bytes(live);
  region->set_unit_bytes(live);
  region->set_num_units(1);
  region->set_units_per_stride(1);
  return pool;
}

// Captures worker payloads instead of dialing sockets.
class FakeTransport final : public FramedTransport {
 public:
  absl::StatusOr<std::string> Call(absl::string_view address,
                                   absl::string_view payload,
                                   absl::Duration /*timeout*/) override {
    {
      absl::MutexLock lock(mu_);
      calls_.emplace_back(std::string(address), std::string(payload));
    }
    if (fail_addr_ == address) {
      tpu_sync::rpc::ControlResponse failed;
      failed.set_success(false);
      failed.set_message("injected arm refusal");
      return failed.SerializeAsString();
    }
    tpu_sync::rpc::ControlResponse ok;
    ok.set_success(true);
    return ok.SerializeAsString();
  }

  void FailFor(const std::string& addr) { fail_addr_ = addr; }

  absl::Mutex mu_;
  std::vector<std::pair<std::string, std::string>> calls_;
  std::string fail_addr_;
};

class ReshardStackTest : public ::testing::Test {
 protected:
  ReshardStackTest() {
    ReshardService::Options options;
    options.port = 0;
    options.transport = &transport_;
    options.clock = [this]() { return now_; };
    service_ = std::make_unique<ReshardService>(options);
  }

  // Registers rank units 0..7 plus the decode unit through the framed
  // surface (byte-level, like the real facade would).
  void RegisterAllUnits(int num_src, int64_t live, int64_t stride,
                        int64_t num_blocks) {
    for (int rank = 0; rank < num_src; ++rank) {
      tpu_sync::rpc::ControlRequest req;
      req.set_command(
          tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
      auto* reg = req.mutable_register_work_unit_request();
      *reg->mutable_unit() = RaidenIdToProto(Unit(rank));
      reg->add_shards(absl::StrCat("10.0.0.1:", 9000 + rank));
      reg->set_control_plane_rpc_address(
          absl::StrCat("10.0.0.1:", 9100 + 2 * rank));
      *reg->add_pools() = MakePool("fa", live, stride, num_blocks);
      reg->set_layout_fingerprint("fp1");
      reg->set_page_tokens(512);
      reg->set_transfer_parallelism(num_src);
      reg->set_transfer_rank(rank);
      tpu_sync::rpc::ControlResponse resp = Handle(req.SerializeAsString());
      ASSERT_TRUE(resp.success()) << resp.message();
    }
    tpu_sync::rpc::ControlRequest req;
    req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
    auto* reg = req.mutable_register_work_unit_request();
    *reg->mutable_unit() = RaidenIdToProto(DstUnit());
    reg->add_shards("10.0.0.2:9400");
    reg->set_control_plane_rpc_address("10.0.0.2:9600");
    *reg->add_pools() = MakePool("fa", live, stride, num_blocks);
    reg->set_layout_fingerprint("fp1");
    reg->set_page_tokens(4096);
    reg->set_transfer_parallelism(num_src);
    reg->set_transfer_rank(0);
    tpu_sync::rpc::ControlResponse resp = Handle(req.SerializeAsString());
    ASSERT_TRUE(resp.success()) << resp.message();
  }

  void RegisterSpans(int rank, const std::string& req_id, int64_t uuid,
                     int64_t live, int64_t src_block, int64_t dst_index,
                     int64_t dst_offset, int64_t size,
                     int32_t dst_space_version = 0) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS);
    auto* block_req = req.mutable_register_request_blocks_request();
    block_req->set_req_id(req_id);
    block_req->set_uuid(uuid);
    *block_req->mutable_unit() = RaidenIdToProto(Unit(rank));
    block_req->add_block_ids(src_block);
    auto* entry = block_req->add_pool_spans();
    entry->set_tag("fa");
    entry->add_block_ids(src_block);
    auto* span = entry->add_spans();
    span->set_src_block_ordinal(0);
    span->set_src_offset_bytes(0);
    span->set_dst_block_index(dst_space_version == 0 ? dst_index : 0);
    span->set_dst_offset_bytes(dst_offset);
    span->set_size_bytes(size);
    span->set_count(1);
    entry->set_declared_bytes(size);
    entry->set_dst_space_version(dst_space_version);
    tpu_sync::rpc::ControllerResponse resp;
    resp.ParseFromString(service_->HandleFrame(req.SerializeAsString()));
    ASSERT_TRUE(resp.success()) << resp.message();
  }

  // One rank's v1 (request-global destination space) declaration with an
  // arbitrary span set over its own source blocks.
  struct GlobalSpan {
    int64_t src_ordinal;
    int64_t src_offset;
    int64_t dst_global_offset;
    int64_t size;
  };
  void RegisterGlobalSpans(int rank, const std::string& req_id, int64_t uuid,
                           const std::vector<int64_t>& src_blocks,
                           const std::vector<GlobalSpan>& spans) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS);
    auto* block_req = req.mutable_register_request_blocks_request();
    block_req->set_req_id(req_id);
    block_req->set_uuid(uuid);
    *block_req->mutable_unit() = RaidenIdToProto(Unit(rank));
    for (int64_t block : src_blocks) {
      block_req->add_block_ids(block);
    }
    auto* entry = block_req->add_pool_spans();
    entry->set_tag("fa");
    for (int64_t block : src_blocks) {
      entry->add_block_ids(block);
    }
    int64_t declared = 0;
    for (const GlobalSpan& span : spans) {
      auto* out = entry->add_spans();
      out->set_src_block_ordinal(span.src_ordinal);
      out->set_src_offset_bytes(span.src_offset);
      out->set_dst_block_index(0);
      out->set_dst_offset_bytes(span.dst_global_offset);
      out->set_size_bytes(span.size);
      out->set_count(1);
      declared += span.size;
    }
    entry->set_declared_bytes(declared);
    entry->set_dst_space_version(1);
    tpu_sync::rpc::ControllerResponse resp;
    resp.ParseFromString(service_->HandleFrame(req.SerializeAsString()));
    ASSERT_TRUE(resp.success()) << resp.message();
  }

  tpu_sync::rpc::ControlResponse Handle(const std::string& bytes) {
    tpu_sync::rpc::ControlResponse resp;
    resp.ParseFromString(service_->HandleFrame(bytes));
    return resp;
  }

  tpu_sync::rpc::ControllerResponse HandleController(const std::string& bytes) {
    tpu_sync::rpc::ControllerResponse resp;
    resp.ParseFromString(service_->HandleFrame(bytes));
    return resp;
  }

  tpu_sync::rpc::ControllerResponse Coordinate(
      const std::string& req_id, int64_t uuid, int num_src,
      std::vector<int64_t> dst_blocks,
      std::vector<int64_t> dst_skip = {}) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_COORDINATE_TRANSFER);
    auto* coord = req.mutable_coordinate_transfer_request();
    for (int rank = 0; rank < num_src; ++rank) {
      *coord->add_src_units() = RaidenIdToProto(Unit(rank));
    }
    *coord->add_dst_units() = RaidenIdToProto(DstUnit());
    coord->set_uuid(uuid);
    coord->set_is_sender(true);
    coord->set_dst_mem_type(tpu_sync::rpc::MEMORY_TYPE_HBM);
    coord->set_use_block_chunks(true);
    coord->set_req_id(req_id);
    for (int64_t block : dst_blocks) {
      coord->add_dst_device_block_ids(block);
    }
    coord->add_transfer_pool_tags("fa");
    for (int64_t skip : dst_skip) {
      coord->add_dst_skip_bytes(skip);
    }
    return HandleController(req.SerializeAsString());
  }

  FakeTransport transport_;
  double now_ = 1000.0;
  std::unique_ptr<ReshardService> service_;
};

TEST(FramedRpcTest, LoopbackRoundTrip) {
  FramedServer server(0, [](const std::string& request) {
    return absl::StrCat("echo:", request);
  });
  ASSERT_TRUE(server.Bind().ok());
  server.Start();
  SocketFramedTransport transport;
  auto response = transport.Call(absl::StrCat("127.0.0.1:", server.port()),
                                 "hello", absl::Seconds(5));
  ASSERT_TRUE(response.ok()) << response.status();
  EXPECT_EQ(*response, "echo:hello");
  server.Stop();
}

TEST_F(ReshardStackTest, FullPoolReshardFlowEmitsArmThenDispatch) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  // Rank 0 covers dst block 0 fully; rank 1 covers a prefix of block 1.
  RegisterSpans(0, "req-1", 42, 1024, /*src_block=*/3, /*dst_index=*/0,
                /*dst_offset=*/0, /*size=*/1024);
  RegisterSpans(1, "req-1", 42, 1024, /*src_block=*/5, /*dst_index=*/1,
                /*dst_offset=*/0, /*size=*/512);
  tpu_sync::rpc::ControllerResponse resp = Coordinate("req-1", 42, 2, {7, 9});
  ASSERT_TRUE(resp.success()) << resp.message();

  // Exactly 3 worker calls: 1 arm (decode control addr), 2 sender
  // dispatches; the arm strictly precedes both dispatches.
  ASSERT_EQ(transport_.calls_.size(), 3u);
  EXPECT_EQ(transport_.calls_[0].first, "10.0.0.2:9600");
  tpu_sync::rpc::ControlRequest arm;
  ASSERT_TRUE(arm.ParseFromString(transport_.calls_[0].second));
  EXPECT_FALSE(arm.start_transfer_request().is_sender());
  ASSERT_EQ(arm.start_transfer_request().pool_groups_size(), 1);
  const auto& group = arm.start_transfer_request().pool_groups(0);
  EXPECT_EQ(group.expected_pushes(), 2);
  ASSERT_EQ(group.dst_expected_extent_bytes_size(), 2);
  EXPECT_EQ(group.dst_expected_extent_bytes(0), 1024);
  EXPECT_EQ(group.dst_expected_extent_bytes(1), 512);
  // Receiver plan carries both source schedules keyed by ordinal.
  EXPECT_EQ(arm.start_transfer_request().shard_push_schedules_size(), 2);

  for (int i = 1; i <= 2; ++i) {
    tpu_sync::rpc::ControlRequest dispatch;
    ASSERT_TRUE(dispatch.ParseFromString(transport_.calls_[i].second));
    EXPECT_TRUE(dispatch.start_transfer_request().is_sender());
    EXPECT_EQ(dispatch.start_transfer_request().shard_push_schedules_size(), 1);
    const auto& schedule =
        dispatch.start_transfer_request().shard_push_schedules().at(0);
    ASSERT_EQ(schedule.entries_size(), 1);
    EXPECT_EQ(schedule.entries(0).dst_peer(), "10.0.0.2:9400");
    EXPECT_EQ(schedule.entries(0).count(), 1);
  }

  // Status is COMPLETED once the synchronous coordination returns.
  tpu_sync::rpc::ControllerRequest status_req;
  status_req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_GET_TRANSFER_STATUS);
  status_req.mutable_get_transfer_status_request()->set_req_id("req-1");
  tpu_sync::rpc::ControllerResponse status_resp =
      HandleController(status_req.SerializeAsString());
  ASSERT_TRUE(status_resp.success());
  EXPECT_EQ(status_resp.get_transfer_status_response().status(),
            tpu_sync::rpc::GetTransferStatusResponse::STATUS_COMPLETED);
}

TEST_F(ReshardStackTest, GlobalSpaceSpansSplitAtPageBoundaries) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  // Rank 0 declares a v1 (request-global) span: 1024 bytes at global
  // offset 512 → the controller must split it at the 1024-byte page
  // boundary into [512,1024) of page 0 and [0,512) of page 1. Rank 1
  // fills page 0's [0,512) prefix in legacy page-indexed form.
  RegisterSpans(0, "req-2", 43, 1024, /*src_block=*/3, /*dst_index=*/0,
                /*dst_offset=*/512, /*size=*/1024, /*dst_space_version=*/1);
  RegisterSpans(1, "req-2", 43, 1024, /*src_block=*/5, /*dst_index=*/0,
                /*dst_offset=*/0, /*size=*/512);
  tpu_sync::rpc::ControllerResponse resp = Coordinate("req-2", 43, 2, {7, 9});
  ASSERT_TRUE(resp.success()) << resp.message();
  ASSERT_EQ(transport_.calls_.size(), 3u);
  tpu_sync::rpc::ControlRequest arm;
  ASSERT_TRUE(arm.ParseFromString(transport_.calls_[0].second));
  const auto& group = arm.start_transfer_request().pool_groups(0);
  ASSERT_EQ(group.dst_expected_extent_bytes_size(), 2);
  EXPECT_EQ(group.dst_expected_extent_bytes(0), 1024);
  EXPECT_EQ(group.dst_expected_extent_bytes(1), 512);
  // Rank 0's dispatch carries the two split entries (page 0 tail, page 1
  // prefix), in (dst_block_index, dst_offset) order. Sender dispatch is
  // parallel, so select rank 0's payload by its control address.
  tpu_sync::rpc::ControlRequest rank0;
  bool found_rank0 = false;
  for (size_t i = 1; i < transport_.calls_.size(); ++i) {
    if (transport_.calls_[i].first == "10.0.0.1:9100") {
      ASSERT_TRUE(rank0.ParseFromString(transport_.calls_[i].second));
      found_rank0 = true;
    }
  }
  ASSERT_TRUE(found_rank0);
  const auto& schedule =
      rank0.start_transfer_request().shard_push_schedules().at(0);
  ASSERT_EQ(schedule.entries_size(), 2);
  EXPECT_EQ(schedule.entries(0).dst_block_id(), 7);
  EXPECT_EQ(schedule.entries(0).dst_offset_bytes(), 512);
  EXPECT_EQ(schedule.entries(0).size_bytes(), 512);
  EXPECT_EQ(schedule.entries(1).dst_block_id(), 9);
  EXPECT_EQ(schedule.entries(1).dst_offset_bytes(), 0);
  EXPECT_EQ(schedule.entries(1).size_bytes(), 512);
}

TEST_F(ReshardStackTest, ClipDropsTrimsAndRebasesGlobalSpans) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  // Full global space is [0, 2048): rank 1 owns [0,512) and [1536,2048),
  // rank 0 owns the straddling middle [512,1536). A one-page clip
  // (dst_skip_bytes=1024) must drop rank 1's first span entirely, trim
  // rank 0's span to [1024,1536), and re-base survivors onto the single
  // suffix destination page.
  RegisterGlobalSpans(0, "req-clip", 50, {3},
                      {{/*src_ordinal=*/0, /*src_offset=*/0,
                        /*dst_global_offset=*/512, /*size=*/1024}});
  RegisterGlobalSpans(1, "req-clip", 50, {5, 6},
                      {{0, 0, /*dst_global_offset=*/0, 512},
                       {1, 0, /*dst_global_offset=*/1536, 512}});
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-clip", 50, 2, {9}, /*dst_skip=*/{1024});
  ASSERT_TRUE(resp.success()) << resp.message();

  ASSERT_EQ(transport_.calls_.size(), 3u);
  tpu_sync::rpc::ControlRequest arm;
  ASSERT_TRUE(arm.ParseFromString(transport_.calls_[0].second));
  const auto& group = arm.start_transfer_request().pool_groups(0);
  EXPECT_EQ(group.expected_pushes(), 2);
  ASSERT_EQ(group.dst_expected_extent_bytes_size(), 1);
  EXPECT_EQ(group.dst_expected_extent_bytes(0), 1024);

  bool found_rank0 = false;
  bool found_rank1 = false;
  for (size_t i = 1; i < transport_.calls_.size(); ++i) {
    tpu_sync::rpc::ControlRequest dispatch;
    ASSERT_TRUE(dispatch.ParseFromString(transport_.calls_[i].second));
    const auto& schedule =
        dispatch.start_transfer_request().shard_push_schedules().at(0);
    ASSERT_EQ(schedule.entries_size(), 1);
    const auto& entry = schedule.entries(0);
    if (transport_.calls_[i].first == "10.0.0.1:9100") {
      // Rank 0's trimmed straddler: source advanced past the clipped 512
      // bytes, destination re-based to the suffix page's origin.
      found_rank0 = true;
      EXPECT_EQ(entry.src_block_id(), 3);
      EXPECT_EQ(entry.src_offset_bytes(), 512);
      EXPECT_EQ(entry.dst_block_id(), 9);
      EXPECT_EQ(entry.dst_offset_bytes(), 0);
      EXPECT_EQ(entry.size_bytes(), 512);
    } else if (transport_.calls_[i].first == "10.0.0.1:9102") {
      // Rank 1's surviving tail span, from its second source block.
      found_rank1 = true;
      EXPECT_EQ(entry.src_block_id(), 6);
      EXPECT_EQ(entry.src_offset_bytes(), 0);
      EXPECT_EQ(entry.dst_block_id(), 9);
      EXPECT_EQ(entry.dst_offset_bytes(), 512);
      EXPECT_EQ(entry.size_bytes(), 512);
    }
  }
  EXPECT_TRUE(found_rank0);
  EXPECT_TRUE(found_rank1);
}

TEST_F(ReshardStackTest, ClipValidationFailsClosed) {
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterGlobalSpans(0, "req-clipv", 52, {3},
                      {{0, 0, /*dst_global_offset=*/0, 1024}});

  // Not a page multiple: rejected before any claim.
  tpu_sync::rpc::ControllerResponse unaligned =
      Coordinate("req-clipv", 52, 1, {9}, /*dst_skip=*/{512});
  EXPECT_FALSE(unaligned.success());
  EXPECT_NE(unaligned.message().find("whole multiple"), std::string::npos)
      << unaligned.message();

  // Wrong arity: one tag, two skips.
  tpu_sync::rpc::ControllerResponse arity =
      Coordinate("req-clipv", 52, 1, {9}, /*dst_skip=*/{1024, 0});
  EXPECT_FALSE(arity.success());
  EXPECT_NE(arity.message().find("align 1:1"), std::string::npos)
      << arity.message();

  // Clip covering the whole declared extent: a full local hit must never
  // reach the planner.
  tpu_sync::rpc::ControllerResponse whole =
      Coordinate("req-clipv", 52, 1, {9}, /*dst_skip=*/{1024});
  EXPECT_FALSE(whole.success());
  EXPECT_NE(whole.message().find("removes the entire tag"), std::string::npos)
      << whole.message();

  // The registration survives the failed attempts (claims abandoned),
  // and an explicit zero clip on the wire is accepted as "no clip".
  tpu_sync::rpc::ControllerResponse ok =
      Coordinate("req-clipv", 52, 1, {9}, /*dst_skip=*/{0});
  EXPECT_TRUE(ok.success()) << ok.message();
}

TEST_F(ReshardStackTest, ClipRejectsPageIndexedDeclarations) {
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  // v0 (page-indexed) declaration: a nonzero clip has no defined meaning.
  RegisterSpans(0, "req-clipv0", 53, 1024, /*src_block=*/3, /*dst_index=*/0,
                /*dst_offset=*/0, /*size=*/1024);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-clipv0", 53, 1, {9}, /*dst_skip=*/{1024});
  EXPECT_FALSE(resp.success());
  EXPECT_NE(resp.message().find("destination-page-agnostic"),
            std::string::npos)
      << resp.message();
}

TEST_F(ReshardStackTest, MissingRegistrationKeepsContractString) {
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  tpu_sync::rpc::ControllerResponse resp = Coordinate("req-3", 44, 1, {7});
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(),
              HasSubstr("Missing producer block registration for "
                        "req_id=req-3, uuid=44, unit=RaidenId(job_name="
                        "'prefill', job_replica_id='0', data_name="
                        "'kv_cache', data_replica_idx=0)"));
}

TEST_F(ReshardStackTest, ArmFailureAbandonsClaimAndSkipsSenders) {
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-4", 45, 1024, 3, 0, 0, 1024);
  transport_.FailFor("10.0.0.2:9600");
  tpu_sync::rpc::ControllerResponse resp = Coordinate("req-4", 45, 1, {7});
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("injected arm refusal"));
  // Only the arm call went out; no sender was contacted.
  ASSERT_EQ(transport_.calls_.size(), 1u);
  // The claim was abandoned: a fresh coordination succeeds end-to-end.
  transport_.FailFor("");
  transport_.calls_.clear();
  tpu_sync::rpc::ControllerResponse retry = Coordinate("req-4", 45, 1, {7});
  ASSERT_TRUE(retry.success()) << retry.message();
  EXPECT_EQ(transport_.calls_.size(), 2u);
}

TEST_F(ReshardStackTest, CancelTombstoneBlocksLateRegistration) {
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  tpu_sync::rpc::ControllerRequest cancel_req;
  cancel_req.set_command(tpu_sync::rpc::ControllerRequest::
                             COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED);
  cancel_req.mutable_cancel_request_blocks_if_unclaimed_request()->set_req_id(
      "req-5");
  cancel_req.mutable_cancel_request_blocks_if_unclaimed_request()->set_uuid(46);
  tpu_sync::rpc::ControllerResponse cancel_resp =
      HandleController(cancel_req.SerializeAsString());
  ASSERT_TRUE(cancel_resp.success());
  EXPECT_EQ(cancel_resp.response_data(), "true");

  tpu_sync::rpc::ControllerRequest req;
  req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS);
  auto* block_req = req.mutable_register_request_blocks_request();
  block_req->set_req_id("req-5");
  block_req->set_uuid(46);
  *block_req->mutable_unit() = RaidenIdToProto(Unit(0));
  block_req->add_block_ids(3);
  tpu_sync::rpc::ControllerResponse resp =
      HandleController(req.SerializeAsString());
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(),
              HasSubstr("Request block registration was cancelled for "
                        "req_id=req-5, uuid=46"));
}

TEST_F(ReshardStackTest, CompletionVotesRetireAfterClaim) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-6", 47, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-6", 47, 1024, 5, 1, 0, 512);
  ASSERT_TRUE(Coordinate("req-6", 47, 2, {7, 9}).success());

  auto complete = [&](int rank) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_COMPLETE_REQUEST_BLOCKS);
    auto* complete_req = req.mutable_complete_request_blocks_request();
    complete_req->set_req_id("req-6");
    complete_req->set_uuid(47);
    *complete_req->mutable_unit() = RaidenIdToProto(Unit(rank));
    return HandleController(req.SerializeAsString());
  };
  tpu_sync::rpc::ControllerResponse first = complete(0);
  ASSERT_TRUE(first.success());
  EXPECT_EQ(first.response_data(), "0");
  tpu_sync::rpc::ControllerResponse second = complete(1);
  ASSERT_TRUE(second.success());
  EXPECT_EQ(second.response_data(), "2");  // both rank rows retired
}

TEST_F(ReshardStackTest, TtlPurgesExpiredRegistrations) {
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-7", 48, 1024, 3, 0, 0, 1024);
  now_ += 601.0;  // past the 600 s TTL
  tpu_sync::rpc::ControllerResponse resp = Coordinate("req-7", 48, 1, {7});
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("Missing producer block registration"));
}

TEST_F(ReshardStackTest, LegacyCommandsFailClosed) {
  tpu_sync::rpc::ControlRequest req;
  req.set_command(
      tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_TRANSFER_SCHEDULE);
  auto* start_req = req.mutable_start_transfer_request();
  start_req->add_transfer_pool_indices(0);
  tpu_sync::rpc::ControlResponse resp = Handle(req.SerializeAsString());
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(),
              HasSubstr("Inter-controller reshard schedule registration is "
                        "retired"));
}

}  // namespace
}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
