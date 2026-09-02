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

#include <algorithm>
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

RaidenId DstUnit(int idx = 0) {
  RaidenId id;
  id.job_name = "decode";
  id.job_replica_id = absl::StrCat(idx);
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

  // Registers one decode unit. Destination idx 0 keeps the legacy
  // addresses (shard 10.0.0.2:9400, control 10.0.0.2:9600); siblings
  // stagger off them. Overrides support mismatch tests (registration is
  // replacement).
  void RegisterDstUnit(int idx, int num_src, int64_t live, int64_t stride,
                       int64_t num_blocks,
                       const std::string& fingerprint = "fp1",
                       const std::string& shard_override = "") {
    tpu_sync::rpc::ControlRequest req;
    req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
    auto* reg = req.mutable_register_work_unit_request();
    *reg->mutable_unit() = RaidenIdToProto(DstUnit(idx));
    reg->add_shards(shard_override.empty()
                        ? absl::StrCat("10.0.0.2:", 9400 + idx)
                        : shard_override);
    reg->set_control_plane_rpc_address(absl::StrCat("10.0.0.2:", 9600 + idx));
    *reg->add_pools() = MakePool("fa", live, stride, num_blocks);
    reg->set_layout_fingerprint(fingerprint);
    reg->set_page_tokens(4096);
    reg->set_transfer_parallelism(num_src);
    reg->set_transfer_rank(0);
    tpu_sync::rpc::ControlResponse resp = Handle(req.SerializeAsString());
    ASSERT_TRUE(resp.success()) << resp.message();
  }

  // Registers rank units 0..7 plus the decode unit(s) through the framed
  // surface (byte-level, like the real facade would).
  void RegisterAllUnits(int num_src, int64_t live, int64_t stride,
                        int64_t num_blocks, int num_dst = 1) {
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
    for (int idx = 0; idx < num_dst; ++idx) {
      RegisterDstUnit(idx, num_src, live, stride, num_blocks);
    }
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

  // One rank's page-indexed "fa" declaration over one source block whose
  // spans are routed per destination unit (dst_unit_ordinal >= 0) or
  // replicated (ordinal -1 = absent on the wire). Returns the controller
  // response so callers can assert on refusals.
  struct RoutedSpan {
    int64_t src_offset;
    int64_t dst_index;
    int64_t dst_offset;
    int64_t size;
    int32_t dst_unit_ordinal;
  };
  tpu_sync::rpc::ControllerResponse RegisterRoutedSpans(
      int rank, const std::string& req_id, int64_t uuid, int64_t src_block,
      const std::vector<RoutedSpan>& spans) {
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
    int64_t declared = 0;
    for (const RoutedSpan& span : spans) {
      auto* out = entry->add_spans();
      out->set_src_block_ordinal(0);
      out->set_src_offset_bytes(span.src_offset);
      out->set_dst_block_index(span.dst_index);
      out->set_dst_offset_bytes(span.dst_offset);
      out->set_size_bytes(span.size);
      out->set_count(1);
      if (span.dst_unit_ordinal != -1) {
        out->set_dst_unit_ordinal(span.dst_unit_ordinal);
      }
      declared += span.size;
    }
    entry->set_declared_bytes(declared);
    entry->set_dst_space_version(0);
    return HandleController(req.SerializeAsString());
  }

  // Two sources (live 1024) feeding two head-shard destinations (live 512):
  // the TP2 shape where each destination receives a different half of
  // every source block.
  void RegisterShardedPair() {
    RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                     /*num_blocks=*/16, /*num_dst=*/0);
    RegisterDstUnit(0, /*num_src=*/2, /*live=*/512, /*stride=*/512,
                    /*num_blocks=*/16);
    RegisterDstUnit(1, /*num_src=*/2, /*live=*/512, /*stride=*/512,
                    /*num_blocks=*/16);
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
      std::vector<int64_t> dst_blocks, std::vector<int64_t> dst_skip = {},
      int num_dst = 1) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_COORDINATE_TRANSFER);
    auto* coord = req.mutable_coordinate_transfer_request();
    for (int rank = 0; rank < num_src; ++rank) {
      *coord->add_src_units() = RaidenIdToProto(Unit(rank));
    }
    for (int idx = 0; idx < num_dst; ++idx) {
      *coord->add_dst_units() = RaidenIdToProto(DstUnit(idx));
    }
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

// One plan replicates the identical byte set to N destinations.
TEST_F(ReshardStackTest, TwoDestinationsArmEachThenDispatchSendersOnce) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16, /*num_dst=*/2);
  RegisterSpans(0, "req-mc", 60, 1024, /*src_block=*/3, /*dst_index=*/0,
                /*dst_offset=*/0, /*size=*/1024);
  RegisterSpans(1, "req-mc", 60, 1024, /*src_block=*/5, /*dst_index=*/1,
                /*dst_offset=*/0, /*size=*/512);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-mc", 60, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_TRUE(resp.success()) << resp.message();

  // 4 calls: 2 concurrent arms (order between them unspecified), then 2
  // sender dispatches — every arm strictly precedes every dispatch.
  ASSERT_EQ(transport_.calls_.size(), 4u);
  std::vector<std::string> arm_addresses = {transport_.calls_[0].first,
                                            transport_.calls_[1].first};
  std::sort(arm_addresses.begin(), arm_addresses.end());
  EXPECT_EQ(arm_addresses[0], "10.0.0.2:9600");
  EXPECT_EQ(arm_addresses[1], "10.0.0.2:9601");

  for (int i = 0; i < 2; ++i) {
    tpu_sync::rpc::ControlRequest arm;
    ASSERT_TRUE(arm.ParseFromString(transport_.calls_[i].second));
    const auto& start_req = arm.start_transfer_request();
    EXPECT_FALSE(start_req.is_sender());
    // The arm names only the armed unit (controller-delivery worker
    // resolution reads dst_units(0)).
    ASSERT_EQ(start_req.dst_units_size(), 1);
    const std::string own_peer = transport_.calls_[i].first == "10.0.0.2:9600"
                                     ? "10.0.0.2:9400"
                                     : "10.0.0.2:9401";
    // Filtered schedule: both sources, each contributing its single pair,
    // every entry addressed to this receiver's own data endpoint.
    ASSERT_EQ(start_req.shard_push_schedules_size(), 2);
    int entry_count = 0;
    for (const auto& keyed_schedule : start_req.shard_push_schedules()) {
      for (const auto& entry : keyed_schedule.second.entries()) {
        EXPECT_EQ(entry.dst_peer(), own_peer);
        ++entry_count;
      }
    }
    EXPECT_EQ(entry_count, 2);
    // Per-destination expected pushes match what the receiver recomputes
    // from its filtered schedule: two senders x one pair each.
    ASSERT_EQ(start_req.pool_groups_size(), 1);
    EXPECT_EQ(start_req.pool_groups(0).expected_pushes(), 2);
  }

  for (int i = 2; i < 4; ++i) {
    tpu_sync::rpc::ControlRequest dispatch;
    ASSERT_TRUE(dispatch.ParseFromString(transport_.calls_[i].second));
    const auto& start_req = dispatch.start_transfer_request();
    EXPECT_TRUE(start_req.is_sender());
    EXPECT_EQ(start_req.dst_units_size(), 2);
    // The sender's local schedule carries one entry per destination for
    // its single chunk — same bytes, both peers, one D2H amortized
    // executor-side.
    ASSERT_EQ(start_req.shard_push_schedules_size(), 1);
    const auto& schedule = start_req.shard_push_schedules().at(0);
    ASSERT_EQ(schedule.entries_size(), 2);
    std::vector<std::string> peers = {schedule.entries(0).dst_peer(),
                                      schedule.entries(1).dst_peer()};
    std::sort(peers.begin(), peers.end());
    EXPECT_EQ(peers[0], "10.0.0.2:9400");
    EXPECT_EQ(peers[1], "10.0.0.2:9401");
    EXPECT_EQ(schedule.entries(0).dst_offset_bytes(),
              schedule.entries(1).dst_offset_bytes());
    EXPECT_EQ(schedule.entries(0).size_bytes(),
              schedule.entries(1).size_bytes());
  }
}

TEST_F(ReshardStackTest, MismatchedDestinationsFailClosed) {
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16, /*num_dst=*/2);
  RegisterSpans(0, "req-mcgeo", 62, 1024, 3, 0, 0, 1024);

  // Destination geometry must be identical across units.
  RegisterDstUnit(1, /*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                  /*num_blocks=*/8);
  tpu_sync::rpc::ControllerResponse geometry =
      Coordinate("req-mcgeo", 62, 1, {7}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(geometry.success());
  EXPECT_THAT(geometry.message(),
              HasSubstr("Destination pool geometry differs across units"));

  // Fingerprints must be identical across every unit of the pair.
  RegisterDstUnit(1, /*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                  /*num_blocks=*/16, /*fingerprint=*/"fp2");
  tpu_sync::rpc::ControllerResponse fingerprint =
      Coordinate("req-mcgeo", 62, 1, {7}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(fingerprint.success());
  EXPECT_THAT(fingerprint.message(), HasSubstr("Layout fingerprint mismatch"));

  // Destinations must expose distinct data-plane endpoints.
  RegisterDstUnit(1, /*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                  /*num_blocks=*/16, /*fingerprint=*/"fp1",
                  /*shard_override=*/"10.0.0.2:9400");
  tpu_sync::rpc::ControllerResponse duplicate =
      Coordinate("req-mcgeo", 62, 1, {7}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(duplicate.success());
  EXPECT_THAT(duplicate.message(), HasSubstr("distinct data-plane endpoints"));

  // All three rejections precede the claim or abandon it: the healthy
  // registration still coordinates.
  RegisterDstUnit(1, /*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                  /*num_blocks=*/16);
  tpu_sync::rpc::ControllerResponse ok =
      Coordinate("req-mcgeo", 62, 1, {7}, /*dst_skip=*/{}, /*num_dst=*/2);
  EXPECT_TRUE(ok.success()) << ok.message();
}

// Sharded destinations: every span names its destination unit; each receiver
// is armed with only its own bytes and push count, while one sender program
// carries both peers with different source offsets.
TEST_F(ReshardStackTest, TwoDestinationsShardedSpansRouteBytesPerReceiver) {
  RegisterShardedPair();
  // Rank 0 owns source block 3 -> destination block index 0; rank 1 owns
  // source block 5 -> destination block index 1. Destination unit d takes
  // the source half at offset d*512.
  ASSERT_TRUE(RegisterRoutedSpans(0, "req-sh", 70, 3,
                                  {{0, 0, 0, 512, 0}, {512, 0, 0, 512, 1}})
                  .success());
  ASSERT_TRUE(RegisterRoutedSpans(1, "req-sh", 70, 5,
                                  {{0, 1, 0, 512, 0}, {512, 1, 0, 512, 1}})
                  .success());
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-sh", 70, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_TRUE(resp.success()) << resp.message();

  // 2 concurrent arms, then 2 sender dispatches.
  ASSERT_EQ(transport_.calls_.size(), 4u);
  for (int i = 0; i < 2; ++i) {
    tpu_sync::rpc::ControlRequest arm;
    ASSERT_TRUE(arm.ParseFromString(transport_.calls_[i].second));
    const auto& start_req = arm.start_transfer_request();
    EXPECT_FALSE(start_req.is_sender());
    ASSERT_EQ(start_req.dst_units_size(), 1);
    const bool is_dst0 = transport_.calls_[i].first == "10.0.0.2:9600";
    const std::string own_peer = is_dst0 ? "10.0.0.2:9400" : "10.0.0.2:9401";
    const int64_t own_src_offset = is_dst0 ? 0 : 512;
    // Both senders contribute exactly one entry each, addressed to this
    // receiver's endpoint, sourced from this destination's half.
    ASSERT_EQ(start_req.shard_push_schedules_size(), 2);
    for (const auto& keyed_schedule : start_req.shard_push_schedules()) {
      ASSERT_EQ(keyed_schedule.second.entries_size(), 1);
      const auto& entry = keyed_schedule.second.entries(0);
      EXPECT_EQ(entry.dst_peer(), own_peer);
      EXPECT_EQ(entry.src_offset_bytes(), own_src_offset);
      EXPECT_EQ(entry.dst_offset_bytes(), 0);
      EXPECT_EQ(entry.size_bytes(), 512);
    }
    ASSERT_EQ(start_req.pool_groups_size(), 1);
    EXPECT_EQ(start_req.pool_groups(0).expected_pushes(), 2);
    ASSERT_EQ(start_req.pool_groups(0).dst_expected_extent_bytes_size(), 2);
    EXPECT_EQ(start_req.pool_groups(0).dst_expected_extent_bytes(0), 512);
    EXPECT_EQ(start_req.pool_groups(0).dst_expected_extent_bytes(1), 512);
  }
  for (int i = 2; i < 4; ++i) {
    tpu_sync::rpc::ControlRequest dispatch;
    ASSERT_TRUE(dispatch.ParseFromString(transport_.calls_[i].second));
    const auto& start_req = dispatch.start_transfer_request();
    EXPECT_TRUE(start_req.is_sender());
    EXPECT_EQ(start_req.dst_units_size(), 2);
    ASSERT_EQ(start_req.shard_push_schedules_size(), 1);
    const auto& schedule = start_req.shard_push_schedules().at(0);
    // One entry per destination, each with that destination's source half.
    ASSERT_EQ(schedule.entries_size(), 2);
    for (const auto& entry : schedule.entries()) {
      const int64_t expected_src =
          entry.dst_peer() == "10.0.0.2:9400" ? 0 : 512;
      EXPECT_EQ(entry.src_offset_bytes(), expected_src);
      EXPECT_EQ(entry.dst_offset_bytes(), 0);
      EXPECT_EQ(entry.size_bytes(), 512);
    }
    EXPECT_NE(schedule.entries(0).dst_peer(), schedule.entries(1).dst_peer());
    // Senders carry the largest per-destination push count.
    ASSERT_EQ(start_req.pool_groups_size(), 1);
    EXPECT_EQ(start_req.pool_groups(0).expected_pushes(), 2);
  }
}

TEST_F(ReshardStackTest, ShardedSpanOrdinalOutOfRangeFailsClosed) {
  RegisterShardedPair();
  // Ordinal 2 with two destination units: accepted by the registry (the
  // upper bound is a plan-time fact), refused by the planner.
  ASSERT_TRUE(RegisterRoutedSpans(0, "req-oob", 71, 3,
                                  {{0, 0, 0, 512, 0}, {512, 0, 0, 512, 2}})
                  .success());
  ASSERT_TRUE(RegisterRoutedSpans(1, "req-oob", 71, 5,
                                  {{0, 1, 0, 512, 0}, {512, 1, 0, 512, 1}})
                  .success());
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-oob", 71, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("dst_unit_ordinal 2 exceeds"));
  EXPECT_TRUE(transport_.calls_.empty());
}

TEST_F(ReshardStackTest, ShardedDestinationWithoutCoverageFailsClosed) {
  RegisterShardedPair();
  // Everything routed to destination 0: destination 1 has no bytes.
  ASSERT_TRUE(RegisterRoutedSpans(0, "req-nocov", 72, 3, {{0, 0, 0, 512, 0}})
                  .success());
  ASSERT_TRUE(RegisterRoutedSpans(1, "req-nocov", 72, 5, {{0, 1, 0, 512, 0}})
                  .success());
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-nocov", 72, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("has no declared coverage"));
  EXPECT_THAT(resp.message(), HasSubstr("at destination unit 1"));
  EXPECT_TRUE(transport_.calls_.empty());
}

TEST_F(ReshardStackTest, ShardedNonUniformExtentsFailClosed) {
  RegisterShardedPair();
  // A single (final) destination block: destination 0 gets 512 bytes,
  // destination 1 only 256 — both prefix-shaped, but the group carries one
  // extent vector, so the shards must agree.
  ASSERT_TRUE(RegisterRoutedSpans(0, "req-ext", 73, 3,
                                  {{0, 0, 0, 512, 0}, {512, 0, 0, 256, 1}})
                  .success());
  ASSERT_TRUE(RegisterRoutedSpans(1, "req-ext", 73, 5, {}).success());
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-ext", 73, 2, {7}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("uniform coverage extents"));
  EXPECT_TRUE(transport_.calls_.empty());
}

TEST_F(ReshardStackTest, NegativeOrdinalRejectedAtRegistration) {
  RegisterShardedPair();
  tpu_sync::rpc::ControllerResponse resp =
      RegisterRoutedSpans(0, "req-neg", 74, 3, {{0, 0, 0, 512, -2}});
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("dst_unit_ordinal must be -1"));
}

// Mixed routing: a replicated span (no ordinal) reaches both destinations
// while routed spans reach one each; coverage and pushes account per
// destination.
TEST_F(ReshardStackTest, ReplicatedAndRoutedSpansCombine) {
  RegisterShardedPair();
  // Rank 0: block index 0 replicated (the same 512 bytes to both);
  // rank 1: block index 1 sharded.
  ASSERT_TRUE(
      RegisterRoutedSpans(0, "req-mix", 75, 3, {{0, 0, 0, 512, -1}}).success());
  ASSERT_TRUE(RegisterRoutedSpans(1, "req-mix", 75, 5,
                                  {{0, 1, 0, 512, 0}, {512, 1, 0, 512, 1}})
                  .success());
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-mix", 75, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_TRUE(resp.success()) << resp.message();
  ASSERT_EQ(transport_.calls_.size(), 4u);
  for (int i = 0; i < 2; ++i) {
    tpu_sync::rpc::ControlRequest arm;
    ASSERT_TRUE(arm.ParseFromString(transport_.calls_[i].second));
    const auto& start_req = arm.start_transfer_request();
    int entries = 0;
    for (const auto& keyed_schedule : start_req.shard_push_schedules()) {
      entries += keyed_schedule.second.entries_size();
    }
    EXPECT_EQ(entries, 2);
    EXPECT_EQ(start_req.pool_groups(0).expected_pushes(), 2);
  }
  for (int i = 2; i < 4; ++i) {
    tpu_sync::rpc::ControlRequest dispatch;
    ASSERT_TRUE(dispatch.ParseFromString(transport_.calls_[i].second));
    const auto& schedule =
        dispatch.start_transfer_request().shard_push_schedules().at(0);
    // Rank 0 (replicated): 2 entries (one per peer); rank 1 (sharded): 2.
    EXPECT_EQ(schedule.entries_size(), 2);
  }
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
