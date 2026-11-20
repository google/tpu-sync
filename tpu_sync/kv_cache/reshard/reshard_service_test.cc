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
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/kv_cache/reshard/declaration_types.h"
#include "tpu_sync/kv_cache/reshard/pool_reshard_planner.h"
#include "tpu_sync/kv_cache/reshard/request_block_registry.h"
#include "tpu_sync/kv_cache/reshard/reshard_client.h"
#include "tpu_sync/kv_cache/reshard/reshard_control_pipe.h"
#include "tpu_sync/kv_cache/reshard/reshard_coordinator.h"
#include "tpu_sync/kv_cache/reshard/work_unit_directory.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/rpc/controller_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
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
class FakeTransport final : public ControlPipeClient {
 public:
  absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> SendRaw(
      absl::string_view endpoint,
      const control_pipe::proto::ControlEnvelope& envelope,
      absl::Duration /*timeout*/) override {
    // Cancels are recorded and answered apart from the other worker calls.
    if (tpu_sync::rpc::ControlRequest worker_req;
        worker_req.ParseFromString(envelope.payload()) &&
        worker_req.command() ==
            tpu_sync::rpc::ControlRequest::COMMAND_CANCEL_TRANSFER) {
      if (worker_req.cancel_transfer_request().is_sender()) {
        {
          absl::MutexLock lock(mu_);
          sender_cancel_addrs_.insert(std::string(endpoint));
        }
        if (on_sender_cancel_) on_sender_cancel_();
      }
      return AnswerCancel(std::string(endpoint), envelope.request_id(),
                          worker_req.cancel_transfer_request().uuid());
    }
    if (unreachable_addr_ == endpoint) {
      return MarkControlPipeNotSent(
          absl::DeadlineExceededError("injected connect deadline"));
    }
    absl::Notification* hold = nullptr;
    {
      absl::MutexLock lock(mu_);
      calls_.emplace_back(std::string(endpoint), envelope.payload());
      if (hold_addr_ == endpoint) {
        size_t calls_to_endpoint = 0;
        for (const auto& [addr, payload] : calls_) {
          if (addr == endpoint) ++calls_to_endpoint;
        }
        if (calls_to_endpoint > hold_after_) hold = hold_release_;
      }
    }
    if (hold != nullptr) hold->WaitForNotification();
    control_pipe::proto::ControlResponseEnvelope resp_env;
    resp_env.set_request_id(envelope.request_id());
    resp_env.set_status_code(0);
    if (fail_addr_ == endpoint) {
      tpu_sync::rpc::ControlResponse failed;
      failed.set_success(false);
      failed.set_message("injected arm refusal");
      resp_env.set_payload(failed.SerializeAsString());
      return resp_env;
    }
    {
      absl::MutexLock lock(mu_);
      size_t calls_to_endpoint = 0;
      for (const auto& [addr, payload] : calls_) {
        if (addr == endpoint) ++calls_to_endpoint;
      }
      if (silent_after_addr_ == endpoint && calls_to_endpoint > silent_after_) {
        return absl::DeadlineExceededError("injected reply timeout");
      }
      if (fail_after_addr_ == endpoint && calls_to_endpoint > fail_after_) {
        tpu_sync::rpc::ControlResponse failed;
        failed.set_success(false);
        failed.set_message("injected arm refusal");
        resp_env.set_payload(failed.SerializeAsString());
        return resp_env;
      }
      if (busy_addr_ == endpoint && calls_to_endpoint > busy_after_ &&
          busy_remaining_ > 0) {
        --busy_remaining_;
        tpu_sync::rpc::ControlResponse busy;
        busy.set_success(false);
        busy.set_message(
            "pool staging: uuid 1 waited 10s for 2 slots on storage 0 "
            "(free=0 of 4, active leases=2)");
        resp_env.set_payload(busy.SerializeAsString());
        return resp_env;
      }
    }
    auto reply_it = replies_.find(std::string(endpoint));
    if (reply_it != replies_.end()) {
      resp_env.set_payload(reply_it->second.SerializeAsString());
      return resp_env;
    }
    tpu_sync::rpc::ControlResponse ok;
    ok.set_success(true);
    resp_env.set_payload(ok.SerializeAsString());
    return resp_env;
  }

  ControlPipeBackendType backend_type() const override {
    return ControlPipeBackendType::kTcp;
  }

  void FailFor(const std::string& addr) { fail_addr_ = addr; }
  // Refuses every call to `addr` after its first `after_calls` calls.
  void FailAfterCalls(const std::string& addr, size_t after_calls) {
    absl::MutexLock lock(mu_);
    fail_after_addr_ = addr;
    fail_after_ = after_calls;
  }
  // Calls to `addr` after its first `after_calls` calls get no answer.
  void DropRepliesAfterCalls(const std::string& addr, size_t after_calls) {
    absl::MutexLock lock(mu_);
    silent_after_addr_ = addr;
    silent_after_ = after_calls;
  }
  // Answers `times` calls to `addr` after its first `after_calls` calls
  // with the receiver's bounded-staging lease-wait refusal.
  void RefuseArmsBusy(const std::string& addr, size_t after_calls,
                      int times) {
    absl::MutexLock lock(mu_);
    busy_addr_ = addr;
    busy_after_ = after_calls;
    busy_remaining_ = times;
  }
  // Blocks every call to `addr` after its first `after_calls` calls until
  // `release` is notified.
  void HoldAfterCalls(const std::string& addr, size_t after_calls,
                      absl::Notification* release) {
    absl::MutexLock lock(mu_);
    hold_addr_ = addr;
    hold_after_ = after_calls;
    hold_release_ = release;
  }

  // `addr` answers its next `times` cancels with "a push is still in
  // flight". A negative `times` never settles.
  void KeepInFlight(const std::string& addr, int times) {
    absl::MutexLock lock(mu_);
    in_flight_[addr] = times;
  }
  // Every cancel sent to `addr` fails.
  void FailCancelFor(const std::string& addr) { cancel_fail_addr_ = addr; }
  // Calls to `addr` never reach it.
  void MakeUnreachable(const std::string& addr) { unreachable_addr_ = addr; }

  // Plan uuids of the cancels sent to `addr`.
  std::vector<int64_t> CancelsTo(const std::string& addr) {
    absl::MutexLock lock(mu_);
    std::vector<int64_t> uuids;
    for (const auto& [address, uuid] : cancel_calls_) {
      if (address == addr) uuids.push_back(uuid);
    }
    return uuids;
  }

  absl::Mutex mu_;
  std::vector<std::pair<std::string, std::string>> calls_;
  std::string fail_addr_;
  std::string fail_after_addr_;
  size_t fail_after_ = 0;
  std::string silent_after_addr_;
  size_t silent_after_ = 0;
  std::string busy_addr_;
  size_t busy_after_ = 0;
  int busy_remaining_ = 0;
  std::string hold_addr_;
  size_t hold_after_ = 0;
  absl::Notification* hold_release_ = nullptr;
  // Reply returned for an endpoint instead of a bare success.
  std::map<std::string, tpu_sync::rpc::ControlResponse> replies_;
  std::vector<std::pair<std::string, int64_t>> cancel_calls_;
  // Addresses that were asked to cancel a send.
  std::set<std::string> sender_cancel_addrs_;
  // Runs when a sender is asked to cancel, before it answers.
  std::function<void()> on_sender_cancel_;
  std::map<std::string, int> in_flight_;
  std::string cancel_fail_addr_;
  std::string unreachable_addr_;

 private:
  absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> AnswerCancel(
      const std::string& endpoint, uint64_t request_id, int64_t uuid) {
    tpu_sync::rpc::ControlResponse reply;
    {
      absl::MutexLock lock(mu_);
      cancel_calls_.emplace_back(endpoint, uuid);
      if (cancel_fail_addr_ == endpoint) {
        reply.set_success(false);
        reply.set_message("injected cancel failure");
      } else {
        int& in_flight = in_flight_[endpoint];
        reply.set_success(true);
        reply.set_cancel_settled(in_flight == 0);
        if (in_flight > 0) --in_flight;
      }
    }
    if (unreachable_addr_ == endpoint) {
      return MarkControlPipeNotSent(
          absl::DeadlineExceededError("injected connect deadline"));
    }
    control_pipe::proto::ControlResponseEnvelope resp_env;
    resp_env.set_request_id(request_id);
    resp_env.set_status_code(0);
    resp_env.set_payload(reply.SerializeAsString());
    return resp_env;
  }
};

class ReshardStackTest : public ::testing::Test {
 protected:
  ReshardStackTest() {
    ReshardService::Options options;
    options.port = 0;
    options.client = &transport_;
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
                       const std::string& shard_override = "",
                       const std::string& peregrine_endpoint = "") {
    tpu_sync::rpc::ControlRequest req;
    req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
    auto* reg = req.mutable_register_work_unit_request();
    *reg->mutable_unit() = RaidenIdToProto(DstUnit(idx));
    reg->add_shards(shard_override.empty()
                        ? absl::StrCat("10.0.0.2:", 9400 + idx)
                        : shard_override);
    if (!peregrine_endpoint.empty()) {
      reg->add_peregrine_endpoints(peregrine_endpoint);
    }
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
    tpu_sync::rpc::ControllerResponse resp =
        service_->HandleControllerCommand(req);
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
    tpu_sync::rpc::ControllerResponse resp =
        service_->HandleControllerCommand(req);
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

  // One rank's request-global (v1) declaration of strided spans over one
  // source block: a head slice of `size` bytes repeated `count` times per
  // page, routed per destination unit.
  struct StridedSpan {
    int64_t src_offset;
    int64_t dst_global_offset;
    int64_t size;
    int64_t src_stride;
    int64_t dst_stride;
    int32_t count;
    int32_t dst_unit_ordinal;
  };
  tpu_sync::rpc::ControllerResponse RegisterStridedGlobalSpans(
      int rank, const std::string& req_id, int64_t uuid, int64_t src_block,
      const std::vector<StridedSpan>& spans) {
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
    for (const StridedSpan& span : spans) {
      auto* out = entry->add_spans();
      out->set_src_block_ordinal(0);
      out->set_src_offset_bytes(span.src_offset);
      out->set_dst_block_index(0);
      out->set_dst_offset_bytes(span.dst_global_offset);
      out->set_size_bytes(span.size);
      out->set_src_stride_bytes(span.src_stride);
      out->set_dst_stride_bytes(span.dst_stride);
      out->set_count(span.count);
      if (span.dst_unit_ordinal != -1) {
        out->set_dst_unit_ordinal(span.dst_unit_ordinal);
      }
      declared += span.size * span.count;
    }
    entry->set_declared_bytes(declared);
    entry->set_dst_space_version(1);
    return HandleController(req.SerializeAsString());
  }

  // Head-split declarations for RegisterShardedPair(): a source page holds
  // four tokens of two 128-byte heads; destination unit d takes head d of
  // every token as one strided span per page.
  void RegisterHeadSplitPages(const std::string& req_id, int64_t uuid) {
    ASSERT_TRUE(RegisterStridedGlobalSpans(0, req_id, uuid, 3,
                                           {{0, 0, 128, 256, 128, 4, 0},
                                            {128, 0, 128, 256, 128, 4, 1}})
                    .success());
    ASSERT_TRUE(RegisterStridedGlobalSpans(1, req_id, uuid, 5,
                                           {{0, 512, 128, 256, 128, 4, 0},
                                            {128, 512, 128, 256, 128, 4, 1}})
                    .success());
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

  // Pipeline-parallel on both sides: source rank r and destination stage r
  // each register only layer r's pool. Stage r's control address is
  // 10.0.0.2:<9600 + r>.
  static RaidenId PipelineStage(int stage) {
    RaidenId unit = DstUnit();
    unit.job_replica_id = absl::StrCat(unit.job_replica_id, "-rank", stage);
    return unit;
  }

  void RegisterPipelinedUnits(int num_stages) {
    for (int rank = 0; rank < num_stages; ++rank) {
      tpu_sync::rpc::ControlRequest req;
      req.set_command(
          tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
      auto* reg = req.mutable_register_work_unit_request();
      *reg->mutable_unit() = RaidenIdToProto(Unit(rank));
      reg->add_shards(absl::StrCat("10.0.0.1:", 9000 + rank));
      reg->set_control_plane_rpc_address(
          absl::StrCat("10.0.0.1:", 9100 + rank));
      *reg->add_pools() = MakePool(absl::StrCat("fa.l", rank), 1024, 1024, 16);
      reg->set_layout_fingerprint("fp1");
      reg->set_page_tokens(512);
      reg->set_transfer_parallelism(num_stages);
      reg->set_transfer_rank(rank);
      tpu_sync::rpc::ControlResponse resp = Handle(req.SerializeAsString());
      ASSERT_TRUE(resp.success()) << resp.message();
    }
    for (int stage = 0; stage < num_stages; ++stage) {
      tpu_sync::rpc::ControlRequest req;
      req.set_command(
          tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
      auto* reg = req.mutable_register_work_unit_request();
      *reg->mutable_unit() = RaidenIdToProto(PipelineStage(stage));
      reg->add_shards(absl::StrCat("10.0.0.2:", 9400 + stage));
      reg->set_control_plane_rpc_address(
          absl::StrCat("10.0.0.2:", 9600 + stage));
      *reg->add_pools() = MakePool(absl::StrCat("fa.l", stage), 1024, 1024, 16);
      reg->set_layout_fingerprint("fp1");
      reg->set_page_tokens(512);
      reg->set_transfer_parallelism(num_stages);
      reg->set_transfer_rank(stage);
      tpu_sync::rpc::ControlResponse resp = Handle(req.SerializeAsString());
      ASSERT_TRUE(resp.success()) << resp.message();
    }
  }

  // Every source rank registers one block of its layer's pool for the
  // request.
  void RegisterPipelinedSpans(const std::string& req_id, int64_t uuid,
                              int num_stages) {
    for (int rank = 0; rank < num_stages; ++rank) {
      tpu_sync::rpc::ControllerRequest req;
      req.set_command(
          tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS);
      auto* block_req = req.mutable_register_request_blocks_request();
      block_req->set_req_id(req_id);
      block_req->set_uuid(uuid);
      *block_req->mutable_unit() = RaidenIdToProto(Unit(rank));
      block_req->add_block_ids(3 + rank);
      auto* entry = block_req->add_pool_spans();
      entry->set_tag(absl::StrCat("fa.l", rank));
      entry->add_block_ids(3 + rank);
      auto* span = entry->add_spans();
      span->set_src_block_ordinal(0);
      span->set_src_offset_bytes(0);
      span->set_dst_block_index(0);
      span->set_dst_offset_bytes(0);
      span->set_size_bytes(1024);
      span->set_count(1);
      entry->set_declared_bytes(1024);
      entry->set_dst_space_version(1);
      tpu_sync::rpc::ControllerResponse resp =
          HandleController(req.SerializeAsString());
      ASSERT_TRUE(resp.success()) << resp.message();
    }
  }

  // One destination stage plans the request against every source rank,
  // pulling its own layer into block 7.
  tpu_sync::rpc::ControllerResponse CoordinateStage(const std::string& req_id,
                                                    int64_t uuid,
                                                    int num_stages, int stage) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_COORDINATE_TRANSFER);
    auto* coord = req.mutable_coordinate_transfer_request();
    for (int rank = 0; rank < num_stages; ++rank) {
      *coord->add_src_units() = RaidenIdToProto(Unit(rank));
    }
    *coord->add_dst_units() = RaidenIdToProto(PipelineStage(stage));
    coord->set_uuid(uuid);
    coord->set_is_sender(true);
    coord->set_dst_mem_type(tpu_sync::rpc::MEMORY_TYPE_HBM);
    coord->set_use_block_chunks(true);
    coord->set_req_id(req_id);
    coord->add_dst_device_block_ids(7);
    coord->add_transfer_pool_tags(absl::StrCat("fa.l", stage));
    coord->add_dst_block_counts(1);
    return HandleController(req.SerializeAsString());
  }

  tpu_sync::rpc::ControllerResponse CancelIfUnclaimed(const std::string& req_id,
                                                      int64_t uuid) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(tpu_sync::rpc::ControllerRequest::
                        COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED);
    req.mutable_cancel_request_blocks_if_unclaimed_request()->set_req_id(
        req_id);
    req.mutable_cancel_request_blocks_if_unclaimed_request()->set_uuid(uuid);
    return HandleController(req.SerializeAsString());
  }

  // One rank's registration of bare `block_ids` (no byte spans). Returns the
  // controller response so callers can assert on refusals.
  tpu_sync::rpc::ControllerResponse RegisterBlockIds(
      int rank, const std::string& req_id, int64_t uuid,
      const std::vector<int64_t>& block_ids) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS);
    auto* block_req = req.mutable_register_request_blocks_request();
    block_req->set_req_id(req_id);
    block_req->set_uuid(uuid);
    *block_req->mutable_unit() = RaidenIdToProto(Unit(rank));
    for (int64_t block : block_ids) block_req->add_block_ids(block);
    return HandleController(req.SerializeAsString());
  }

  // One rank's completion vote for the request generation.
  tpu_sync::rpc::ControllerResponse CompleteRequestBlocks(
      const std::string& req_id, int64_t uuid, int rank) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_COMPLETE_REQUEST_BLOCKS);
    auto* complete_req = req.mutable_complete_request_blocks_request();
    complete_req->set_req_id(req_id);
    complete_req->set_uuid(uuid);
    *complete_req->mutable_unit() = RaidenIdToProto(Unit(rank));
    return HandleController(req.SerializeAsString());
  }

  tpu_sync::rpc::ControllerResponse ReleaseRequestBlocks(
      const std::string& req_id, int64_t uuid, bool force) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_RELEASE_REQUEST_BLOCKS);
    auto* release_req = req.mutable_release_request_blocks_request();
    release_req->set_req_id(req_id);
    release_req->set_uuid(uuid);
    release_req->set_force(force);
    return HandleController(req.SerializeAsString());
  }

  std::vector<int> RequestBlockStatuses(const std::string& req_id,
                                        int64_t uuid) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_GET_REQUEST_BLOCK_STATUS);
    auto* key = req.mutable_get_request_block_status_request()->add_keys();
    key->set_req_id(req_id);
    key->set_uuid(uuid);
    tpu_sync::rpc::ControllerResponse resp =
        HandleController(req.SerializeAsString());
    EXPECT_TRUE(resp.success()) << resp.message();
    std::vector<int> statuses;
    for (int status : resp.get_request_block_status_response().statuses()) {
      statuses.push_back(status);
    }
    return statuses;
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
      int num_dst = 1, std::vector<int64_t> dst_window = {}) {
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
    for (int64_t window : dst_window) {
      coord->add_dst_window_blocks(window);
    }
    return HandleController(req.SerializeAsString());
  }

  // Waits for the request's chained sub-transfers (the first one runs inside
  // the coordination RPC; the rest on a coordinator thread).
  void WaitForChains() { service_->coordinator()->WaitForSubTransferChains(); }

  // Parses worker call `index` as a START_TRANSFER control request.
  tpu_sync::rpc::StartTransferRequest WorkerStartTransfer(size_t index) {
    tpu_sync::rpc::ControlRequest req;
    EXPECT_TRUE(req.ParseFromString(transport_.calls_.at(index).second));
    return req.start_transfer_request();
  }

  // Rank r declares whole pages 2r and 2r+1 of the destination's four (v1
  // "fa" spans from its source blocks 3+2r and 4+2r), so each rank pushes two
  // (src, dst) block pairs to the destination and the receiver's expected
  // push count depends on the plan's stream count. Plan it with
  // Coordinate(req_id, uuid, 2, {7, 8, 9, 10}).
  void RegisterTwoPagesPerRank(const std::string& req_id, int64_t uuid) {
    RegisterGlobalSpans(0, req_id, uuid, {3, 4},
                        {{/*src_ordinal=*/0, /*src_offset=*/0,
                          /*dst_global_offset=*/0, /*size=*/1024},
                         {1, 0, /*dst_global_offset=*/1024, 1024}});
    RegisterGlobalSpans(1, req_id, uuid, {5, 6},
                        {{0, 0, /*dst_global_offset=*/2048, 1024},
                         {1, 0, /*dst_global_offset=*/3072, 1024}});
  }

  // The planner request the coordinator builds for
  // Coordinate(req_id, uuid, 2, {7, 8, 9, 10}): no explicit stream count.
  absl::StatusOr<PlanRequest> TwoRankPlanRequest(const std::string& req_id,
                                                 int64_t uuid) {
    PlanRequest request;
    request.src_units = {Unit(0), Unit(1)};
    request.dst_units = {DstUnit(0)};
    ABSL_ASSIGN_OR_RETURN(
        request.src_metadata,
        service_->directory()->LocalMetadata(request.src_units));
    ABSL_ASSIGN_OR_RETURN(
        request.dst_metadata,
        service_->directory()->LocalMetadata(request.dst_units));
    request.req_id = req_id;
    request.uuid = uuid;
    request.dst_device_block_ids = {7, 8, 9, 10};
    request.transfer_pool_tags = {"fa"};
    return request;
  }

  FakeTransport transport_;
  double now_ = 1000.0;
  std::unique_ptr<ReshardService> service_;
};

TEST(ReshardControlPipeTest, TcpBackendRoundTrip) {
  ReshardService::Options options;
  options.port = 0;
  options.backend_type = ControlPipeBackendType::kTcp;
  ReshardService service(options);
  ASSERT_TRUE(service.StartServer().ok());

  const std::string addr = absl::StrCat("127.0.0.1:", service.port());
  std::unique_ptr<ControlPipeClient> client =
      CreateReshardControlPipeClient(ControlPipeBackendType::kTcp);
  ReshardClient reshard_client(addr, client.get());

  RegisterWorkUnitArgs args;
  args.unit = Unit(0);
  args.shards = {"10.0.0.1:9000"};
  args.peregrine_endpoints = {"10.0.0.1:19000"};
  args.control_plane_rpc_address = "10.0.0.1:9100";
  ASSERT_TRUE(reshard_client.RegisterWorkUnit(args).ok());

  auto metadata = reshard_client.GetMetadata();
  ASSERT_TRUE(metadata.ok()) << metadata.status();
  EXPECT_EQ(metadata->size(), 1u);

  service.StopServer();
}

TEST(ReshardControlPipeTest, GrpcBackendEnvVarRoundTrip) {
  setenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "grpc", 1);
  auto env_cleanup =
      absl::MakeCleanup([] { unsetenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND"); });
  ReshardService::Options options;
  options.port = 0;
  ReshardService service(options);
  ASSERT_TRUE(service.StartServer().ok());

  const std::string addr = absl::StrCat("127.0.0.1:", service.port());
  ReshardClient reshard_client(addr);

  RegisterWorkUnitArgs args;
  args.unit = Unit(0);
  args.shards = {"10.0.0.1:9000"};
  args.control_plane_rpc_address = "10.0.0.1:9100";
  ASSERT_TRUE(reshard_client.RegisterWorkUnit(args).ok());

  auto metadata = reshard_client.GetMetadata();
  ASSERT_TRUE(metadata.ok()) << metadata.status();
  EXPECT_EQ(metadata->size(), 1u);

  service.StopServer();
}

TEST(ReshardControlPipeTest, ZmqBackendRoundTrip) {
  ReshardService::Options options;
  options.port = 0;
  options.backend_type = ControlPipeBackendType::kZmq;
  ReshardService service(options);
  ASSERT_TRUE(service.StartServer().ok());

  const std::string addr = absl::StrCat("127.0.0.1:", service.port());
  std::unique_ptr<ControlPipeClient> client =
      CreateReshardControlPipeClient(ControlPipeBackendType::kZmq);
  ReshardClient reshard_client(addr, client.get());

  RegisterWorkUnitArgs args;
  args.unit = Unit(0);
  args.shards = {"10.0.0.1:9000"};
  args.control_plane_rpc_address = "10.0.0.1:9100";
  ASSERT_TRUE(reshard_client.RegisterWorkUnit(args).ok());

  service.StopServer();
}

TEST_F(ReshardStackTest, FullPoolReshardFlowEmitsArmThenDispatch) {
  // Receiver address forwarding is opt-in; this test covers the enabled path.
  setenv("TPU_RAIDEN_FORWARD_RECEIVER_ADDRS", "1", 1);
  auto env_cleanup = absl::MakeCleanup(
      [] { unsetenv("TPU_RAIDEN_FORWARD_RECEIVER_ADDRS"); });
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  // Rank 0 covers dst block 0 fully; rank 1 covers a prefix of block 1.
  RegisterSpans(0, "req-1", 42, 1024, /*src_block=*/3, /*dst_index=*/0,
                /*dst_offset=*/0, /*size=*/1024);
  RegisterSpans(1, "req-1", 42, 1024, /*src_block=*/5, /*dst_index=*/1,
                /*dst_offset=*/0, /*size=*/512);
  // The receiver reports its pool 0 addresses in the arm reply.
  tpu_sync::rpc::ControlResponse arm_reply;
  arm_reply.set_success(true);
  tpu_sync::rpc::PoolHostAddrsProto& reply_pool =
      (*arm_reply.mutable_receiver_pool_addrs())[0];
  reply_pool.add_host_base_addrs(uint64_t{0x7f00'0000'0000});
  reply_pool.set_block_stride_bytes(1024);
  reply_pool.set_num_blocks(16);
  transport_.replies_["10.0.0.2:9600"] = arm_reply;
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
  EXPECT_TRUE(arm.start_transfer_request().receiver_addrs().empty());

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
    // Senders get the receiver's pool addresses keyed by its data endpoint.
    const auto& receiver_addrs =
        dispatch.start_transfer_request().receiver_addrs();
    ASSERT_EQ(receiver_addrs.size(), 1);
    ASSERT_TRUE(receiver_addrs.contains("10.0.0.2:9400"));
    const auto& pools = receiver_addrs.at("10.0.0.2:9400").pools();
    ASSERT_TRUE(pools.contains(0));
    ASSERT_EQ(pools.at(0).host_base_addrs_size(), 1);
    EXPECT_EQ(pools.at(0).host_base_addrs(0), uint64_t{0x7f00'0000'0000});
    EXPECT_EQ(pools.at(0).block_stride_bytes(), 1024);
    EXPECT_EQ(pools.at(0).num_blocks(), 16);
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

TEST_F(ReshardStackTest,
       RegisteredDestinationPeregrineEndpointsEmittedOnScheduleEntries) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16, /*num_dst=*/2);
  // Dst 0 advertises Peregrine; Dst 1 is socket-only.
  RegisterDstUnit(/*idx=*/0, /*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                  /*num_blocks=*/16, /*fingerprint=*/"fp1",
                  /*shard_override=*/"",
                  /*peregrine_endpoint=*/"10.0.0.2:19400");

  RegisterSpans(0, "req-pg", 420, 1024, /*src_block=*/3, /*dst_index=*/0,
                /*dst_offset=*/0, /*size=*/1024);
  RegisterSpans(1, "req-pg", 420, 1024, /*src_block=*/5, /*dst_index=*/1,
                /*dst_offset=*/0, /*size=*/512);

  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-pg", 420, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_TRUE(resp.success()) << resp.message();
  ASSERT_EQ(transport_.calls_.size(), 4u);

  for (int i = 2; i < 4; ++i) {
    tpu_sync::rpc::ControlRequest dispatch;
    ASSERT_TRUE(dispatch.ParseFromString(transport_.calls_[i].second));
    const auto& schedule =
        dispatch.start_transfer_request().shard_push_schedules().at(0);
    ASSERT_EQ(schedule.entries_size(), 2);
    EXPECT_EQ(schedule.entries(0).dst_peer(), "10.0.0.2:9400");
    EXPECT_EQ(schedule.entries(0).dst_peregrine_peer(), "10.0.0.2:19400");
    EXPECT_EQ(schedule.entries(1).dst_peer(), "10.0.0.2:9401");
    EXPECT_EQ(schedule.entries(1).dst_peregrine_peer(), "");
  }
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

// ---------------------------------------------------------------------------
// Bounded-staging windows: a request larger than its per-tag window is
// coordinated as a chain of sub-transfers.

TEST(PlanSubTransfersTest, NoWindowsPlanOneTransfer) {
  PoolReshardArgs args;
  args.transfer_pool_tags = {"fa"};
  args.dst_device_block_ids = {7, 8, 9, 10};
  auto chain = PlanSubTransfers(args);
  ASSERT_TRUE(chain.ok()) << chain.status();
  EXPECT_EQ(chain->count, 1);
  EXPECT_TRUE(chain->sub_plans.empty());

  // Zero windows and windows that fit the request plan one transfer too.
  args.dst_window_blocks = {0};
  chain = PlanSubTransfers(args);
  ASSERT_TRUE(chain.ok()) << chain.status();
  EXPECT_EQ(chain->count, 1);
  args.dst_window_blocks = {4};
  chain = PlanSubTransfers(args);
  ASSERT_TRUE(chain.ok()) << chain.status();
  EXPECT_EQ(chain->count, 1);
}

TEST(PlanSubTransfersTest, WindowSplitsPagedTagInDestinationOrder) {
  PoolReshardArgs args;
  args.transfer_pool_tags = {"fa"};
  args.dst_device_block_ids = {7, 8, 9, 10, 11};
  args.dst_window_blocks = {2};
  auto chain = PlanSubTransfers(args);
  ASSERT_TRUE(chain.ok()) << chain.status();
  ASSERT_EQ(chain->count, 3);
  ASSERT_EQ(chain->sub_plans.size(), 3u);
  for (int j = 0; j < 3; ++j) {
    ASSERT_EQ(chain->sub_plans[j].size(), 1u);
    const SubTransferSlice& slice = chain->sub_plans[j][0];
    EXPECT_EQ(slice.tag_index, 0u);
    EXPECT_EQ(slice.offset, 2 * j);
    EXPECT_EQ(slice.count, j == 2 ? 1 : 2);
    EXPECT_EQ(slice.total, 5);
  }
}

TEST(PlanSubTransfersTest, UnwindowedTagsRideInTheLastSubTransfer) {
  // FA pages windowed one per sub-transfer; two single-block state tags are
  // not windowed and must land with the final FA window (order_rank keeps
  // state after FA inside that plan).
  // The two state tags name the same block (a group's conv and ssm pools
  // address one state block), which must not disable the window: only a
  // windowed tag sharing ids would.
  PoolReshardArgs args;
  args.transfer_pool_tags = {"fa", "gdn.conv.g0", "gdn.ssm.g0"};
  args.dst_device_block_ids = {7, 8, 9, 20, 20};
  args.dst_block_counts = {3, 1, 1};
  args.dst_window_blocks = {1, 0, 0};
  auto chain = PlanSubTransfers(args);
  ASSERT_TRUE(chain.ok()) << chain.status();
  ASSERT_EQ(chain->count, 3);
  EXPECT_EQ(chain->sub_plans[0].size(), 1u);
  EXPECT_EQ(chain->sub_plans[1].size(), 1u);
  ASSERT_EQ(chain->sub_plans[2].size(), 3u);
  EXPECT_EQ(chain->sub_plans[2][0].tag_index, 0u);
  EXPECT_EQ(chain->sub_plans[2][0].offset, 2);
  EXPECT_EQ(chain->sub_plans[2][0].count, 1);
  EXPECT_EQ(chain->sub_plans[2][0].total, 3);
  for (size_t i = 1; i < 3; ++i) {
    EXPECT_EQ(chain->sub_plans[2][i].tag_index, i);
    EXPECT_EQ(chain->sub_plans[2][i].offset, 0);
    EXPECT_EQ(chain->sub_plans[2][i].count, 1);
    EXPECT_EQ(chain->sub_plans[2][i].total, 0);
  }
}

TEST(PlanSubTransfersTest, WindowedTagSharingIdsDisablesWindows) {
  // A windowed tag whose ids another tag also names (aliased arena pages)
  // keeps the intra-plan H2D ordering guarantee: the request is never split.
  PoolReshardArgs args;
  args.transfer_pool_tags = {"fa", "state"};
  args.dst_device_block_ids = {7, 8, 9, 7};
  args.dst_block_counts = {3, 1};
  args.dst_window_blocks = {1, 0};
  auto chain = PlanSubTransfers(args);
  ASSERT_TRUE(chain.ok()) << chain.status();
  EXPECT_EQ(chain->count, 1);
  // The same ids shared only between two unwindowed tags are fine.
  args.transfer_pool_tags = {"fa", "conv", "ssm"};
  args.dst_device_block_ids = {7, 8, 9, 11, 11};
  args.dst_block_counts = {3, 1, 1};
  args.dst_window_blocks = {2, 0, 0};
  chain = PlanSubTransfers(args);
  ASSERT_TRUE(chain.ok()) << chain.status();
  EXPECT_EQ(chain->count, 2);
}

TEST(PlanSubTransfersTest, MalformedWindowsFailClosed) {
  PoolReshardArgs args;
  args.transfer_pool_tags = {"fa"};
  args.dst_device_block_ids = {7, 8};
  args.dst_window_blocks = {-1};
  EXPECT_THAT(PlanSubTransfers(args).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("non-negative")));
  args.dst_window_blocks = {1, 1};
  EXPECT_THAT(PlanSubTransfers(args).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("align 1:1")));
}

TEST(SubPlanUuidTest, DerivedUuidsArePositiveAndDistinct) {
  const int64_t uuid = 0x7000'1234'5678'9abc;
  EXPECT_EQ(SubPlanUuid(uuid, 0), uuid);
  std::set<int64_t> seen{uuid};
  for (int32_t j = 1; j < 64; ++j) {
    const int64_t derived = SubPlanUuid(uuid, j);
    EXPECT_GT(derived, 0);
    EXPECT_TRUE(seen.insert(derived).second) << "collision at " << j;
  }
  EXPECT_NE(SubPlanUuid(uuid, 1), SubPlanUuid(uuid + 1, 1));
}

TEST_F(ReshardStackTest, WindowedRequestChainsOneSubTransferPerWindow) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  // Four destination pages, two per rank; a one-page window makes four
  // sub-transfers: arm then dispatch each, in destination order, each with
  // its own engine uuid under one registry claim.
  RegisterTwoPagesPerRank("req-win", 60);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-win", 60, 2, {7, 8, 9, 10}, /*dst_skip=*/{},
                 /*num_dst=*/1, /*dst_window=*/{1});
  ASSERT_TRUE(resp.success()) << resp.message();
  WaitForChains();

  // Per sub-transfer: one arm plus the one sender owning that page.
  ASSERT_EQ(transport_.calls_.size(), 8u);
  const int64_t src_blocks[] = {3, 4, 5, 6};
  const char* sender_addrs[] = {"10.0.0.1:9100", "10.0.0.1:9100",
                                "10.0.0.1:9102", "10.0.0.1:9102"};
  std::set<int64_t> uuids;
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(transport_.calls_[2 * j].first, "10.0.0.2:9600");
    const tpu_sync::rpc::StartTransferRequest arm = WorkerStartTransfer(2 * j);
    EXPECT_FALSE(arm.is_sender());
    EXPECT_EQ(arm.sub_plan_index(), j);
    EXPECT_EQ(arm.sub_plan_count(), 4);
    EXPECT_EQ(arm.uuid(), SubPlanUuid(60, j));
    EXPECT_TRUE(uuids.insert(arm.uuid()).second);
    EXPECT_EQ(arm.req_id(), "req-win");
    EXPECT_EQ(arm.expected_block_count(), 1);
    ASSERT_EQ(arm.pool_groups_size(), 1);
    const auto& group = arm.pool_groups(0);
    ASSERT_EQ(group.dst_device_block_ids_size(), 1);
    EXPECT_EQ(group.dst_device_block_ids(0), 7 + j);
    EXPECT_EQ(group.expected_pushes(), 1);
    ASSERT_EQ(group.dst_expected_extent_bytes_size(), 1);
    EXPECT_EQ(group.dst_expected_extent_bytes(0), 1024);

    EXPECT_EQ(transport_.calls_[2 * j + 1].first, sender_addrs[j]);
    const tpu_sync::rpc::StartTransferRequest dispatch =
        WorkerStartTransfer(2 * j + 1);
    EXPECT_TRUE(dispatch.is_sender());
    EXPECT_EQ(dispatch.uuid(), arm.uuid());
    EXPECT_EQ(dispatch.sub_plan_index(), j);
    EXPECT_EQ(dispatch.sub_plan_count(), 4);
    const auto& schedule = dispatch.shard_push_schedules().at(0);
    ASSERT_EQ(schedule.entries_size(), 1);
    EXPECT_EQ(schedule.entries(0).src_block_id(), src_blocks[j]);
    EXPECT_EQ(schedule.entries(0).src_offset_bytes(), 0);
    EXPECT_EQ(schedule.entries(0).dst_block_id(), 7 + j);
    EXPECT_EQ(schedule.entries(0).dst_offset_bytes(), 0);
    EXPECT_EQ(schedule.entries(0).size_bytes(), 1024);
  }
  EXPECT_EQ(uuids.count(60), 1u);

  // One claim for the whole chain: the registration reads as claimed and
  // cannot be cancelled.
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  EXPECT_EQ(RequestBlockStatuses("req-win", 60),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
  EXPECT_EQ(CancelIfUnclaimed("req-win", 60).response_data(), "false");
}

TEST_F(ReshardStackTest, WindowClipsStraddlingSpansAtWindowBoundaries) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  // Global space [0, 4096) = four destination pages. Rank 0 owns [0, 1536)
  // (block 3 whole, block 4's first half); rank 1 owns [1536, 4096) as three
  // spans, the middle one ([2560, 3584) from block 6) straddling the
  // three-page window boundary at 3072. The first sub-transfer trims that
  // span at the window end and the second resumes it with the source offset
  // advanced past the bytes the first took.
  RegisterGlobalSpans(0, "req-win2", 61, {3, 4},
                      {{/*src_ordinal=*/0, /*src_offset=*/0,
                        /*dst_global_offset=*/0, /*size=*/1024},
                       {1, 0, /*dst_global_offset=*/1024, 512}});
  RegisterGlobalSpans(1, "req-win2", 61, {5, 6, 7},
                      {{0, 0, /*dst_global_offset=*/1536, 1024},
                       {1, 0, /*dst_global_offset=*/2560, 1024},
                       {2, 0, /*dst_global_offset=*/3584, 512}});
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-win2", 61, 2, {7, 8, 9, 10}, /*dst_skip=*/{},
                 /*num_dst=*/1, /*dst_window=*/{3});
  ASSERT_TRUE(resp.success()) << resp.message();
  WaitForChains();

  // Sub-transfer 0: arm + two senders; sub-transfer 1: arm + rank 1 only.
  ASSERT_EQ(transport_.calls_.size(), 5u);
  const tpu_sync::rpc::StartTransferRequest arm0 = WorkerStartTransfer(0);
  EXPECT_EQ(arm0.sub_plan_index(), 0);
  EXPECT_EQ(arm0.sub_plan_count(), 2);
  ASSERT_EQ(arm0.pool_groups_size(), 1);
  EXPECT_EQ(arm0.pool_groups(0).dst_device_block_ids_size(), 3);
  // Two streams per sender (rank 0: 2 pairs, rank 1: 3 pairs, 2 streams).
  EXPECT_EQ(arm0.pool_groups(0).expected_pushes(), 4);
  ASSERT_EQ(arm0.pool_groups(0).dst_expected_extent_bytes_size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(arm0.pool_groups(0).dst_expected_extent_bytes(i), 1024);
  }
  bool saw_rank1_first = false;
  for (size_t i = 1; i <= 2; ++i) {
    if (transport_.calls_[i].first != "10.0.0.1:9102") continue;
    saw_rank1_first = true;
    const tpu_sync::rpc::StartTransferRequest dispatch = WorkerStartTransfer(i);
    const auto& schedule = dispatch.shard_push_schedules().at(0);
    // [1536, 2048) and [2048, 2560) of block 5 (split at the page boundary),
    // then [2560, 3072) of block 6 trimmed at the window end.
    ASSERT_EQ(schedule.entries_size(), 3);
    EXPECT_EQ(schedule.entries(0).src_block_id(), 5);
    EXPECT_EQ(schedule.entries(0).src_offset_bytes(), 0);
    EXPECT_EQ(schedule.entries(0).dst_block_id(), 8);
    EXPECT_EQ(schedule.entries(0).dst_offset_bytes(), 512);
    EXPECT_EQ(schedule.entries(0).size_bytes(), 512);
    EXPECT_EQ(schedule.entries(1).src_block_id(), 5);
    EXPECT_EQ(schedule.entries(1).src_offset_bytes(), 512);
    EXPECT_EQ(schedule.entries(1).dst_block_id(), 9);
    EXPECT_EQ(schedule.entries(1).dst_offset_bytes(), 0);
    EXPECT_EQ(schedule.entries(1).size_bytes(), 512);
    EXPECT_EQ(schedule.entries(2).src_block_id(), 6);
    EXPECT_EQ(schedule.entries(2).src_offset_bytes(), 0);
    EXPECT_EQ(schedule.entries(2).dst_block_id(), 9);
    EXPECT_EQ(schedule.entries(2).dst_offset_bytes(), 512);
    EXPECT_EQ(schedule.entries(2).size_bytes(), 512);
  }
  EXPECT_TRUE(saw_rank1_first);

  const tpu_sync::rpc::StartTransferRequest arm1 = WorkerStartTransfer(3);
  EXPECT_EQ(arm1.sub_plan_index(), 1);
  ASSERT_EQ(arm1.pool_groups_size(), 1);
  ASSERT_EQ(arm1.pool_groups(0).dst_device_block_ids_size(), 1);
  EXPECT_EQ(arm1.pool_groups(0).dst_device_block_ids(0), 10);
  EXPECT_EQ(arm1.pool_groups(0).expected_pushes(), 2);
  EXPECT_EQ(transport_.calls_[4].first, "10.0.0.1:9102");
  const tpu_sync::rpc::StartTransferRequest dispatch1 = WorkerStartTransfer(4);
  const auto& schedule1 = dispatch1.shard_push_schedules().at(0);
  ASSERT_EQ(schedule1.entries_size(), 2);
  // The tail of block 6's span: source offset advanced by the 512 bytes the
  // first window took; then block 7 fills the page's second half.
  EXPECT_EQ(schedule1.entries(0).src_block_id(), 6);
  EXPECT_EQ(schedule1.entries(0).src_offset_bytes(), 512);
  EXPECT_EQ(schedule1.entries(0).dst_block_id(), 10);
  EXPECT_EQ(schedule1.entries(0).dst_offset_bytes(), 0);
  EXPECT_EQ(schedule1.entries(0).size_bytes(), 512);
  EXPECT_EQ(schedule1.entries(1).src_block_id(), 7);
  EXPECT_EQ(schedule1.entries(1).src_offset_bytes(), 0);
  EXPECT_EQ(schedule1.entries(1).dst_block_id(), 10);
  EXPECT_EQ(schedule1.entries(1).dst_offset_bytes(), 512);
  EXPECT_EQ(schedule1.entries(1).size_bytes(), 512);
}

TEST_F(ReshardStackTest, WindowComposesWithPrefixClip) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  // Four declared pages; the consumer already holds page 0 (skip 1024) and
  // pulls the three-page suffix {8, 9, 10} with a two-page window: pages
  // {8, 9} then {10}.
  RegisterTwoPagesPerRank("req-win3", 62);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-win3", 62, 2, {8, 9, 10}, /*dst_skip=*/{1024},
                 /*num_dst=*/1, /*dst_window=*/{2});
  ASSERT_TRUE(resp.success()) << resp.message();
  WaitForChains();

  ASSERT_EQ(transport_.calls_.size(), 5u);
  const tpu_sync::rpc::StartTransferRequest arm0 = WorkerStartTransfer(0);
  ASSERT_EQ(arm0.pool_groups(0).dst_device_block_ids_size(), 2);
  EXPECT_EQ(arm0.pool_groups(0).dst_device_block_ids(0), 8);
  EXPECT_EQ(arm0.pool_groups(0).dst_device_block_ids(1), 9);
  // Rank 0's second page (global page 1 -> dst 8) and rank 1's first (global
  // page 2 -> dst 9).
  std::map<int64_t, int64_t> dst_by_src;
  for (size_t i = 1; i <= 2; ++i) {
    const tpu_sync::rpc::StartTransferRequest dispatch = WorkerStartTransfer(i);
    const auto& schedule = dispatch.shard_push_schedules().at(0);
    ASSERT_EQ(schedule.entries_size(), 1);
    dst_by_src[schedule.entries(0).src_block_id()] =
        schedule.entries(0).dst_block_id();
  }
  EXPECT_EQ(dst_by_src, (std::map<int64_t, int64_t>{{4, 8}, {5, 9}}));
  const tpu_sync::rpc::StartTransferRequest arm1 = WorkerStartTransfer(3);
  ASSERT_EQ(arm1.pool_groups(0).dst_device_block_ids_size(), 1);
  EXPECT_EQ(arm1.pool_groups(0).dst_device_block_ids(0), 10);
  const tpu_sync::rpc::StartTransferRequest dispatch1 = WorkerStartTransfer(4);
  const auto& schedule1 = dispatch1.shard_push_schedules().at(0);
  ASSERT_EQ(schedule1.entries_size(), 1);
  EXPECT_EQ(schedule1.entries(0).src_block_id(), 6);
  EXPECT_EQ(schedule1.entries(0).dst_block_id(), 10);
}

TEST_F(ReshardStackTest, ChainedArmRetriesWhileReceiverStagingIsBusy) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterTwoPagesPerRank("req-win4", 63);
  // The receiver refuses the second sub-transfer's arm twice with the
  // bounded-staging lease-wait message before accepting it.
  transport_.RefuseArmsBusy("10.0.0.2:9600", /*after_calls=*/1, /*times=*/2);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-win4", 63, 2, {7, 8, 9, 10}, /*dst_skip=*/{},
                 /*num_dst=*/1, /*dst_window=*/{2});
  ASSERT_TRUE(resp.success()) << resp.message();
  WaitForChains();

  // Sub-transfer 0 (pages 7, 8 = rank 0's pages): arm + rank 0's dispatch.
  // Sub-transfer 1: 2 refused arms, the accepted arm, then rank 1's
  // dispatch.
  ASSERT_EQ(transport_.calls_.size(), 6u);
  EXPECT_EQ(transport_.calls_[1].first, "10.0.0.1:9100");
  for (size_t i = 2; i <= 4; ++i) {
    EXPECT_EQ(transport_.calls_[i].first, "10.0.0.2:9600");
    EXPECT_EQ(WorkerStartTransfer(i).sub_plan_index(), 1);
  }
  EXPECT_EQ(transport_.calls_[5].first, "10.0.0.1:9102");
  EXPECT_EQ(service_->coordinator()->GetTransferStatus("req-win4"),
            ReshardCoordinator::Status::kCompleted);
}

TEST_F(ReshardStackTest, ChainedRequestStaysInProgressUntilTheChainEnds) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterTwoPagesPerRank("req-win6", 65);
  // Hold the second sub-transfer's arm: the coordination RPC returns after
  // sub-transfer 0, while sub-transfer 1 is still on the chain thread.
  absl::Notification release;
  transport_.HoldAfterCalls("10.0.0.2:9600", /*after_calls=*/1, &release);
  absl::Cleanup release_on_exit = [&release] {
    if (!release.HasBeenNotified()) release.Notify();
  };
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-win6", 65, 2, {7, 8, 9, 10}, /*dst_skip=*/{},
                 /*num_dst=*/1, /*dst_window=*/{2});
  ASSERT_TRUE(resp.success()) << resp.message();
  EXPECT_EQ(service_->coordinator()->GetTransferStatus("req-win6"),
            ReshardCoordinator::Status::kInProgress);

  release.Notify();
  WaitForChains();
  EXPECT_EQ(service_->coordinator()->GetTransferStatus("req-win6"),
            ReshardCoordinator::Status::kCompleted);
}

TEST_F(ReshardStackTest, ChainedArmFailureMarksTheTransferFailed) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterTwoPagesPerRank("req-win5", 64);
  // A hard refusal of the second sub-transfer's arm ends the chain; the
  // first sub-transfer was already dispatched, so the claim stays with it.
  transport_.FailAfterCalls("10.0.0.2:9600", /*after_calls=*/1);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-win5", 64, 2, {7, 8, 9, 10}, /*dst_skip=*/{},
                 /*num_dst=*/1, /*dst_window=*/{2});
  ASSERT_TRUE(resp.success()) << resp.message();
  WaitForChains();

  // Sub-transfer 0: arm + rank 0's dispatch; sub-transfer 1: the refused
  // arm and nothing after it.
  ASSERT_EQ(transport_.calls_.size(), 3u);
  EXPECT_EQ(transport_.calls_[2].first, "10.0.0.2:9600");
  EXPECT_EQ(service_->coordinator()->GetTransferStatus("req-win5"),
            ReshardCoordinator::Status::kFailed);
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  EXPECT_EQ(RequestBlockStatuses("req-win5", 64),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
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

TEST_F(ReshardStackTest, RequestBlockStatusProbeTracksLifecycle) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  auto probe = [&](std::vector<std::pair<std::string, int64_t>> keys) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_GET_REQUEST_BLOCK_STATUS);
    for (const auto& [req_id, uuid] : keys) {
      auto* key = req.mutable_get_request_block_status_request()->add_keys();
      key->set_req_id(req_id);
      key->set_uuid(uuid);
    }
    tpu_sync::rpc::ControllerResponse resp =
        HandleController(req.SerializeAsString());
    EXPECT_TRUE(resp.success()) << resp.message();
    std::vector<int> statuses;
    for (int status : resp.get_request_block_status_response().statuses()) {
      statuses.push_back(status);
    }
    return statuses;
  };
  auto cancel = [&](const std::string& req_id, int64_t uuid) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(tpu_sync::rpc::ControllerRequest::
                        COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED);
    req.mutable_cancel_request_blocks_if_unclaimed_request()->set_req_id(
        req_id);
    req.mutable_cancel_request_blocks_if_unclaimed_request()->set_uuid(uuid);
    return HandleController(req.SerializeAsString());
  };

  // Never registered.
  EXPECT_EQ(probe({{"req-s", 60}}),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});

  // Registered (one rank is enough) and unclaimed.
  RegisterSpans(0, "req-s", 60, 1024, 3, 0, 0, 1024);
  EXPECT_EQ(probe({{"req-s", 60}}),
            std::vector<int>{ProtoStatus::STATUS_REGISTERED});

  // The consumer's release-only cancel: the row is gone, the tombstone
  // reports it — this is what the producer worker polls for.
  tpu_sync::rpc::ControllerResponse cancel_resp = cancel("req-s", 60);
  ASSERT_TRUE(cancel_resp.success());
  EXPECT_EQ(cancel_resp.response_data(), "true");
  EXPECT_EQ(probe({{"req-s", 60}}),
            std::vector<int>{ProtoStatus::STATUS_CANCELLED});

  // Claimed registrations refuse cancellation and read as claimed; the
  // batch keeps key order.
  RegisterSpans(0, "req-c", 61, 1024, 5, 0, 0, 1024);
  RegisterSpans(1, "req-c", 61, 1024, 7, 1, 0, 512);
  ASSERT_TRUE(Coordinate("req-c", 61, 2, {7, 9}).success());
  EXPECT_EQ(probe({{"req-c", 61}, {"req-s", 60}, {"req-none", 1}}),
            (std::vector<int>{ProtoStatus::STATUS_CLAIMED,
                              ProtoStatus::STATUS_CANCELLED,
                              ProtoStatus::STATUS_UNKNOWN}));
  EXPECT_EQ(cancel("req-c", 61).response_data(), "false");

  // Tombstones lapse with the TTL and read as unknown.
  now_ += 601.0;
  EXPECT_EQ(probe({{"req-s", 60}}),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});
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

TEST_F(ReshardStackTest, SubsetSourceManifestsPairPoolsByTag) {
  // Pipeline-parallel source: rank r registers only its own layer's pool
  // (tag fa.l<r>), and the destination registers both layers. Each rank
  // covers the whole request for its layer; the plan pairs pools by tag,
  // hands the receiver destination-canonical pool indices, and rewrites
  // every sender's request into that sender's own pool index space.
  for (int rank = 0; rank < 2; ++rank) {
    tpu_sync::rpc::ControlRequest req;
    req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
    auto* reg = req.mutable_register_work_unit_request();
    *reg->mutable_unit() = RaidenIdToProto(Unit(rank));
    reg->add_shards(absl::StrCat("10.0.0.1:", 9000 + rank));
    reg->set_control_plane_rpc_address(absl::StrCat("10.0.0.1:", 9100 + rank));
    *reg->add_pools() = MakePool(absl::StrCat("fa.l", rank), 1024, 1024, 16);
    reg->set_layout_fingerprint("fp1");
    reg->set_page_tokens(512);
    reg->set_transfer_parallelism(2);
    reg->set_transfer_rank(rank);
    tpu_sync::rpc::ControlResponse resp = Handle(req.SerializeAsString());
    ASSERT_TRUE(resp.success()) << resp.message();
  }
  {
    tpu_sync::rpc::ControlRequest req;
    req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
    auto* reg = req.mutable_register_work_unit_request();
    *reg->mutable_unit() = RaidenIdToProto(DstUnit(0));
    reg->add_shards("10.0.0.2:9400");
    reg->set_control_plane_rpc_address("10.0.0.2:9600");
    *reg->add_pools() = MakePool("fa.l0", 1024, 1024, 16);
    *reg->add_pools() = MakePool("fa.l1", 1024, 1024, 16);
    reg->set_layout_fingerprint("fp1");
    reg->set_page_tokens(512);
    reg->set_transfer_parallelism(2);
    reg->set_transfer_rank(0);
    tpu_sync::rpc::ControlResponse resp = Handle(req.SerializeAsString());
    ASSERT_TRUE(resp.success()) << resp.message();
  }
  for (int rank = 0; rank < 2; ++rank) {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS);
    auto* block_req = req.mutable_register_request_blocks_request();
    block_req->set_req_id("req-pp");
    block_req->set_uuid(77);
    *block_req->mutable_unit() = RaidenIdToProto(Unit(rank));
    block_req->add_block_ids(3 + rank);
    auto* entry = block_req->add_pool_spans();
    entry->set_tag(absl::StrCat("fa.l", rank));
    entry->add_block_ids(3 + rank);
    auto* span = entry->add_spans();
    span->set_src_block_ordinal(0);
    span->set_src_offset_bytes(0);
    span->set_dst_block_index(0);
    span->set_dst_offset_bytes(0);
    span->set_size_bytes(1024);
    span->set_count(1);
    entry->set_declared_bytes(1024);
    entry->set_dst_space_version(1);
    tpu_sync::rpc::ControllerResponse resp =
        HandleController(req.SerializeAsString());
    ASSERT_TRUE(resp.success()) << resp.message();
  }

  tpu_sync::rpc::ControllerRequest req;
  req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_COORDINATE_TRANSFER);
  auto* coord = req.mutable_coordinate_transfer_request();
  *coord->add_src_units() = RaidenIdToProto(Unit(0));
  *coord->add_src_units() = RaidenIdToProto(Unit(1));
  *coord->add_dst_units() = RaidenIdToProto(DstUnit(0));
  coord->set_uuid(77);
  coord->set_is_sender(true);
  coord->set_dst_mem_type(tpu_sync::rpc::MEMORY_TYPE_HBM);
  coord->set_use_block_chunks(true);
  coord->set_req_id("req-pp");
  // Both tags land on the same destination page (one page per layer pool).
  coord->add_dst_device_block_ids(7);
  coord->add_dst_device_block_ids(7);
  coord->add_transfer_pool_tags("fa.l0");
  coord->add_transfer_pool_tags("fa.l1");
  coord->add_dst_block_counts(1);
  coord->add_dst_block_counts(1);
  tpu_sync::rpc::ControllerResponse resp =
      HandleController(req.SerializeAsString());
  ASSERT_TRUE(resp.success()) << resp.message();

  // One arm, two sender dispatches.
  ASSERT_EQ(transport_.calls_.size(), 3u);
  tpu_sync::rpc::ControlRequest arm;
  ASSERT_TRUE(arm.ParseFromString(transport_.calls_[0].second));
  const auto& arm_req = arm.start_transfer_request();
  EXPECT_FALSE(arm_req.is_sender());
  // Receiver: destination-canonical pools 0 and 1, one group per tag, each
  // fed by exactly one sender.
  ASSERT_EQ(arm_req.transfer_pool_indices_size(), 2);
  EXPECT_EQ(arm_req.transfer_pool_indices(0), 0);
  EXPECT_EQ(arm_req.transfer_pool_indices(1), 1);
  ASSERT_EQ(arm_req.pool_dtype_tags_size(), 2);
  ASSERT_EQ(arm_req.pool_groups_size(), 2);
  for (int g = 0; g < 2; ++g) {
    ASSERT_EQ(arm_req.pool_groups(g).pool_indices_size(), 1);
    EXPECT_EQ(arm_req.pool_groups(g).pool_indices(0), g);
    EXPECT_EQ(arm_req.pool_groups(g).expected_pushes(), 1);
  }
  EXPECT_EQ(arm_req.shard_push_schedules_size(), 2);
  EXPECT_EQ(arm_req.wire_pool_indices_size(), 0);

  // Senders: each request is rewritten into the sender's own single-pool
  // index space; the group it does not feed is kept (positionally) but
  // names no local pool.
  for (int i = 1; i <= 2; ++i) {
    tpu_sync::rpc::ControlRequest dispatch;
    ASSERT_TRUE(dispatch.ParseFromString(transport_.calls_[i].second));
    const auto& send_req = dispatch.start_transfer_request();
    EXPECT_TRUE(send_req.is_sender());
    ASSERT_EQ(send_req.transfer_pool_indices_size(), 1);
    EXPECT_EQ(send_req.transfer_pool_indices(0), 0);
    ASSERT_EQ(send_req.pool_dtype_tags_size(), 1);
    ASSERT_EQ(send_req.pool_groups_size(), 2);
    const int rank = transport_.calls_[i].first == "10.0.0.1:9100" ? 0 : 1;
    for (int g = 0; g < 2; ++g) {
      if (g == rank) {
        ASSERT_EQ(send_req.pool_groups(g).pool_indices_size(), 1);
        EXPECT_EQ(send_req.pool_groups(g).pool_indices(0), 0);
      } else {
        EXPECT_EQ(send_req.pool_groups(g).pool_indices_size(), 0);
      }
    }
    ASSERT_EQ(send_req.shard_push_schedules_size(), 1);
    const auto& schedule = send_req.shard_push_schedules().at(0);
    ASSERT_EQ(schedule.entries_size(), 1);
    EXPECT_EQ(schedule.entries(0).pool_group(), rank);
    EXPECT_EQ(schedule.entries(0).src_block_id(), 3 + rank);
    EXPECT_EQ(schedule.entries(0).dst_block_id(), 7);
    // The sender's single local pool is named on the wire by the
    // destination index it feeds.
    ASSERT_EQ(send_req.wire_pool_indices_size(), 1);
    EXPECT_EQ(send_req.wire_pool_indices().at(0), rank);
  }
}

TEST_F(ReshardStackTest, PipelinedDestinationStagesShareOneRequestClaim) {
  // Every destination stage plans the same request against the whole source
  // rank set; the planner keeps the source with its layer, and the request
  // claim is shared by the stages.
  RegisterPipelinedUnits(/*num_stages=*/2);
  RegisterPipelinedSpans("req-pp2", 78, /*num_stages=*/2);
  for (int stage = 0; stage < 2; ++stage) {
    tpu_sync::rpc::ControllerResponse resp =
        CoordinateStage("req-pp2", 78, /*num_stages=*/2, stage);
    ASSERT_TRUE(resp.success()) << "stage " << stage << ": " << resp.message();
  }

  // Per stage: one arm on that stage, one sender on the rank with its layer.
  ASSERT_EQ(transport_.calls_.size(), 4u);
  EXPECT_EQ(transport_.calls_[0].first, "10.0.0.2:9600");
  EXPECT_EQ(transport_.calls_[1].first, "10.0.0.1:9100");
  EXPECT_EQ(transport_.calls_[2].first, "10.0.0.2:9601");
  EXPECT_EQ(transport_.calls_[3].first, "10.0.0.1:9101");
  // Stage 1's arm carries its single source under that source's transfer
  // rank, which is the node id the receiver resolves the pushes by.
  tpu_sync::rpc::ControlRequest arm;
  ASSERT_TRUE(arm.ParseFromString(transport_.calls_[2].second));
  const auto& arm_req = arm.start_transfer_request();
  ASSERT_EQ(arm_req.shard_push_schedules_size(), 1);
  EXPECT_EQ(arm_req.shard_push_schedules().count(1), 1u);
}

TEST_F(ReshardStackTest, AbandonedStageLeavesTheSiblingClaimInPlace) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterPipelinedUnits(/*num_stages=*/2);
  RegisterPipelinedSpans("req-pp3", 79, /*num_stages=*/2);

  // Stage 1's receiver refuses its arm, so stage 1's planning attempt
  // abandons the claim it shares with stage 0.
  transport_.FailFor("10.0.0.2:9601");
  tpu_sync::rpc::ControllerResponse first =
      CoordinateStage("req-pp3", 79, /*num_stages=*/2, /*stage=*/0);
  ASSERT_TRUE(first.success()) << first.message();
  tpu_sync::rpc::ControllerResponse refused =
      CoordinateStage("req-pp3", 79, /*num_stages=*/2, /*stage=*/1);
  ASSERT_FALSE(refused.success());
  EXPECT_THAT(refused.message(), HasSubstr("injected arm refusal"));

  // Stage 0 still holds the claim: the request cannot be cancelled and reads
  // as claimed.
  EXPECT_EQ(CancelIfUnclaimed("req-pp3", 79).response_data(), "false");
  EXPECT_EQ(RequestBlockStatuses("req-pp3", 79),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});

  // Stage 1 rejoins the claim once its receiver accepts the arm.
  transport_.FailFor("");
  tpu_sync::rpc::ControllerResponse retry =
      CoordinateStage("req-pp3", 79, /*num_stages=*/2, /*stage=*/1);
  ASSERT_TRUE(retry.success()) << retry.message();
}

TEST_F(ReshardStackTest, AbandoningTheLastStageFreesTheClaim) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterPipelinedUnits(/*num_stages=*/2);
  RegisterPipelinedSpans("req-pp4", 80, /*num_stages=*/2);

  // Both stages' receivers refuse their arms: every planning attempt
  // abandons the claim, and the last one leaving frees it.
  for (int stage = 0; stage < 2; ++stage) {
    transport_.FailFor(absl::StrCat("10.0.0.2:", 9600 + stage));
    tpu_sync::rpc::ControllerResponse resp =
        CoordinateStage("req-pp4", 80, /*num_stages=*/2, stage);
    ASSERT_FALSE(resp.success()) << "stage " << stage;
    EXPECT_THAT(resp.message(), HasSubstr("injected arm refusal"));
  }
  EXPECT_EQ(RequestBlockStatuses("req-pp4", 80),
            std::vector<int>{ProtoStatus::STATUS_REGISTERED});

  // Nothing holds the claim: the request can be cancelled, and a stage that
  // plans afterwards is refused.
  EXPECT_EQ(CancelIfUnclaimed("req-pp4", 80).response_data(), "true");
  EXPECT_EQ(RequestBlockStatuses("req-pp4", 80),
            std::vector<int>{ProtoStatus::STATUS_CANCELLED});
  transport_.FailFor("");
  tpu_sync::rpc::ControllerResponse late =
      CoordinateStage("req-pp4", 80, /*num_stages=*/2, /*stage=*/0);
  ASSERT_FALSE(late.success());
  EXPECT_THAT(late.message(), HasSubstr("was cancelled"));
}

TEST_F(ReshardStackTest, SpansForAnUnregisteredTagAreRefusedAtRegistration) {
  // A rank that registers no pool for a tag cannot declare spans for it:
  // the registry refuses the declaration before it can reach planning.
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  tpu_sync::rpc::ControllerRequest req;
  req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS);
  auto* block_req = req.mutable_register_request_blocks_request();
  block_req->set_req_id("req-x");
  block_req->set_uuid(43);
  *block_req->mutable_unit() = RaidenIdToProto(Unit(1));
  block_req->add_block_ids(5);
  auto* entry = block_req->add_pool_spans();
  entry->set_tag("fa.other");
  entry->add_block_ids(5);
  auto* span = entry->add_spans();
  span->set_dst_block_index(1);
  span->set_size_bytes(1024);
  span->set_count(1);
  entry->set_declared_bytes(1024);
  tpu_sync::rpc::ControllerResponse resp =
      HandleController(req.SerializeAsString());
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("does not match any registered pool"));
}

TEST_F(ReshardStackTest, SourceTagsAbsentOnTheDestinationAreIgnored) {
  // Pipeline-parallel on both sides: prefill stage 0 registers layers 0
  // and 1, decode stage 0 registers layer 0 only. Planning layer 0 pairs
  // the shared tag and ignores the source's extra layer.
  {
    tpu_sync::rpc::ControlRequest req;
    req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
    auto* reg = req.mutable_register_work_unit_request();
    *reg->mutable_unit() = RaidenIdToProto(Unit(0));
    reg->add_shards("10.0.0.1:9000");
    reg->set_control_plane_rpc_address("10.0.0.1:9100");
    *reg->add_pools() = MakePool("fa.l0", 1024, 1024, 16);
    *reg->add_pools() = MakePool("fa.l1", 1024, 1024, 16);
    reg->set_layout_fingerprint("fp1");
    reg->set_page_tokens(512);
    reg->set_transfer_parallelism(1);
    reg->set_transfer_rank(0);
    tpu_sync::rpc::ControlResponse resp = Handle(req.SerializeAsString());
    ASSERT_TRUE(resp.success()) << resp.message();
  }
  {
    tpu_sync::rpc::ControlRequest req;
    req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
    auto* reg = req.mutable_register_work_unit_request();
    *reg->mutable_unit() = RaidenIdToProto(DstUnit());
    reg->add_shards("10.0.0.2:9400");
    reg->set_control_plane_rpc_address("10.0.0.2:9600");
    *reg->add_pools() = MakePool("fa.l0", 1024, 1024, 16);
    reg->set_layout_fingerprint("fp1");
    reg->set_page_tokens(512);
    reg->set_transfer_parallelism(1);
    reg->set_transfer_rank(0);
    tpu_sync::rpc::ControlResponse resp = Handle(req.SerializeAsString());
    ASSERT_TRUE(resp.success()) << resp.message();
  }
  {
    tpu_sync::rpc::ControllerRequest req;
    req.set_command(
        tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS);
    auto* block_req = req.mutable_register_request_blocks_request();
    block_req->set_req_id("req-pp2pp");
    block_req->set_uuid(78);
    *block_req->mutable_unit() = RaidenIdToProto(Unit(0));
    block_req->add_block_ids(3);
    for (const char* tag : {"fa.l0", "fa.l1"}) {
      auto* entry = block_req->add_pool_spans();
      entry->set_tag(tag);
      entry->add_block_ids(3);
      auto* span = entry->add_spans();
      span->set_size_bytes(1024);
      span->set_count(1);
      entry->set_declared_bytes(1024);
      entry->set_dst_space_version(1);
    }
    tpu_sync::rpc::ControllerResponse resp =
        HandleController(req.SerializeAsString());
    ASSERT_TRUE(resp.success()) << resp.message();
  }
  tpu_sync::rpc::ControllerRequest req;
  req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_COORDINATE_TRANSFER);
  auto* coord = req.mutable_coordinate_transfer_request();
  *coord->add_src_units() = RaidenIdToProto(Unit(0));
  *coord->add_dst_units() = RaidenIdToProto(DstUnit());
  coord->set_uuid(78);
  coord->set_is_sender(true);
  coord->set_dst_mem_type(tpu_sync::rpc::MEMORY_TYPE_HBM);
  coord->set_use_block_chunks(true);
  coord->set_req_id("req-pp2pp");
  coord->add_dst_device_block_ids(7);
  coord->add_transfer_pool_tags("fa.l0");
  tpu_sync::rpc::ControllerResponse resp =
      HandleController(req.SerializeAsString());
  ASSERT_TRUE(resp.success()) << resp.message();
  ASSERT_EQ(transport_.calls_.size(), 2u);
  tpu_sync::rpc::ControlRequest dispatch;
  ASSERT_TRUE(dispatch.ParseFromString(transport_.calls_[1].second));
  const auto& send_req = dispatch.start_transfer_request();
  ASSERT_EQ(send_req.transfer_pool_indices_size(), 1);
  EXPECT_EQ(send_req.transfer_pool_indices(0), 0);
  ASSERT_EQ(send_req.pool_dtype_tags_size(), 2);
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

TEST_F(ReshardStackTest, StridedGlobalSpansStayOneEntryPerDestination) {
  RegisterShardedPair();
  RegisterHeadSplitPages("req-st", 71);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-st", 71, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_TRUE(resp.success()) << resp.message();

  // 2 concurrent arms, then 2 sender dispatches.
  ASSERT_EQ(transport_.calls_.size(), 4u);
  for (int i = 0; i < 2; ++i) {
    tpu_sync::rpc::ControlRequest arm;
    ASSERT_TRUE(arm.ParseFromString(transport_.calls_[i].second));
    const auto& start_req = arm.start_transfer_request();
    EXPECT_FALSE(start_req.is_sender());
    const bool is_dst0 = transport_.calls_[i].first == "10.0.0.2:9600";
    const std::string own_peer = is_dst0 ? "10.0.0.2:9400" : "10.0.0.2:9401";
    const int64_t own_src_offset = is_dst0 ? 0 : 128;
    // Each sender contributes one strided entry: four tokens of this
    // destination's head, not four entries.
    ASSERT_EQ(start_req.shard_push_schedules_size(), 2);
    for (const auto& keyed_schedule : start_req.shard_push_schedules()) {
      ASSERT_EQ(keyed_schedule.second.entries_size(), 1);
      const auto& entry = keyed_schedule.second.entries(0);
      EXPECT_EQ(entry.dst_peer(), own_peer);
      EXPECT_EQ(entry.src_offset_bytes(), own_src_offset);
      EXPECT_EQ(entry.dst_offset_bytes(), 0);
      EXPECT_EQ(entry.size_bytes(), 128);
      EXPECT_EQ(entry.src_stride_bytes(), 256);
      EXPECT_EQ(entry.dst_stride_bytes(), 128);
      EXPECT_EQ(entry.count(), 4);
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
    ASSERT_EQ(start_req.shard_push_schedules_size(), 1);
    const auto& schedule = start_req.shard_push_schedules().at(0);
    ASSERT_EQ(schedule.entries_size(), 2);
    for (const auto& entry : schedule.entries()) {
      const int64_t expected_src =
          entry.dst_peer() == "10.0.0.2:9400" ? 0 : 128;
      EXPECT_EQ(entry.src_offset_bytes(), expected_src);
      EXPECT_EQ(entry.src_stride_bytes(), 256);
      EXPECT_EQ(entry.dst_stride_bytes(), 128);
      EXPECT_EQ(entry.count(), 4);
    }
    EXPECT_NE(schedule.entries(0).dst_peer(), schedule.entries(1).dst_peer());
  }
}

TEST_F(ReshardStackTest, StridedGlobalSpansClipByWholePages) {
  RegisterShardedPair();
  RegisterHeadSplitPages("req-sc", 72);
  // Skipping the first destination page drops rank 0's page whole and
  // re-bases rank 1's page to destination index 0.
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-sc", 72, 2, {9}, /*dst_skip=*/{512}, /*num_dst=*/2);
  ASSERT_TRUE(resp.success()) << resp.message();
  // 2 arms, then the one sender with surviving spans.
  ASSERT_EQ(transport_.calls_.size(), 3u);
  tpu_sync::rpc::ControlRequest dispatch;
  ASSERT_TRUE(dispatch.ParseFromString(transport_.calls_[2].second));
  const auto& start_req = dispatch.start_transfer_request();
  EXPECT_TRUE(start_req.is_sender());
  EXPECT_EQ(transport_.calls_[2].first, "10.0.0.1:9102");
  const auto& schedule = start_req.shard_push_schedules().at(0);
  ASSERT_EQ(schedule.entries_size(), 2);
  for (const auto& entry : schedule.entries()) {
    EXPECT_EQ(entry.src_block_id(), 5);
    EXPECT_EQ(entry.dst_block_id(), 9);
    EXPECT_EQ(entry.dst_offset_bytes(), 0);
    EXPECT_EQ(entry.count(), 4);
  }
  for (int i = 0; i < 2; ++i) {
    tpu_sync::rpc::ControlRequest arm;
    ASSERT_TRUE(arm.ParseFromString(transport_.calls_[i].second));
    const auto& group = arm.start_transfer_request().pool_groups(0);
    EXPECT_EQ(group.expected_pushes(), 1);
    ASSERT_EQ(group.dst_expected_extent_bytes_size(), 1);
    EXPECT_EQ(group.dst_expected_extent_bytes(0), 512);
  }
}

TEST_F(ReshardStackTest, StridedGlobalSpansMustStayInsideOnePage) {
  RegisterShardedPair();
  // Eight repeats of 128 bytes at stride 128 span 1024 bytes: two
  // destination pages.
  tpu_sync::rpc::ControllerResponse resp = RegisterStridedGlobalSpans(
      0, "req-sx", 73, 3, {{0, 0, 128, 128, 128, 8, -1}});
  ASSERT_TRUE(resp.success()) << resp.message();
  resp = Coordinate("req-sx", 73, 1, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  EXPECT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("inside one destination page"));
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

// Sets TPU_RAIDEN_TRANSFER_STREAMS for one test (unsets it when `value` is
// null) and restores it afterwards. Planning reads it on the test thread;
// the coordinator's dispatch threads are joined before Coordinate() returns.
class ScopedTransferStreamsEnv {
 public:
  explicit ScopedTransferStreamsEnv(const char* value) {
    if (const char* prev = std::getenv(kTransferStreamsEnvVar)) {
      previous_ = prev;
    }
    if (value == nullptr) {
      unsetenv(kTransferStreamsEnvVar);
    } else {
      setenv(kTransferStreamsEnvVar, value, 1);
    }
  }
  ~ScopedTransferStreamsEnv() {
    if (previous_.has_value()) {
      setenv(kTransferStreamsEnvVar, previous_->c_str(), 1);
    } else {
      unsetenv(kTransferStreamsEnvVar);
    }
  }

 private:
  std::optional<std::string> previous_;
};

TEST_F(ReshardStackTest, UnsetTransferStreamsPlansOneStreamPerSourceRank) {
  ScopedTransferStreamsEnv env(/*value=*/nullptr);
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterTwoPagesPerRank("req-streams-default", 70);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-streams-default", 70, 2, {7, 8, 9, 10});
  ASSERT_TRUE(resp.success()) << resp.message();

  // One arm plus two sender dispatches, all carrying the admitted stream count.
  ASSERT_EQ(transport_.calls_.size(), 3u);
  for (const auto& [endpoint, payload] : transport_.calls_) {
    tpu_sync::rpc::ControlRequest request;
    ASSERT_TRUE(request.ParseFromString(payload));
    EXPECT_EQ(request.start_transfer_request().parallelism(), 2) << endpoint;
  }
  // Two streams per rank: each rank's two block pairs arrive as two pushes.
  tpu_sync::rpc::ControlRequest arm;
  ASSERT_TRUE(arm.ParseFromString(transport_.calls_[0].second));
  EXPECT_EQ(arm.start_transfer_request().pool_groups(0).expected_pushes(), 4);
}

TEST_F(ReshardStackTest, TransferStreamsEnvCapsPlanStreamsAndExpectedPushes) {
  ScopedTransferStreamsEnv env("1");
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterTwoPagesPerRank("req-streams-env", 71);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-streams-env", 71, 2, {7, 8, 9, 10});
  ASSERT_TRUE(resp.success()) << resp.message();

  // The receiver and both senders take the capped stream count from the plan.
  ASSERT_EQ(transport_.calls_.size(), 3u);
  for (const auto& [endpoint, payload] : transport_.calls_) {
    tpu_sync::rpc::ControlRequest request;
    ASSERT_TRUE(request.ParseFromString(payload));
    EXPECT_EQ(request.start_transfer_request().parallelism(), 1) << endpoint;
  }
  // One stream per rank: each rank's two block pairs arrive as one push.
  tpu_sync::rpc::ControlRequest arm;
  ASSERT_TRUE(arm.ParseFromString(transport_.calls_[0].second));
  EXPECT_EQ(arm.start_transfer_request().pool_groups(0).expected_pushes(), 2);
}

TEST_F(ReshardStackTest, ExplicitPlanParallelismOverridesTransferStreamsEnv) {
  ScopedTransferStreamsEnv env("1");
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterTwoPagesPerRank("req-streams-explicit", 72);
  absl::StatusOr<PlanRequest> request =
      TwoRankPlanRequest("req-streams-explicit", 72);
  ABSL_ASSERT_OK(request);
  request->parallelism = 2;
  const int claim_owner = 0;

  absl::StatusOr<PoolReshardPlan> plan =
      BuildPoolReshardPlan(*request, service_->registry(), &claim_owner);

  ABSL_ASSERT_OK(plan);
  EXPECT_EQ(plan->parallelism, 2);
  EXPECT_EQ(plan->pool_groups[0].expected_pushes_by_dst.at(DstUnit(0)), 4);
}

TEST_F(ReshardStackTest, InvalidTransferStreamsEnvFailsThePlan) {
  ScopedTransferStreamsEnv env("many");
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterTwoPagesPerRank("req-streams-invalid", 73);
  absl::StatusOr<PlanRequest> request =
      TwoRankPlanRequest("req-streams-invalid", 73);
  ABSL_ASSERT_OK(request);
  const int claim_owner = 0;

  EXPECT_THAT(
      BuildPoolReshardPlan(*request, service_->registry(), &claim_owner),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstr("TPU_RAIDEN_TRANSFER_STREAMS"), HasSubstr("many"))));
}

TEST_F(ReshardStackTest,
       DuplicateRegistrationRefreshesTtlWhileConflictsAndPostClaimWritesFail) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-reg-refresh", 80, 1024, /*src_block=*/3,
                /*dst_index=*/0, /*dst_offset=*/0, /*size=*/1024);

  // Identical re-registration refreshes the 600 s TTL.
  now_ += 400.0;
  RegisterSpans(0, "req-reg-refresh", 80, 1024, /*src_block=*/3,
                /*dst_index=*/0, /*dst_offset=*/0, /*size=*/1024);
  now_ += 300.0;
  EXPECT_EQ(RequestBlockStatuses("req-reg-refresh", 80),
            std::vector<int>{ProtoStatus::STATUS_REGISTERED});

  // Conflicting payload for the same (req_id, unit) is rejected.
  tpu_sync::rpc::ControllerResponse conflict_resp =
      RegisterBlockIds(0, "req-reg-refresh", 80, {4});
  EXPECT_FALSE(conflict_resp.success());
  EXPECT_THAT(conflict_resp.message(),
              HasSubstr("Conflicting request block registration"));

  // Once claimed, late or duplicate producer registration is rejected.
  ASSERT_TRUE(Coordinate("req-reg-refresh", 80, 1, {7}).success());
  tpu_sync::rpc::ControllerResponse claimed_resp =
      RegisterBlockIds(0, "req-reg-refresh", 80, {3});
  EXPECT_FALSE(claimed_resp.success());
  EXPECT_THAT(claimed_resp.message(), HasSubstr("already claimed"));
  EXPECT_EQ(RequestBlockStatuses("req-reg-refresh", 80),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
}

TEST_F(ReshardStackTest,
       IncompatibleClaimUnitSetAndNonOwnerAbandonPreserveClaim) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-claim-guard", 81, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-claim-guard", 81, 1024, 5, 1, 0, 1024);

  // Claim owners are compared by address; the values are irrelevant.
  const int owner_a = 0;
  const int owner_b = 0;
  ABSL_ASSERT_OK(service_->registry()->LookupAndClaim(
      "req-claim-guard", 81, {Unit(0), Unit(1)}, &owner_a));

  EXPECT_THAT(
      service_->registry()->LookupAndClaim("req-claim-guard", 81, {Unit(0)},
                                           &owner_b),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("already claimed for a different source unit set")));

  EXPECT_FALSE(
      service_->registry()->AbandonClaim("req-claim-guard", 81, &owner_b));
  EXPECT_EQ(RequestBlockStatuses("req-claim-guard", 81),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});

  EXPECT_TRUE(
      service_->registry()->AbandonClaim("req-claim-guard", 81, &owner_a));
  EXPECT_EQ(RequestBlockStatuses("req-claim-guard", 81),
            std::vector<int>{ProtoStatus::STATUS_REGISTERED});
}

TEST_F(ReshardStackTest, CancelClaimCancelsOnlyWithNoOtherOwnerLeft) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-claim-cancel", 107, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-claim-cancel", 107, 1024, 5, 1, 0, 1024);

  // Claim owners are compared by address; the values are irrelevant.
  const int owner_a = 0;
  const int owner_b = 0;
  ABSL_ASSERT_OK(service_->registry()->LookupAndClaim(
      "req-claim-cancel", 107, {Unit(0), Unit(1)}, &owner_a));
  ABSL_ASSERT_OK(service_->registry()->LookupAndClaim(
      "req-claim-cancel", 107, {Unit(0), Unit(1)}, &owner_b));

  // Another coordination still holds the claim.
  EXPECT_TRUE(
      service_->registry()->ClaimIsShared("req-claim-cancel", 107, &owner_b));
  EXPECT_FALSE(
      service_->registry()->CancelClaim("req-claim-cancel", 107, &owner_b));
  EXPECT_EQ(RequestBlockStatuses("req-claim-cancel", 107),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
  EXPECT_FALSE(
      service_->registry()->ClaimIsShared("req-claim-cancel", 107, &owner_a));

  // The last owner cancels the request.
  EXPECT_TRUE(
      service_->registry()->CancelClaim("req-claim-cancel", 107, &owner_a));
  EXPECT_EQ(RequestBlockStatuses("req-claim-cancel", 107),
            std::vector<int>{ProtoStatus::STATUS_CANCELLED});
  EXPECT_FALSE(
      service_->registry()->CancelClaim("req-claim-cancel", 107, &owner_a));
}

TEST_F(ReshardStackTest, CompletionNoOpsAndPreClaimGuard) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);

  // Unknown generation: idempotent no-op.
  tpu_sync::rpc::ControllerResponse unknown_resp =
      CompleteRequestBlocks("req-comp-unknown", 82, 0);
  ASSERT_TRUE(unknown_resp.success()) << unknown_resp.message();
  EXPECT_EQ(unknown_resp.response_data(), "0");
  EXPECT_EQ(RequestBlockStatuses("req-comp-unknown", 82),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});

  // Registered generation: non-empty registered unit is refused pre-claim,
  // while an unregistered sibling unit is a no-op.
  RegisterSpans(0, "req-comp-guard", 83, 1024, 3, 0, 0, 1024);
  tpu_sync::rpc::ControllerResponse pre_claim =
      CompleteRequestBlocks("req-comp-guard", 83, 0);
  EXPECT_FALSE(pre_claim.success());
  EXPECT_THAT(pre_claim.message(),
              HasSubstr("Only an empty producer registration may complete"));
  tpu_sync::rpc::ControllerResponse unreg_unit =
      CompleteRequestBlocks("req-comp-guard", 83, 1);
  ASSERT_TRUE(unreg_unit.success()) << unreg_unit.message();
  EXPECT_EQ(unreg_unit.response_data(), "0");
  EXPECT_EQ(RequestBlockStatuses("req-comp-guard", 83),
            std::vector<int>{ProtoStatus::STATUS_REGISTERED});

  // Claimed generation: a unit outside the claimed set is a no-op; the
  // claimed unit retires the generation, and repeating after retirement is a
  // no-op.
  ASSERT_TRUE(Coordinate("req-comp-guard", 83, 1, {7}).success());
  tpu_sync::rpc::ControllerResponse unclaimed_unit =
      CompleteRequestBlocks("req-comp-guard", 83, 1);
  ASSERT_TRUE(unclaimed_unit.success()) << unclaimed_unit.message();
  EXPECT_EQ(unclaimed_unit.response_data(), "0");
  EXPECT_EQ(RequestBlockStatuses("req-comp-guard", 83),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});

  tpu_sync::rpc::ControllerResponse retired =
      CompleteRequestBlocks("req-comp-guard", 83, 0);
  ASSERT_TRUE(retired.success()) << retired.message();
  EXPECT_EQ(retired.response_data(), "1");
  EXPECT_EQ(RequestBlockStatuses("req-comp-guard", 83),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});
  tpu_sync::rpc::ControllerResponse after_retire =
      CompleteRequestBlocks("req-comp-guard", 83, 0);
  ASSERT_TRUE(after_retire.success()) << after_retire.message();
  EXPECT_EQ(after_retire.response_data(), "0");
  EXPECT_EQ(RequestBlockStatuses("req-comp-guard", 83),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});

  // Cancelled generation: repeated CancelIfUnclaimed stays true, and Complete
  // is a no-op leaving the tombstone in place.
  EXPECT_EQ(CancelIfUnclaimed("req-comp-cancel", 84).response_data(), "true");
  EXPECT_EQ(CancelIfUnclaimed("req-comp-cancel", 84).response_data(), "true");
  tpu_sync::rpc::ControllerResponse cancelled_comp =
      CompleteRequestBlocks("req-comp-cancel", 84, 0);
  ASSERT_TRUE(cancelled_comp.success()) << cancelled_comp.message();
  EXPECT_EQ(cancelled_comp.response_data(), "0");
  EXPECT_EQ(RequestBlockStatuses("req-comp-cancel", 84),
            std::vector<int>{ProtoStatus::STATUS_CANCELLED});
}

TEST_F(ReshardStackTest,
       EmptyRegistrationCompletesBeforeClaimAndRetiresOnClaim) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);

  ASSERT_TRUE(RegisterBlockIds(0, "req-empty-pre", 85, {}).success());
  EXPECT_EQ(RequestBlockStatuses("req-empty-pre", 85),
            std::vector<int>{ProtoStatus::STATUS_REGISTERED});

  // An empty registration may vote before the claim; the vote is recorded
  // without retiring anything yet.
  tpu_sync::rpc::ControllerResponse comp_resp =
      CompleteRequestBlocks("req-empty-pre", 85, 0);
  ASSERT_TRUE(comp_resp.success()) << comp_resp.message();
  EXPECT_EQ(comp_resp.response_data(), "0");
  EXPECT_EQ(RequestBlockStatuses("req-empty-pre", 85),
            std::vector<int>{ProtoStatus::STATUS_REGISTERED});

  // The claim finds every claimed unit already voted and retires at once.
  const int owner = 0;
  auto claimed = service_->registry()->LookupAndClaim("req-empty-pre", 85,
                                                      {Unit(0)}, &owner);
  ABSL_ASSERT_OK(claimed);
  EXPECT_EQ(claimed->size(), 1u);
  EXPECT_EQ(RequestBlockStatuses("req-empty-pre", 85),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});
}

TEST_F(ReshardStackTest, ForcedReleaseClearsEveryLifecycleState) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);

  tpu_sync::rpc::ControllerResponse unforced =
      ReleaseRequestBlocks("req-rel-u", 86, /*force=*/false);
  EXPECT_FALSE(unforced.success());
  EXPECT_THAT(unforced.message(),
              HasSubstr("Legacy rank-local release is unsafe"));

  // Unknown (fresh and repeated).
  tpu_sync::rpc::ControllerResponse rel_u1 =
      ReleaseRequestBlocks("req-rel-u", 86, /*force=*/true);
  ASSERT_TRUE(rel_u1.success()) << rel_u1.message();
  EXPECT_EQ(rel_u1.response_data(), "0");
  EXPECT_EQ(RequestBlockStatuses("req-rel-u", 86),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});
  tpu_sync::rpc::ControllerResponse rel_u2 =
      ReleaseRequestBlocks("req-rel-u", 86, /*force=*/true);
  ASSERT_TRUE(rel_u2.success()) << rel_u2.message();
  EXPECT_EQ(rel_u2.response_data(), "0");
  EXPECT_EQ(RequestBlockStatuses("req-rel-u", 86),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});

  // Registered.
  RegisterSpans(0, "req-rel-r", 87, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rel-r", 87, 1024, 5, 1, 0, 1024);
  tpu_sync::rpc::ControllerResponse rel_r =
      ReleaseRequestBlocks("req-rel-r", 87, /*force=*/true);
  ASSERT_TRUE(rel_r.success()) << rel_r.message();
  EXPECT_EQ(rel_r.response_data(), "2");
  EXPECT_EQ(RequestBlockStatuses("req-rel-r", 87),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});

  // Claimed, followed by cancel-after-retire and release.
  RegisterSpans(0, "req-rel-c", 88, 1024, 3, 0, 0, 1024);
  ASSERT_TRUE(Coordinate("req-rel-c", 88, 1, {7}).success());
  tpu_sync::rpc::ControllerResponse rel_c =
      ReleaseRequestBlocks("req-rel-c", 88, /*force=*/true);
  ASSERT_TRUE(rel_c.success()) << rel_c.message();
  EXPECT_EQ(rel_c.response_data(), "1");
  EXPECT_EQ(RequestBlockStatuses("req-rel-c", 88),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});
  EXPECT_EQ(CancelIfUnclaimed("req-rel-c", 88).response_data(), "true");
  EXPECT_EQ(RequestBlockStatuses("req-rel-c", 88),
            std::vector<int>{ProtoStatus::STATUS_CANCELLED});
  tpu_sync::rpc::ControllerResponse rel_after_retire_cancel =
      ReleaseRequestBlocks("req-rel-c", 88, /*force=*/true);
  ASSERT_TRUE(rel_after_retire_cancel.success())
      << rel_after_retire_cancel.message();
  EXPECT_EQ(rel_after_retire_cancel.response_data(), "0");
  EXPECT_EQ(RequestBlockStatuses("req-rel-c", 88),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});

  // Cancelled on a fresh generation.
  EXPECT_EQ(CancelIfUnclaimed("req-rel-x", 89).response_data(), "true");
  tpu_sync::rpc::ControllerResponse rel_x =
      ReleaseRequestBlocks("req-rel-x", 89, /*force=*/true);
  ASSERT_TRUE(rel_x.success()) << rel_x.message();
  EXPECT_EQ(rel_x.response_data(), "0");
  EXPECT_EQ(RequestBlockStatuses("req-rel-x", 89),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});
}

TEST_F(ReshardStackTest,
       TwoDestinationsOneArmRefusalAbandonsClaimAndSkipsSenders) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16, /*num_dst=*/2);
  RegisterSpans(0, "req-arm2-fail", 90, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-arm2-fail", 90, 1024, 5, 1, 0, 512);

  // Destination 1 refuses its arm while destination 0 accepts. The arms are
  // joined before the verdict, so one refusal fails the plan, abandons the
  // claim, and dispatches no sender.
  transport_.FailFor("10.0.0.2:9601");
  tpu_sync::rpc::ControllerResponse resp = Coordinate(
      "req-arm2-fail", 90, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("injected arm refusal"));

  // Both arms went out (their order is unspecified); no sender did.
  ASSERT_EQ(transport_.calls_.size(), 2u);
  std::vector<std::string> arm_addresses = {transport_.calls_[0].first,
                                            transport_.calls_[1].first};
  std::sort(arm_addresses.begin(), arm_addresses.end());
  EXPECT_EQ(arm_addresses,
            (std::vector<std::string>{"10.0.0.2:9600", "10.0.0.2:9601"}));
  EXPECT_FALSE(WorkerStartTransfer(0).is_sender());
  EXPECT_FALSE(WorkerStartTransfer(1).is_sender());
  EXPECT_EQ(RequestBlockStatuses("req-arm2-fail", 90),
            std::vector<int>{ProtoStatus::STATUS_REGISTERED});
  EXPECT_EQ(service_->coordinator()->GetTransferStatus("req-arm2-fail"),
            ReshardCoordinator::Status::kFailed);

  // The abandoned claim lets a fresh coordination arm both and dispatch both.
  transport_.FailFor("");
  transport_.calls_.clear();
  tpu_sync::rpc::ControllerResponse retry = Coordinate(
      "req-arm2-fail", 90, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_TRUE(retry.success()) << retry.message();
  EXPECT_EQ(transport_.calls_.size(), 4u);
}

TEST_F(ReshardStackTest, SenderDispatchFailurePropagatesErrorAfterReceiverArm) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-send-fail", 91, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-send-fail", 91, 1024, 5, 1, 0, 512);

  tpu_sync::rpc::ControlResponse sender_failure;
  sender_failure.set_success(false);
  sender_failure.set_message("injected sender failure");
  transport_.replies_["10.0.0.1:9100"] = sender_failure;

  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-send-fail", 91, 2, {7, 9});
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("injected sender failure"));

  // The arm precedes both sender dispatches, which are joined before the
  // first failure is reported.
  ASSERT_EQ(transport_.calls_.size(), 3u);
  EXPECT_EQ(transport_.calls_[0].first, "10.0.0.2:9600");
  EXPECT_FALSE(WorkerStartTransfer(0).is_sender());
  std::vector<std::string> sender_addresses = {transport_.calls_[1].first,
                                               transport_.calls_[2].first};
  std::sort(sender_addresses.begin(), sender_addresses.end());
  EXPECT_EQ(sender_addresses,
            (std::vector<std::string>{"10.0.0.1:9100", "10.0.0.1:9102"}));
  EXPECT_TRUE(WorkerStartTransfer(1).is_sender());
  EXPECT_TRUE(WorkerStartTransfer(2).is_sender());

  // The receiver is already armed, so a sender failure keeps the claim and
  // the request reads as failed.
  EXPECT_EQ(RequestBlockStatuses("req-send-fail", 91),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
  EXPECT_EQ(service_->coordinator()->GetTransferStatus("req-send-fail"),
            ReshardCoordinator::Status::kFailed);
}

TEST_F(ReshardStackTest,
       ClaimRefreshesRegistrationTtlForSiblingStageAndCompletion) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterPipelinedUnits(/*num_stages=*/2);
  RegisterPipelinedSpans("req-claim-ttl", 92, /*num_stages=*/2);

  // Stage 0 claims the request 400 s into the 600 s registration TTL, so the
  // claim and the claimed producer rows now both expire at +1000 s.
  now_ += 400.0;
  tpu_sync::rpc::ControllerResponse first =
      CoordinateStage("req-claim-ttl", 92, /*num_stages=*/2, /*stage=*/0);
  ASSERT_TRUE(first.success()) << first.message();

  // At +700 s (past the registration TTL, within the claim TTL) the producer
  // rows are still live: stage 1 joins the claim, a registration under
  // another uuid for the same (req_id, unit) is still refused, and completion
  // votes retire both rank rows.
  now_ += 300.0;
  EXPECT_EQ(RequestBlockStatuses("req-claim-ttl", 92),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
  EXPECT_EQ(CancelIfUnclaimed("req-claim-ttl", 92).response_data(), "false");

  tpu_sync::rpc::ControllerResponse second =
      CoordinateStage("req-claim-ttl", 92, /*num_stages=*/2, /*stage=*/1);
  ASSERT_TRUE(second.success()) << second.message();

  tpu_sync::rpc::ControllerResponse conflict =
      RegisterBlockIds(/*rank=*/1, "req-claim-ttl", /*uuid=*/93, {4});
  ASSERT_FALSE(conflict.success());
  EXPECT_THAT(conflict.message(),
              HasSubstr("Conflicting request block registration"));

  tpu_sync::rpc::ControllerResponse vote0 =
      CompleteRequestBlocks("req-claim-ttl", 92, /*rank=*/0);
  ASSERT_TRUE(vote0.success()) << vote0.message();
  EXPECT_EQ(vote0.response_data(), "0");

  tpu_sync::rpc::ControllerResponse vote1 =
      CompleteRequestBlocks("req-claim-ttl", 92, /*rank=*/1);
  ASSERT_TRUE(vote1.success()) << vote1.message();
  EXPECT_EQ(vote1.response_data(), "2");
  EXPECT_EQ(RequestBlockStatuses("req-claim-ttl", 92),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});
}

TEST_F(ReshardStackTest, FailedClaimDoesNotRefreshPartialRegistrationTtl) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-partial-ttl", 94, 1024, 3, 0, 0, 1024);

  // Rank 1 has not registered, so a planning attempt at +400 s fails before
  // the claim linearization point and leaves rank 0's 600 s TTL untouched.
  now_ += 400.0;
  tpu_sync::rpc::ControllerResponse poll =
      Coordinate("req-partial-ttl", 94, 2, {7, 9});
  ASSERT_FALSE(poll.success());
  EXPECT_THAT(poll.message(), HasSubstr("Missing producer block registration"));

  // At +601 s rank 0's registration has lapsed on its original TTL.
  now_ += 201.0;
  EXPECT_EQ(RequestBlockStatuses("req-partial-ttl", 94),
            std::vector<int>{ProtoStatus::STATUS_UNKNOWN});
}

// Rollback of a failed coordination (TPU_RAIDEN_ROLLBACK_FAILED_TRANSFERS).

class ReshardRollbackTest : public ReshardStackTest {
 protected:
  ReshardRollbackTest() {
    setenv("TPU_RAIDEN_ROLLBACK_FAILED_TRANSFERS", "1", 1);
  }
  ~ReshardRollbackTest() override {
    unsetenv("TPU_RAIDEN_ROLLBACK_FAILED_TRANSFERS");
  }
};

TEST_F(ReshardStackTest, FailedTransferIsNotReleasedWithoutTheFlag) {
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-off", 95, 1024, 3, 0, 0, 1024);
  transport_.FailFor("10.0.0.2:9600");
  tpu_sync::rpc::ControllerResponse resp = Coordinate("req-rb-off", 95, 1, {7});
  ASSERT_FALSE(resp.success());
  EXPECT_FALSE(resp.transfer_released());
  EXPECT_TRUE(transport_.cancel_calls_.empty());
}

TEST_F(ReshardStackTest, SenderFailureCancelsNothingWithoutTheFlag) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-off-send", 109, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-off-send", 109, 1024, 5, 1, 0, 512);

  // Sender 0 fails to start.
  tpu_sync::rpc::ControlResponse sender_failure;
  sender_failure.set_success(false);
  sender_failure.set_message("injected sender failure");
  transport_.replies_["10.0.0.1:9100"] = sender_failure;

  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-off-send", 109, 2, {7, 9});
  ASSERT_FALSE(resp.success());
  EXPECT_FALSE(resp.transfer_released());
  // No receiver or sender is cancelled, and the claim stays.
  EXPECT_TRUE(transport_.cancel_calls_.empty());
  EXPECT_EQ(RequestBlockStatuses("req-rb-off-send", 109),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
}

TEST_F(ReshardRollbackTest, FailureBeforeAnyPrepareIsReleased) {
  RegisterAllUnits(/*num_src=*/1, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);

  // No producer registration: planning fails before any prepare is sent.
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-plan", 96, 1, {7});
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("Missing producer block registration"));
  EXPECT_TRUE(resp.transfer_released());
  EXPECT_TRUE(transport_.calls_.empty());
  EXPECT_TRUE(transport_.cancel_calls_.empty());
}

TEST_F(ReshardRollbackTest, PrepareFailureCancelsThePreparedReceivers) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16, /*num_dst=*/2);
  RegisterSpans(0, "req-rb-arm", 97, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-arm", 97, 1024, 5, 1, 0, 512);

  // Destination 0 is prepared, destination 1 refuses.
  transport_.FailFor("10.0.0.2:9601");
  tpu_sync::rpc::ControllerResponse resp = Coordinate(
      "req-rb-arm", 97, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("injected arm refusal"));
  EXPECT_TRUE(resp.transfer_released());

  // Both receivers got one cancel for the plan. No sender was contacted.
  EXPECT_EQ(transport_.CancelsTo("10.0.0.2:9600"), std::vector<int64_t>{97});
  EXPECT_EQ(transport_.CancelsTo("10.0.0.2:9601"), std::vector<int64_t>{97});
  EXPECT_EQ(transport_.calls_.size(), 2u);
  // The claim is abandoned, as without the flag.
  EXPECT_EQ(RequestBlockStatuses("req-rb-arm", 97),
            std::vector<int>{ProtoStatus::STATUS_REGISTERED});
}

TEST_F(ReshardRollbackTest, ReceiverThePrepareNeverReachedIsNotCancelled) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16, /*num_dst=*/2);
  RegisterSpans(0, "req-rb-unsent", 98, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-unsent", 98, 1024, 5, 1, 0, 512);

  transport_.MakeUnreachable("10.0.0.2:9601");
  tpu_sync::rpc::ControllerResponse resp = Coordinate(
      "req-rb-unsent", 98, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(resp.success());
  EXPECT_TRUE(resp.transfer_released());
  EXPECT_EQ(transport_.CancelsTo("10.0.0.2:9600").size(), 1u);
  EXPECT_TRUE(transport_.CancelsTo("10.0.0.2:9601").empty());
}

TEST_F(ReshardRollbackTest, FailedCancelIsNotReleased) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16, /*num_dst=*/2);
  RegisterSpans(0, "req-rb-unconfirmed", 99, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-unconfirmed", 99, 1024, 5, 1, 0, 512);

  // Destination 0 is prepared but its cancel fails: it may still be prepared.
  transport_.FailFor("10.0.0.2:9601");
  transport_.FailCancelFor("10.0.0.2:9600");
  tpu_sync::rpc::ControllerResponse resp = Coordinate(
      "req-rb-unconfirmed", 99, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("injected arm refusal"));
  EXPECT_FALSE(resp.transfer_released());
}

TEST_F(ReshardRollbackTest, SecondCoordinationLeavesTheFirstOnesReceive) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16, /*num_dst=*/2);
  RegisterSpans(0, "req-rb-twice", 104, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-twice", 104, 1024, 5, 1, 0, 512);

  // The first coordination succeeds and keeps its claim.
  ASSERT_TRUE(Coordinate("req-rb-twice", 104, 2, {7, 9}, /*dst_skip=*/{},
                         /*num_dst=*/2)
                  .success());
  ASSERT_TRUE(transport_.cancel_calls_.empty());

  // Coordinated again: destination 1 refuses (the first receive is there).
  transport_.FailAfterCalls("10.0.0.2:9601", 1);
  tpu_sync::rpc::ControllerResponse resp = Coordinate(
      "req-rb-twice", 104, 2, {7, 9}, /*dst_skip=*/{}, /*num_dst=*/2);
  ASSERT_FALSE(resp.success());

  // The first transfer may be running: no cancel there, and not released.
  EXPECT_TRUE(transport_.CancelsTo("10.0.0.2:9601").empty());
  EXPECT_FALSE(resp.transfer_released());
  // What this coordination prepared itself is cancelled.
  EXPECT_EQ(transport_.CancelsTo("10.0.0.2:9600"), std::vector<int64_t>{104});
  // The first coordination's claim stays.
  EXPECT_EQ(RequestBlockStatuses("req-rb-twice", 104),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
}

TEST_F(ReshardRollbackTest, SenderFailureCancelsTheReceivers) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-send", 100, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-send", 100, 1024, 5, 1, 0, 512);

  // Sender 0 fails to start. Sender 1 started: it and the receiver have a
  // push in flight when they are first cancelled.
  tpu_sync::rpc::ControlResponse sender_failure;
  sender_failure.set_success(false);
  sender_failure.set_message("injected sender failure");
  transport_.replies_["10.0.0.1:9100"] = sender_failure;
  transport_.KeepInFlight("10.0.0.1:9102", 1);
  transport_.KeepInFlight("10.0.0.2:9600", 1);

  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-send", 100, 2, {7, 9});
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("injected sender failure"));
  EXPECT_TRUE(resp.transfer_released());

  // The receiver and both senders were cancelled, the busy ones asked again
  // until their push had ended.
  EXPECT_EQ(transport_.CancelsTo("10.0.0.2:9600"),
            (std::vector<int64_t>{100, 100}));
  EXPECT_EQ(transport_.CancelsTo("10.0.0.1:9100"), std::vector<int64_t>{100});
  EXPECT_EQ(transport_.CancelsTo("10.0.0.1:9102"),
            (std::vector<int64_t>{100, 100}));
  EXPECT_EQ(transport_.sender_cancel_addrs_,
            (std::set<std::string>{"10.0.0.1:9100", "10.0.0.1:9102"}));
  // Every sender stopped, so the claim is cancelled and a stopped sender's
  // completion vote is a no-op.
  EXPECT_EQ(RequestBlockStatuses("req-rb-send", 100),
            std::vector<int>{ProtoStatus::STATUS_CANCELLED});
  tpu_sync::rpc::ControllerResponse vote =
      CompleteRequestBlocks("req-rb-send", 100, /*rank=*/1);
  ASSERT_TRUE(vote.success()) << vote.message();
  EXPECT_EQ(vote.response_data(), "0");
}

TEST_F(ReshardRollbackTest, ClaimStaysWhileASenderIsStillCopying) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-copying", 102, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-copying", 102, 1024, 5, 1, 0, 512);

  // Sender 0 fails to start; sender 1 never stops within the rollback timeout.
  tpu_sync::rpc::ControlResponse sender_failure;
  sender_failure.set_success(false);
  sender_failure.set_message("injected sender failure");
  transport_.replies_["10.0.0.1:9100"] = sender_failure;
  transport_.KeepInFlight("10.0.0.1:9102", -1);

  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-copying", 102, 2, {7, 9});
  ASSERT_FALSE(resp.success());
  EXPECT_TRUE(resp.transfer_released());
  EXPECT_GT(transport_.CancelsTo("10.0.0.1:9102").size(), 1u);
  EXPECT_EQ(RequestBlockStatuses("req-rb-copying", 102),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
}

TEST_F(ReshardRollbackTest, SenderTheStartNeverReachedIsNotCancelled) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-unstarted", 105, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-unstarted", 105, 1024, 5, 1, 0, 512);

  // Sender 0 cannot be reached, so its start was never sent. Sender 1 started.
  transport_.MakeUnreachable("10.0.0.1:9100");
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-unstarted", 105, 2, {7, 9});
  ASSERT_FALSE(resp.success());
  EXPECT_TRUE(resp.transfer_released());

  // Only the started sender is asked to stop, and the claim is cancelled.
  EXPECT_TRUE(transport_.CancelsTo("10.0.0.1:9100").empty());
  EXPECT_EQ(transport_.CancelsTo("10.0.0.1:9102"), std::vector<int64_t>{105});
  EXPECT_EQ(RequestBlockStatuses("req-rb-unstarted", 105),
            std::vector<int>{ProtoStatus::STATUS_CANCELLED});
}

TEST_F(ReshardRollbackTest, SecondCoordinationLeavesTheFirstOnesSender) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-twice-send", 106, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-twice-send", 106, 1024, 5, 1, 0, 512);

  // The first coordination succeeds and keeps its claim.
  ASSERT_TRUE(Coordinate("req-rb-twice-send", 106, 2, {7, 9}).success());

  // Coordinated again: sender 0 refuses (the first send is still there).
  tpu_sync::rpc::ControlResponse refusal;
  refusal.set_success(false);
  refusal.set_message("pool reshard send UUID already active: 106");
  transport_.replies_["10.0.0.1:9100"] = refusal;
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-twice-send", 106, 2, {7, 9});
  ASSERT_FALSE(resp.success());

  // The first transfer's sender is left alone. The sender this coordination
  // started is stopped.
  EXPECT_TRUE(transport_.CancelsTo("10.0.0.1:9100").empty());
  EXPECT_EQ(transport_.CancelsTo("10.0.0.1:9102"), std::vector<int64_t>{106});
  // The first coordination's claim stays.
  EXPECT_EQ(RequestBlockStatuses("req-rb-twice-send", 106),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
}

TEST_F(ReshardRollbackTest, SenderThatDidNotAnswerIsAskedToStop) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-silent", 110, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-silent", 110, 1024, 5, 1, 0, 512);

  // The start to sender 0 gets no answer, so it may have started.
  transport_.DropRepliesAfterCalls("10.0.0.1:9100", 0);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-silent", 110, 2, {7, 9});
  ASSERT_FALSE(resp.success());

  // Both senders are asked to stop, and the claim is cancelled.
  EXPECT_EQ(transport_.CancelsTo("10.0.0.1:9100"), std::vector<int64_t>{110});
  EXPECT_EQ(transport_.CancelsTo("10.0.0.1:9102"), std::vector<int64_t>{110});
  EXPECT_EQ(RequestBlockStatuses("req-rb-silent", 110),
            std::vector<int>{ProtoStatus::STATUS_CANCELLED});
}

TEST_F(ReshardRollbackTest, SecondCoordinationLeavesASenderThatDidNotAnswer) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-twice-silent", 111, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-twice-silent", 111, 1024, 5, 1, 0, 512);

  // The first coordination succeeds and keeps its claim.
  ASSERT_TRUE(Coordinate("req-rb-twice-silent", 111, 2, {7, 9}).success());

  // Coordinated again: the start to sender 0 gets no answer. The first send
  // may still run there.
  transport_.DropRepliesAfterCalls("10.0.0.1:9100", 1);
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-twice-silent", 111, 2, {7, 9});
  ASSERT_FALSE(resp.success());

  // That sender is left alone. The sender that accepted this coordination's
  // start is stopped.
  EXPECT_TRUE(transport_.CancelsTo("10.0.0.1:9100").empty());
  EXPECT_EQ(transport_.CancelsTo("10.0.0.1:9102"), std::vector<int64_t>{111});
  // The first coordination's claim stays.
  EXPECT_EQ(RequestBlockStatuses("req-rb-twice-silent", 111),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
}

TEST_F(ReshardRollbackTest, ClaimStaysWhenTheOtherCoordinationDropsOut) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-twice-drop", 112, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-twice-drop", 112, 1024, 5, 1, 0, 512);

  // Another coordination holds the claim too. Owners are compared by address.
  const int other = 0;
  ABSL_ASSERT_OK(service_->registry()->LookupAndClaim(
      "req-rb-twice-drop", 112, {Unit(0), Unit(1)}, &other));

  // Sender 0 refuses, so it is not asked to stop. The other coordination
  // drops out while sender 1 is being stopped.
  tpu_sync::rpc::ControlResponse refusal;
  refusal.set_success(false);
  refusal.set_message("pool reshard send UUID already active: 112");
  transport_.replies_["10.0.0.1:9100"] = refusal;
  transport_.on_sender_cancel_ = [this, &other]() {
    service_->registry()->AbandonClaim("req-rb-twice-drop", 112, &other);
  };
  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-twice-drop", 112, 2, {7, 9});
  ASSERT_FALSE(resp.success());

  // Sender 0 may still run a send, so the claim must stay.
  EXPECT_TRUE(transport_.CancelsTo("10.0.0.1:9100").empty());
  EXPECT_EQ(transport_.CancelsTo("10.0.0.1:9102"), std::vector<int64_t>{112});
  EXPECT_EQ(RequestBlockStatuses("req-rb-twice-drop", 112),
            std::vector<int>{ProtoStatus::STATUS_CLAIMED});
}

TEST_F(ReshardRollbackTest, ControllerModeStopsSendersAndCancelsTheClaim) {
  using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
  // Senders are started through a controller. worker_0 has no transfer
  // manager, so it refuses a start. worker_1 is not registered, so its start
  // gets no answer from a worker.
  std::unique_ptr<controller::TestWorkerServer> worker =
      controller::CreateTestWorkerServer();
  tpu_sync::rpc::RaidenIdProto controller_unit;
  controller_unit.set_job_name("prefill");
  controller_unit.set_job_replica_id("0");
  controller_unit.set_data_name("kv_cache");
  auto raiden_controller = controller::RaidenController::Create(
      controller_unit, /*num_blocks=*/1, /*num_shards=*/1,
      /*shard_size_bytes=*/512, /*raiden_controller_address=*/"",
      /*preprovision_worker_buffers=*/false);
  ABSL_ASSERT_OK(raiden_controller);
  ABSL_ASSERT_OK(core::controller::RaidenControllerClient(
                     (*raiden_controller)->controller_address())
                     .RegisterWorker(
                         "worker_0", worker->server_address,
                         {RaidenTransferEndpoint{worker->server_address, {}}}));

  ReshardService::Options options;
  options.port = 0;
  options.client = &transport_;
  options.clock = [this]() { return now_; };
  options.delivery.mode = WorkerDelivery::Mode::kController;
  options.delivery.controller = raiden_controller->get();
  service_ = std::make_unique<ReshardService>(options);
  // The service borrows the controller, so it goes first.
  absl::Cleanup drop_service = [this]() { service_.reset(); };

  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-controller", 108, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-controller", 108, 1024, 5, 1, 0, 512);

  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-controller", 108, 2, {7, 9});
  ASSERT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("Transfer manager is not initialized"));
  EXPECT_TRUE(resp.transfer_released());

  // Both starts may have reached a worker, so both senders are asked to stop,
  // and the claim is cancelled.
  EXPECT_EQ(transport_.CancelsTo("10.0.0.1:9100"), std::vector<int64_t>{108});
  EXPECT_EQ(transport_.CancelsTo("10.0.0.1:9102"), std::vector<int64_t>{108});
  EXPECT_EQ(RequestBlockStatuses("req-rb-controller", 108),
            std::vector<int>{ProtoStatus::STATUS_CANCELLED});
}

TEST_F(ReshardRollbackTest, ReceiverStillInFlightIsNotReleased) {
  RegisterAllUnits(/*num_src=*/2, /*live=*/1024, /*stride=*/1024,
                   /*num_blocks=*/16);
  RegisterSpans(0, "req-rb-drain", 101, 1024, 3, 0, 0, 1024);
  RegisterSpans(1, "req-rb-drain", 101, 1024, 5, 1, 0, 512);

  // The receiver's push never ends within the rollback timeout.
  tpu_sync::rpc::ControlResponse sender_failure;
  sender_failure.set_success(false);
  sender_failure.set_message("injected sender failure");
  transport_.replies_["10.0.0.1:9100"] = sender_failure;
  transport_.KeepInFlight("10.0.0.2:9600", -1);

  tpu_sync::rpc::ControllerResponse resp =
      Coordinate("req-rb-drain", 101, 2, {7, 9});
  ASSERT_FALSE(resp.success());
  EXPECT_FALSE(resp.transfer_released());
  EXPECT_GT(transport_.CancelsTo("10.0.0.2:9600").size(), 1u);
}

TEST(ReshardControlPipeTest, ClientSeesAReleasedTransfer) {
  for (const bool rollback : {false, true}) {
    if (rollback) setenv("TPU_RAIDEN_ROLLBACK_FAILED_TRANSFERS", "1", 1);
    auto env_cleanup = absl::MakeCleanup(
        [] { unsetenv("TPU_RAIDEN_ROLLBACK_FAILED_TRANSFERS"); });
    ReshardService::Options options;
    options.port = 0;
    options.backend_type = ControlPipeBackendType::kTcp;
    ReshardService service(options);
    ASSERT_TRUE(service.StartServer().ok());

    const std::string addr = absl::StrCat("127.0.0.1:", service.port());
    std::unique_ptr<ControlPipeClient> client =
        CreateReshardControlPipeClient(ControlPipeBackendType::kTcp);
    ReshardClient reshard_client(addr, client.get());

    // No unit is registered, so the call fails before any prepare.
    StartTransferArgs args;
    args.src_units = {Unit(0)};
    args.dst_units = {DstUnit()};
    args.req_id = "req-client";
    args.uuid = 7;
    args.use_block_chunks = true;
    args.dst_mem_type = tpu_sync::rpc::MEMORY_TYPE_HBM;
    args.has_dst_device_block_ids = true;
    args.dst_device_block_ids = {7};
    args.has_transfer_pool_tags = true;
    args.transfer_pool_tags = {"fa"};
    absl::StatusOr<bool> started = reshard_client.StartTransfer(args);
    ASSERT_FALSE(started.ok());
    EXPECT_THAT(
        started.status().message(),
        ::testing::StartsWith("Remote Controller Server execution failed: "));
    EXPECT_EQ(IsTransferReleased(started.status()), rollback);

    service.StopServer();
  }
}

}  // namespace
}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
