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

#include "tpu_sync/core/transfer_program_reshard.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "tpu_sync/proto/transfer_program.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace core {

namespace {

using ::testing::HasSubstr;

std::string Canonical(const ::tpu_sync::rpc::StartTransferRequest& msg) {
  std::string out;
  google::protobuf::io::StringOutputStream stream(&out);
  google::protobuf::io::CodedOutputStream coded(&stream);
  coded.SetSerializationDeterministic(true);
  EXPECT_TRUE(msg.SerializeToCodedStream(&coded));
  coded.Trim();
  return out;
}

::tpu_sync::rpc::ShardPushEntryProto MakeEntry(int32_t src_block,
                                               int32_t dst_block,
                                               int64_t src_off, int64_t dst_off,
                                               int64_t size) {
  ::tpu_sync::rpc::ShardPushEntryProto entry;
  entry.set_dst_peer("10.0.0.2:14579");
  entry.set_dst_shard_idx(1);
  entry.set_dst_offset_bytes(dst_off);
  entry.set_src_offset_bytes(src_off);
  entry.set_size_bytes(size);
  entry.set_src_block_id(src_block);
  entry.set_dst_block_id(dst_block);
  entry.set_src_stride_bytes(4096);
  entry.set_dst_stride_bytes(8192);
  entry.set_count(3);
  return entry;
}

// A representative multi-rank pool request exercising every
// presence-sensitive field: skip_d2h present-false, per-entry layer_idx
// present/absent, pool_group present/absent, empty skip_tiling.
::tpu_sync::rpc::StartTransferRequest MakePoolRequest(bool is_sender) {
  ::tpu_sync::rpc::StartTransferRequest req;
  auto* src_unit = req.add_src_units();
  src_unit->set_job_name("p");
  src_unit->set_job_replica_id("0");
  src_unit->set_data_name("kv");
  auto* dst_unit = req.add_dst_units();
  dst_unit->set_job_name("d");
  dst_unit->set_job_replica_id("3");
  dst_unit->set_data_name("kv");
  dst_unit->set_data_replica_idx(3);
  req.set_uuid(int64_t{0x7fee'dd00'1234'5678});
  req.set_is_sender(is_sender);
  req.set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_HBM);
  req.set_use_block_chunks(true);
  req.set_expected_block_count(10);
  req.set_req_id("req-roundtrip-0");
  req.add_pool_dtype_tags("float8_e4m3fn");
  req.add_pool_dtype_tags("bfloat16");
  req.add_transfer_pool_indices(0);
  req.add_transfer_pool_indices(4);
  req.set_parallelism(8);
  req.set_skip_d2h(false);  // present-false must survive the round trip

  auto* group0 = req.add_pool_groups();
  group0->add_pool_indices(0);
  group0->add_dst_device_block_ids(11);
  group0->add_dst_device_block_ids(12);
  group0->set_expected_pushes(8);
  group0->add_dst_expected_extent_bytes(1 << 20);
  group0->add_dst_expected_extent_bytes(1 << 19);
  group0->set_order_rank(0);
  auto* group1 = req.add_pool_groups();
  group1->add_pool_indices(4);
  group1->add_dst_device_block_ids(3);
  group1->set_expected_pushes(2);
  group1->add_dst_expected_extent_bytes(49152);
  group1->set_order_rank(1);

  // Rank 0: two entries, one with layer_idx+pool_group, one bare.
  auto& schedule0 = (*req.mutable_shard_push_schedules())[0];
  {
    ::tpu_sync::rpc::ShardPushEntryProto entry = MakeEntry(5, 11, 0, 0, 65536);
    entry.set_layer_idx(7);
    entry.set_pool_group(0);
    *schedule0.add_entries() = entry;
    *schedule0.add_entries() = MakeEntry(6, 12, 128, 256, 4096);
  }
  if (!is_sender) {
    // Receiver arm carries every source rank's schedule.
    auto& schedule3 = (*req.mutable_shard_push_schedules())[3];
    ::tpu_sync::rpc::ShardPushEntryProto entry = MakeEntry(9, 3, 64, 32, 49152);
    entry.set_pool_group(1);
    *schedule3.add_entries() = entry;
  }
  return req;
}

TEST(TransferProgramReshard, SenderRoundTripsByteIdentical) {
  const ::tpu_sync::rpc::StartTransferRequest original = MakePoolRequest(true);
  auto program = CompileStartTransfer(original);
  ASSERT_TRUE(program.ok()) << program.status();
  auto lowering_class = NormalizeReshardProgram(*program);
  ASSERT_TRUE(lowering_class.ok()) << lowering_class.status();
  EXPECT_EQ(*lowering_class, ReshardLoweringClass::kPoolReshardSender);
  auto lowered = LowerToStartTransfer(*program);
  ASSERT_TRUE(lowered.ok()) << lowered.status();
  EXPECT_EQ(Canonical(original), Canonical(*lowered));
  EXPECT_EQ(DeriveSenderSourceBlockIds(original),
            DeriveSenderSourceBlockIds(*lowered));
}

TEST(TransferProgramReshard, ArmRoundTripsByteIdentical) {
  const ::tpu_sync::rpc::StartTransferRequest original = MakePoolRequest(false);
  auto program = CompileStartTransfer(original);
  ASSERT_TRUE(program.ok()) << program.status();
  auto lowering_class = NormalizeReshardProgram(*program);
  ASSERT_TRUE(lowering_class.ok()) << lowering_class.status();
  EXPECT_EQ(*lowering_class, ReshardLoweringClass::kPoolReshardArm);
  auto lowered = LowerToStartTransfer(*program);
  ASSERT_TRUE(lowered.ok()) << lowered.status();
  EXPECT_EQ(Canonical(original), Canonical(*lowered));
  EXPECT_EQ(DeriveArmChipBlockIds(original),
            DeriveArmChipBlockIds(*lowered));
}

TEST(TransferProgramReshard, SkipTilingMapRoundTrips) {
  ::tpu_sync::rpc::StartTransferRequest original = MakePoolRequest(true);
  (*original.mutable_skip_tiling())[2] = true;
  (*original.mutable_skip_tiling())[5] = false;
  auto program = CompileStartTransfer(original);
  ASSERT_TRUE(program.ok()) << program.status();
  auto lowered = LowerToStartTransfer(*program);
  ASSERT_TRUE(lowered.ok()) << lowered.status();
  EXPECT_EQ(Canonical(original), Canonical(*lowered));
}

TEST(TransferProgramReshard, WirePoolIndexMapRoundTrips) {
  ::tpu_sync::rpc::StartTransferRequest original = MakePoolRequest(true);
  (*original.mutable_wire_pool_indices())[0] = 9;
  (*original.mutable_wire_pool_indices())[1] = 10;
  auto program = CompileStartTransfer(original);
  ASSERT_TRUE(program.ok()) << program.status();
  auto lowered = LowerToStartTransfer(*program);
  ASSERT_TRUE(lowered.ok()) << lowered.status();
  EXPECT_EQ(Canonical(original), Canonical(*lowered));
}

TEST(TransferProgramReshard, LegacyDensePlanRefusesToCompile) {
  ::tpu_sync::rpc::StartTransferRequest legacy;
  legacy.set_uuid(1);
  legacy.set_is_sender(true);
  auto program = CompileStartTransfer(legacy);
  EXPECT_EQ(program.status().code(), absl::StatusCode::kInvalidArgument);
}

// Conformance pins for completion contracts that the pool-reshard worker
// entry does not support.
TEST(TransferProgramReshard, AwaitInlineIsUnimplemented) {
  ::tpu_sync::proto::TransferProgramRequest request;
  request.mutable_program()->mutable_completion()->mutable_await_inline();
  auto result = NormalizeReshardProgram(request);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(std::string(result.status().message()),
              HasSubstr("AwaitInline"));
}

TEST(TransferProgramReshard, BarrierIsUnimplemented) {
  ::tpu_sync::proto::TransferProgramRequest request;
  request.mutable_program()->mutable_completion()->mutable_barrier();
  auto result = NormalizeReshardProgram(request);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(std::string(result.status().message()), HasSubstr("Barrier"));
}

TEST(TransferProgramReshard, MalformedProgramsAreInvalid) {
  // No contract at all.
  {
    ::tpu_sync::proto::TransferProgramRequest request;
    EXPECT_EQ(NormalizeReshardProgram(request).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  // A valid program mutated: two extents on one step.
  auto program = CompileStartTransfer(MakePoolRequest(true));
  ASSERT_TRUE(program.ok());
  {
    ::tpu_sync::proto::TransferProgramRequest mutated = *program;
    *mutated.mutable_program()->mutable_steps(0)->add_extents() =
        mutated.program().steps(0).extents(0);
    EXPECT_EQ(NormalizeReshardProgram(mutated).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  // MemoryRef.space set on a reshard step.
  {
    ::tpu_sync::proto::TransferProgramRequest mutated = *program;
    mutated.mutable_program()->mutable_steps(0)->mutable_src()
        ->set_pool_index(0);
    EXPECT_EQ(NormalizeReshardProgram(mutated).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  // Out-of-range group reference.
  {
    ::tpu_sync::proto::TransferProgramRequest mutated = *program;
    mutated.mutable_program()->mutable_steps(0)->set_group(99);
    EXPECT_EQ(NormalizeReshardProgram(mutated).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  // Unknown role.
  {
    ::tpu_sync::proto::TransferProgramRequest mutated = *program;
    mutated.mutable_envelope()->set_role(
        ::tpu_sync::proto::TRANSFER_ROLE_UNSPECIFIED);
    EXPECT_EQ(NormalizeReshardProgram(mutated).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace core
}  // namespace tpu_raiden
