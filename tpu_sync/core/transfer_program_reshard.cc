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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/proto/transfer_program.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace core {

namespace {

// Mirror of kv_cache_listener.cc HasPoolReshardFields: either field opts
// into the pool executor so partially populated plans fail closed there
// instead of silently taking the legacy path.
bool HasPoolReshardFields(
    const ::tpu_sync::rpc::StartTransferRequest& request) {
  return !request.transfer_pool_indices().empty() ||
         !request.pool_groups().empty();
}

absl::Status CheckInt32(int64_t value, absl::string_view what) {
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError(
        absl::StrCat(what, " ", value, " overflows int32"));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<ReshardLoweringClass> NormalizeReshardProgram(
    const ::tpu_sync::proto::TransferProgramRequest& request) {
  const ::tpu_sync::proto::TransferProgram& program = request.program();
  const ::tpu_sync::proto::TransferEnvelope& envelope = request.envelope();

  // Dispatch keys on the completion contract, never on `kind`.
  switch (program.completion().mode_case()) {
    case ::tpu_sync::proto::CompletionContract::kAwaitInline:
      return absl::UnimplementedError(
          "AwaitInline transfer programs are not supported by the "
          "pool-reshard worker entry");
    case ::tpu_sync::proto::CompletionContract::kBarrier:
      return absl::UnimplementedError(
          "Barrier transfer programs are not supported by the pool-reshard "
          "worker entry");
    case ::tpu_sync::proto::CompletionContract::kFanIn:
      break;
    case ::tpu_sync::proto::CompletionContract::MODE_NOT_SET:
      return absl::InvalidArgumentError(
          "transfer program carries no completion contract");
  }

  // `kind` is observability-only, but an inconsistent envelope is a caller
  // bug worth failing loudly on.
  if (envelope.kind() != ::tpu_sync::proto::TRANSFER_KIND_POOL_RESHARD &&
      envelope.kind() != ::tpu_sync::proto::TRANSFER_KIND_UNSPECIFIED) {
    return absl::InvalidArgumentError(
        absl::StrCat("fan-in transfer program carries inconsistent kind ",
                     ::tpu_sync::proto::TransferKind_Name(envelope.kind())));
  }

  const ::tpu_sync::proto::FanIn& fan_in = program.completion().fan_in();
  if (fan_in.groups().empty()) {
    return absl::InvalidArgumentError(
        "pool-reshard program declares no fan-in groups");
  }
  if (program.reshard().transfer_pool_indices().empty()) {
    return absl::InvalidArgumentError(
        "pool-reshard program names no transfer pool indices");
  }
  if (program.steps().empty()) {
    return absl::InvalidArgumentError("pool-reshard program has no steps");
  }
  for (const ::tpu_sync::proto::TransferStep& step : program.steps()) {
    if (step.extents_size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "pool-reshard steps carry exactly one extent, got ",
          step.extents_size()));
    }
    if (step.src().space_case() !=
            ::tpu_sync::proto::MemoryRef::SPACE_NOT_SET ||
        step.dst().space_case() !=
            ::tpu_sync::proto::MemoryRef::SPACE_NOT_SET) {
      return absl::InvalidArgumentError(
          "pool-reshard steps address blocks through their fan-in group "
          "scope; MemoryRef.space must be unset");
    }
    if (step.has_group() &&
        (step.group() < 0 || step.group() >= fan_in.groups_size())) {
      return absl::InvalidArgumentError(
          absl::StrCat("step group ", step.group(), " out of range [0, ",
                       fan_in.groups_size(), ")"));
    }
  }

  switch (envelope.role()) {
    case ::tpu_sync::proto::TRANSFER_ROLE_SENDER:
      return ReshardLoweringClass::kPoolReshardSender;
    case ::tpu_sync::proto::TRANSFER_ROLE_RECEIVER_ARM:
      return ReshardLoweringClass::kPoolReshardArm;
    case ::tpu_sync::proto::TRANSFER_ROLE_LOCAL_COPY:
      return absl::UnimplementedError(
          "LOCAL_COPY transfer programs are not supported by the "
          "pool-reshard worker entry");
    default:
      return absl::InvalidArgumentError(
          "pool-reshard program carries no role");
  }
}

absl::StatusOr<::tpu_sync::proto::TransferProgramRequest> CompileStartTransfer(
    const ::tpu_sync::rpc::StartTransferRequest& request) {
  if (!HasPoolReshardFields(request)) {
    return absl::InvalidArgumentError(
        "only pool-reshard StartTransferRequests compile to transfer "
        "programs (legacy dense plans stay on the framed door)");
  }

  ::tpu_sync::proto::TransferProgramRequest out;
  ::tpu_sync::proto::TransferEnvelope* envelope = out.mutable_envelope();
  envelope->set_uuid(static_cast<uint64_t>(request.uuid()));
  envelope->set_req_id(request.req_id());
  envelope->set_kind(::tpu_sync::proto::TRANSFER_KIND_POOL_RESHARD);
  envelope->set_role(request.is_sender()
                         ? ::tpu_sync::proto::TRANSFER_ROLE_SENDER
                         : ::tpu_sync::proto::TRANSFER_ROLE_RECEIVER_ARM);

  ::tpu_sync::proto::TransferProgram* program = out.mutable_program();

  ::tpu_sync::proto::ExecutionPolicy* policy = program->mutable_policy();
  policy->set_parallelism(request.parallelism());
  if (request.has_skip_d2h()) {
    policy->set_skip_d2h(request.skip_d2h());
  }
  for (const auto& [layer, skip] : request.skip_tiling()) {
    (*policy->mutable_skip_tiling())[layer] = skip;
  }
  policy->set_dst_mem_type(request.dst_mem_type());
  policy->set_use_block_chunks(request.use_block_chunks());

  ::tpu_sync::proto::ReshardBinding* binding = program->mutable_reshard();
  *binding->mutable_src_units() = request.src_units();
  *binding->mutable_dst_units() = request.dst_units();
  *binding->mutable_transfer_pool_indices() =
      request.transfer_pool_indices();
  *binding->mutable_pool_dtype_tags() = request.pool_dtype_tags();
  for (const auto& [local, wire] : request.wire_pool_indices()) {
    (*binding->mutable_wire_pool_indices())[local] = wire;
  }

  ::tpu_sync::proto::FanIn* fan_in =
      program->mutable_completion()->mutable_fan_in();
  fan_in->set_expected_block_count(request.expected_block_count());
  for (const ::tpu_sync::rpc::PoolGroupProto& group : request.pool_groups()) {
    ::tpu_sync::proto::FanInGroup* out_group = fan_in->add_groups();
    *out_group->mutable_pool_indices() = group.pool_indices();
    *out_group->mutable_dst_device_block_ids() =
        group.dst_device_block_ids();
    out_group->set_expected_pushes(group.expected_pushes());
    *out_group->mutable_dst_expected_extent_bytes() =
        group.dst_expected_extent_bytes();
    out_group->set_order_rank(group.order_rank());
  }

  // Emit steps rank-block by rank-block in ascending rank order so the
  // program is deterministic; entry order inside each rank is preserved
  // verbatim (the plan shape is order-sensitive).
  std::vector<int32_t> ranks;
  ranks.reserve(request.shard_push_schedules_size());
  for (const auto& [rank, schedule] : request.shard_push_schedules()) {
    ranks.push_back(rank);
  }
  std::sort(ranks.begin(), ranks.end());
  for (int32_t rank : ranks) {
    const ::tpu_sync::rpc::ShardPushScheduleProto& schedule =
        request.shard_push_schedules().at(rank);
    for (const ::tpu_sync::rpc::ShardPushEntryProto& entry :
         schedule.entries()) {
      ::tpu_sync::proto::TransferStep* step = program->add_steps();
      step->mutable_src()->set_block_id(entry.src_block_id());
      step->mutable_dst()->set_block_id(entry.dst_block_id());
      ::tpu_sync::proto::Extent* extent = step->add_extents();
      extent->set_src_offset_bytes(entry.src_offset_bytes());
      extent->set_dst_offset_bytes(entry.dst_offset_bytes());
      extent->set_size_bytes(entry.size_bytes());
      extent->set_src_stride_bytes(entry.src_stride_bytes());
      extent->set_dst_stride_bytes(entry.dst_stride_bytes());
      extent->set_count(entry.count());
      step->set_dst_peer(entry.dst_peer());
      step->set_dst_shard_idx(entry.dst_shard_idx());
      if (entry.has_pool_group()) {
        step->set_group(entry.pool_group());
      }
      step->set_source_rank(rank);
      if (entry.has_layer_idx()) {
        step->set_layer_idx(entry.layer_idx());
      }
    }
  }
  return out;
}

absl::StatusOr<::tpu_sync::rpc::StartTransferRequest> LowerToStartTransfer(
    const ::tpu_sync::proto::TransferProgramRequest& request) {
  const ::tpu_sync::proto::TransferProgram& program = request.program();
  if (program.completion().mode_case() !=
      ::tpu_sync::proto::CompletionContract::kFanIn) {
    return absl::InvalidArgumentError(
        "only fan-in programs lower to StartTransferRequest");
  }

  ::tpu_sync::rpc::StartTransferRequest out;
  out.set_uuid(static_cast<int64_t>(request.envelope().uuid()));
  out.set_req_id(request.envelope().req_id());
  out.set_is_sender(request.envelope().role() ==
                    ::tpu_sync::proto::TRANSFER_ROLE_SENDER);

  const ::tpu_sync::proto::ReshardBinding& binding = program.reshard();
  *out.mutable_src_units() = binding.src_units();
  *out.mutable_dst_units() = binding.dst_units();
  *out.mutable_transfer_pool_indices() = binding.transfer_pool_indices();
  *out.mutable_pool_dtype_tags() = binding.pool_dtype_tags();
  for (const auto& [local, wire] : binding.wire_pool_indices()) {
    (*out.mutable_wire_pool_indices())[local] = wire;
  }

  const ::tpu_sync::proto::ExecutionPolicy& policy = program.policy();
  out.set_parallelism(policy.parallelism());
  if (policy.has_skip_d2h()) {
    out.set_skip_d2h(policy.skip_d2h());
  }
  for (const auto& [layer, skip] : policy.skip_tiling()) {
    (*out.mutable_skip_tiling())[layer] = skip;
  }
  out.set_dst_mem_type(policy.dst_mem_type());
  out.set_use_block_chunks(policy.use_block_chunks());

  const ::tpu_sync::proto::FanIn& fan_in = program.completion().fan_in();
  out.set_expected_block_count(fan_in.expected_block_count());
  for (const ::tpu_sync::proto::FanInGroup& group : fan_in.groups()) {
    ::tpu_sync::rpc::PoolGroupProto* out_group = out.add_pool_groups();
    *out_group->mutable_pool_indices() = group.pool_indices();
    *out_group->mutable_dst_device_block_ids() =
        group.dst_device_block_ids();
    out_group->set_expected_pushes(group.expected_pushes());
    *out_group->mutable_dst_expected_extent_bytes() =
        group.dst_expected_extent_bytes();
    out_group->set_order_rank(group.order_rank());
  }

  for (const ::tpu_sync::proto::TransferStep& step : program.steps()) {
    if (step.extents_size() != 1) {
      return absl::InvalidArgumentError(
          "pool-reshard steps carry exactly one extent");
    }
    const ::tpu_sync::proto::Extent& extent = step.extents(0);
    if (absl::Status s = CheckInt32(step.src().block_id(), "src block id");
        !s.ok()) {
      return s;
    }
    if (absl::Status s = CheckInt32(step.dst().block_id(), "dst block id");
        !s.ok()) {
      return s;
    }
    ::tpu_sync::rpc::ShardPushEntryProto* entry =
        (*out.mutable_shard_push_schedules())[step.source_rank()].add_entries();
    entry->set_dst_peer(step.dst_peer());
    entry->set_dst_shard_idx(step.dst_shard_idx());
    entry->set_dst_offset_bytes(extent.dst_offset_bytes());
    entry->set_src_offset_bytes(extent.src_offset_bytes());
    entry->set_size_bytes(extent.size_bytes());
    entry->set_src_block_id(static_cast<int32_t>(step.src().block_id()));
    entry->set_dst_block_id(static_cast<int32_t>(step.dst().block_id()));
    entry->set_src_stride_bytes(extent.src_stride_bytes());
    entry->set_dst_stride_bytes(extent.dst_stride_bytes());
    entry->set_count(extent.count());
    if (step.has_layer_idx()) {
      entry->set_layer_idx(step.layer_idx());
    }
    if (step.has_group()) {
      entry->set_pool_group(step.group());
    }
  }
  return out;
}

std::vector<int64_t> DeriveSenderSourceBlockIds(
    const ::tpu_sync::rpc::StartTransferRequest& request) {
  std::set<int64_t> src_id_set;
  for (const auto& [schedule_key, schedule] :
       request.shard_push_schedules()) {
    for (const auto& entry : schedule.entries()) {
      src_id_set.insert(static_cast<int64_t>(entry.src_block_id()));
    }
  }
  return std::vector<int64_t>(src_id_set.begin(), src_id_set.end());
}

std::vector<int64_t> DeriveArmChipBlockIds(
    const ::tpu_sync::rpc::StartTransferRequest& request) {
  std::vector<int64_t> chip_block_ids;
  for (const auto& group : request.pool_groups()) {
    chip_block_ids.insert(chip_block_ids.end(),
                          group.dst_device_block_ids().begin(),
                          group.dst_device_block_ids().end());
  }
  return chip_block_ids;
}

}  // namespace core
}  // namespace tpu_raiden
