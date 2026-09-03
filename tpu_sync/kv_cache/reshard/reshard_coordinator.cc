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

#include "tpu_sync/kv_cache/reshard/reshard_coordinator.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/transfer_program_reshard.h"
#include "tpu_sync/kv_cache/reshard/declaration_types.h"
#include "tpu_sync/kv_cache/reshard/framed_rpc.h"
#include "tpu_sync/kv_cache/reshard/pool_reshard_planner.h"
#include "tpu_sync/kv_cache/reshard/request_block_registry.h"
#include "tpu_sync/kv_cache/reshard/work_unit_directory.h"
#include "tpu_sync/proto/transfer_program.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

namespace {

// Worker RPC timeout: WorkerRpcClient uses connect_socket(timeout=60).
constexpr absl::Duration kWorkerRpcTimeout = absl::Seconds(60);

int64_t MonotonicNs() { return absl::GetCurrentTimeNanos(); }

std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\r':
        out += "\\r";
        break;
      default:
        out += c;
    }
  }
  return out;
}

std::string JsonFloat(double value) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6f", value);
  return buf;
}

// Sends one framed worker RPC and applies WorkerRpcClient._verify_response.
absl::Status SendWorkerRpc(FramedTransport* transport,
                           const std::string& address,
                           const std::string& payload) {
  auto response = transport->Call(address, payload, kWorkerRpcTimeout);
  if (!response.ok()) return response.status();
  tpu_sync::rpc::ControlResponse resp;
  if (!resp.ParseFromString(*response)) {
    return absl::InternalError(
        "Failed to parse worker ControlResponse payload");
  }
  if (!resp.success()) {
    return absl::InternalError(absl::StrCat(
        "Raiden remote native execution failed: ", resp.message()));
  }
  return absl::OkStatus();
}

}  // namespace

tpu_sync::rpc::StartTransferRequest BuildStartTransferForTarget(
    const PoolReshardPlan& plan, const RaidenId& target) {
  const bool is_sender = std::find(plan.src_units.begin(), plan.src_units.end(),
                                   target) != plan.src_units.end();
  const bool is_receiver =
      std::find(plan.dst_units.begin(), plan.dst_units.end(), target) !=
      plan.dst_units.end();
  tpu_sync::rpc::StartTransferRequest start_req;
  for (const RaidenId& unit : plan.src_units) {
    *start_req.add_src_units() = RaidenIdToProto(unit);
  }
  if (is_receiver) {
    *start_req.add_dst_units() = RaidenIdToProto(target);
  } else {
    for (const RaidenId& unit : plan.dst_units) {
      *start_req.add_dst_units() = RaidenIdToProto(unit);
    }
  }
  start_req.set_uuid(plan.uuid);
  start_req.set_is_sender(is_sender);
  start_req.set_dst_mem_type(tpu_sync::rpc::MEMORY_TYPE_HBM);
  start_req.set_use_block_chunks(true);
  start_req.set_expected_block_count(plan.expected_block_count);
  start_req.set_req_id(plan.req_id);
  // The plan addresses pools in the destination's index space. A sender
  // executes against its own pool table, which may register only a subset
  // of the destination's tags, so its request is rewritten into its own
  // index space: transferred pools it does not register are dropped (their
  // groups keep their position so entry pool_group references hold), and
  // the dtype tag list is the sender's own.
  const std::map<int32_t, int32_t>* sender_remap = nullptr;
  if (is_sender && !is_receiver) {
    auto remap_it = plan.src_pool_indices.find(target);
    if (remap_it != plan.src_pool_indices.end()) {
      sender_remap = &remap_it->second;
    }
  }
  auto local_pool_index = [sender_remap](int32_t index) -> int32_t {
    if (sender_remap == nullptr) return index;
    auto it = sender_remap->find(index);
    return it == sender_remap->end() ? -1 : it->second;
  };
  for (int32_t index : plan.transfer_pool_indices) {
    const int32_t local = local_pool_index(index);
    if (local >= 0) start_req.add_transfer_pool_indices(local);
  }
  // The receiver resolves each push against its own pool table, so a
  // rewritten sender still names pools on the wire by destination index.
  if (sender_remap != nullptr) {
    for (const auto& [dst_index, local] : *sender_remap) {
      (*start_req.mutable_wire_pool_indices())[local] = dst_index;
    }
  }
  if (sender_remap != nullptr) {
    for (const std::string& tag : plan.src_pool_dtype_tags.at(target)) {
      start_req.add_pool_dtype_tags(tag);
    }
  } else {
    for (const std::string& tag : plan.pool_dtype_tags) {
      start_req.add_pool_dtype_tags(tag);
    }
  }
  start_req.set_parallelism(plan.parallelism);
  // Python assigns transfer_plan.skip_d2h unconditionally, which marks the
  // optional field present even when false — mirror that for G1.
  start_req.set_skip_d2h(plan.skip_d2h);
  // Pool plans leave skip_tiling empty: absent map on the wire.

  for (const PlanPoolGroup& group : plan.pool_groups) {
    tpu_sync::rpc::PoolGroupProto* group_proto = start_req.add_pool_groups();
    for (int32_t index : group.pool_indices) {
      const int32_t local = local_pool_index(index);
      if (local >= 0) group_proto->add_pool_indices(local);
    }
    for (int64_t block_id : group.dst_device_block_ids) {
      group_proto->add_dst_device_block_ids(block_id);
    }
    int32_t expected_pushes = 0;
    if (is_receiver) {
      auto by_dst_it = group.expected_pushes_by_dst.find(target);
      if (by_dst_it != group.expected_pushes_by_dst.end()) {
        expected_pushes = by_dst_it->second;
      }
    } else {
      // Senders carry the largest per-destination count: sender-side plan
      // validation only requires a positive value, and with sharded
      // destinations the per-receiver counts differ.
      for (const auto& [unit, count] : group.expected_pushes_by_dst) {
        expected_pushes = std::max(expected_pushes, count);
      }
    }
    group_proto->set_expected_pushes(expected_pushes);
    for (int64_t extent : group.dst_expected_extent_bytes) {
      group_proto->add_dst_expected_extent_bytes(extent);
    }
    group_proto->set_order_rank(group.order_rank);
  }

  auto entries_to_proto =
      [](const std::vector<ScheduleEntry>& entries,
         tpu_sync::rpc::ShardPushScheduleProto* schedule_proto) {
        for (const ScheduleEntry& entry : entries) {
          tpu_sync::rpc::ShardPushEntryProto* entry_proto =
              schedule_proto->add_entries();
          entry_proto->set_dst_peer(entry.dst_peer);
          entry_proto->set_dst_shard_idx(entry.dst_shard_idx);
          entry_proto->set_dst_offset_bytes(entry.dst_offset_bytes);
          entry_proto->set_src_offset_bytes(entry.src_offset_bytes);
          entry_proto->set_size_bytes(entry.size_bytes);
          entry_proto->set_src_block_id(
              static_cast<int32_t>(entry.src_block_id));
          entry_proto->set_dst_block_id(
              static_cast<int32_t>(entry.dst_block_id));
          entry_proto->set_src_stride_bytes(entry.src_stride_bytes);
          entry_proto->set_dst_stride_bytes(entry.dst_stride_bytes);
          entry_proto->set_count(entry.count);
          entry_proto->set_layer_idx(entry.layer_idx);
          entry_proto->set_pool_group(entry.pool_group);
        }
      };

  if (is_receiver) {
    // Receiver path: every source's schedule, keyed by the source's transfer
    // rank (the node id its pushes carry), filtered to entries targeting
    // this receiver (single-endpoint pool plans always match).
    const std::string& target_peer = plan.dst_peers.at(target);
    for (const auto& [src_unit, entries] : plan.schedules) {
      auto key_it = plan.src_schedule_keys.find(src_unit);
      if (key_it == plan.src_schedule_keys.end()) continue;
      std::vector<ScheduleEntry> filtered;
      for (const ScheduleEntry& entry : entries) {
        if (entry.dst_peer == target_peer) filtered.push_back(entry);
      }
      if (filtered.empty()) continue;
      tpu_sync::rpc::ShardPushScheduleProto schedule_proto;
      entries_to_proto(filtered, &schedule_proto);
      (*start_req.mutable_shard_push_schedules())[key_it->second] =
          schedule_proto;
    }
  } else {
    auto schedule_it = plan.schedules.find(target);
    if (schedule_it != plan.schedules.end() && !schedule_it->second.empty()) {
      tpu_sync::rpc::ShardPushScheduleProto schedule_proto;
      entries_to_proto(schedule_it->second, &schedule_proto);
      (*start_req.mutable_shard_push_schedules())[0] = schedule_proto;
    }
  }

  return start_req;
}

std::string EncodeStartTransfer(const PoolReshardPlan& plan,
                                const RaidenId& target) {
  tpu_sync::rpc::ControlRequest req;
  req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER);
  // peers: the destination units' data endpoints (one dst, one endpoint).
  for (const RaidenId& unit : plan.dst_units) {
    req.add_peers(plan.dst_peers.at(unit));
  }
  *req.mutable_start_transfer_request() =
      BuildStartTransferForTarget(plan, target);
  return req.SerializeAsString();
}

ReshardCoordinator::ReshardCoordinator(WorkUnitDirectory* directory,
                                       RequestBlockRegistry* registry,
                                       FramedTransport* transport,
                                       WorkerDelivery delivery)
    : directory_(directory),
      registry_(registry),
      transport_(transport),
      delivery_(delivery) {}

absl::StatusOr<std::vector<tpu_sync::rpc::RegisterWorkUnitRequest>>
ReshardCoordinator::QueryRemoteMetadata(const std::string& address) {
  tpu_sync::rpc::ControlRequest req;
  req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_GET_METADATA);
  auto response =
      transport_->Call(address, req.SerializeAsString(), kWorkerRpcTimeout);
  if (!response.ok()) return response.status();
  tpu_sync::rpc::ControlResponse resp;
  if (!resp.ParseFromString(*response)) {
    return absl::InternalError(
        "Failed to parse remote metadata ControlResponse");
  }
  if (!resp.success()) {
    return absl::InternalError(
        absl::StrCat("Failed to query remote metadata: ", resp.message()));
  }
  std::vector<tpu_sync::rpc::RegisterWorkUnitRequest> metadata(
      resp.get_metadata_response().metadata().begin(),
      resp.get_metadata_response().metadata().end());
  return metadata;
}

absl::Status ReshardCoordinator::StartPoolReshard(const PoolReshardArgs& args) {
  // _start_pool_reshard_transfer's synchronous validation.
  if (args.src_units.empty()) {
    return absl::InvalidArgumentError("src_units must not be empty");
  }
  if (args.dst_units.empty()) {
    return absl::InvalidArgumentError("dst_units must not be empty");
  }
  if (args.req_id.empty()) {
    return absl::InvalidArgumentError(
        "req_id must not be empty for pool resharding");
  }
  if (!args.has_dst_device_block_ids) {
    return absl::InvalidArgumentError(
        "dst_device_block_ids are required for pool resharding");
  }
  const int64_t num_tokens = args.num_tokens.value_or(0);
  if (num_tokens < 0) {
    return absl::InvalidArgumentError(
        "num_tokens must be non-negative for pool resharding");
  }
  if (args.transfer_pool_tags.empty()) {
    return absl::InvalidArgumentError(
        "transfer_pool_tags must name the pools to move for pool "
        "resharding");
  }
  if (!args.use_block_chunks) {
    return absl::InvalidArgumentError(
        "Pool resharding requires use_block_chunks=true");
  }
  if (args.dst_mem_type != tpu_sync::rpc::MEMORY_TYPE_HBM) {
    return absl::InvalidArgumentError(
        "Pool resharding requires dst_mem_type=HBM");
  }
  if (args.parallelism.has_value() && *args.parallelism <= 0) {
    return absl::InvalidArgumentError("parallelism must be positive");
  }

  {
    absl::MutexLock lock(status_mu_);
    transfer_status_[args.req_id] = Status::kInProgress;
  }
  absl::Status status = ExecutePoolReshard(args);
  {
    absl::MutexLock lock(status_mu_);
    transfer_status_[args.req_id] =
        status.ok() ? Status::kCompleted : Status::kFailed;
  }
  return status;
}

absl::Status ReshardCoordinator::ExecutePoolReshard(
    const PoolReshardArgs& args) {
  if (!args.is_sender) {
    return absl::InvalidArgumentError(
        "Pool reshard coordination must target the source controller: "
        "the consumer coordinates with src_controller_address directly "
        "(the destination-side relay is retired)");
  }

  const int64_t controller_start_ns = MonotonicNs();
  const int64_t plan_build_start_ns = controller_start_ns;

  auto src_metadata = directory_->LocalMetadata(args.src_units);
  if (!src_metadata.ok()) return src_metadata.status();
  std::vector<tpu_sync::rpc::RegisterWorkUnitRequest> dst_metadata;
  if (!args.dst_controller_address.empty()) {
    auto remote = QueryRemoteMetadata(args.dst_controller_address);
    if (!remote.ok()) return remote.status();
    dst_metadata = *std::move(remote);
  } else {
    auto local = directory_->LocalMetadata(args.dst_units);
    if (!local.ok()) return local.status();
    dst_metadata = *std::move(local);
  }

  int64_t uuid = args.uuid;
  if (uuid <= 0) {
    // Python: random.randint(1, 2**63 - 1) when the wire carries no uuid.
    absl::BitGen gen;
    uuid = absl::Uniform<int64_t>(gen, 1, std::numeric_limits<int64_t>::max());
  }

  PlanRequest plan_request;
  plan_request.src_units = args.src_units;
  plan_request.dst_units = args.dst_units;
  plan_request.src_metadata = *std::move(src_metadata);
  plan_request.dst_metadata = std::move(dst_metadata);
  plan_request.req_id = args.req_id;
  plan_request.uuid = uuid;
  plan_request.dst_device_block_ids = args.dst_device_block_ids;
  plan_request.num_tokens = args.num_tokens.value_or(0);
  plan_request.transfer_pool_tags = args.transfer_pool_tags;
  plan_request.parallelism = args.parallelism;
  plan_request.dst_block_counts = args.dst_block_counts;
  plan_request.dst_skip_bytes = args.dst_skip_bytes;

  // _build_pool_reshard_plan: claim under a unique owner; abandon on any
  // planning failure.
  static std::atomic<uintptr_t> claim_counter{1};
  const void* claim_owner =
      reinterpret_cast<const void*>(claim_counter.fetch_add(1));
  auto plan_or = BuildPoolReshardPlan(plan_request, registry_, claim_owner);
  if (!plan_or.ok()) {
    registry_->AbandonClaim(args.req_id, uuid, claim_owner);
    return plan_or.status();
  }
  PoolReshardPlan& plan = *plan_or;
  plan.skip_d2h = args.skip_d2h;
  const int64_t plan_build_end_ns = MonotonicNs();

  // Receivers must be armed before any sender can put bytes on the wire.
  // Multi-destination plans arm every receiver concurrently and join
  // before dispatching senders.
  const int64_t receiver_arm_start_ns = MonotonicNs();
  {
    std::vector<absl::Status> arm_status(plan.dst_units.size());
    std::vector<std::thread> armers;
    armers.reserve(plan.dst_units.size());
    for (size_t i = 0; i < plan.dst_units.size(); ++i) {
      const RaidenId& unit = plan.dst_units[i];
      armers.emplace_back([this, &plan, &arm_status, i, unit]() {
        auto addr_it = plan.worker_rpc_addresses.find(unit);
        if (addr_it == plan.worker_rpc_addresses.end()) {
          arm_status[i] = absl::InternalError(absl::StrCat(
              "No control endpoint recorded for ", PythonRepr(unit)));
          return;
        }
        std::fprintf(stderr,
                     "{\"event\": \"raiden_pool_reshard_arm\", \"dst_unit\": "
                     "\"%s\", \"pool_dtype_tags\": \"%s\", \"uuid\": %lld}\n",
                     PythonRepr(unit).c_str(),
                     absl::StrJoin(plan.pool_dtype_tags, ",").c_str(),
                     static_cast<long long>(plan.uuid));
        const std::string arm_payload = EncodeStartTransfer(plan, unit);
        arm_status[i] = SendWorkerRpc(transport_, addr_it->second, arm_payload);
      });
    }
    for (std::thread& t : armers) t.join();
    for (const absl::Status& status : arm_status) {
      if (!status.ok()) {
        registry_->AbandonClaim(args.req_id, uuid, claim_owner);
        return status;
      }
    }
  }
  const int64_t receiver_arm_ack_ns = MonotonicNs();

  // Dispatch the senders in parallel, joining before the RPC returns
  // (Python gathers the same fan-out on its per-connection event loop).
  const int64_t sender_dispatch_ns = MonotonicNs();
  std::vector<absl::Status> sender_status(plan.src_units.size());
  if (delivery_.mode == WorkerDelivery::Mode::kController) {
    if (delivery_.controller == nullptr) {
      registry_->AbandonClaim(args.req_id, uuid, claim_owner);
      return absl::InternalError(
          "WorkerDelivery::Mode::kController requires a non-null controller");
    }
    // Senders route via the local controller's persistent WorkerService
    // channels. Workers are registered as "worker_<schedule_key>" (where
    // schedule_key matches the shard_id the worker was initialized with).
    std::vector<tsl::Future<::tpu_sync::proto::TransferProgramResponse>>
        futures;
    futures.reserve(plan.src_units.size());
    for (size_t i = 0; i < plan.src_units.size(); ++i) {
      const RaidenId& unit = plan.src_units[i];
      auto key_it = plan.src_schedule_keys.find(unit);
      if (key_it == plan.src_schedule_keys.end()) {
        sender_status[i] = absl::InternalError(absl::StrCat(
            "No schedule key assigned for ", PythonRepr(unit)));
        continue;
      }
      const std::string worker_id = absl::StrCat("worker_", key_it->second);
      auto compiled =
          core::CompileStartTransfer(BuildStartTransferForTarget(plan, unit));
      if (!compiled.ok()) {
        sender_status[i] = compiled.status();
        continue;
      }
      futures.push_back(
          delivery_.controller->SubmitTransferProgram(worker_id, *compiled));
    }
    for (size_t i = 0; i < futures.size(); ++i) {
      auto resp = futures[i].Await();
      if (!resp.ok()) {
        sender_status[i] = resp.status();
      } else if (!resp->success()) {
        sender_status[i] = absl::InternalError(absl::StrCat(
            "Raiden remote native execution failed: ", resp->message()));
      }
    }
  } else {
    std::vector<std::thread> senders;
    senders.reserve(plan.src_units.size());
    for (size_t i = 0; i < plan.src_units.size(); ++i) {
      const RaidenId& unit = plan.src_units[i];
      senders.emplace_back([this, &plan, &sender_status, i, unit]() {
        auto addr_it = plan.worker_rpc_addresses.find(unit);
        if (addr_it == plan.worker_rpc_addresses.end()) {
          sender_status[i] = absl::InternalError(absl::StrCat(
              "No control endpoint recorded for ", PythonRepr(unit)));
          return;
        }
        const std::string payload = EncodeStartTransfer(plan, unit);
        sender_status[i] = SendWorkerRpc(transport_, addr_it->second, payload);
      });
    }
    for (std::thread& t : senders) t.join();
  }
  const int64_t sender_dispatch_end_ns = MonotonicNs();
  for (const absl::Status& status : sender_status) {
    if (!status.ok()) return status;
  }

  // Milestone event (D5): schema-identical JSON line on stderr,
  // sort_keys=True ordering.
  std::vector<std::string> sorted_tags(args.transfer_pool_tags.begin(),
                                       args.transfer_pool_tags.end());
  std::sort(sorted_tags.begin(), sorted_tags.end());
  sorted_tags.erase(std::unique(sorted_tags.begin(), sorted_tags.end()),
                    sorted_tags.end());
  std::string tags_json;
  for (size_t i = 0; i < sorted_tags.size(); ++i) {
    absl::StrAppend(&tags_json, i ? ", " : "", "\"", JsonEscape(sorted_tags[i]),
                    "\"");
  }
  std::string skipped_json;
  {
    bool first = true;
    for (const auto& [tag, count] : plan.skipped_pool_counts) {
      absl::StrAppend(&skipped_json, first ? "" : ", ", "\"", JsonEscape(tag),
                      "\": ", count);
      first = false;
    }
  }
  const std::string event = absl::StrCat(
      "{\"active_source_ranks\": ", plan.src_units.size(),
      ", \"controller_total_ms\": ",
      JsonFloat((sender_dispatch_end_ns - controller_start_ns) / 1e6),
      ", \"destination_pages\": ", plan.expected_block_count,
      ", \"destination_units\": ", plan.dst_units.size(),
      ", \"event\": \"raiden_pool_reshard_senders_dispatched\"",
      ", \"expected_pushes_per_pool\": ", plan.expected_pushes_per_pool,
      ", \"num_tokens\": ", plan.num_tokens, ", \"plan_build_ms\": ",
      JsonFloat((plan_build_end_ns - plan_build_start_ns) / 1e6),
      ", \"receiver_arm_ack_monotonic_ns\": ", receiver_arm_ack_ns,
      ", \"receiver_arm_rpc_ms\": ",
      JsonFloat((receiver_arm_ack_ns - receiver_arm_start_ns) / 1e6),
      ", \"receiver_armed_before_sender_dispatch\": true", ", \"req_id\": \"",
      JsonEscape(plan.req_id), "\"",
      ", \"sender_dispatch_monotonic_ns\": ", sender_dispatch_ns,
      ", \"sender_dispatch_ms\": ",
      JsonFloat((sender_dispatch_end_ns - sender_dispatch_ns) / 1e6),
      ", \"skipped_pool_counts\": {", skipped_json, "}",
      ", \"transfer_pool_tags\": [", tags_json, "]",
      ", \"transferred_pools\": ", plan.transfer_pool_indices.size(),
      ", \"uuid\": ", plan.uuid, "}");
  std::fprintf(stderr, "%s\n", event.c_str());
  std::fflush(stderr);
  return absl::OkStatus();
}

ReshardCoordinator::Status ReshardCoordinator::GetTransferStatus(
    const std::string& req_id) const {
  absl::MutexLock lock(status_mu_);
  auto it = transfer_status_.find(req_id);
  if (it == transfer_status_.end()) return Status::kNotStarted;
  return it->second;
}

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
