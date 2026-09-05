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

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/transfer_program_reshard.h"
#include "tpu_sync/kv_cache/reshard/declaration_types.h"
#include "tpu_sync/kv_cache/reshard/framed_rpc.h"
#include "tpu_sync/kv_cache/reshard/request_block_registry.h"
#include "tpu_sync/kv_cache/reshard/reshard_coordinator.h"
#include "tpu_sync/kv_cache/reshard/work_unit_directory.h"
#include "tpu_sync/proto/transfer_program.pb.h"
#include "tpu_sync/rpc/controller_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

double SteadyClockSeconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// str(e) for absl statuses: the raw message, no code prefix (Python
// ValueError parity for every contract-bearing string).
std::string StatusMessage(const absl::Status& status) {
  return std::string(status.message());
}

}  // namespace

ReshardService::ReshardService(const Options& options)
    : delivery_(options.delivery), requested_port_(options.port) {
  FramedTransport* transport = options.transport;
  if (transport == nullptr) {
    default_transport_ = std::make_unique<SocketFramedTransport>();
    transport = default_transport_.get();
  }
  transport_ = transport;
  directory_ = std::make_unique<WorkUnitDirectory>(&state_mu_);
  RequestBlockRegistry::Clock clock = options.clock;
  if (!clock) clock = SteadyClockSeconds;
  registry_ = std::make_unique<RequestBlockRegistry>(
      directory_.get(), options.request_registry_ttl_s, std::move(clock));
  coordinator_ = std::make_unique<ReshardCoordinator>(
      directory_.get(), registry_.get(), transport, delivery_);
}

ReshardService::~ReshardService() { StopServer(); }

absl::Status ReshardService::StartServer() {
  server_ = std::make_unique<FramedServer>(
      requested_port_,
      [this](const std::string& request) { return HandleFrame(request); });
  absl::Status bound = server_->Bind();
  if (!bound.ok()) return bound;
  server_->Start();
  return absl::OkStatus();
}

void ReshardService::StopServer() {
  if (server_ != nullptr) server_->Stop();
}

int ReshardService::port() const {
  return server_ != nullptr ? server_->port() : requested_port_;
}

std::string ReshardService::HandleFrame(const std::string& request_bytes) {
  tpu_sync::rpc::ControllerRequest req;
  if (!req.ParseFromString(request_bytes)) {
    req.set_command(tpu_sync::rpc::ControllerRequest::COMMAND_UNSPECIFIED);
  }

  const bool is_controller_command =
      (req.command() ==
           tpu_sync::rpc::ControllerRequest::COMMAND_COORDINATE_TRANSFER &&
       req.has_coordinate_transfer_request()) ||
      (req.command() ==
           tpu_sync::rpc::ControllerRequest::COMMAND_GET_TRANSFER_STATUS &&
       req.has_get_transfer_status_request()) ||
      (req.command() ==
           tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS &&
       req.has_register_request_blocks_request()) ||
      (req.command() ==
           tpu_sync::rpc::ControllerRequest::COMMAND_RELEASE_REQUEST_BLOCKS &&
       req.has_release_request_blocks_request()) ||
      (req.command() ==
           tpu_sync::rpc::ControllerRequest::COMMAND_COMPLETE_REQUEST_BLOCKS &&
       req.has_complete_request_blocks_request()) ||
      (req.command() == tpu_sync::rpc::ControllerRequest::
                            COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED &&
       req.has_cancel_request_blocks_if_unclaimed_request()) ||
      (req.command() ==
           tpu_sync::rpc::ControllerRequest::COMMAND_GET_REQUEST_BLOCK_STATUS &&
       req.has_get_request_block_status_request());
  if (is_controller_command) {
    return HandleControllerCommand(req);
  }
  return HandleRaidenCommand(request_bytes);
}

std::string ReshardService::HandleControllerCommand(
    const tpu_sync::rpc::ControllerRequest& req) {
  tpu_sync::rpc::ControllerResponse resp;
  resp.set_success(false);

  switch (req.command()) {
    case tpu_sync::rpc::ControllerRequest::COMMAND_COORDINATE_TRANSFER: {
      const auto& coord_req = req.coordinate_transfer_request();
      // Reshard requests are recognized by their destination block list;
      // num_tokens was retired from the wire.
      const bool is_pool_request = coord_req.dst_device_block_ids_size() > 0 ||
                                   coord_req.transfer_pool_tags_size() > 0;
      if (!is_pool_request) {
        resp.set_message(
            "Legacy dense transfers are not implemented by the C++ reshard "
            "controller; keep the Python sidecar for dense/weight-sync "
            "paths");
        break;
      }
      PoolReshardArgs args;
      for (const auto& unit : coord_req.src_units()) {
        args.src_units.push_back(RaidenIdFromProto(unit));
      }
      for (const auto& unit : coord_req.dst_units()) {
        args.dst_units.push_back(RaidenIdFromProto(unit));
      }
      args.req_id = coord_req.req_id();
      args.dst_device_block_ids.assign(coord_req.dst_device_block_ids().begin(),
                                       coord_req.dst_device_block_ids().end());
      args.has_dst_device_block_ids = coord_req.dst_device_block_ids_size() > 0;
      args.use_block_chunks = coord_req.use_block_chunks();
      args.dst_mem_type = coord_req.dst_mem_type();
      args.dst_controller_address = coord_req.dst_controller_address();
      args.uuid = coord_req.uuid() > 0 ? coord_req.uuid() : 0;
      args.is_sender = coord_req.is_sender();
      // num_tokens = 0 iff the wire carries destination blocks (server
      // decode rule); the coordinator re-checks non-negativity.
      if (coord_req.dst_device_block_ids_size() > 0) {
        args.num_tokens = 0;
      }
      args.transfer_pool_tags.assign(coord_req.transfer_pool_tags().begin(),
                                     coord_req.transfer_pool_tags().end());
      args.has_transfer_pool_tags = coord_req.transfer_pool_tags_size() > 0;
      args.dst_block_counts.assign(coord_req.dst_block_counts().begin(),
                                   coord_req.dst_block_counts().end());
      args.dst_skip_bytes.assign(coord_req.dst_skip_bytes().begin(),
                                 coord_req.dst_skip_bytes().end());
      // Wire carries no parallelism (planner derives it from admission);
      // dst_device_block_ids emptiness surfaces in the coordinator as the
      // Python path would report it.
      absl::Status status = coordinator_->StartPoolReshard(args);
      if (!status.ok()) {
        resp.set_message(StatusMessage(status));
        break;
      }
      resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControllerRequest::COMMAND_GET_TRANSFER_STATUS: {
      const auto& status_req = req.get_transfer_status_request();
      resp.mutable_get_transfer_status_response()->set_status(
          static_cast<tpu_sync::rpc::GetTransferStatusResponse::Status>(
              static_cast<int>(
                  coordinator_->GetTransferStatus(status_req.req_id()))));
      resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS: {
      const auto& block_req = req.register_request_blocks_request();
      std::vector<PoolSpanRegistration> pool_spans;
      for (const auto& entry : block_req.pool_spans()) {
        PoolSpanRegistration registration;
        registration.tag = entry.tag();
        registration.block_ids.assign(entry.block_ids().begin(),
                                      entry.block_ids().end());
        for (const auto& span : entry.spans()) {
          PoolByteSpan byte_span;
          byte_span.src_block_ordinal = span.src_block_ordinal();
          byte_span.src_offset_bytes = span.src_offset_bytes();
          byte_span.dst_block_index = span.dst_block_index();
          byte_span.dst_offset_bytes = span.dst_offset_bytes();
          byte_span.size_bytes = span.size_bytes();
          byte_span.src_stride_bytes = span.src_stride_bytes();
          byte_span.dst_stride_bytes = span.dst_stride_bytes();
          byte_span.count = span.count();
          byte_span.dst_unit_ordinal =
              span.has_dst_unit_ordinal() ? span.dst_unit_ordinal() : -1;
          registration.spans.push_back(byte_span);
        }
        registration.declared_bytes = entry.declared_bytes();
        registration.dst_space_version = entry.dst_space_version();
        pool_spans.push_back(std::move(registration));
      }
      absl::Status status = registry_->Register(
          block_req.req_id(), block_req.uuid(),
          RaidenIdFromProto(block_req.unit()),
          std::vector<int64_t>(block_req.block_ids().begin(),
                               block_req.block_ids().end()),
          pool_spans);
      if (!status.ok()) {
        resp.set_message(StatusMessage(status));
        break;
      }
      resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControllerRequest::COMMAND_RELEASE_REQUEST_BLOCKS: {
      const auto& release_req = req.release_request_blocks_request();
      if (!release_req.force()) {
        resp.set_message(
            "Legacy rank-local release is unsafe; use the aggregate "
            "completion-vote RPC");
        break;
      }
      auto released =
          registry_->Release(release_req.req_id(), release_req.uuid());
      if (!released.ok()) {
        resp.set_message(StatusMessage(released.status()));
        break;
      }
      resp.set_response_data(absl::StrCat(*released));
      resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControllerRequest::COMMAND_COMPLETE_REQUEST_BLOCKS: {
      const auto& complete_req = req.complete_request_blocks_request();
      auto released =
          registry_->Complete(complete_req.req_id(), complete_req.uuid(),
                              RaidenIdFromProto(complete_req.unit()));
      if (!released.ok()) {
        resp.set_message(StatusMessage(released.status()));
        break;
      }
      resp.set_response_data(absl::StrCat(*released));
      resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControllerRequest::
        COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED: {
      const auto& cancel_req = req.cancel_request_blocks_if_unclaimed_request();
      auto cancelled =
          registry_->CancelIfUnclaimed(cancel_req.req_id(), cancel_req.uuid());
      if (!cancelled.ok()) {
        resp.set_message(StatusMessage(cancelled.status()));
        break;
      }
      resp.set_response_data(*cancelled ? "true" : "false");
      resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControllerRequest::COMMAND_GET_REQUEST_BLOCK_STATUS: {
      const auto& status_req = req.get_request_block_status_request();
      std::vector<RequestBlockRegistry::RequestBlockKey> keys;
      keys.reserve(status_req.keys_size());
      for (const auto& key : status_req.keys()) {
        keys.emplace_back(key.req_id(), key.uuid());
      }
      auto* out = resp.mutable_get_request_block_status_response();
      for (RequestBlockRegistry::RequestBlockStatus status :
           registry_->Status(keys)) {
        using ProtoStatus = tpu_sync::rpc::GetRequestBlockStatusResponse;
        switch (status) {
          case RequestBlockRegistry::RequestBlockStatus::kRegistered:
            out->add_statuses(ProtoStatus::STATUS_REGISTERED);
            break;
          case RequestBlockRegistry::RequestBlockStatus::kClaimed:
            out->add_statuses(ProtoStatus::STATUS_CLAIMED);
            break;
          case RequestBlockRegistry::RequestBlockStatus::kCancelled:
            out->add_statuses(ProtoStatus::STATUS_CANCELLED);
            break;
          case RequestBlockRegistry::RequestBlockStatus::kUnknown:
            out->add_statuses(ProtoStatus::STATUS_UNKNOWN);
            break;
        }
      }
      resp.set_success(true);
      break;
    }
    default:
      resp.set_message("COMMAND_UNSPECIFIED");
      break;
  }
  return resp.SerializeAsString();
}

std::string ReshardService::HandleRaidenCommand(
    const std::string& request_bytes) {
  tpu_sync::rpc::ControlRequest raiden_req;
  tpu_sync::rpc::ControlResponse raiden_resp;
  raiden_resp.set_success(false);
  if (!raiden_req.ParseFromString(request_bytes)) {
    raiden_resp.set_message("Failed to parse ControlRequest");
    return raiden_resp.SerializeAsString();
  }

  switch (raiden_req.command()) {
    case tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT: {
      const auto& reg = raiden_req.register_work_unit_request();
      WorkUnitRegistration registration;
      registration.unit = RaidenIdFromProto(reg.unit());
      registration.shards.assign(reg.shards().begin(), reg.shards().end());
      if (!reg.control_plane_rpc_address().empty()) {
        registration.control_plane_rpc_address =
            reg.control_plane_rpc_address();
      }
      if (reg.mesh_shape_size() > 0) {
        registration.mesh_shape.emplace(reg.mesh_shape().begin(),
                                        reg.mesh_shape().end());
      }
      if (reg.layout_size() > 0) {
        registration.layout.emplace(reg.layout().begin(), reg.layout().end());
      }
      if (reg.global_shape_size() > 0) {
        registration.global_shape.emplace(reg.global_shape().begin(),
                                          reg.global_shape().end());
      }
      if (reg.itemsize() > 0) registration.itemsize = reg.itemsize();
      if (reg.pools_size() > 0) {
        registration.pool_manifest.emplace(reg.pools().begin(),
                                           reg.pools().end());
      }
      if (!reg.layout_fingerprint().empty()) {
        registration.layout_fingerprint = reg.layout_fingerprint();
      }
      if (reg.page_tokens() > 0) {
        registration.page_tokens = reg.page_tokens();
      }
      if (reg.transfer_parallelism() > 0) {
        registration.transfer_parallelism = reg.transfer_parallelism();
      }
      // Python quirk preserved: transfer_rank rides only with a manifest.
      if (registration.pool_manifest.has_value()) {
        registration.transfer_rank = reg.transfer_rank();
      }
      if (reg.variables_size() > 0) {
        registration.variables.emplace(reg.variables().begin(),
                                       reg.variables().end());
      }
      absl::Status status = WorkUnitDirectory::Validate(registration);
      if (!status.ok()) {
        raiden_resp.set_message(StatusMessage(status));
        break;
      }
      {
        absl::MutexLock lock(state_mu_);
        directory_->mu()->AssertHeld();
        directory_->RegisterLocked(registration);
        // A worker restart invalidates every request-local physical block
        // ID previously reported by that identity.
        registry_->mu()->AssertHeld();
        registry_->InvalidateUnitLocked(registration.unit);
      }
      raiden_resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControlRequest::COMMAND_GET_METADATA: {
      for (auto& metadata : directory_->AllMetadata()) {
        *raiden_resp.mutable_get_metadata_response()->add_metadata() =
            std::move(metadata);
      }
      raiden_resp.set_success(true);
      break;
    }
    case tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_TRANSFER_SCHEDULE: {
      const auto& start_req = raiden_req.start_transfer_request();
      if (start_req.transfer_pool_indices_size() > 0 ||
          start_req.pool_groups_size() > 0) {
        raiden_resp.set_message(
            "Inter-controller reshard schedule registration is retired: "
            "the planning controller arms destination workers directly");
      } else {
        raiden_resp.set_message(
            "Legacy transfer scheduling is not implemented by the C++ "
            "reshard controller; keep the Python sidecar for "
            "dense/weight-sync paths");
      }
      break;
    }
    case tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER: {
      return HandleRaidenStartTransfer(raiden_req);
    }
    case tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN: {
      // Best-effort worker shutdown fan-out, then stop (the Python
      // controller's shutdown_workers + stop()).
      std::set<std::string> addresses;
      for (const auto& metadata : directory_->AllMetadata()) {
        if (!metadata.control_plane_rpc_address().empty()) {
          addresses.insert(metadata.control_plane_rpc_address());
        }
      }
      tpu_sync::rpc::ControlRequest shutdown_req;
      shutdown_req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN);
      const std::string payload = shutdown_req.SerializeAsString();
      for (const std::string& address : addresses) {
        transport_->Call(address, payload, absl::Seconds(5))
            .status()
            .IgnoreError();
      }
      raiden_resp.set_success(true);
      if (shutdown_callback_) shutdown_callback_();
      break;
    }
    default:
      raiden_resp.set_message("COMMAND_UNSPECIFIED");
      break;
  }
  return raiden_resp.SerializeAsString();
}

std::string ReshardService::HandleRaidenStartTransfer(
    const tpu_sync::rpc::ControlRequest& req) {
  tpu_sync::rpc::ControlResponse resp;
  resp.set_success(false);
  if (delivery_.mode != WorkerDelivery::Mode::kController) {
    // In framed mode, START_TRANSFER is a worker-side RPC; the framed
    // controller is not the destination for it.
    resp.set_message(
        "START_TRANSFER is handled by worker listeners in framed mode");
    return resp.SerializeAsString();
  }
  if (delivery_.controller == nullptr) {
    resp.set_message("ReshardService has no RaidenController configured");
    return resp.SerializeAsString();
  }
  const auto& start_req = req.start_transfer_request();
  auto compiled = core::CompileStartTransfer(start_req);
  if (!compiled.ok()) {
    resp.set_message(StatusMessage(compiled.status()));
    return resp.SerializeAsString();
  }
  // The receiver arm's destination unit names the worker to target.
  // In single-host deployments this is worker_<transfer_rank> (defaulting
  // to worker_0).
  std::string worker_id = "worker_0";
  if (start_req.dst_units_size() > 0) {
    auto meta = directory_->LocalMetadata(
        {RaidenIdFromProto(start_req.dst_units(0))});
    if (meta.ok() && !meta->empty() && (*meta)[0].transfer_rank() >= 0) {
      worker_id = absl::StrCat("worker_", (*meta)[0].transfer_rank());
    }
  }
  auto submit_resp =
      delivery_.controller->SubmitTransferProgram(worker_id, *compiled).Await();
  if (!submit_resp.ok()) {
    resp.set_message(StatusMessage(submit_resp.status()));
    return resp.SerializeAsString();
  }
  resp.set_success(submit_resp->success());
  resp.set_message(submit_resp->message());
  return resp.SerializeAsString();
}

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
