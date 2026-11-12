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

#include "tpu_sync/weight_sync/weight_synchronizer_listener.h"

#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

constexpr char kStoppingMessage[] = "WeightSynchronizerListener is stopping";

// Validates the expected chunk counts of the receiver-side `start_req` and
// registers them, with the transfer's tiling settings, so that `engine` can
// account for the chunks of transfer `start_req.uuid()` as they arrive.
absl::Status RegisterIncomingTransfer(
    WeightSynchronizerBase& engine,
    const ::tpu_sync::rpc::StartTransferRequest& start_req) {
  const int64_t expected_block_count = start_req.expected_block_count();
  const uint64_t uuid = start_req.uuid();
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1))) {
    VLOG(1) << "RAIDEN_DIAG recv arm uuid=" << uuid
            << " req_id=" << start_req.req_id()
            << " expected_block_count=" << expected_block_count;
  }
  if (expected_block_count <= 0 ||
      expected_block_count > std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError(
        "expected_block_count must be positive and fit in 32-bit uint");
  }
  const auto& layer_counts_proto = start_req.expected_layer_chunk_counts();
  // Every received chunk must be attributed to a layer so that each layer
  // with data fires OnLayerDataReceived before OnDataReceived.
  int64_t total_layer_chunks = 0;
  for (const auto& [layer_idx, count] : layer_counts_proto) {
    if (layer_idx < 0 ||
        static_cast<size_t>(layer_idx) >= engine.num_layers() || count < 0 ||
        count > std::numeric_limits<uint32_t>::max()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "expected_layer_chunk_counts maps layer ", layer_idx, " to ", count,
          " chunks; layers must be in [0, ", engine.num_layers(),
          ") and counts in [0, ", std::numeric_limits<uint32_t>::max(), "]"));
    }
    total_layer_chunks += count;
  }
  if (layer_counts_proto.empty() ||
      total_layer_chunks != expected_block_count) {
    return absl::InvalidArgumentError(
        absl::StrCat("expected_layer_chunk_counts must be non-empty and sum to "
                     "expected_block_count=",
                     expected_block_count, "; got ", layer_counts_proto.size(),
                     " layer(s) summing to ", total_layer_chunks));
  }
  engine.StoreSkipTiling(uuid, start_req);

  absl::flat_hash_map<size_t, uint32_t> layer_counts;
  for (const auto& [layer_idx, count] : layer_counts_proto) {
    layer_counts[static_cast<size_t>(layer_idx)] = static_cast<uint32_t>(count);
  }
  ABSL_RETURN_IF_ERROR(engine.RegisterExpectedLayerChunks(uuid, layer_counts));
  return engine.RegisterExpectedChunks(
      uuid, static_cast<uint32_t>(expected_block_count));
}

// Returns FailedPrecondition if `target_incarnation_id` is neither 0 nor
// `engine`'s incarnation ID, i.e. the `rpc` request was meant for another
// incarnation of this work unit.
absl::Status CheckTargetIncarnation(const WeightSynchronizerBase& engine,
                                    absl::string_view rpc, int64_t uuid,
                                    int64_t target_incarnation_id) {
  if (target_incarnation_id == 0 ||
      target_incarnation_id == engine.incarnation_id()) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(absl::StrCat(
      rpc, " (uuid=", uuid, ") targets incarnation_id=", target_incarnation_id,
      ", but this worker's is ", engine.incarnation_id()));
}

}  // namespace

ControlPipeBackendType WeightSynchronizerListener::ResolveBackendType(
    std::optional<ControlPipeBackendType> backend_type) {
  if (backend_type.has_value()) {
    return *backend_type;
  }
  if (const char* env_backend = std::getenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND");
      env_backend != nullptr && env_backend[0] != '\0') {
    if (std::optional<ControlPipeBackendType> parsed =
            ParseControlPipeBackendType(env_backend);
        parsed.has_value()) {
      return *parsed;
    }
    LOG(WARNING) << "Unknown TPU_RAIDEN_CONTROL_PLANE_BACKEND='" << env_backend
                 << "', falling back to default.";
    return ControlPipeBackendType::kGrpc;
  }
  if (const char* env_flag = std::getenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE")) {
    std::string val = absl::AsciiStrToLower(env_flag);
    if (val == "0" || val == "false") {
      return ControlPipeBackendType::kTcp;
    }
    if (val == "1" || val == "true") {
      return ControlPipeBackendType::kGrpc;
    }
  }
  return ControlPipeBackendType::kGrpc;
}

template <typename Request, typename Response>
void WeightSynchronizerListener::RegisterHandler(
    absl::StatusOr<Response> (WeightSynchronizerListener::*handle)(
        const Request&),
    size_t max_payload_bytes) {
  pipe_server_->dispatcher().RegisterHandler<Request, Response>(
      [this, handle](const ControlContext& /*ctx*/,
                     const Request& req) -> absl::StatusOr<Response> {
        absl::StatusOr<Response> resp = (this->*handle)(req);
        if (!resp.ok()) {
          LOG(ERROR) << Request::descriptor()->name()
                     << " failed: " << resp.status();
        }
        return resp;
      },
      HandlerOptions<Request>().WithMaxPayloadBytes(max_payload_bytes));
}

WeightSynchronizerListener::WeightSynchronizerListener(
    WeightSynchronizerBase* engine, int listener_port,
    std::optional<ControlPipeBackendType> backend_type)
    : engine_(engine), listener_port_(listener_port) {
  ControlPipeConfig cfg;
  cfg.backend_type = ResolveBackendType(backend_type);
  cfg.requested_port = listener_port;
  cfg.allow_legacy_framing = (cfg.backend_type == ControlPipeBackendType::kTcp);

  pipe_server_ = CreateControlPipeServer(cfg);
  RegisterHandler(&WeightSynchronizerListener::HandleControlRequest,
                  cfg.max_frame_bytes);
  RegisterHandler(&WeightSynchronizerListener::HandlePrepareWeightReceive,
                  cfg.max_frame_bytes);
  RegisterHandler(&WeightSynchronizerListener::HandleStartWeightPush,
                  cfg.max_frame_bytes);

  if (pipe_server_->backend_type() == ControlPipeBackendType::kGrpc) {
    CHECK_OK(
        pipe_server_->RegisterGrpcService(engine_->peregrine_control_service()))
        << "Failed to register PeregrineService on the listener";
  } else {
    // No gRPC server exists on this backend, so PeregrineService cannot be
    // served. That is only acceptable when PSP-TCP is not required.
    CHECK(!engine_->psp_tcp_required())
        << "PSP-TCP is required but the listener backend '"
        << ControlPipeBackendTypeName(pipe_server_->backend_type())
        << "' cannot serve PeregrineService; use the gRPC control plane";
  }

  absl::StatusOr<int> port = pipe_server_->Start(listener_port);
  CHECK_OK(port.status())
      << "Failed to start WeightSynchronizerListener ControlPipeServer";
  listener_port_ = *port;
  LOG(INFO) << "Native C++ WeightSynchronizerListener actively listening "
               "on port: "
            << listener_port_ << " (backend="
            << ControlPipeBackendTypeName(pipe_server_->backend_type()) << ")";
}

WeightSynchronizerListener::~WeightSynchronizerListener() { Shutdown(); }

void WeightSynchronizerListener::Shutdown() {
  stopping_.store(true);
  if (pipe_server_) {
    pipe_server_->Stop();
  }
}

absl::StatusOr<::tpu_sync::rpc::ControlResponse>
WeightSynchronizerListener::HandleControlRequest(
    const ::tpu_sync::rpc::ControlRequest& req) {
  ::tpu_sync::rpc::ControlResponse resp;
  if (req.command() ==
          ::tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER &&
      stopping_.load()) {
    resp.set_success(false);
    resp.set_message(kStoppingMessage);
    return resp;
  }
  ExecuteControlRequest(engine_, req, &resp,
                        [this]() { stopping_.store(true); });
  return resp;
}

void WeightSynchronizerListener::ExecuteControlRequest(
    WeightSynchronizerBase* engine, const ::tpu_sync::rpc::ControlRequest& req,
    ::tpu_sync::rpc::ControlResponse* resp,
    std::function<void()> shutdown_callback) {
  resp->set_success(true);
  resp->set_message("SUCCESS");

  if (req.command() ==
      ::tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER) {
    bool is_sender = true;
    bool is_resharded = false;
    if (req.has_start_transfer_request()) {
      const auto& start_req = req.start_transfer_request();
      is_sender = start_req.is_sender();
      is_resharded = !start_req.shard_push_schedules().empty();

      if (start_req.broadcast_round_destinations_size() > 0 ||
          start_req.has_broadcast_round()) {
        std::string src_unit_str =
            start_req.src_units().empty()
                ? "unknown"
                : absl::StrCat(start_req.src_units(0).job_name(), ":",
                               start_req.src_units(0).job_replica_id());
        std::string round_str = start_req.has_broadcast_round()
                                    ? absl::StrCat(start_req.broadcast_round())
                                    : "none";
        std::vector<std::string> round_summaries;
        round_summaries.reserve(start_req.broadcast_round_destinations_size());
        for (const auto& rd : start_req.broadcast_round_destinations()) {
          std::vector<std::string> dest_entries;
          if (rd.dst_units_size() == 1 && rd.dst_peers_size() > 1) {
            dest_entries.reserve(rd.dst_peers_size());
            std::string u = rd.dst_units(0);
            for (const auto& p : rd.dst_peers()) {
              dest_entries.push_back(absl::StrCat(u, " (", p, ")"));
            }
          } else {
            int count = std::max(rd.dst_units_size(), rd.dst_peers_size());
            dest_entries.reserve(count);
            for (int i = 0; i < count; ++i) {
              std::string u = (i < rd.dst_units_size()) ? rd.dst_units(i) : "";
              std::string p = (i < rd.dst_peers_size()) ? rd.dst_peers(i) : "";
              if (!u.empty() && !p.empty()) {
                dest_entries.push_back(absl::StrCat(u, " (", p, ")"));
              } else if (!u.empty()) {
                dest_entries.push_back(u);
              } else if (!p.empty()) {
                dest_entries.push_back(p);
              }
            }
          }
          round_summaries.push_back(
              absl::StrCat("Round ", rd.round_idx(), ": [",
                           absl::StrJoin(dest_entries, ", "), "]"));
        }
        std::string schedule_summary =
            round_summaries.empty() ? "none"
                                    : absl::StrJoin(round_summaries, ", ");
        LOG(INFO) << "WeightSynchronizerListener [src_unit=" << src_unit_str
                  << ", req_id=" << start_req.req_id()
                  << ", uuid=" << start_req.uuid()
                  << ", active_round=" << round_str
                  << "] broadcast round destinations: " << schedule_summary;
      }
    }

    if (is_sender) {
      if (is_resharded) {
        LOG(INFO) << "C++ Listener executing PushWeightsResharded";
        const auto& start_req = req.start_transfer_request();
        if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1))) {
          int64_t total_entries = 0;
          for (const auto& [shard_idx, schedule] :
               start_req.shard_push_schedules()) {
            total_entries += schedule.entries_size();
          }
          std::string round_str =
              start_req.has_broadcast_round()
                  ? absl::StrCat(start_req.broadcast_round())
                  : "none";
          VLOG(1)
              << "RAIDEN_DIAG push C++ Listener executing PushWeightsResharded"
              << " req_id=" << start_req.req_id()
              << " uuid=" << start_req.uuid() << " is_sender=" << is_sender
              << " broadcast_round=" << round_str << " pid=" << getpid()
              << " shards=" << start_req.shard_push_schedules_size()
              << " schedule_entries=" << total_entries;
        }
        absl::Status status = engine->PushWeightsResharded(start_req);
        if (!status.ok()) {
          resp->set_success(false);
          resp->set_message(std::string(status.message()));
          LOG(ERROR) << "PushWeightsResharded native execution failed: "
                     << status;
        }
      } else {
        std::vector<std::string> peers(req.peers().begin(), req.peers().end());
        LOG(INFO) << "C++ Listener executing PushWeights to " << peers.size()
                  << " peers";
        if (!peers.empty()) {
          // PushWeights() takes no request, so the channels for its PSP-TCP
          // key exchanges are recorded here.
          absl::Status status = engine->RecordPeerControlEndpoints(
              peers, req.start_transfer_request());
          if (status.ok()) status = engine->PushWeights(peers);
          if (!status.ok()) {
            resp->set_success(false);
            resp->set_message(std::string(status.message()));
            LOG(ERROR) << "PushWeights native execution failed: " << status;
          }
        }
      }
    } else {
      LOG(INFO) << "C++ Listener received START_TRANSFER (Receiver) - "
                   "registering expected block count";
      if (absl::Status status =
              RegisterIncomingTransfer(*engine, req.start_transfer_request());
          !status.ok()) {
        resp->set_success(false);
        resp->set_message(std::string(status.message()));
        LOG(ERROR) << "START_TRANSFER (Receiver) failed: " << status;
      }
    }
  } else if (req.command() ==
             ::tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN) {
    LOG(INFO) << "C++ Listener received SHUTDOWN command. Draining pending H2D "
                 "and initiating clean exit.";
    if (engine != nullptr) {
      engine->DrainPendingH2d();
    }
    if (shutdown_callback) {
      shutdown_callback();
    }
    resp->set_success(true);
  } else {
    resp->set_success(false);
    resp->set_message("COMMAND_UNSPECIFIED");
    LOG(WARNING) << "C++ Listener received unknown or unspecified "
                    "Protobuf command";
  }
}

absl::StatusOr<::tpu_sync::rpc::PrepareWeightReceiveResponse>
WeightSynchronizerListener::HandlePrepareWeightReceive(
    const ::tpu_sync::rpc::PrepareWeightReceiveRequest& req) {
  if (stopping_.load()) {
    return absl::UnavailableError(kStoppingMessage);
  }
  LOG(INFO) << "C++ Listener executing PrepareWeightReceive (uuid="
            << req.uuid() << ", req_id=" << req.req_id() << ")";
  if (req.uuid() <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "PrepareWeightReceive requires a positive uuid; got ", req.uuid()));
  }
  // TODO(raiden-eng): Commit the version into the staging buffer once the
  // buffer tracks versions; until then nothing on the worker reads it.
  if (req.version().epoch() == 0 && req.version().version_id() == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "PrepareWeightReceive (uuid=", req.uuid(), ") carries no version"));
  }
  ABSL_RETURN_IF_ERROR(CheckTargetIncarnation(*engine_, "PrepareWeightReceive",
                                              req.uuid(),
                                              req.target_incarnation_id()));

  ::tpu_sync::rpc::StartTransferRequest recv;
  recv.set_is_sender(false);
  recv.set_uuid(req.uuid());
  recv.set_expected_block_count(req.expected_block_count());
  *recv.mutable_expected_layer_chunk_counts() =
      req.expected_layer_chunk_counts();
  recv.set_host_tiling_mode(req.host_tiling_mode());
  *recv.mutable_skip_tiling() = req.skip_tiling();
  recv.set_req_id(req.req_id());
  ABSL_RETURN_IF_ERROR(RegisterIncomingTransfer(*engine_, recv));
  return ::tpu_sync::rpc::PrepareWeightReceiveResponse();
}

absl::StatusOr<::tpu_sync::rpc::StartWeightPushResponse>
WeightSynchronizerListener::HandleStartWeightPush(
    const ::tpu_sync::rpc::StartWeightPushRequest& req) {
  if (stopping_.load()) {
    return absl::UnavailableError(kStoppingMessage);
  }
  LOG(INFO) << "C++ Listener executing StartWeightPush (uuid=" << req.uuid()
            << ", req_id=" << req.req_id() << ")";
  if (req.uuid() <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "StartWeightPush requires a positive uuid; got ", req.uuid()));
  }
  // Without StartTransferRequest.src_units, PushWeightsResharded() cannot
  // infer the layer of an entry.
  for (const auto& [shard, schedule] : req.shard_push_schedules()) {
    for (const ::tpu_sync::rpc::ShardPushEntryProto& entry :
         schedule.entries()) {
      if (!entry.has_layer_idx()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "StartWeightPush (uuid=", req.uuid(),
            ") has a schedule entry without layer_idx for shard ", shard));
      }
    }
  }
  ABSL_RETURN_IF_ERROR(CheckTargetIncarnation(
      *engine_, "StartWeightPush", req.uuid(), req.target_incarnation_id()));

  ::tpu_sync::rpc::StartTransferRequest push;
  push.set_is_sender(true);
  push.set_uuid(req.uuid());
  *push.mutable_shard_push_schedules() = req.shard_push_schedules();
  push.set_skip_d2h(req.skip_d2h());
  push.set_host_tiling_mode(req.host_tiling_mode());
  *push.mutable_skip_tiling() = req.skip_tiling();
  push.set_parallelism(req.parallelism());
  push.set_req_id(req.req_id());
  *push.mutable_peer_control_endpoints() = req.peer_control_endpoints();
  ABSL_RETURN_IF_ERROR(engine_->PushWeightsResharded(push));
  return ::tpu_sync::rpc::StartWeightPushResponse();
}

}  // namespace weight_sync
}  // namespace tpu_raiden
