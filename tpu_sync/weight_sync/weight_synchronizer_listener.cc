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
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"
#include "tpu_sync/weight_sync/host_tiling_validation.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"
#include "tpu_sync/weight_sync/weight_version.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

constexpr char kStoppingMessage[] = "WeightSynchronizerListener is stopping";

// The longest request timeout (for example `receive_timeout_ms`) of
// weight_transfer_worker_service.proto.
constexpr absl::Duration kMaxTransferTimeout = absl::Hours(1);

// Returns InvalidArgument unless `value_ms` milliseconds lie in [0, `max`].
absl::Status CheckDurationMs(int64_t value_ms, absl::Duration max) {
  if (value_ms >= 0 && absl::Milliseconds(value_ms) <= max) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "must be in [0, ", absl::FormatDuration(max), "]; got ", value_ms));
}

// Returns an error `status` with the concatenated `prefix_parts` prepended to
// its message, and an OK `status` unchanged without building the prefix. It
// stands in for the status builder of ABSL_RETURN_IF_ERROR, which the
// open-source Abseil backport (third_party/abseil/status_macros.patch) lacks.
template <typename... Args>
absl::Status Prepend(absl::Status status, const Args&... prefix_parts) {
  if (status.ok()) return status;
  return absl::Status(status.code(),
                      absl::StrCat(prefix_parts..., status.message()));
}

// Prefix of the sender ids that Raiden reserves for its own senders.
constexpr absl::string_view kReservedSenderPrefix = "raiden/";

// Returns InvalidArgument if `req` names a sender with the reserved prefix.
// Only the RPC handler applies this check: a puller prepares its own receive
// as "raiden/pull" through WeightSynchronizerBase, so the check must not move
// into code the engine shares when the coordinated prepare moves into
// WeightSynchronizerBase.
absl::Status RejectReservedSenders(
    const ::tpu_sync::rpc::PrepareWeightReceiveRequest& req) {
  std::vector<absl::string_view> senders(req.expected_senders().begin(),
                                         req.expected_senders().end());
  for (const ::tpu_sync::rpc::Chunks& chunks : req.chunks()) {
    senders.push_back(chunks.sender());
  }
  for (absl::string_view sender : senders) {
    if (absl::StartsWith(sender, kReservedSenderPrefix)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "PrepareWeightReceive (uuid=", req.uuid(), ") names sender '", sender,
          "'; ids starting with ", kReservedSenderPrefix, " are reserved"));
    }
  }
  return absl::OkStatus();
}

// Returns InvalidArgument naming the field if direct push `direct` of
// `push_name`, with push deadline `timeout_ms`, is malformed. Schedule keys and
// identity shards need the engine's shards and are not checked yet.
absl::Status CheckDirectPush(absl::string_view push_name,
                             const ::tpu_sync::rpc::DirectPush& direct,
                             int64_t timeout_ms) {
  const std::string& sender = direct.sender();
  if (sender.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat(push_name, " sets no direct.sender"));
  }
  if (absl::StartsWith(sender, kReservedSenderPrefix)) {
    return absl::InvalidArgumentError(absl::StrCat(
        push_name, " sets direct.sender '", sender, "'; ids starting with ",
        kReservedSenderPrefix, " are reserved"));
  }
  if (direct.destinations().empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat(push_name, " sets no direct.destinations"));
  }
  for (int i = 0; i < direct.destinations_size(); ++i) {
    const ::tpu_sync::rpc::PushDestination& destination =
        direct.destinations(i);
    const std::string field = absl::StrCat("direct.destinations[", i, "]");
    if (destination.listener_endpoint().empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat(push_name, " sets no ", field, ".listener_endpoint"));
    }
    if (destination.plan_case() ==
        ::tpu_sync::rpc::PushDestination::PLAN_NOT_SET) {
      return absl::InvalidArgumentError(absl::StrCat(
          push_name, " sets no ", field, ".plan; set schedules or identity"));
    }
    const auto& senders = destination.expected_senders();
    if (std::find(senders.begin(), senders.end(), sender) == senders.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat(push_name, " sets ", field, ".expected_senders to [",
                       absl::StrJoin(senders, ", "),
                       "], which lacks direct.sender '", sender, "'"));
    }
  }
  // The receive must not time out before the push. 0 sets no deadline (valid
  // only until deadlines are implemented), so only positive deadlines compare.
  if (timeout_ms > 0 && direct.receive_timeout_ms() > 0 &&
      direct.receive_timeout_ms() < timeout_ms) {
    return absl::InvalidArgumentError(absl::StrCat(
        push_name, " sets direct.receive_timeout_ms=",
        direct.receive_timeout_ms(), ", less than timeout_ms=", timeout_ms));
  }
  ABSL_RETURN_IF_ERROR(
      CheckKnownHostTilingMode(push_name, "direct.receiver_host_tiling_mode",
                               direct.receiver_host_tiling_mode()));
  return absl::OkStatus();
}

// The most chunks a receive can expect, in total and from one sender per
// layer: the transport counts a transfer's chunks in 32 bits.
constexpr int64_t kMaxReceiveChunks = std::numeric_limits<uint32_t>::max();

// Sets the expected chunk counts of `recv` from the chunks of `req`: per layer,
// the sum of every expected sender's chunk counts, and their total. Validates
// every sender's chunks before they are summed. A valid prepare that covers
// only some expected senders is not supported yet.
absl::Status SumChunkCounts(
    const WeightSynchronizerBase& engine,
    const ::tpu_sync::rpc::PrepareWeightReceiveRequest& req,
    ::tpu_sync::rpc::StartTransferRequest& recv) {
  // Expected sender -> whether `req.chunks()` has an entry of it.
  absl::flat_hash_map<absl::string_view, bool> has_chunks;
  for (const std::string& sender : req.expected_senders()) {
    if (sender.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("PrepareWeightReceive (uuid=", req.uuid(),
                       ") expects a sender with an empty id"));
    }
    if (!has_chunks.emplace(sender, false).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("PrepareWeightReceive (uuid=", req.uuid(),
                       ") expects sender '", sender, "' twice"));
    }
  }
  if (has_chunks.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "PrepareWeightReceive (uuid=", req.uuid(), ") expects no senders"));
  }
  if (req.chunks().empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "PrepareWeightReceive (uuid=", req.uuid(), ") carries no chunks"));
  }
  // Bounds every partial sum, so no addition below can overflow.
  int64_t total_chunks = 0;
  for (const ::tpu_sync::rpc::Chunks& chunks : req.chunks()) {
    const std::string& sender = chunks.sender();
    auto it = has_chunks.find(sender);
    if (it == has_chunks.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("PrepareWeightReceive (uuid=", req.uuid(),
                       ") carries chunks of sender '", sender,
                       "', which is not in expected_senders"));
    }
    if (it->second) {
      return absl::InvalidArgumentError(
          absl::StrCat("PrepareWeightReceive (uuid=", req.uuid(),
                       ") carries chunks of sender '", sender, "' twice"));
    }
    it->second = true;
    // Rejects `sender`'s count of `count` chunks of layer `layer_idx`.
    auto invalid_count = [&req, &sender](int32_t layer_idx, int64_t count,
                                         absl::string_view reason) {
      return absl::InvalidArgumentError(absl::StrCat(
          "PrepareWeightReceive (uuid=", req.uuid(), "): sender '", sender,
          "' maps layer ", layer_idx, " to ", count, " chunks; ", reason));
    };
    int64_t sender_chunks = 0;
    for (const auto& [layer_idx, count] : chunks.layer_chunk_counts()) {
      if (layer_idx < 0 ||
          static_cast<size_t>(layer_idx) >= engine.num_layers()) {
        return invalid_count(
            layer_idx, count,
            absl::StrCat("layers lie in [0, ", engine.num_layers(), ")"));
      }
      if (count < 0 || count > kMaxReceiveChunks) {
        return invalid_count(
            layer_idx, count,
            absl::StrCat("counts lie in [0, ", kMaxReceiveChunks, "]"));
      }
      sender_chunks += count;
      total_chunks += count;
      if (total_chunks > kMaxReceiveChunks) {
        return invalid_count(layer_idx, count,
                             absl::StrCat("the summed counts exceed ",
                                          kMaxReceiveChunks, " chunks"));
      }
      (*recv.mutable_expected_layer_chunk_counts())[layer_idx] += count;
    }
    if (sender_chunks == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("PrepareWeightReceive (uuid=", req.uuid(), "): sender '",
                       sender, "' sends no chunks"));
    }
  }
  std::vector<absl::string_view> missing;
  for (const std::string& sender : req.expected_senders()) {
    if (!has_chunks.at(sender)) missing.push_back(sender);
  }
  if (!missing.empty()) {
    return absl::UnimplementedError(absl::StrCat(
        "PrepareWeightReceive (uuid=", req.uuid(),
        ") carries no chunks of expected sender(s) ",
        absl::StrJoin(missing, ", "),
        "; prepares that cover only some expected_senders are not implemented "
        "yet"));
  }
  recv.set_expected_block_count(total_chunks);
  return absl::OkStatus();
}

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
  ABSL_RETURN_IF_ERROR(CheckKnownHostTilingMode(
      absl::StrCat("COMMAND_START_TRANSFER (uuid=", uuid, ")"),
      "host_tiling_mode", start_req.host_tiling_mode()));
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
  RegisterHandler(&WeightSynchronizerListener::HandleAwaitWeightReceive,
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
  ABSL_RETURN_IF_ERROR(CheckKnownHostTilingMode(
      absl::StrCat("PrepareWeightReceive (uuid=", req.uuid(), ")"),
      "host_tiling_mode", req.host_tiling_mode()));
  // TODO(raiden-eng): Commit the version into the staging buffer once the
  // buffer tracks versions; until then nothing on the worker reads it.
  ABSL_RETURN_IF_ERROR(
      WeightVersion::FromProto(
          req.version(),
          absl::StrCat("PrepareWeightReceive (uuid=", req.uuid(), ") version"))
          .status());
  ABSL_RETURN_IF_ERROR(CheckTargetIncarnation(*engine_, "PrepareWeightReceive",
                                              req.uuid(),
                                              req.target_incarnation_id()));
  ABSL_RETURN_IF_ERROR(RejectReservedSenders(req));
  ABSL_RETURN_IF_ERROR(Prepend(
      CheckDurationMs(req.receive_timeout_ms(), kMaxTransferTimeout),
      "PrepareWeightReceive (uuid=", req.uuid(), ") receive_timeout_ms "));
  if (req.has_report_to() && (req.report_to().endpoint().empty() ||
                              req.report_to().receiver().empty())) {
    return absl::InvalidArgumentError(
        absl::StrCat("PrepareWeightReceive (uuid=", req.uuid(),
                     ") sets report_to with an empty endpoint or receiver"));
  }

  ::tpu_sync::rpc::StartTransferRequest recv;
  recv.set_is_sender(false);
  recv.set_uuid(req.uuid());
  ABSL_RETURN_IF_ERROR(SumChunkCounts(*engine_, req, recv));
  if (req.receive_timeout_ms() != 0) {
    return absl::UnimplementedError(
        absl::StrCat("PrepareWeightReceive (uuid=", req.uuid(),
                     ") sets receive_timeout_ms=", req.receive_timeout_ms(),
                     "; receive deadlines are not implemented yet"));
  }
  if (req.has_report_to()) {
    return absl::UnimplementedError(absl::StrCat(
        "PrepareWeightReceive (uuid=", req.uuid(),
        ") sets report_to; reporting a receive's outcome is not implemented "
        "yet"));
  }
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
  const std::string push_name =
      absl::StrCat("StartWeightPush (uuid=", req.uuid(), ")");
  if (req.mode_case() ==
      ::tpu_sync::rpc::StartWeightPushRequest::MODE_NOT_SET) {
    return absl::InvalidArgumentError(
        absl::StrCat(push_name, " sets no mode; set coordinated or direct"));
  }
  ABSL_RETURN_IF_ERROR(CheckKnownHostTilingMode(push_name, "host_tiling_mode",
                                                req.host_tiling_mode()));
  const ::tpu_sync::rpc::CoordinatedPush& coordinated = req.coordinated();
  // Without StartTransferRequest.src_units, PushWeightsResharded() cannot
  // infer the layer of an entry.
  for (const auto& [shard, schedule] : coordinated.shard_push_schedules()) {
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
  ABSL_RETURN_IF_ERROR(
      Prepend(CheckDurationMs(req.timeout_ms(), kMaxTransferTimeout), push_name,
              " timeout_ms "));
  ABSL_RETURN_IF_ERROR(Prepend(
      CheckDurationMs(req.direct().receive_timeout_ms(), kMaxTransferTimeout),
      push_name, " direct.receive_timeout_ms "));
  // TODO(raiden-eng): Bind the version in the staging buffer and enforce
  // `timeout_ms` once versioned pushes and deadlines are implemented; an empty
  // `version` tag denotes the default version (version 0).
  ABSL_RETURN_IF_ERROR(WeightVersion::FromProto(
                           req.version(), absl::StrCat(push_name, " version"))
                           .status());
  if (req.has_direct()) {
    ABSL_RETURN_IF_ERROR(
        CheckDirectPush(push_name, req.direct(), req.timeout_ms()));
    return absl::UnimplementedError(absl::StrCat(
        push_name, " sets mode direct; direct pushes are not implemented yet"));
  }

  ::tpu_sync::rpc::StartTransferRequest push;
  push.set_is_sender(true);
  push.set_uuid(req.uuid());
  *push.mutable_shard_push_schedules() = coordinated.shard_push_schedules();
  push.set_skip_d2h(req.skip_d2h());
  push.set_host_tiling_mode(req.host_tiling_mode());
  *push.mutable_skip_tiling() = req.skip_tiling();
  push.set_parallelism(req.parallelism());
  push.set_req_id(req.req_id());
  *push.mutable_peer_control_endpoints() = coordinated.peer_control_endpoints();
  ABSL_RETURN_IF_ERROR(engine_->PushWeightsResharded(push));
  return ::tpu_sync::rpc::StartWeightPushResponse();
}

absl::StatusOr<::tpu_sync::rpc::AwaitWeightReceiveResponse>
WeightSynchronizerListener::HandleAwaitWeightReceive(
    const ::tpu_sync::rpc::AwaitWeightReceiveRequest& req) {
  if (stopping_.load()) {
    return absl::UnavailableError(kStoppingMessage);
  }
  ABSL_RETURN_IF_ERROR(CheckTargetIncarnation(
      *engine_, "AwaitWeightReceive", req.uuid(), req.target_incarnation_id()));
  if (req.uuid() <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "AwaitWeightReceive requires a positive uuid; got ", req.uuid()));
  }
  const std::string await_name =
      absl::StrCat("AwaitWeightReceive (uuid=", req.uuid(), ")");
  ABSL_ASSIGN_OR_RETURN(
      const WeightVersion version,
      WeightVersion::FromProto(req.version(),
                               absl::StrCat(await_name, " version")));
  if (req.timeout_ms() <= 0 ||
      absl::Milliseconds(req.timeout_ms()) > kMaxTransferTimeout) {
    return absl::InvalidArgumentError(
        absl::StrCat(await_name, " timeout_ms must be in (0, ",
                     absl::FormatDuration(kMaxTransferTimeout), "]; got ",
                     req.timeout_ms()));
  }
  return absl::UnimplementedError(
      absl::StrCat(await_name, " for ", version,
                   ": awaiting a receive's commit is not implemented yet"));
}

}  // namespace weight_sync
}  // namespace tpu_raiden
