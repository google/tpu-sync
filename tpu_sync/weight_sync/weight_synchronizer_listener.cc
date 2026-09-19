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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {

WeightSynchronizerListener::WeightSynchronizerListener(
    WeightSynchronizerBase* engine, int listener_port,
    ControlPipeBackendType backend_type)
    : engine_(engine), listener_port_(listener_port) {
  ControlPipeConfig cfg;
  cfg.backend_type = backend_type;
  cfg.requested_port = listener_port;
  cfg.allow_legacy_framing = true;

  pipe_server_ = CreateControlPipeServer(cfg);
  pipe_server_->dispatcher()
      .RegisterHandler<::tpu_sync::rpc::ControlRequest,
                       ::tpu_sync::rpc::ControlResponse>(
          [this](const ControlContext& /*ctx*/,
                 const ::tpu_sync::rpc::ControlRequest& req)
              -> absl::StatusOr<::tpu_sync::rpc::ControlResponse> {
            ::tpu_sync::rpc::ControlResponse resp;
            ExecuteControlRequest(engine_, req, &resp, [this]() {
              stopping_.store(true);
              if (pipe_server_) {
                pipe_server_->StopAccepting();
              }
            });
            return resp;
          },
          HandlerOptions<::tpu_sync::rpc::ControlRequest>().WithMaxPayloadBytes(
              64 * 1024 * 1024));

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
      is_sender = req.start_transfer_request().is_sender();
      is_resharded =
          !req.start_transfer_request().shard_push_schedules().empty();
    }

    if (is_sender) {
      if (is_resharded) {
        LOG(INFO) << "C++ Listener executing PushWeightsResharded";
        absl::Status status =
            engine->PushWeightsResharded(req.start_transfer_request());
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
          absl::Status status = engine->PushWeights(peers);
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
      int64_t expected_block_count =
          req.start_transfer_request().expected_block_count();
      if (expected_block_count <= 0 ||
          expected_block_count > std::numeric_limits<uint32_t>::max()) {
        resp->set_success(false);
        resp->set_message(
            "expected_block_count must be positive and fit in 32-bit uint");
        LOG(ERROR) << "Invalid expected_block_count: " << expected_block_count;
        return;
      }
      uint64_t uuid = req.start_transfer_request().uuid();
      engine->StoreSkipTiling(uuid, req.start_transfer_request());

      const auto& layer_counts_proto =
          req.start_transfer_request().expected_layer_chunk_counts();
      if (!layer_counts_proto.empty()) {
        absl::flat_hash_map<size_t, uint32_t> layer_counts;
        for (const auto& [layer_idx, count] : layer_counts_proto) {
          layer_counts[static_cast<size_t>(layer_idx)] =
              static_cast<uint32_t>(count);
        }
        absl::Status layer_status =
            engine->RegisterExpectedLayerChunks(uuid, layer_counts);
        if (!layer_status.ok()) {
          LOG(WARNING) << "RegisterExpectedLayerChunks failed: "
                       << layer_status;
        }
      }

      absl::Status status = engine->RegisterExpectedChunks(
          uuid, static_cast<uint32_t>(expected_block_count));
      if (!status.ok()) {
        resp->set_success(false);
        resp->set_message(std::string(status.message()));
        LOG(ERROR) << "RegisterExpectedChunks failed: " << status;
      }
    }
  } else if (req.command() ==
             ::tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN) {
    LOG(INFO) << "C++ Listener received SHUTDOWN command. Draining pending H2D "
                 "and initiating clean exit.";
    if (engine != nullptr) {
      if (engine->control_delegate() != nullptr) {
        engine->control_delegate()->DrainPendingH2d();
      } else {
        engine->DrainPendingH2d();
      }
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

}  // namespace weight_sync
}  // namespace tpu_raiden
