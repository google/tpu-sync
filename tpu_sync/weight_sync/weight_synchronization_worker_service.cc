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

// Branched from:
// //tpu_sync/weight_sync/weight_synchronizer_listener.cc

#include "tpu_sync/weight_sync/weight_synchronization_worker_service.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {

WeightSynchronizationWorkerServiceImpl::WeightSynchronizationWorkerServiceImpl(
    WeightSynchronizerBase* engine, std::function<void()> shutdown_callback)
    : engine_(engine), shutdown_callback_(std::move(shutdown_callback)) {}

grpc::Status WeightSynchronizationWorkerServiceImpl::HandleControl(
    grpc::ServerContext* context,
    const ::tpu_sync::rpc::ControlRequest* request,
    ::tpu_sync::rpc::ControlResponse* response) {
  if (request == nullptr || response == nullptr) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Null request or response");
  }

  response->set_success(true);
  response->set_message("SUCCESS");

  if (request->command() ==
      tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER) {
    RAIDEN_TRACE("WSyncService::HandleStartTransfer");
    bool is_sender = true;
    bool is_resharded = false;
    if (request->has_start_transfer_request()) {
      is_sender = request->start_transfer_request().is_sender();
      is_resharded =
          !request->start_transfer_request().shard_push_schedules().empty();
    }

    if (is_sender) {
      if (is_resharded) {
        LOG(INFO) << "gRPC WeightSynchronizationWorkerService executing "
                     "PushWeightsResharded";
        absl::Status status =
            engine_->PushWeightsResharded(request->start_transfer_request());
        if (!status.ok()) {
          response->set_success(false);
          response->set_message(std::string(status.message()));
          LOG(ERROR) << "PushWeightsResharded native execution failed: "
                     << status;
        }
      } else {
        std::vector<std::string> peers(request->peers().begin(),
                                       request->peers().end());
        LOG(INFO) << "gRPC WeightSynchronizationWorkerService executing "
                     "PushWeights to "
                  << peers.size() << " peers";
        if (!peers.empty()) {
          absl::Status status = engine_->PushWeights(peers);
          if (!status.ok()) {
            response->set_success(false);
            response->set_message(std::string(status.message()));
            LOG(ERROR) << "PushWeights native execution failed: " << status;
          }
        }
      }
    } else {
      LOG(INFO)
          << "gRPC WeightSynchronizationWorkerService received START_TRANSFER "
             "(Receiver) - registering expected block count";
      int64_t expected_block_count =
          request->start_transfer_request().expected_block_count();
      if (expected_block_count <= 0 ||
          expected_block_count > std::numeric_limits<uint32_t>::max()) {
        response->set_success(false);
        response->set_message(
            "expected_block_count must be positive and fit in 32-bit uint");
        LOG(ERROR) << "Invalid expected_block_count: " << expected_block_count;
        return grpc::Status::OK;
      }
      uint64_t uuid = request->start_transfer_request().uuid();
      engine_->StoreSkipTiling(uuid, request->start_transfer_request());

      const auto& layer_counts_proto =
          request->start_transfer_request().expected_layer_chunk_counts();
      if (!layer_counts_proto.empty()) {
        absl::flat_hash_map<size_t, uint32_t> layer_counts;
        for (const auto& [layer_idx, count] : layer_counts_proto) {
          layer_counts[static_cast<size_t>(layer_idx)] =
              static_cast<uint32_t>(count);
        }
        absl::Status layer_status =
            engine_->RegisterExpectedLayerChunks(uuid, layer_counts);
        if (!layer_status.ok()) {
          LOG(WARNING) << "RegisterExpectedLayerChunks failed: "
                       << layer_status;
        }
      }

      absl::Status status = engine_->RegisterExpectedChunks(
          uuid, static_cast<uint32_t>(expected_block_count));
      if (!status.ok()) {
        response->set_success(false);
        response->set_message(std::string(status.message()));
        LOG(ERROR) << "RegisterExpectedChunks failed: " << status;
      }
    }
  } else if (request->command() ==
             tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN) {
    LOG(INFO) << "gRPC WeightSynchronizationWorkerService received SHUTDOWN "
                 "command. Initiating clean exit.";
    if (shutdown_callback_) {
      std::thread([cb = shutdown_callback_]() {
        absl::SleepFor(absl::Milliseconds(50));
        cb();
      }).detach();
    }
  } else {
    response->set_success(false);
    response->set_message("COMMAND_UNSPECIFIED");
    LOG(WARNING) << "gRPC WeightSynchronizationWorkerService received unknown "
                    "or unspecified Protobuf command";
  }

  return grpc::Status::OK;
}

WeightSynchronizationWorkerService::WeightSynchronizationWorkerService(
    WeightSynchronizerBase* engine, int server_port)
    : engine_(engine), server_port_(server_port) {
  service_impl_ = std::make_unique<WeightSynchronizationWorkerServiceImpl>(
      engine_, [this]() { Shutdown(); });

  std::string server_address = absl::StrCat("[::]:", server_port_);
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(service_impl_.get());
  grpc_server_ = builder.BuildAndStart();

  if (!grpc_server_ || selected_port == 0) {
    LOG(FATAL) << "Failed to start WeightSynchronizationWorkerService gRPC "
                  "server on port: "
               << server_port_;
  }

  server_port_ = selected_port;
  LOG(INFO) << "WeightSynchronizationWorkerService actively listening on gRPC "
               "port: "
            << server_port_;
}

WeightSynchronizationWorkerService::~WeightSynchronizationWorkerService() {
  Shutdown();
}

void WeightSynchronizationWorkerService::Shutdown() {
  if (stopping_.exchange(true)) {
    return;
  }
  if (grpc_server_) {
    grpc_server_->Shutdown();
  }
}

}  // namespace weight_sync
}  // namespace tpu_raiden
