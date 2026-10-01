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

#include "tpu_sync/kv_cache/kv_cache_listener.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

using ::tpu_sync::rpc::ControlRequest;
using ::tpu_sync::rpc::ControlResponse;

namespace {

bool HasPoolReshardFields(const tpu_sync::rpc::StartTransferRequest& request) {
  // Treat either field as opting into the pool executor. This intentionally
  // sends partially populated pool plans to PoolReshardPush /
  // PoolReshardRegisterRecv so their validation fails closed instead of
  // silently taking the legacy path.
  return !request.transfer_pool_indices().empty() ||
         request.pool_groups_size() > 0;
}

}  // namespace

KVCacheListener::KVCacheListener(
    EngineCallbacks callbacks, int listener_port,
    std::optional<ControlPipeBackendType> backend_type)
    : callbacks_(std::move(callbacks)),
      backend_type_(ResolveControlPipeBackendType(backend_type)) {
  ControlPipeConfig cfg;
  cfg.backend_type = backend_type_;
  cfg.requested_port = listener_port;
  cfg.allow_legacy_framing = true;

  pipe_server_ = CreateControlPipeServer(cfg);
  pipe_server_->dispatcher()
      .RegisterHandler<ControlRequest, ControlResponse>(
          [this](const ControlContext& /*ctx*/, const ControlRequest& req)
              -> absl::StatusOr<ControlResponse> {
            ControlResponse resp;
            HandleControlRequest(req, &resp);
            return resp;
          },
          HandlerOptions<ControlRequest>().WithMaxPayloadBytes(
              cfg.max_frame_bytes));

  absl::StatusOr<int> bound_port = pipe_server_->Start(listener_port);
  CHECK_OK(bound_port.status())
      << "Failed to start KVCacheListener ControlPipeServer on port "
      << listener_port;
  listener_port_ = *bound_port;

  LOG(INFO) << "Native C++ KVCacheListener actively listening on port: "
            << listener_port_ << " (backend="
            << ControlPipeBackendTypeName(pipe_server_->backend_type()) << ")";
}

KVCacheListener::~KVCacheListener() { Shutdown(); }

void KVCacheListener::Shutdown() {
  stopping_.store(true);
  if (pipe_server_ != nullptr) {
    pipe_server_->Stop();
  }
  if (!work_drained_.exchange(true)) {
    absl::Status status = callbacks_.wait_for_pending_work();
    if (!status.ok()) {
      LOG(ERROR) << "WaitForPendingWork failed during shutdown: " << status;
    }
  }
}

void KVCacheListener::HandleControlRequest(const ControlRequest& req,
                                           ControlResponse* resp) {
  resp->set_success(true);
  resp->set_message("SUCCESS");

  if (req.command() == ControlRequest::COMMAND_START_TRANSFER) {
    if (stopping_.load()) {
      resp->set_success(false);
      resp->set_message("KVCacheListener is stopping");
      return;
    }
    if (req.has_start_transfer_request()) {
      const auto& start_req = req.start_transfer_request();
      const bool is_pool_reshard = HasPoolReshardFields(start_req);
      if (is_pool_reshard && start_req.is_sender()) {
        LOG(INFO) << "C++ KVCacheListener received pool START_TRANSFER "
                     "(Sender)";
        // The sender's local block list is the union of its schedule
        // entries' source blocks; the wire carries no scalar mirror of it.
        std::set<int64_t> src_id_set;
        for (const auto& [schedule_key, schedule] :
             start_req.shard_push_schedules()) {
          for (const auto& entry : schedule.entries()) {
            src_id_set.insert(static_cast<int64_t>(entry.src_block_id()));
          }
        }
        const std::vector<int64_t> src_block_ids(src_id_set.begin(),
                                                 src_id_set.end());
        absl::Status status = callbacks_.pool_reshard_push(
            start_req, src_block_ids, start_req.parallelism());
        if (!status.ok()) {
          resp->set_success(false);
          resp->set_message(std::string(status.message()));
          LOG(ERROR) << "PoolReshardPush native execution failed: " << status;
        }
      } else if (is_pool_reshard) {
        LOG(INFO) << "C++ KVCacheListener received pool START_TRANSFER "
                     "(Receiver)";
        // The flat destination list is the groups' concatenation; the wire
        // carries no scalar mirror of it.
        std::vector<int64_t> chip_block_ids;
        for (const auto& group : start_req.pool_groups()) {
          chip_block_ids.insert(chip_block_ids.end(),
                                group.dst_device_block_ids().begin(),
                                group.dst_device_block_ids().end());
        }
        absl::Status status =
            callbacks_.pool_reshard_register_recv(start_req, chip_block_ids);
        if (!status.ok()) {
          resp->set_success(false);
          resp->set_message(std::string(status.message()));
          LOG(ERROR) << "PoolReshardRegisterRecv native execution failed: "
                     << status;
        } else {
          // Report this receiver's pool addresses so the sender can
          // compute raddr.
          for (int32_t pool_idx : start_req.transfer_pool_indices()) {
            absl::StatusOr<tpu_sync::rpc::PoolHostAddrsProto> addrs =
                callbacks_.pool_host_addrs(start_req.uuid(),
                                           static_cast<size_t>(pool_idx));
            if (!addrs.ok()) {
              LOG(WARNING) << "No host base address for pool " << pool_idx
                           << ": " << addrs.status();
              continue;
            }
            if (addrs->host_base_addrs().empty()) continue;
            (*resp->mutable_receiver_pool_addrs())[pool_idx] =
                *std::move(addrs);
          }
        }
      } else if (start_req.is_sender()) {
        // Preserve the pre-pool controller protocol for existing callers.
        LOG(INFO) << "C++ KVCacheListener received legacy START_TRANSFER "
                     "(Sender)";
        absl::Status status = callbacks_.push_kv_cache_resharded(start_req);
        if (!status.ok()) {
          resp->set_success(false);
          resp->set_message(std::string(status.message()));
          LOG(ERROR) << "PushKVCacheResharded native execution failed: "
                     << status;
        }
      } else {
        LOG(INFO) << "C++ KVCacheListener received START_TRANSFER (Receiver), "
                     "registering expected buffers for uuid "
                  << start_req.uuid()
                  << ", expected blocks: " << start_req.expected_block_count();
        absl::Status status = callbacks_.register_active_plan(
            start_req.uuid(), start_req, /*is_sender=*/false);
        if (!status.ok()) {
          resp->set_success(false);
          resp->set_message(std::string(status.message()));
          LOG(ERROR) << "RegisterActivePlan native execution failed: "
                     << status;
        } else {
          // Report this receiver's layer addresses so the sender can compute
          // raddr.
          std::vector<tpu_sync::rpc::PoolHostAddrsProto> addrs =
              callbacks_.layer_host_addrs(start_req.uuid());
          for (size_t l = 0; l < addrs.size(); ++l) {
            (*resp->mutable_receiver_pool_addrs())[static_cast<int32_t>(l)] =
                std::move(addrs[l]);
          }
        }
      }
    } else {
      resp->set_success(false);
      resp->set_message("Missing start_transfer_request");
      LOG(ERROR) << "Missing start_transfer_request in START_TRANSFER command";
    }
  } else if (req.command() == ControlRequest::COMMAND_SHUTDOWN) {
    LOG(INFO) << "C++ KVCacheListener received SHUTDOWN command. Initiating "
                 "clean exit.";
    stopping_.store(true);
    if (!work_drained_.exchange(true)) {
      absl::Status status = callbacks_.wait_for_pending_work();
      if (!status.ok()) {
        LOG(ERROR) << "WaitForPendingWork failed during shutdown: " << status;
      }
    }
  } else {
    resp->set_success(false);
    resp->set_message("COMMAND_UNSPECIFIED");
    LOG(WARNING)
        << "C++ KVCacheListener received unknown or unspecified Protobuf "
           "command";
  }
}

}  // namespace kv_cache
}  // namespace tpu_raiden
