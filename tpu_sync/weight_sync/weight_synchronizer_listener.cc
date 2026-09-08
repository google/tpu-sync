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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "tpu_sync/common/trace.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {

WeightSynchronizerListener::WeightSynchronizerListener(
    WeightSynchronizerBase* engine, int listener_port)
    : engine_(engine), listener_port_(listener_port) {
  int sock = socket(AF_INET6, SOCK_STREAM, 0);
  server_fd_.store(sock);
  if (sock < 0) {
    LOG(FATAL) << "Failed to create C++ Listener socket: "
               << std::strerror(errno);
  }

  int opt = 1;
  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
    LOG(WARNING) << "setsockopt SO_REUSEADDR failed";
  }

  sockaddr_in6 address{};
  address.sin6_family = AF_INET6;
  address.sin6_addr = in6addr_any;
  address.sin6_port = htons(listener_port_);

  // Bind to the requested port (0 for OS auto-allocation)
  if (bind(sock, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
    LOG(FATAL) << "C++ Listener bind failed on port " << listener_port_ << ": "
               << std::strerror(errno);
  }

  if (listen(sock, 128) < 0) {
    LOG(FATAL) << "C++ Listener listen failed: " << std::strerror(errno);
  }

  socklen_t addr_len = sizeof(address);
  if (getsockname(sock, reinterpret_cast<sockaddr*>(&address), &addr_len) ==
      0) {
    listener_port_ = ntohs(address.sin6_port);
  }

  LOG(INFO) << "Native C++ WeightSynchronizerListener actively listening "
               "on port: "
            << listener_port_;

  listener_thread_ =
      std::thread(&WeightSynchronizerListener::ListenerLoop, this);
}

WeightSynchronizerListener::~WeightSynchronizerListener() {
  stopping_ = true;
  int fd = server_fd_.exchange(-1);
  if (fd >= 0) {
    shutdown(fd, SHUT_RDWR);
    close(fd);
  }

  if (listener_thread_.joinable()) {
    listener_thread_.join();
  }

  for (auto& t : worker_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void WeightSynchronizerListener::ListenerLoop() {
  while (!stopping_) {
    sockaddr_in6 client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd =
        accept(server_fd_.load(), reinterpret_cast<sockaddr*>(&client_addr),
               &client_len);
    if (client_fd < 0) {
      if (stopping_) break;
      continue;
    }

    worker_threads_.push_back(std::thread(
        &WeightSynchronizerListener::ConnectionWorker, this, client_fd));
  }
}

void WeightSynchronizerListener::ConnectionWorker(int client_fd) {
  uint32_t net_len = 0;
  if (read(client_fd, &net_len, sizeof(net_len)) != sizeof(net_len)) {
    close(client_fd);
    return;
  }
  uint32_t payload_len = ntohl(net_len);

  std::vector<char> buffer(payload_len);
  size_t total_read = 0;
  while (total_read < payload_len) {
    ssize_t n =
        read(client_fd, buffer.data() + total_read, payload_len - total_read);
    if (n <= 0) {
      close(client_fd);
      return;
    }
    total_read += n;
  }

  tpu_sync::rpc::ControlRequest req;
  if (!req.ParseFromString(absl::string_view(buffer.data(), buffer.size()))) {
    LOG(ERROR) << "Failed to parse ControlRequest Protobuf";
    close(client_fd);
    return;
  }

  tpu_sync::rpc::ControlResponse resp;
  resp.set_success(true);
  resp.set_message("SUCCESS");

  if (req.command() == tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER) {
    RAIDEN_TRACE("WSyncListener::HandleStartTransfer");
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
            engine_->PushWeightsResharded(req.start_transfer_request());
        if (!status.ok()) {
          resp.set_success(false);
          resp.set_message(std::string(status.message()));
          LOG(ERROR) << "PushWeightsResharded native execution failed: "
                     << status;
        }
      } else {
        std::vector<std::string> peers(req.peers().begin(), req.peers().end());
        LOG(INFO) << "C++ Listener executing PushWeights to " << peers.size()
                  << " peers";
        if (!peers.empty()) {
          absl::Status status = engine_->PushWeights(peers);
          if (!status.ok()) {
            resp.set_success(false);
            resp.set_message(std::string(status.message()));
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
        resp.set_success(false);
        resp.set_message(
            "expected_block_count must be positive and fit in 32-bit uint");
        LOG(ERROR) << "Invalid expected_block_count: " << expected_block_count;
        std::string resp_str;
        if (resp.SerializeToString(&resp_str)) {
          uint32_t resp_net_len = htonl(resp_str.size());
          write(client_fd, &resp_net_len, sizeof(resp_net_len));
          write(client_fd, resp_str.data(), resp_str.size());
        }
        close(client_fd);
        return;
      }
      uint64_t uuid = req.start_transfer_request().uuid();
      engine_->StoreSkipTiling(uuid, req.start_transfer_request());

      const auto& layer_counts_proto =
          req.start_transfer_request().expected_layer_chunk_counts();
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
        resp.set_success(false);
        resp.set_message(std::string(status.message()));
        LOG(ERROR) << "RegisterExpectedChunks failed: " << status;
      }
    }
  } else if (req.command() == tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN) {
    LOG(INFO) << "C++ Listener received SHUTDOWN command. Draining pending H2D "
                 "and initiating clean exit.";
    if (engine_) {
      if (engine_->control_delegate()) {
        engine_->control_delegate()->DrainPendingH2d();
      } else {
        engine_->DrainPendingH2d();
      }
    }
    stopping_ = true;
    int fd = server_fd_.exchange(-1);
    if (fd >= 0) {
      shutdown(fd, SHUT_RDWR);
      close(fd);
    }
    resp.set_success(true);
  } else {
    resp.set_success(false);
    resp.set_message("COMMAND_UNSPECIFIED");
    LOG(WARNING) << "C++ Listener received unknown or unspecified "
                    "Protobuf command";
  }

  std::string resp_str;
  if (resp.SerializeToString(&resp_str)) {
    uint32_t resp_net_len = htonl(resp_str.size());
    write(client_fd, &resp_net_len, sizeof(resp_net_len));
    write(client_fd, resp_str.data(), resp_str.size());
  }
  close(client_fd);
}

}  // namespace weight_sync
}  // namespace tpu_raiden
