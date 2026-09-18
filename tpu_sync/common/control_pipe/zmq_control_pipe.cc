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

#include "tpu_sync/common/control_pipe/zmq_control_pipe.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "zmq.h"

namespace tpu_raiden {
namespace {

std::string FormatZmqEndpoint(absl::string_view endpoint) {
  if (absl::StartsWith(endpoint, "tcp://") ||
      absl::StartsWith(endpoint, "inproc://") ||
      absl::StartsWith(endpoint, "ipc://")) {
    return std::string(endpoint);
  }
  return absl::StrCat("tcp://", endpoint);
}

bool RecvMultipart(void* sock, std::vector<std::string>* frames,
                   int flags = 0) {
  frames->clear();
  while (true) {
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int rc = zmq_msg_recv(&msg, sock, flags);
    if (rc < 0) {
      zmq_msg_close(&msg);
      return false;
    }
    frames->emplace_back(static_cast<const char*>(zmq_msg_data(&msg)),
                         zmq_msg_size(&msg));
    int more = zmq_msg_more(&msg);
    zmq_msg_close(&msg);
    if (!more) {
      break;
    }
    flags = 0;
  }
  return !frames->empty();
}

bool SendMultipart(void* sock, const std::vector<std::string>& frames) {
  for (size_t i = 0; i < frames.size(); ++i) {
    int flags = (i + 1 < frames.size()) ? ZMQ_SNDMORE : 0;
    if (zmq_send(sock, frames[i].data(), frames[i].size(), flags) < 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

ZmqControlPipeServer::ZmqControlPipeServer(const ControlPipeConfig& config)
    : config_(config) {}

ZmqControlPipeServer::~ZmqControlPipeServer() { Stop(); }

absl::StatusOr<int> ZmqControlPipeServer::Start(int requested_port) {
  if (zmq_ctx_ != nullptr) {
    return absl::FailedPreconditionError(
        "ZmqControlPipeServer is already running");
  }
  zmq_ctx_ = zmq_ctx_new();
  if (zmq_ctx_ == nullptr) {
    return absl::InternalError(
        absl::StrCat("zmq_ctx_new failed: ", zmq_strerror(zmq_errno())));
  }

  router_socket_ = zmq_socket(zmq_ctx_, ZMQ_ROUTER);
  if (router_socket_ == nullptr) {
    int err = zmq_errno();
    Stop();
    return absl::InternalError(absl::StrCat(
        "Failed to create ZMQ_ROUTER socket: ", zmq_strerror(err)));
  }

  int router_linger = 100;
  zmq_setsockopt(router_socket_, ZMQ_LINGER, &router_linger,
                 sizeof(router_linger));
  int64_t max_msg_size = static_cast<int64_t>(config_.max_frame_bytes);
  zmq_setsockopt(router_socket_, ZMQ_MAXMSGSIZE, &max_msg_size,
                 sizeof(max_msg_size));

  std::string bind_addr = absl::StrCat("tcp://*:", requested_port);
  if (zmq_bind(router_socket_, bind_addr.c_str()) != 0) {
    int err = zmq_errno();
    Stop();
    return absl::InternalError(absl::StrCat("zmq_bind to ", bind_addr,
                                            " failed: ", zmq_strerror(err)));
  }

  char endpoint_buf[256] = {0};
  size_t endpoint_len = sizeof(endpoint_buf);
  if (zmq_getsockopt(router_socket_, ZMQ_LAST_ENDPOINT, endpoint_buf,
                     &endpoint_len) != 0) {
    int err = zmq_errno();
    Stop();
    return absl::InternalError(absl::StrCat(
        "zmq_getsockopt(ZMQ_LAST_ENDPOINT) failed: ", zmq_strerror(err)));
  }
  absl::string_view ep(endpoint_buf);
  if (!ep.empty() && ep.back() == '\0') {
    ep.remove_suffix(1);
  }
  size_t colon_pos = ep.rfind(':');
  if (colon_pos == absl::string_view::npos ||
      !absl::SimpleAtoi(ep.substr(colon_pos + 1), &bound_port_)) {
    Stop();
    return absl::InternalError(
        absl::StrCat("Failed to parse bound port from endpoint: ", ep));
  }

  reply_inproc_addr_ =
      absl::StrCat("inproc://reply_", reinterpret_cast<uintptr_t>(this));
  inproc_reply_pull_ = zmq_socket(zmq_ctx_, ZMQ_PULL);
  if (inproc_reply_pull_ == nullptr) {
    int err = zmq_errno();
    Stop();
    return absl::InternalError(absl::StrCat(
        "Failed to create inproc ZMQ_PULL socket: ", zmq_strerror(err)));
  }
  int linger = 0;
  zmq_setsockopt(inproc_reply_pull_, ZMQ_LINGER, &linger, sizeof(linger));
  if (zmq_bind(inproc_reply_pull_, reply_inproc_addr_.c_str()) != 0) {
    int err = zmq_errno();
    Stop();
    return absl::InternalError(
        absl::StrCat("Failed to bind inproc pull socket: ", zmq_strerror(err)));
  }

  {
    absl::MutexLock lock(mu_);
    stopping_ = false;
  }
  poll_thread_ = std::thread(&ZmqControlPipeServer::PollLoop, this);
  return bound_port_;
}

void ZmqControlPipeServer::Stop() {
  {
    absl::MutexLock lock(mu_);
    stopping_ = true;
  }
  if (poll_thread_.joinable()) {
    poll_thread_.join();
  }
  {
    absl::MutexLock lock(mu_);
    auto cond = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return active_handlers_ == 0;
    };
    mu_.Await(absl::Condition(&cond));
  }
  if (router_socket_ != nullptr) {
    zmq_close(router_socket_);
    router_socket_ = nullptr;
  }
  if (inproc_reply_pull_ != nullptr) {
    zmq_close(inproc_reply_pull_);
    inproc_reply_pull_ = nullptr;
  }
  if (zmq_ctx_ != nullptr) {
    zmq_ctx_term(zmq_ctx_);
    zmq_ctx_ = nullptr;
  }
}

std::string ZmqControlPipeServer::ProcessPayload(absl::string_view payload) {
  control_pipe::proto::ControlResponseEnvelope resp_env;
  if (payload.size() > config_.max_frame_bytes) {
    resp_env.set_status_code(
        static_cast<int32_t>(absl::StatusCode::kResourceExhausted));
    resp_env.set_error_message(absl::StrCat("ZMQ frame size (", payload.size(),
                                            ") exceeds max_frame_bytes (",
                                            config_.max_frame_bytes, ")"));
  } else {
    control_pipe::proto::ControlEnvelope envelope;
    if (!envelope.ParseFromString(payload)) {
      resp_env.set_status_code(
          static_cast<int32_t>(absl::StatusCode::kInvalidArgument));
      resp_env.set_error_message("Failed to parse ControlEnvelope");
    } else {
      ControlContext ctx;
      ctx.peer_ip = "zmq_peer";
      ctx.request_id = envelope.request_id();
      ctx.backend_type = ControlPipeBackendType::kZmq;
      for (const auto& [k, v] : envelope.metadata()) {
        ctx.metadata[k] = v;
      }
      resp_env = dispatcher_.Dispatch(ctx, envelope);
    }
  }
  std::string out;
  resp_env.SerializeToString(&out);
  return out;
}

void ZmqControlPipeServer::PollLoop() {
  while (true) {
    bool is_stopping = false;
    {
      absl::MutexLock lock(mu_);
      is_stopping = stopping_;
      if (is_stopping && active_handlers_ == 0) {
        std::vector<std::string> reply_frames;
        while (RecvMultipart(inproc_reply_pull_, &reply_frames, ZMQ_DONTWAIT)) {
          SendMultipart(router_socket_, reply_frames);
        }
        break;
      }
    }

    zmq_pollitem_t items[2] = {
        {router_socket_, 0, ZMQ_POLLIN, 0},
        {inproc_reply_pull_, 0, ZMQ_POLLIN, 0},
    };
    int rc = zmq_poll(items, 2, 10);
    if (rc < 0) {
      if (zmq_errno() == EINTR) continue;
      break;
    }
    if (rc == 0) continue;

    if (!is_stopping && (items[0].revents & ZMQ_POLLIN)) {
      std::vector<std::string> frames;
      while (RecvMultipart(router_socket_, &frames, ZMQ_DONTWAIT)) {
        if (frames.size() < 2) continue;
        std::vector<std::string> reply_frames(frames.begin(), frames.end() - 1);
        std::string payload = std::move(frames.back());

        if (config_.executor == nullptr) {
          reply_frames.push_back(ProcessPayload(payload));
          SendMultipart(router_socket_, reply_frames);
        } else {
          {
            absl::MutexLock lock(mu_);
            if (stopping_) continue;
            ++active_handlers_;
          }
          config_.executor([this, reply_frames = std::move(reply_frames),
                            payload = std::move(payload)]() mutable {
            reply_frames.push_back(ProcessPayload(payload));
            void* push_sock = zmq_socket(zmq_ctx_, ZMQ_PUSH);
            if (push_sock != nullptr) {
              int linger = 1000;
              zmq_setsockopt(push_sock, ZMQ_LINGER, &linger, sizeof(linger));
              if (zmq_connect(push_sock, reply_inproc_addr_.c_str()) == 0) {
                SendMultipart(push_sock, reply_frames);
              }
              zmq_close(push_sock);
            }
            absl::MutexLock lock(mu_);
            --active_handlers_;
          });
        }
      }
    }

    if (items[1].revents & ZMQ_POLLIN) {
      std::vector<std::string> reply_frames;
      while (RecvMultipart(inproc_reply_pull_, &reply_frames, ZMQ_DONTWAIT)) {
        SendMultipart(router_socket_, reply_frames);
      }
    }
  }

  std::vector<std::string> final_frames;
  while (RecvMultipart(inproc_reply_pull_, &final_frames, ZMQ_DONTWAIT)) {
    SendMultipart(router_socket_, final_frames);
  }
}

ZmqConnectionPool::ZmqConnectionPool(void* zmq_ctx,
                                     const ControlPipeConfig& config)
    : zmq_ctx_(zmq_ctx), config_(config) {}

ZmqConnectionPool::~ZmqConnectionPool() { CloseAll(); }

absl::StatusOr<void*> ZmqConnectionPool::Acquire(absl::string_view endpoint,
                                                 absl::Duration timeout) {
  std::string norm_endpoint = FormatZmqEndpoint(endpoint);
  void* sock = nullptr;
  {
    absl::MutexLock lock(mu_);
    auto it = pool_.find(norm_endpoint);
    if (it != pool_.end() && !it->second.empty()) {
      sock = it->second.front();
      it->second.pop_front();
    }
  }

  if (sock == nullptr) {
    sock = zmq_socket(zmq_ctx_, ZMQ_REQ);
    if (sock == nullptr) {
      return absl::InternalError(absl::StrCat("zmq_socket(ZMQ_REQ) failed: ",
                                              zmq_strerror(zmq_errno())));
    }
    int linger = 0;
    zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
    int64_t max_msg_size = static_cast<int64_t>(config_.max_frame_bytes);
    zmq_setsockopt(sock, ZMQ_MAXMSGSIZE, &max_msg_size, sizeof(max_msg_size));
    if (zmq_connect(sock, norm_endpoint.c_str()) != 0) {
      int err = zmq_errno();
      zmq_close(sock);
      return absl::UnavailableError(absl::StrCat(
          "zmq_connect to ", norm_endpoint, " failed: ", zmq_strerror(err)));
    }
  }

  absl::Duration effective_timeout =
      (timeout <= absl::ZeroDuration()) ? config_.default_timeout : timeout;
  int timeout_ms = -1;
  if (effective_timeout != absl::InfiniteDuration()) {
    int64_t ms = absl::ToInt64Milliseconds(effective_timeout);
    timeout_ms =
        (ms <= 0) ? 1 : static_cast<int>(std::min<int64_t>(ms, INT_MAX));
  }
  zmq_setsockopt(sock, ZMQ_SNDTIMEO, &timeout_ms, sizeof(timeout_ms));
  zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
  return sock;
}

void ZmqConnectionPool::Release(absl::string_view endpoint, void* sock) {
  if (sock == nullptr) return;
  std::string norm_endpoint = FormatZmqEndpoint(endpoint);
  absl::MutexLock lock(mu_);
  if (!config_.enable_tcp_connection_pooling ||
      pool_[norm_endpoint].size() >=
          config_.max_idle_connections_per_endpoint) {
    zmq_close(sock);
  } else {
    pool_[norm_endpoint].push_back(sock);
  }
}

void ZmqConnectionPool::Discard(void* sock) {
  if (sock != nullptr) {
    zmq_close(sock);
  }
}

void ZmqConnectionPool::CloseAll() {
  absl::MutexLock lock(mu_);
  for (auto& [endpoint, sockets] : pool_) {
    for (void* sock : sockets) {
      zmq_close(sock);
    }
  }
  pool_.clear();
}

ZmqControlPipeClient::ZmqControlPipeClient(const ControlPipeConfig& config)
    : config_(config) {
  zmq_ctx_ = zmq_ctx_new();
  pool_ = std::make_unique<ZmqConnectionPool>(zmq_ctx_, config_);
}

ZmqControlPipeClient::~ZmqControlPipeClient() {
  if (pool_ != nullptr) {
    pool_->CloseAll();
    pool_.reset();
  }
  if (zmq_ctx_ != nullptr) {
    zmq_ctx_term(zmq_ctx_);
    zmq_ctx_ = nullptr;
  }
}

absl::StatusOr<control_pipe::proto::ControlResponseEnvelope>
ZmqControlPipeClient::SendRaw(
    absl::string_view endpoint,
    const control_pipe::proto::ControlEnvelope& envelope,
    absl::Duration timeout) {
  std::string req_bytes;
  if (!envelope.SerializeToString(&req_bytes)) {
    return absl::InternalError("Failed to serialize ControlEnvelope");
  }
  if (req_bytes.size() > config_.max_frame_bytes) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Serialized ControlEnvelope size (", req_bytes.size(),
        ") exceeds max_frame_bytes (", config_.max_frame_bytes, ")"));
  }

  absl::Duration effective_timeout =
      (timeout <= absl::ZeroDuration()) ? config_.default_timeout : timeout;
  ABSL_ASSIGN_OR_RETURN(void* sock,
                        pool_->Acquire(endpoint, effective_timeout));

  if (zmq_send(sock, req_bytes.data(), req_bytes.size(), 0) < 0) {
    int err = zmq_errno();
    pool_->Discard(sock);
    if (err == EAGAIN) {
      return absl::DeadlineExceededError(
          absl::StrCat("ZMQ send timed out to ", endpoint));
    }
    return absl::UnavailableError(
        absl::StrCat("ZMQ send failed to ", endpoint, ": ", zmq_strerror(err)));
  }

  std::vector<std::string> reply_frames;
  if (!RecvMultipart(sock, &reply_frames, 0)) {
    int err = zmq_errno();
    pool_->Discard(sock);
    if (err == EAGAIN) {
      return absl::DeadlineExceededError(
          absl::StrCat("ZMQ recv timed out from ", endpoint));
    }
    return absl::UnavailableError(absl::StrCat(
        "ZMQ recv failed from ", endpoint, ": ", zmq_strerror(err)));
  }

  pool_->Release(endpoint, sock);

  control_pipe::proto::ControlResponseEnvelope resp_env;
  if (!resp_env.ParseFromString(reply_frames.back())) {
    return absl::InternalError(
        "Failed to parse ControlResponseEnvelope from ZMQ response");
  }
  return resp_env;
}

}  // namespace tpu_raiden
