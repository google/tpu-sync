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

#include "tpu_sync/transport/lib/conn/pool.h"

#include <sys/poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/transport/lib/socket/util.h"

namespace tpu_raiden::transport::lib {

namespace {
bool HasReadableData(const int fd) {
  struct pollfd pfd = {.fd = fd, .events = POLLIN};
  return poll(&pfd, /*nfds=*/1, /*timeout=*/0) > 0;
}

void CloseSocket(const int fd) {
  DCHECK_GE(fd, 0);
  ::shutdown(fd, SHUT_RDWR);
  ::close(fd);
}
}  // namespace

absl::StatusOr<int> ConnPool::Borrow(
    absl::string_view peer, absl::string_view local_ip, bool require_psp,
    std::shared_ptr<grpc::Channel> channel) {
  const bool diag_vlog = VLOG_IS_ON(1);
  absl::Time borrow_start =
      ABSL_PREDICT_FALSE(diag_vlog) ? absl::Now() : absl::InfinitePast();
  const Key key = GenPoolKey(peer, local_ip);
  int reused_fd = -1;
  size_t current_pool_size = 0;
  {
    absl::MutexLock lock(mu_);
    if (stop_) {
      return absl::FailedPreconditionError("ConnPool is closed.");
    }
    auto it = pool_.find(key);
    if ABSL_PREDICT_TRUE (it != pool_.end()) {
      Fds& fds = it->second;
      while (!fds.empty()) {
        const int fd = fds.back();
        fds.pop_back();

        if (HasReadableData(fd)) {
          CloseSocket(fd);
          continue;
        }
        reused_fd = fd;
        break;
      }
      current_pool_size = fds.size();
    }
  }
  if (reused_fd >= 0) {
    FaultInjectSocket(hooks::kConnPoolBorrowReuse, reused_fd);
    if (ABSL_PREDICT_FALSE(diag_vlog)) {
      double wait_ms = absl::ToDoubleMilliseconds(absl::Now() - borrow_start);
      if (wait_ms > 10.0) {
        VLOG(1) << "RAIDEN_DIAG conn borrow_wait wait_ms=" << wait_ms
                << " peer=" << peer << " local_ip=" << local_ip;
      }
    }
    return reused_fd;
  }
  ABSL_RETURN_IF_ERROR(FaultInjectStatus(hooks::kConnPoolBorrowConnect,
                                         absl::StatusCode::kUnavailable));
  std::string blackhole_peer;
  if (!FaultInjectStatus(hooks::kConnPoolBorrowConnectBlackhole).ok()) {
    const size_t colon = peer.rfind(':');
    blackhole_peer = absl::StrCat("192.0.2.1", colon == absl::string_view::npos
                                                   ? ":80"
                                                   : peer.substr(colon));
    peer = blackhole_peer;
  }
  ConnectTiming timing;
  ABSL_ASSIGN_OR_RETURN(int sock_fd,
                        ConnectToPeer(peer, local_ip, require_psp, channel,
                                      diag_vlog ? &timing : nullptr));
  if (ABSL_PREDICT_FALSE(diag_vlog)) {
    std::string local_endpoint = GetLocalEndpoint(sock_fd);
    std::string psp_str =
        require_psp ? absl::StrCat(" psp_kex_ms=", timing.psp_key_exchange_ms)
                    : "";
    VLOG(1) << "RAIDEN_DIAG conn new_socket fd=" << sock_fd << " peer=" << peer
            << " local_ip=" << (local_ip.empty() ? local_endpoint : local_ip)
            << " bound_local=" << local_endpoint
            << " connect_ms=" << timing.connect_ms << psp_str
            << " pool_size=" << current_pool_size;
  }
  return sock_fd;
}

void ConnPool::Return(bool ok, int fd, absl::string_view peer,
                      absl::string_view local_ip) {
  if ABSL_PREDICT_FALSE (fd < 0) {
    return;
  }

  DCHECK_GE(fd, 0);
  if ABSL_PREDICT_FALSE (!ok) {
    CloseSocket(fd);
    return;
  }

  absl::MutexLock lock(mu_);
  if ABSL_PREDICT_FALSE (stop_) {
    CloseSocket(fd);
  } else {
    const Key key = GenPoolKey(peer, local_ip);
    pool_[key].push_back(fd);
  }
}

void ConnPool::Close() {
  absl::MutexLock lock(mu_);
  stop_ = true;
  for (auto& [_, fds] : pool_) {
    for (const int fd : fds) {
      CloseSocket(fd);
    }
  }
  pool_.clear();
}

}  // namespace tpu_raiden::transport::lib
