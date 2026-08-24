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

#include <memory>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/channel.h"
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
  const Key key = GenPoolKey(peer, local_ip);
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
        return fd;
      }
    }
  }
  return ConnectToPeer(peer, local_ip, require_psp, channel);
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
