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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TCP_CONTROL_PLANE_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TCP_CONTROL_PLANE_BACKEND_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "xla/tsl/concurrency/future.h"

namespace tpu_raiden {

class TcpControlPlaneBackend : public ControlPlaneBackend {
 public:
  static constexpr int kMaxNics = 8;

  struct alignas(8) ControlRequestHeader {
    uint32_t magic = 0x52414944;  // "RAID"
    uint32_t op = 0;
    uint64_t uuid = 0;
    uint32_t ep_idx = 0;
    uint32_t consumer_data_port = 0;
    uint64_t num_blocks = 0;
    uint32_t num_ips = 0;
    uint8_t consumer_ips[kMaxNics][16] = {{0}};
    uint32_t padding = 0;
  };

  struct alignas(8) ControlResponseHeader {
    uint32_t magic = 0x44494152;  // "DIAR"
    int32_t status = 0;
    uint32_t num_layers = 0;
    uint32_t data_port = 0;
    uint64_t message_len = 0;
  };

  static constexpr uint32_t kControlMagic = 0x52414944;
  static constexpr uint32_t kResponseMagic = 0x44494152;
  static constexpr uint32_t kOpAck = 2;
  static constexpr uint32_t kOpPullStream = 3;
  static constexpr uint64_t kMaxControlErrorMessageBytes = 4 * 1024;

  explicit TcpControlPlaneBackend(
      TaskExecutor executor = nullptr,
      absl::Duration default_timeout = absl::Seconds(120));
  ~TcpControlPlaneBackend() override;

  absl::StatusOr<int> StartServer(int requested_port,
                                  ControlPlaneHandler* handler) override;
  void StopServer() override;

  // Runs the whole exchange on the calling thread and returns a ready future.
  tsl::Future<PullStreamResponseSpec> SendPullRequest(
      absl::string_view remote_endpoint, const PullStreamRequestSpec& req,
      absl::Duration timeout) override;

  absl::Status SendAck(absl::string_view remote_endpoint, uint64_t uuid,
                       absl::Duration timeout) override;

  absl::string_view Name() const override { return "tcp"; }

  static bool EncodeIpToIpv6Bytes(const std::string& ip, uint8_t out[16]);
  static absl::StatusOr<std::pair<std::string, int>> SplitEndpoint(
      absl::string_view endpoint);
  static absl::Status SetSocketTimeouts(int fd, double timeout_s);
  static absl::StatusOr<int> ConnectTcp(absl::string_view endpoint,
                                        double timeout_s);
  static absl::Status WriteExact(int fd, const void* buffer, size_t length);
  static absl::Status ReadExact(int fd, void* buffer, size_t length);
  static ControlResponseHeader ReadControlResponseHeader(int fd);
  static absl::StatusOr<std::string> GetPeerIp(int fd);
  static absl::Status WriteBlockIds(int fd,
                                    const std::vector<int64_t>& block_ids);
  static absl::StatusOr<std::vector<int64_t>> ReadBlockIds(int fd,
                                                           uint64_t num_blocks);

  void HandleControlConnection(int fd, ControlPlaneHandler* handler = nullptr);

 private:
  absl::StatusOr<PullStreamResponseSpec> SendPullRequestBlocking(
      absl::string_view remote_endpoint, const PullStreamRequestSpec& req,
      absl::Duration timeout);
  void ControlServerLoop();
  void SendErrorResponse(int fd, absl::string_view message);

  TaskExecutor executor_;
  absl::Duration default_timeout_;
  ControlPlaneHandler* handler_ = nullptr;
  int control_fd_ = -1;
  std::thread control_thread_;

  absl::Mutex handlers_mu_;
  bool stopping_ ABSL_GUARDED_BY(handlers_mu_) = false;
  int active_handlers_ ABSL_GUARDED_BY(handlers_mu_) = 0;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TCP_CONTROL_PLANE_BACKEND_H_
