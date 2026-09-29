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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_SOCKET_TRANSPORT_ADAPTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_SOCKET_TRANSPORT_ADAPTER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

namespace tpu_raiden {
namespace transport {
namespace lib {

// Returns true if source IP binding is enabled via environment variable
// TPU_RAIDEN_ENABLE_SOURCE_IP_BIND.
bool SourceBindEnabled();

// Source address for stream `i`, or "" to let the kernel choose by route.
std::string SelectSourceIp(absl::Span<const std::string> local_ips, size_t i);

// TCP Socket implementation of TransportAdapter.
class SocketTransportAdapter : public TransportAdapter {
 public:
  explicit SocketTransportAdapter(RawBufferTransport* raw_transport,
                                  int parallelism = 1);
  ~SocketTransportAdapter() override;

  absl::StatusOr<Handle> Post(
      absl::Span<const std::string> peers, absl::Span<const Request> requests,
      absl::Span<const int> src_block_ids = {},
      absl::Span<const int> dst_block_ids = {},
      CompletionCallback on_complete = nullptr) override;

  absl::StatusOr<Status> Poll(Handle handle) override;

  absl::Duration handshake_ack_read_timeout() const {
    return handshake_ack_read_timeout_;
  }
  absl::Duration final_ack_read_timeout() const {
    return final_ack_read_timeout_;
  }
  absl::Duration initial_peer_backoff() const { return initial_peer_backoff_; }
  absl::Duration max_peer_backoff() const { return max_peer_backoff_; }

  int GetConsecutiveHandshakeTimeoutsForPeer(absl::string_view peer);
  bool IsPeerInBackoff(absl::string_view peer);

 private:
  struct WriteTask {
    uint64_t uuid;
    std::string peer;
    std::function<void()> run;
    std::function<void(absl::Status)> cancel;
  };

  struct PeerQueue {
    std::deque<std::unique_ptr<WriteTask>> tasks;
    int active_streams = 0;
  };

  struct PeerHealthState {
    int consecutive_handshake_timeouts = 0;
    std::optional<uint64_t> last_timeout_uuid;
    int backoff_count = 0;
    absl::Time backoff_until = absl::InfinitePast();
  };

  void SocketWorkerLoop();
  std::unique_ptr<WriteTask> SelectNextTask(
      absl::Time* next_wake_time = nullptr)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(scheduler_mu_);

  void CancelPendingTasksForRequestLocked(uint64_t uuid, absl::Status error)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(scheduler_mu_);
  void DrainPeerLocked(absl::string_view peer)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(scheduler_mu_);
  void RecordHandshakeTimeout(absl::string_view peer, uint64_t uuid);
  void RecordHandshakeSuccess(absl::string_view peer);

  // Block-level Socket Operations (Op 1, 6).
  absl::StatusOr<Handle> PostSocketPush(absl::Span<const std::string> peers,
                                        absl::Span<const Request> requests,
                                        absl::Span<const int> src_block_ids,
                                        absl::Span<const int> dst_block_ids,
                                        CompletionCallback on_complete);

  absl::Status PostSocketPushInternal(
      absl::string_view peer, absl::string_view local_ip,
      absl::string_view dst_ip, absl::Span<const Request> requests,
      absl::Span<const int> src_block_ids, absl::Span<const int> dst_block_ids,
      size_t block_offset, std::vector<int>& allocated_ids);

  // Block-level Socket Pull (Op 2).
  absl::StatusOr<Handle> PostSocketPull(absl::Span<const std::string> peers,
                                        absl::Span<const Request> requests,
                                        CompletionCallback on_complete);

  absl::Status PostSocketPullInternal(absl::string_view peer,
                                      absl::string_view local_ip,
                                      absl::Span<const Request> requests);

 private:
  RawBufferTransport* const raw_transport_;
  const int parallelism_;
  const absl::Duration handshake_ack_read_timeout_;
  const absl::Duration final_ack_read_timeout_;
  const absl::Duration initial_peer_backoff_;
  const absl::Duration max_peer_backoff_;

  absl::Mutex scheduler_mu_;
  absl::CondVar scheduler_cv_;
  absl::flat_hash_map<std::string, PeerQueue> peer_queues_
      ABSL_GUARDED_BY(scheduler_mu_);
  absl::flat_hash_map<std::string, PeerHealthState> peer_health_
      ABSL_GUARDED_BY(scheduler_mu_);
  std::vector<std::string> active_peers_ ABSL_GUARDED_BY(scheduler_mu_);
  size_t rr_index_ ABSL_GUARDED_BY(scheduler_mu_);
  // TODO(swasthi): Guard scheduler_stopping_ with scheduler_mu_ instead of
  // atomic.
  std::atomic<bool> scheduler_stopping_;

  std::vector<std::thread> socket_workers_;
};

}  // namespace lib
}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_SOCKET_TRANSPORT_ADAPTER_H_
