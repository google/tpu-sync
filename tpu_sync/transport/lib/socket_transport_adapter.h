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
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

namespace tpu_raiden {
namespace transport {
namespace lib {

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

 private:
  struct WriteTask {
    uint64_t uuid;
    int layer_idx;
    int stream_idx;
    std::string peer;
    std::function<void()> run;
  };

  struct PeerQueue {
    std::deque<std::unique_ptr<WriteTask>> tasks;
    int active_streams = 0;
  };

  void SocketWorkerLoop();
  std::unique_ptr<WriteTask> SelectNextTask()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(scheduler_mu_);

  // Block-level Socket Operations (Op 1, 6).
  absl::StatusOr<Handle> PostSocketPush(absl::Span<const std::string> peers,
                                        absl::Span<const Request> requests,
                                        absl::Span<const int> src_block_ids,
                                        absl::Span<const int> dst_block_ids,
                                        CompletionCallback on_complete);

  absl::Status PostSocketPushInternal(absl::string_view peer,
                                      absl::string_view local_ip,
                                      absl::Span<const Request> requests,
                                      absl::Span<const int> src_block_ids,
                                      absl::Span<const int> dst_block_ids,
                                      size_t block_offset,
                                      std::vector<int>& allocated_ids);

 private:
  RawBufferTransport* const raw_transport_;
  const int parallelism_;

  absl::Mutex scheduler_mu_;
  absl::CondVar scheduler_cv_;
  absl::flat_hash_map<std::string, PeerQueue> peer_queues_
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
