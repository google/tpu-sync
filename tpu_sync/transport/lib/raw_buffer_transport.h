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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_RAW_BUFFER_TRANSPORT_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_RAW_BUFFER_TRANSPORT_H_

#include <sys/uio.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/conn/pool.h"
#include "tpu_sync/transport/lib/raw_buffer_transport_delegate.h"
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

namespace tpu_raiden::transport::lib {

struct RawProgress {
  uint32_t completed_chunks = 0;
  std::optional<uint32_t> expected_chunks;
  absl::flat_hash_map<size_t, uint32_t> completed_chunks_per_layer;
  absl::flat_hash_map<size_t, uint32_t> expected_chunks_per_layer;
  absl::flat_hash_set<size_t> triggered_layers;
};

// Builds a single Request struct for buffer push operations.
absl::StatusOr<Request> BuildBufferRequest(size_t buffer_id,
                                           size_t dst_shard_idx,
                                           size_t dst_offset_bytes,
                                           const uint8_t* data_ptr,
                                           size_t size_bytes, uint64_t uuid,
                                           uint8_t socket_opcode);

// Builds a batch of Requests for a span of BufferPushTasks.
absl::StatusOr<std::vector<Request>> BuildBufferRequests(
    absl::Span<const BufferPushTask> tasks, uint64_t uuid,
    uint8_t socket_opcode);

// Standalone raw buffer TCP socket transport engine.
class RawBufferTransport final {
 public:
  using CustomRequestHandler = absl::AnyInvocable<absl::Status(
      int client_fd, const ChunkHeader& header)>;

  // Constructor sets up a TCP listening socket on the given `local_port`.
  // IPv6 is preferred with IPv4 as a fallback.
  RawBufferTransport(RawBufferTransportDelegate* delegate, int local_port,
                     const std::vector<std::string>& local_ips = {},
                     CustomRequestHandler custom_request_handler = nullptr,
                     size_t coalesce_window_bytes = 0);

  // Destructor closes all sockets and joins all threads.
  ~RawBufferTransport();

  // Stops accepting work, interrupts accepted and borrowed sockets, and
  // prevents future connector borrows. Socket owners retain close ownership.
  void CancelPendingOperations();

  // Waits until the listener and every accepted-socket worker have stopped.
  bool WaitForPendingOperations(std::chrono::milliseconds timeout);

  // Return the TCP listening socket port.
  int local_port() const { return local_port_; }

  // Return the bound IP address.
  // It is the first IP in `local_ips` if provided, otherwise "127.0.0.1".
  const std::string& bound_ip() const { return bound_ip_; }

  // Return the local IP addresses.
  absl::Span<const std::string> local_ips() const { return local_ips_; }

  // Borrows a connection from the connection pool, resolving the gRPC channel
  // for PSP key exchange if enabled.
  absl::StatusOr<int> BorrowConnection(absl::string_view peer,
                                       absl::string_view local_ip = "") {
    std::shared_ptr<grpc::Channel> channel = nullptr;
    if (require_psp_tcp_ && raw_delegate_ != nullptr) {
      channel = raw_delegate_->GetPeregrineChannel(peer);
    }
    return conn_pool_.Borrow(peer, local_ip, require_psp_tcp_, channel);
  }

  // Returns a connection to the connection pool.
  void ReturnConnection(bool ok_to_pool, int fd, absl::string_view peer,
                        absl::string_view local_ip = "") {
    conn_pool_.Return(ok_to_pool, fd, peer, local_ip);
  }

  // Synchronously pulls a buffer identified by `buffer_id` from the remote
  // `peer`, by sending out a `kOpBufferPull ChunkHeader` and then receiving
  // the data from the peer.
  // Note: This function is only used in RawBufferTransportTest, nowhere else.
  absl::Status PullBuffer(absl::string_view peer, size_t buffer_id,
                          size_t src_shard_idx, size_t src_offset_bytes,
                          size_t dst_shard_idx, size_t dst_offset_bytes,
                          size_t size_bytes);

  // Synchronously pushes a buffer (Op 5) by sending out a
  // `kOpBufferPush ChunkHeader` followed by the data. It waits for a one-byte
  // ack from the `peer` before it returns.
  absl::Status ProcessSocketBufferPush(absl::string_view peer,
                                       const Request& request);

  // Pushes a vector of buffers to multiple peers using
  // `ProcessSocketBufferBatchPush()`.
  absl::Status PushBuffers(const std::vector<BufferPushTask>& tasks,
                           int parallelism, uint64_t uuid);

  // Registers the expected number of chunks for the given `uuid`.
  // If the completed number of chunks is equal to the expected, it triggers
  // the delegate's `OnDataReceived()` H2D callback.
  absl::Status RegisterExpectedChunks(uint64_t uuid, uint32_t expected_chunks);

  // Registers the per-layer expected number of chunks for the given `uuid`.
  // When all chunks for a layer arrive, it triggers
  // `OnLayerDataReceived(layer_idx, uuid)`.
  absl::Status RegisterExpectedLayerChunks(
      uint64_t uuid,
      const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks);

  // Drops receive-progress counters belonging to the give `uuid`.
  void ForgetPushProgress(uint64_t uuid);

  // Registers incoming client PSP key and returns server's allocated RX key.
  // Triggered by ExchangePspKey() in PeregrineControlServiceImpl.
  absl::StatusOr<PspPeerKey> RegisterPspPeer(uint32_t client_spi,
                                             absl::string_view client_key);

 private:
  // Pushes buffer requests to `peer` over a borrowed TCP connection.
  absl::Status ProcessSocketBufferBatchPush(absl::string_view peer,
                                            absl::Span<const Request> requests);

  // Processes a single peer request from the given `client_fd`. These requests
  // are those sent by `PullBuffer`, `PushBuffer`, `PushBuffers` calls.
  absl::Status ProcessPeerRequest(int client_fd);

  // Accepts incoming connections and creates worker threads.
  void ListenerLoop();

  // Polls the socket `client_fd` and processes incoming peer requests.
  void ConnectionWorker(int client_fd);

 private:
  RawBufferTransportDelegate* const raw_delegate_;
  CustomRequestHandler custom_request_handler_;

  const size_t coalesce_window_bytes_;
  const std::string bound_ip_;
  const std::vector<std::string> local_ips_;
  int local_port_;
  const bool require_psp_tcp_;
  std::atomic<bool> stopping_;

  // listener_thread_ owns close(server_fd_). Cancellation only shutdowns it
  // while holding mu_, and listener_thread_ closes and clears it under the
  // same mutex. This prevents stale-fd shutdown after descriptor reuse.
  absl::Mutex mu_;
  absl::CondVar idle_cv_;
  int server_fd_ ABSL_GUARDED_BY(mu_) = -1;
  bool listener_exited_ ABSL_GUARDED_BY(mu_) = false;
  // Each connection worker owns close(client_fd). Cancellation only shutdowns
  // descriptors while holding mu_; workers close and erase under that mutex.
  absl::flat_hash_set<int> active_client_fds_ ABSL_GUARDED_BY(mu_);

  // The conn_pool_ owns the sockets that connect to peers. In comparison, the
  // active_client_fds above are those sockets accepted from peers.
  ConnPool conn_pool_;

  absl::Mutex raw_progress_mu_;
  absl::flat_hash_map<uint64_t, RawProgress> raw_progress_
      ABSL_GUARDED_BY(raw_progress_mu_);

  absl::Mutex push_pool_mu_;
  std::unique_ptr<tpu_raiden::NumaThreadPool> push_pool_
      ABSL_GUARDED_BY(push_pool_mu_);

  // To protect multiple gRPC threads can call RegisterPspPeer concurrently
  absl::Mutex psp_mu_;

  std::thread listener_thread_;
  std::vector<std::thread> worker_threads_;
};

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_RAW_BUFFER_TRANSPORT_H_
