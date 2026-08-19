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

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/transport/lib/chunk_serializer.h"
#include "xla/future.h"
#include "xla/tsl/platform/test.h"

namespace tpu_raiden {
namespace {

class MetadataTransferManager : public KVCacheManagerWithTransfer {
 public:
  using KVCacheManagerWithTransfer::ControlRequestHeader;
  using KVCacheManagerWithTransfer::ControlResponseHeader;

  explicit MetadataTransferManager(bool start_control_server = false,
                                   int64_t num_slots = 2, int parallelism = 1,
                                   double timeout_s = 1.0)
      : KVCacheManagerWithTransfer(
            /*num_layers=*/0, /*num_shards=*/0, /*slice_byte_size=*/1,
            /*local_port=*/std::nullopt,
            /*host_blocks_to_allocate=*/std::nullopt, parallelism,
            /*node_id=*/0,
            /*local_control_port=*/start_control_server ? 0 : -1,
            /*max_blocks=*/1, num_slots, timeout_s) {
    if (!start_control_server) {
      absl::Status status = InitializeSlotPool(num_slots);
      if (!status.ok()) {
        throw std::runtime_error(std::string(status.message()));
      }
    }
  }

  static uint32_t RenewOp() { return kOpRenewLeases; }
  static uint32_t CancelOp() { return kOpCancelLeases; }
  static uint32_t ProtocolVersion() { return kLeaseProtocolVersion; }
  static size_t MaxBatchSize() { return kMaxLeaseBatchSize; }

  std::vector<int32_t> Apply(uint32_t op, const std::vector<uint64_t>& uuids) {
    absl::MutexLock lock(mu_);
    return ApplyLeaseBatchLocked(op, uuids, std::chrono::steady_clock::now());
  }

  void SetDeadline(uint64_t uuid,
                   std::chrono::steady_clock::time_point deadline) {
    absl::MutexLock lock(mu_);
    send_entries_.at(uuid)->deadline = deadline;
  }

  void SetRetentionDeadline(uint64_t uuid,
                            std::chrono::steady_clock::time_point deadline) {
    absl::MutexLock lock(mu_);
    send_entries_.at(uuid)->retention_deadline = deadline;
  }

  void SetTransferDeadline(uint64_t uuid,
                           std::chrono::steady_clock::time_point deadline) {
    absl::MutexLock lock(mu_);
    send_entries_.at(uuid)->transfer_deadline = deadline;
  }

  void HoldAsyncLayerForTest(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto entry = send_entries_.at(uuid);
    ++entry->pending_layer_callbacks;
    ++active_send_callbacks_;
  }

  void FinishAsyncLayerForTest(uint64_t uuid) {
    std::shared_ptr<SendEntry> entry;
    {
      absl::MutexLock lock(mu_);
      auto live = send_entries_.find(uuid);
      if (live != send_entries_.end()) {
        entry = live->second;
      } else {
        auto draining = draining_send_entries_.find(uuid);
        if (draining != draining_send_entries_.end()) entry = draining->second;
      }
    }
    FinishSendLayer(entry, absl::OkStatus(), "");
  }

  std::chrono::steady_clock::time_point GetDeadline(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    return send_entries_.at(uuid)->deadline;
  }

  bool HasEntry(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    return send_entries_.contains(uuid);
  }

  bool HasTombstone(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    PurgeSendTombstonesLocked(std::chrono::steady_clock::now());
    return send_tombstones_.contains(uuid);
  }

  bool IsClaimed(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = send_entries_.find(uuid);
    return it != send_entries_.end() &&
           it->second->phase == SendEntry::Phase::kTransferring;
  }

  bool IsReceiveTransferring(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    return it != active_recv_entries_.end() &&
           it->second.phase == RecvEntry::Phase::kTransferring;
  }

  size_t FreeSlotCount() {
    absl::MutexLock lock(mu_);
    return free_slots_.size();
  }

  void SetControlIoTimeout(std::chrono::milliseconds timeout) {
    control_io_timeout_ = timeout;
  }

  void SetRegistrationRetryBackoff(std::chrono::milliseconds base,
                                   std::chrono::milliseconds maximum) {
    registration_retry_base_delay_ = base;
    registration_retry_max_delay_ = maximum;
  }

  std::chrono::milliseconds RegistrationRetryDelay(uint64_t uuid,
                                                   size_t retry_number) {
    return ComputeRegistrationRetryDelay(uuid, retry_number);
  }

  uint64_t RetryableUnknownPullResponses() const {
    return retryable_unknown_pull_responses_.load(std::memory_order_relaxed);
  }

  void SetLocalDataPort(int port) { local_data_port_ = port; }

  void SetTombstoneCapacity(size_t capacity) {
    absl::MutexLock lock(mu_);
    max_send_tombstones_ = capacity;
    PurgeSendTombstonesLocked(std::chrono::steady_clock::now());
  }

  void SetPendingAckCapacity(size_t capacity) {
    absl::MutexLock lock(mu_);
    max_pending_acks_ = capacity;
    PurgePendingAcksLocked(std::chrono::steady_clock::now());
  }

  size_t TombstoneCount() {
    absl::MutexLock lock(mu_);
    PurgeSendTombstonesLocked(std::chrono::steady_clock::now());
    return send_tombstones_.size();
  }

  size_t PendingAckCount() {
    absl::MutexLock lock(mu_);
    PurgePendingAcksLocked(std::chrono::steady_clock::now());
    return pending_acks_.size();
  }

  bool HasPendingAck(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    PurgePendingAcksLocked(std::chrono::steady_clock::now());
    return pending_acks_.contains(uuid);
  }

  void FinishForTest(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = send_entries_.find(uuid);
    if (it != send_entries_.end()) {
      (void)TerminalizeSendEntryLocked(uuid, it->second, "test cleanup",
                                       /*waiting_only=*/false);
    }
  }

  void FinishReceiveForTest(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it == active_recv_entries_.end()) return;
    ReleaseSlotLocked(it->second.slot_idx);
    active_recv_entries_.erase(it);
  }

  void WaitForActiveControlFd() {
    for (int attempt = 0; attempt < 1000; ++attempt) {
      {
        absl::MutexLock lock(mu_);
        if (!active_control_fds_.empty()) return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    FAIL() << "control server did not track its accepted socket";
  }

  void StopForTest() { StopControlServer(); }

  std::shared_ptr<void> HoldPoolReshardCallbackForTest() {
    return TrackAsyncCallback();
  }

  void CancelTransportForTest() { CancelTransportOperations(); }

  absl::Status SyncPullForTest(const std::string& endpoint) {
    auto transport = GetTransportServer();
    if (!transport) {
      return absl::FailedPreconditionError("transport was not initialized");
    }
    absl::StatusOr<std::vector<int>> result = transport->SyncPull(
        {endpoint}, /*src_block_ids=*/{0}, /*local_block_ids=*/{0});
    return result.ok() ? absl::OkStatus() : result.status();
  }

  absl::Status ParallelLeasePullForTest(const std::string& endpoint) {
    auto transport = GetTransportServer();
    if (!transport) {
      return absl::FailedPreconditionError("transport was not initialized");
    }
    absl::StatusOr<std::vector<int>> result = transport->SyncPull(
        {endpoint}, /*src_block_ids=*/{0, 1}, /*local_block_ids=*/{0, 1},
        /*explicit_dst_ptrs=*/{}, /*parallelism=*/2,
        transport::MajorOrder::kLayerMajor, /*on_block_received=*/{},
        kLeaseAuthorizedPullUuid);
    return result.ok() ? absl::OkStatus() : result.status();
  }

  void StartTrackedStuckPushForTest(
      uint64_t uuid, const std::string& endpoint,
      std::shared_ptr<std::atomic<int>> completion_count = nullptr) {
    std::shared_ptr<SendEntry> entry;
    {
      absl::MutexLock lock(mu_);
      entry = send_entries_.at(uuid);
      ++entry->pending_layer_callbacks;
      ++active_send_callbacks_;
    }
    H2hWriteDirectAsync(
        {endpoint}, {0}, {0}, uuid, /*layer_idx=*/0,
        [this, entry,
         completion_count](absl::StatusOr<std::vector<int>> result) {
          if (completion_count) {
            completion_count->fetch_add(1, std::memory_order_relaxed);
          }
          FinishSendLayer(entry,
                          result.ok() ? absl::OkStatus() : result.status(),
                          result.ok() ? "" : "test transport push failed");
        });
  }

 protected:
  void StartPushInternal(uint64_t, const std::vector<std::string>&,
                         const std::vector<int64_t>&,
                         const std::vector<int64_t>&) override {
    // Keep the entry claimed without requiring TPU buffers.
  }
};

class DelayedPullReadiness {
 public:
  using Callback = transport::BlockTransportDelegate::HostBlockReadyCallback;

  void Add(Callback callback) {
    bool cancel_now = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      cancel_now = cancelling_;
      if (!cancel_now) callbacks_.push_back(std::move(callback));
    }
    if (cancel_now) {
      callback(absl::CancelledError("readiness controller is stopping"));
    }
  }

  bool WaitForPending(size_t count, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (callbacks_.size() >= count) return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  }

  bool CompleteOne(const absl::Status& status) {
    Callback callback;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (callbacks_.empty()) return false;
      callback = std::move(callbacks_.back());
      callbacks_.pop_back();
    }
    callback(status);
    return true;
  }

  void DropAll() {
    std::lock_guard<std::mutex> lock(mu_);
    callbacks_.clear();
  }

  void CancelAll() {
    std::vector<Callback> callbacks;
    {
      std::lock_guard<std::mutex> lock(mu_);
      cancelling_ = true;
      callbacks.swap(callbacks_);
    }
    for (auto& callback : callbacks) {
      callback(absl::CancelledError("readiness controller is stopping"));
    }
  }

 private:
  std::mutex mu_;
  std::vector<Callback> callbacks_;
  bool cancelling_ = false;
};

class DelayedPullResponseManager : public kv_cache::KVCacheManagerBase {
 public:
  explicit DelayedPullResponseManager(
      std::shared_ptr<DelayedPullReadiness> readiness)
      : KVCacheManagerBase(/*num_layers=*/1, /*num_shards=*/1,
                           /*slice_byte_size=*/1,
                           /*local_port=*/std::nullopt,
                           /*host_blocks_to_allocate=*/4,
                           /*parallelism=*/2),
        readiness_(std::move(readiness)) {}

  std::shared_ptr<transport::BlockTransport> TransportForTest() {
    return GetTransportServer();
  }

  void CancelTransportForTest() { CancelTransportOperations(); }

  void RegisterBlockReadinessCallback(
      size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
      transport::BlockTransportDelegate::HostBlockReadyCallback callback)
      override {
    (void)layer_idx;
    (void)shard_idx;
    (void)block_id;
    (void)uuid;
    readiness_->Add(std::move(callback));
  }

 private:
  std::shared_ptr<DelayedPullReadiness> readiness_;
};

class DelayedD2hController {
 public:
  void Add(xla::Promise<> promise) {
    std::lock_guard<std::mutex> lock(mu_);
    promises_.push_back(std::move(promise));
  }

  bool WaitForPending(size_t count, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (promises_.size() >= count) return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  }

  void CompleteAll(const absl::Status& status) {
    std::vector<xla::Promise<>> promises;
    {
      std::lock_guard<std::mutex> lock(mu_);
      promises.swap(promises_);
    }
    for (auto& promise : promises) {
      if (status.ok()) {
        promise.Set();
      } else {
        promise.Set(status);
      }
    }
  }

 private:
  std::mutex mu_;
  std::vector<xla::Promise<>> promises_;
};

class DelayedD2hManager : public kv_cache::KVCacheManagerBase {
 public:
  explicit DelayedD2hManager(std::shared_ptr<DelayedD2hController> controller)
      : KVCacheManagerBase(/*num_layers=*/1, /*num_shards=*/1,
                           /*slice_byte_size=*/1,
                           /*local_port=*/std::nullopt,
                           /*host_blocks_to_allocate=*/4,
                           /*parallelism=*/1),
        controller_(std::move(controller)) {}

 protected:
  absl::StatusOr<std::vector<raiden::PjRtCopyFuture>> DispatchD2hChunks(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes,
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt,
      int64_t device_id = -1) override {
    (void)src_offsets;
    (void)dst_offsets;
    (void)copy_sizes;
    (void)slot_idx;
    (void)layer_idx;
    (void)shard_idx;
    (void)device_id;
    auto [promise, future] = xla::MakePromise();
    controller_->Add(std::move(promise));
    std::vector<raiden::PjRtCopyFuture> futures;
    futures.emplace_back(std::move(future), raiden::BufferHolders{});
    return futures;
  }

 private:
  std::shared_ptr<DelayedD2hController> controller_;
};

bool WriteAll(int fd, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  while (size > 0) {
    const ssize_t written = write(fd, bytes, size);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (written == 0) return false;
    bytes += written;
    size -= written;
  }
  return true;
}

bool ReadAll(int fd, void* data, size_t size) {
  auto* bytes = static_cast<uint8_t*>(data);
  while (size > 0) {
    const ssize_t bytes_read = read(fd, bytes, size);
    if (bytes_read < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (bytes_read == 0) return false;
    bytes += bytes_read;
    size -= bytes_read;
  }
  return true;
}

bool SendNoSignal(int fd, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  while (size > 0) {
    int flags = 0;
#ifdef MSG_NOSIGNAL
    flags |= MSG_NOSIGNAL;
#endif
    const ssize_t written = send(fd, bytes, size, flags);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (written == 0) return false;
    bytes += written;
    size -= written;
  }
  return true;
}

bool ConnectExistingLoopbackSocket(int fd, int port) {
  sockaddr_in address;
  std::memset(&address, 0, sizeof(address));
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  return inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1 &&
         connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
             0;
}

int ConnectLoopback(int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error("socket failed");
  if (!ConnectExistingLoopbackSocket(fd, port)) {
    close(fd);
    throw std::runtime_error("loopback connect failed");
  }
  return fd;
}

std::optional<int> FindAcceptedSocketForPort(int port) {
  constexpr int kMaxTestFd = 4096;
  for (int fd = 0; fd < kMaxTestFd; ++fd) {
    sockaddr_storage local_address = {};
    sockaddr_storage peer_address = {};
    socklen_t local_length = sizeof(local_address);
    socklen_t peer_length = sizeof(peer_address);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&local_address),
                    &local_length) != 0 ||
        getpeername(fd, reinterpret_cast<sockaddr*>(&peer_address),
                    &peer_length) != 0) {
      continue;
    }

    int local_port = -1;
    if (local_address.ss_family == AF_INET) {
      local_port =
          ntohs(reinterpret_cast<const sockaddr_in*>(&local_address)->sin_port);
    } else if (local_address.ss_family == AF_INET6) {
      local_port = ntohs(
          reinterpret_cast<const sockaddr_in6*>(&local_address)->sin6_port);
    }
    if (local_port == port) return fd;
  }
  return std::nullopt;
}

bool WaitForDescriptorClose(int fd, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    errno = 0;
    if (fcntl(fd, F_GETFD) == -1 && errno == EBADF) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

bool WaitForAcceptedSocketAtFd(int fd, int port,
                               std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    const std::optional<int> accepted = FindAcceptedSocketForPort(port);
    if (accepted.has_value() && *accepted == fd) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

std::optional<std::vector<int>> HoldFreeDescriptorsBelow(int occupied_fd) {
  std::vector<int> held;
  while (true) {
    const int fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      for (int held_fd : held) close(held_fd);
      return std::nullopt;
    }
    if (fd >= occupied_fd) {
      close(fd);
      return held;
    }
    held.push_back(fd);
  }
}

struct ReusedSocketPair {
  int reused_fd = -1;
  int peer_fd = -1;
};

std::optional<ReusedSocketPair> ReuseDescriptorWithSocketPair(int target_fd) {
  int pair[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) return std::nullopt;
  if (pair[0] == target_fd) return ReusedSocketPair{pair[0], pair[1]};
  if (pair[1] == target_fd) return ReusedSocketPair{pair[1], pair[0]};
  if (dup2(pair[0], target_fd) != target_fd) {
    close(pair[0]);
    close(pair[1]);
    return std::nullopt;
  }
  close(pair[0]);
  return ReusedSocketPair{target_fd, pair[1]};
}

bool SocketPairRoundTrip(const ReusedSocketPair& pair, uint8_t value) {
  uint8_t received = 0;
  return SendNoSignal(pair.reused_fd, &value, sizeof(value)) &&
         ReadAll(pair.peer_fd, &received, sizeof(received)) &&
         received == value;
}

struct LoopbackListener {
  int fd = -1;
  int port = 0;
};

LoopbackListener OpenLoopbackListener() {
  LoopbackListener listener;
  listener.fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listener.fd < 0) throw std::runtime_error("socket failed");
  sockaddr_in address;
  std::memset(&address, 0, sizeof(address));
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(listener.fd, reinterpret_cast<sockaddr*>(&address),
           sizeof(address)) < 0) {
    close(listener.fd);
    throw std::runtime_error("bind failed");
  }
  socklen_t address_length = sizeof(address);
  if (getsockname(listener.fd, reinterpret_cast<sockaddr*>(&address),
                  &address_length) < 0 ||
      listen(listener.fd, 1) < 0) {
    close(listener.fd);
    throw std::runtime_error("listen failed");
  }
  listener.port = ntohs(address.sin_port);
  return listener;
}

struct WireResponse {
  int32_t status = -999;
  uint32_t version = 0;
  std::string message;
  std::vector<int32_t> lease_results;
};

WireResponse ReadResponse(int fd) {
  MetadataTransferManager::ControlResponseHeader response;
  WireResponse result;
  if (!ReadAll(fd, &response, sizeof(response))) return result;
  result.status = response.status;
  result.version = response.num_layers;
  if (response.message_len == 0) return result;
  if (response.status == 0 && response.message_len % sizeof(int32_t) == 0) {
    result.lease_results.resize(response.message_len / sizeof(int32_t));
    EXPECT_TRUE(ReadAll(fd, result.lease_results.data(), response.message_len));
  } else {
    result.message.resize(response.message_len);
    EXPECT_TRUE(ReadAll(fd, result.message.data(), result.message.size()));
  }
  return result;
}

WireResponse RawExchange(MetadataTransferManager* manager,
                         MetadataTransferManager::ControlRequestHeader request,
                         const std::vector<uint64_t>& body = {}) {
  const int fd = ConnectLoopback(manager->local_control_port());
  EXPECT_TRUE(WriteAll(fd, &request, sizeof(request)));
  if (!body.empty()) {
    EXPECT_TRUE(WriteAll(fd, body.data(), body.size() * sizeof(body.front())));
  }
  WireResponse response = ReadResponse(fd);
  (void)shutdown(fd, SHUT_RDWR);
  close(fd);
  return response;
}

WireResponse Pull(MetadataTransferManager* manager, uint64_t uuid) {
  const int fd = ConnectLoopback(manager->local_control_port());
  MetadataTransferManager::ControlRequestHeader request;
  request.op = 3;
  request.uuid = uuid;
  request.num_blocks = 1;
  request.consumer_data_port = 1;
  EXPECT_TRUE(WriteAll(fd, &request, sizeof(request)));
  const int64_t block_id = 0;
  EXPECT_TRUE(WriteAll(fd, &block_id, sizeof(block_id)));
  EXPECT_TRUE(WriteAll(fd, &block_id, sizeof(block_id)));
  WireResponse response = ReadResponse(fd);
  (void)shutdown(fd, SHUT_RDWR);
  close(fd);
  return response;
}

WireResponse Ack(MetadataTransferManager* manager, uint64_t uuid,
                 uint32_t op = 2) {
  MetadataTransferManager::ControlRequestHeader request;
  request.op = op;
  request.uuid = uuid;
  return RawExchange(manager, request);
}

bool WaitForTransferRegistration(MetadataTransferManager* producer,
                                 MetadataTransferManager* consumer,
                                 uint64_t uuid,
                                 std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (producer->IsClaimed(uuid) && consumer->IsReceiveTransferring(uuid)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

bool WaitForFailedReceive(MetadataTransferManager* manager,
                          const std::string& req_id,
                          std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto [done_sending, done_recving, failed_recving] =
        manager->CompleteReadRaw();
    (void)done_sending;
    (void)done_recving;
    if (std::find(failed_recving.begin(), failed_recving.end(), req_id) !=
        failed_recving.end()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

bool WaitForRetryableUnknowns(MetadataTransferManager* manager,
                              uint64_t minimum,
                              std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (manager->RetryableUnknownPullResponses() >= minimum) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

TEST(SendLifecycleTest, ExpiryCreatesTombstoneAndReportsDoneOnce) {
  MetadataTransferManager manager;
  manager.NotifyForRead("expired", 101, {0});
  manager.SetDeadline(
      101, std::chrono::steady_clock::now() - std::chrono::milliseconds(1));

  EXPECT_EQ(std::get<0>(manager.CompleteReadRaw()),
            std::vector<std::string>({"expired"}));
  EXPECT_TRUE(manager.HasTombstone(101));
  EXPECT_TRUE(std::get<0>(manager.CompleteReadRaw()).empty());
}

TEST(SendLifecycleTest, UnknownAndLatePullsFailWithoutBlockingWorkers) {
  MetadataTransferManager manager(/*start_control_server=*/true);
  manager.NotifyForRead("late", 102, {0});
  manager.SetDeadline(
      102, std::chrono::steady_clock::now() - std::chrono::milliseconds(1));
  (void)manager.CompleteReadRaw();

  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(Pull(&manager, 102).status, -1);
  EXPECT_EQ(Pull(&manager, 103).status, 1);
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
                .count(),
            500);
}

TEST(ReceiveLifecycleTest, EarlyUnknownIsRetriedUntilProducerRegisters) {
  MetadataTransferManager producer(/*start_control_server=*/true,
                                   /*num_slots=*/2,
                                   /*parallelism=*/1,
                                   /*timeout_s=*/0.5);
  producer.SetControlIoTimeout(std::chrono::milliseconds(100));
  MetadataTransferManager consumer(/*start_control_server=*/false,
                                   /*num_slots=*/2,
                                   /*parallelism=*/1,
                                   /*timeout_s=*/0.5);
  consumer.SetControlIoTimeout(std::chrono::milliseconds(100));
  consumer.SetLocalDataPort(1);

  consumer.StartRead(
      "retry", 104,
      "127.0.0.1:" + std::to_string(producer.local_control_port()), {0}, {0});
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  producer.NotifyForRead("retry", 104, {0});

  EXPECT_TRUE(WaitForTransferRegistration(&producer, &consumer, 104,
                                          std::chrono::milliseconds(500)));
  producer.FinishForTest(104);
  consumer.FinishReceiveForTest(104);
}

TEST(ReceiveLifecycleTest, TrulyUnknownUuidStopsAtOverallDeadline) {
  MetadataTransferManager producer(/*start_control_server=*/true);
  producer.SetControlIoTimeout(std::chrono::milliseconds(25));
  MetadataTransferManager consumer(/*start_control_server=*/false,
                                   /*num_slots=*/2,
                                   /*parallelism=*/1,
                                   /*timeout_s=*/0.08);
  consumer.SetControlIoTimeout(std::chrono::milliseconds(25));
  consumer.SetLocalDataPort(1);
  consumer.StartRead(
      "unknown", 105,
      "127.0.0.1:" + std::to_string(producer.local_control_port()), {0}, {0});

  EXPECT_TRUE(WaitForFailedReceive(&consumer, "unknown",
                                   std::chrono::milliseconds(500)));
  EXPECT_EQ(consumer.FreeSlotCount(), 2);
}

TEST(ReceiveLifecycleTest, RetryableUnknownUsesBoundedBackoffAndDeadline) {
  MetadataTransferManager producer(/*start_control_server=*/true);
  producer.SetControlIoTimeout(std::chrono::milliseconds(25));
  MetadataTransferManager consumer(/*start_control_server=*/false,
                                   /*num_slots=*/2,
                                   /*parallelism=*/1,
                                   /*timeout_s=*/0.18);
  consumer.SetControlIoTimeout(std::chrono::milliseconds(25));
  consumer.SetRegistrationRetryBackoff(std::chrono::milliseconds(20),
                                       std::chrono::milliseconds(80));
  consumer.SetLocalDataPort(1);

  const auto first_delay = consumer.RegistrationRetryDelay(1050, 1);
  const auto second_delay = consumer.RegistrationRetryDelay(1050, 2);
  const auto fourth_delay = consumer.RegistrationRetryDelay(1050, 4);
  EXPECT_GE(first_delay.count(), 15);
  EXPECT_LE(first_delay.count(), 25);
  EXPECT_GE(second_delay.count(), 30);
  EXPECT_LE(second_delay.count(), 50);
  EXPECT_LE(fourth_delay.count(), 80);

  consumer.StartRead(
      "paced-unknown", 1050,
      "127.0.0.1:" + std::to_string(producer.local_control_port()), {0}, {0});
  ASSERT_TRUE(
      WaitForRetryableUnknowns(&producer, 1, std::chrono::milliseconds(200)));
  std::this_thread::sleep_for(std::chrono::milliseconds(90));
  const uint64_t attempts_in_window = producer.RetryableUnknownPullResponses();
  EXPECT_GE(attempts_in_window, 2);
  EXPECT_LE(attempts_in_window, 4);

  EXPECT_TRUE(WaitForFailedReceive(&consumer, "paced-unknown",
                                   std::chrono::milliseconds(500)));
  const uint64_t attempts_at_deadline =
      producer.RetryableUnknownPullResponses();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(producer.RetryableUnknownPullResponses(), attempts_at_deadline);
}

TEST(ReceiveLifecycleTest, ShutdownCancelsDelayedRegistrationRetry) {
  MetadataTransferManager producer(/*start_control_server=*/true);
  auto consumer = std::make_unique<MetadataTransferManager>(
      /*start_control_server=*/false, /*num_slots=*/2,
      /*parallelism=*/1, /*timeout_s=*/5.0);
  consumer->SetControlIoTimeout(std::chrono::milliseconds(25));
  consumer->SetRegistrationRetryBackoff(std::chrono::milliseconds(500),
                                        std::chrono::milliseconds(500));
  consumer->SetLocalDataPort(1);
  consumer->StartRead(
      "shutdown-backoff", 1051,
      "127.0.0.1:" + std::to_string(producer.local_control_port()), {0}, {0});
  ASSERT_TRUE(
      WaitForRetryableUnknowns(&producer, 1, std::chrono::milliseconds(200)));

  const auto start = std::chrono::steady_clock::now();
  consumer.reset();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  EXPECT_LT(elapsed.count(), 500);
  const uint64_t attempts_after_shutdown =
      producer.RetryableUnknownPullResponses();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(producer.RetryableUnknownPullResponses(), attempts_after_shutdown);
}

TEST(SendLifecycleTest, DuplicatePullAndAckDoNotReleaseClaimedSlot) {
  MetadataTransferManager manager(/*start_control_server=*/true,
                                  /*num_slots=*/1);
  manager.NotifyForRead("claimed", 106, {0});
  ASSERT_EQ(Pull(&manager, 106).status, 0);
  EXPECT_TRUE(manager.IsClaimed(106));
  EXPECT_EQ(manager.FreeSlotCount(), 0);
  EXPECT_EQ(Pull(&manager, 106).status, -1);
  EXPECT_EQ(Ack(&manager, 106).status, 0);
  EXPECT_TRUE(manager.IsClaimed(106));
  EXPECT_EQ(manager.FreeSlotCount(), 0);
  manager.FinishForTest(106);
  EXPECT_EQ(manager.FreeSlotCount(), 1);
}

TEST(SendLifecycleTest, TransferTimeoutReportsButRetainsSlotUntilAsyncDrain) {
  MetadataTransferManager manager(/*start_control_server=*/true,
                                  /*num_slots=*/1);
  manager.NotifyForRead("transfer-timeout", 1060, {0});
  ASSERT_EQ(Pull(&manager, 1060).status, 0);
  manager.HoldAsyncLayerForTest(1060);
  manager.SetTransferDeadline(
      1060, std::chrono::steady_clock::now() - std::chrono::milliseconds(1));

  EXPECT_EQ(std::get<0>(manager.CompleteReadRaw()),
            std::vector<std::string>({"transfer-timeout"}));
  EXPECT_FALSE(manager.HasEntry(1060));
  EXPECT_TRUE(manager.HasTombstone(1060));
  EXPECT_EQ(manager.FreeSlotCount(), 0);
  manager.FinishAsyncLayerForTest(1060);
  EXPECT_EQ(manager.FreeSlotCount(), 1);
}

TEST(SendLifecycleTest, DestructorDrainsOwnedAsyncCallbacksBeforeReturning) {
  auto manager = std::make_unique<MetadataTransferManager>(
      /*start_control_server=*/true, /*num_slots=*/1);
  manager->NotifyForRead("shutdown-drain", 1061, {0});
  ASSERT_EQ(Pull(manager.get(), 1061).status, 0);
  manager->HoldAsyncLayerForTest(1061);
  MetadataTransferManager* manager_ptr = manager.get();
  std::thread completion([manager_ptr]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    manager_ptr->FinishAsyncLayerForTest(1061);
  });

  const auto start = std::chrono::steady_clock::now();
  manager.reset();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  completion.join();
  EXPECT_GE(
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
      30);
}

TEST(PoolReshardLifecycleTest,
     DestructorDrainsTrackedPoolReshardCallbackBeforeReturning) {
  auto manager = std::make_unique<MetadataTransferManager>();
  std::shared_ptr<void> callback = manager->HoldPoolReshardCallbackForTest();

  auto destroy =
      std::async(std::launch::async, [&manager]() { manager.reset(); });
  EXPECT_EQ(destroy.wait_for(std::chrono::milliseconds(40)),
            std::future_status::timeout);
  callback.reset();
  EXPECT_EQ(destroy.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  destroy.get();
}

TEST(KVCacheManagerLifecycleTest,
     DestructorWaitsForActualDelayedD2hWriteCallbacks) {
  auto controller = std::make_shared<DelayedD2hController>();
  auto manager = std::make_unique<DelayedD2hManager>(controller);
  auto transfer_or = manager->D2hWrite(
      "unused-after-transport-cancellation", /*src_device=*/{0, 1},
      /*src_host=*/{0, 1}, /*dst_host=*/{0, 1}, /*copy_sizes=*/{1, 1});
  ASSERT_TRUE(transfer_or.ok()) << transfer_or.status();
  raiden::PjRtCopyFuture transfer = std::move(*transfer_or);
  ASSERT_TRUE(controller->WaitForPending(2, std::chrono::seconds(1)));

  auto destroy =
      std::async(std::launch::async, [&manager]() { manager.reset(); });
  const std::future_status before_d2h =
      destroy.wait_for(std::chrono::milliseconds(40));
  if (before_d2h != std::future_status::timeout) {
    ADD_FAILURE() << "manager destruction returned while D2H was unresolved";
    // The unguarded implementation's failure callback does not dereference
    // the already-destroyed manager, so fail the promises for safe cleanup.
    controller->CompleteAll(
        absl::CancelledError("test cleanup after early destruction"));
    destroy.get();
    return;
  }

  controller->CompleteAll(absl::OkStatus());
  EXPECT_EQ(destroy.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  destroy.get();
  EXPECT_FALSE(transfer.Await().ok());
}

TEST(SendLifecycleTest, DestructorCancelsStuckTransportAndDrainsCallback) {
  LoopbackListener listener = OpenLoopbackListener();
  auto manager = std::make_unique<MetadataTransferManager>(
      /*start_control_server=*/true, /*num_slots=*/2,
      /*parallelism=*/1);
  manager->NotifyForRead("stuck-transport", 1062, {0});
  if (Pull(manager.get(), 1062).status != 0) {
    close(listener.fd);
    FAIL() << "producer did not accept the test pull";
    return;
  }
  manager->NotifyForRead("queued-transport", 1063, {0});
  if (Pull(manager.get(), 1063).status != 0) {
    close(listener.fd);
    FAIL() << "producer did not accept the queued test pull";
    return;
  }

  std::promise<int> accepted;
  std::future<int> accepted_future = accepted.get_future();
  std::thread stalled_peer([&accepted, listener]() {
    const int fd = accept(listener.fd, nullptr, nullptr);
    accepted.set_value(fd);
    if (fd < 0) return;
    std::array<uint8_t, 256> buffer;
    while (read(fd, buffer.data(), buffer.size()) > 0) {
    }
    close(fd);
  });
  auto completion_count = std::make_shared<std::atomic<int>>(0);
  manager->StartTrackedStuckPushForTest(
      1062, "127.0.0.1:" + std::to_string(listener.port), completion_count);

  if (accepted_future.wait_for(std::chrono::seconds(1)) !=
      std::future_status::ready) {
    manager.reset();
    (void)shutdown(listener.fd, SHUT_RDWR);
    close(listener.fd);
    stalled_peer.join();
    FAIL() << "stalled peer did not accept the transport connection";
    return;
  }
  const int accepted_fd = accepted_future.get();
  manager->StartTrackedStuckPushForTest(
      1063, "127.0.0.1:" + std::to_string(listener.port), completion_count);
  close(listener.fd);
  if (accepted_fd < 0) {
    manager.reset();
    stalled_peer.join();
    FAIL() << "stalled peer accept failed";
    return;
  }

  const auto start = std::chrono::steady_clock::now();
  manager.reset();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  stalled_peer.join();
  EXPECT_LT(elapsed.count(), 1000);
  EXPECT_EQ(completion_count->load(std::memory_order_relaxed), 2);
}

TEST(TransportShutdownTest, ActiveSyncPullIsCancelledWithoutInitMutexDeadlock) {
  LoopbackListener listener = OpenLoopbackListener();
  auto manager = std::make_unique<MetadataTransferManager>();
  ASSERT_TRUE(manager->local_port().has_value());
  MetadataTransferManager* manager_ptr = manager.get();

  std::promise<int> accepted;
  std::future<int> accepted_future = accepted.get_future();
  std::thread stalled_peer([&accepted, listener]() {
    const int fd = accept(listener.fd, nullptr, nullptr);
    accepted.set_value(fd);
    if (fd >= 0) {
      std::array<uint8_t, 256> buffer;
      while (read(fd, buffer.data(), buffer.size()) > 0) {
      }
      close(fd);
    }
  });
  std::future<absl::Status> pull =
      std::async(std::launch::async, [manager_ptr, port = listener.port]() {
        return manager_ptr->SyncPullForTest("127.0.0.1:" +
                                            std::to_string(port));
      });

  if (accepted_future.wait_for(std::chrono::seconds(1)) !=
      std::future_status::ready) {
    manager.reset();
    (void)shutdown(listener.fd, SHUT_RDWR);
    close(listener.fd);
    stalled_peer.join();
    FAIL() << "stalled peer did not accept the pull connection";
    return;
  }
  const int accepted_fd = accepted_future.get();
  close(listener.fd);
  if (accepted_fd < 0) {
    manager.reset();
    stalled_peer.join();
    FAIL() << "stalled peer accept failed";
    return;
  }

  const auto start = std::chrono::steady_clock::now();
  manager.reset();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  EXPECT_EQ(pull.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_FALSE(pull.get().ok());
  stalled_peer.join();
  EXPECT_LT(elapsed.count(), 1000);
}

TEST(TransportShutdownTest,
     ParallelSameUuidPullStreamsAllDrainBeforeDestruction) {
  auto readiness = std::make_shared<DelayedPullReadiness>();
  auto server = std::make_unique<DelayedPullResponseManager>(readiness);
  MetadataTransferManager client(/*start_control_server=*/false,
                                 /*num_slots=*/2,
                                 /*parallelism=*/2);
  ASSERT_TRUE(server->local_port().has_value());
  const std::string endpoint =
      "127.0.0.1:" + std::to_string(*server->local_port());
  std::future<absl::Status> pull =
      std::async(std::launch::async, [&client, endpoint]() {
        return client.ParallelLeasePullForTest(endpoint);
      });

  if (!readiness->WaitForPending(2, std::chrono::seconds(1))) {
    readiness->CancelAll();
    server.reset();
    (void)pull.wait_for(std::chrono::seconds(1));
    FAIL() << "parallel pull did not register both response streams";
    return;
  }

  auto destroy =
      std::async(std::launch::async, [&server]() { server.reset(); });
  EXPECT_EQ(destroy.wait_for(std::chrono::milliseconds(40)),
            std::future_status::timeout);
  ASSERT_TRUE(readiness->CompleteOne(
      absl::CancelledError("release first pull response stream")));

  const std::future_status after_one =
      destroy.wait_for(std::chrono::milliseconds(40));
  if (after_one != std::future_status::timeout) {
    ADD_FAILURE()
        << "one same-UUID stream completion hid another active stream";
    readiness->DropAll();
    destroy.get();
    EXPECT_EQ(pull.wait_for(std::chrono::seconds(1)),
              std::future_status::ready);
    return;
  }

  ASSERT_TRUE(readiness->CompleteOne(
      absl::CancelledError("release second pull response stream")));
  EXPECT_EQ(destroy.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  destroy.get();
  EXPECT_EQ(pull.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_FALSE(pull.get().ok());
}

TEST(TransportShutdownTest,
     DelayedPullCallbackCannotTouchReusedAcceptedDescriptor) {
  auto readiness = std::make_shared<DelayedPullReadiness>();
  auto server = std::make_unique<DelayedPullResponseManager>(readiness);
  std::shared_ptr<transport::BlockTransport> server_transport =
      server->TransportForTest();
  ASSERT_TRUE(server->local_port().has_value());
  const int server_port = *server->local_port();

  const int original_client = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(original_client, 0);
  ASSERT_TRUE(ConnectExistingLoopbackSocket(original_client, server_port));
  transport::lib::ChunkHeader request = {};
  request.version = 1;
  request.op = 2;
  request.flags = static_cast<uint8_t>(transport::MajorOrder::kLayerMajor);
  request.remote_id = 0;
  request.count_or_size = 1;
  request.uuid = kLeaseAuthorizedPullUuid;
  const auto serialized = transport::lib::SerializeChunkHeader(request);
  ASSERT_TRUE(WriteAll(original_client, serialized.data(), serialized.size()));
  std::array<char, transport::lib::kChunkHeaderSize> response;
  ASSERT_TRUE(ReadAll(original_client, response.data(), response.size()));

  if (!readiness->WaitForPending(1, std::chrono::seconds(1))) {
    (void)shutdown(original_client, SHUT_RDWR);
    close(original_client);
    readiness->CancelAll();
    server_transport.reset();
    server.reset();
    FAIL() << "pull response did not register its delayed callback";
    return;
  }

  const std::optional<int> accepted_fd = FindAcceptedSocketForPort(server_port);
  if (!accepted_fd.has_value()) {
    (void)shutdown(original_client, SHUT_RDWR);
    close(original_client);
    readiness->CancelAll();
    server_transport.reset();
    server.reset();
    FAIL() << "could not identify the original accepted descriptor";
    return;
  }

  // Create the replacement client while the original accepted descriptor is
  // still occupied, then fill every lower fd gap. Keeping original_client open
  // after shutdown ensures the next accept deterministically reuses exactly
  // the server descriptor retained by the delayed callback in the old code.
  const int replacement_client = socket(AF_INET, SOCK_STREAM, 0);
  std::optional<std::vector<int>> held_fds =
      HoldFreeDescriptorsBelow(*accepted_fd);
  if (replacement_client < 0 || !held_fds.has_value()) {
    if (replacement_client >= 0) close(replacement_client);
    (void)shutdown(original_client, SHUT_RDWR);
    close(original_client);
    readiness->CancelAll();
    server_transport.reset();
    server.reset();
    FAIL() << "could not prepare deterministic descriptor reuse";
    return;
  }

  (void)shutdown(original_client, SHUT_RDWR);
  if (!WaitForDescriptorClose(*accepted_fd, std::chrono::seconds(1)) ||
      !ConnectExistingLoopbackSocket(replacement_client, server_port) ||
      !WaitForAcceptedSocketAtFd(*accepted_fd, server_port,
                                 std::chrono::seconds(1))) {
    readiness->CancelAll();
    (void)shutdown(replacement_client, SHUT_RDWR);
    close(replacement_client);
    close(original_client);
    for (int fd : *held_fds) close(fd);
    server_transport.reset();
    server.reset();
    FAIL() << "server did not reuse the original accepted descriptor";
    return;
  }

  ASSERT_TRUE(readiness->CompleteOne(
      absl::CancelledError("release delayed pull response")));
  const ReusedSocketPair replacement_pair{*accepted_fd, replacement_client};
  EXPECT_TRUE(SocketPairRoundTrip(replacement_pair, 0x51));

  // Let RawBufferTransport retire the replacement accepted socket, then reuse
  // the exact number once more outside the transport. Repeated transport-level
  // cancellation, manager-level cancellation, and destruction must not retain
  // or act on either old generation.
  (void)shutdown(replacement_client, SHUT_RDWR);
  close(replacement_client);
  close(original_client);
  for (int fd : *held_fds) close(fd);
  ASSERT_TRUE(WaitForDescriptorClose(*accepted_fd, std::chrono::seconds(1)));
  std::optional<ReusedSocketPair> unrelated =
      ReuseDescriptorWithSocketPair(*accepted_fd);
  ASSERT_TRUE(unrelated.has_value());

  server_transport->CancelPendingOperations();
  EXPECT_TRUE(SocketPairRoundTrip(*unrelated, 0x52));
  server_transport->CancelPendingOperations();
  EXPECT_TRUE(SocketPairRoundTrip(*unrelated, 0x53));
  server->CancelTransportForTest();
  EXPECT_TRUE(SocketPairRoundTrip(*unrelated, 0x54));
  server_transport.reset();
  server.reset();
  EXPECT_TRUE(SocketPairRoundTrip(*unrelated, 0x55));

  close(unrelated->reused_fd);
  close(unrelated->peer_fd);
}

TEST(TransportShutdownTest, ListenerCloseOwnershipSurvivesDescriptorReuse) {
  auto manager = std::make_unique<MetadataTransferManager>();
  ASSERT_TRUE(manager->local_port().has_value());
  manager->CancelTransportForTest();

  int pair[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, pair), 0);
  manager.reset();  // Repeated cancellation must not touch either reused fd.

  const uint8_t sent = 0x5a;
  uint8_t received = 0;
  EXPECT_TRUE(WriteAll(pair[0], &sent, sizeof(sent)));
  EXPECT_TRUE(ReadAll(pair[1], &received, sizeof(received)));
  EXPECT_EQ(received, sent);
  close(pair[0]);
  close(pair[1]);
}

TEST(SendLifecycleTest, SimultaneousPullsHaveExactlyOneWinner) {
  MetadataTransferManager manager(/*start_control_server=*/true,
                                  /*num_slots=*/1,
                                  /*parallelism=*/2);
  manager.NotifyForRead("race", 107, {0});
  std::atomic<bool> go = false;
  auto first = std::async(std::launch::async, [&]() {
    while (!go.load()) std::this_thread::yield();
    return Pull(&manager, 107);
  });
  auto second = std::async(std::launch::async, [&]() {
    while (!go.load()) std::this_thread::yield();
    return Pull(&manager, 107);
  });
  go.store(true);

  const WireResponse first_response = first.get();
  const WireResponse second_response = second.get();
  EXPECT_EQ((first_response.status == 0) + (second_response.status == 0), 1);
  EXPECT_TRUE(manager.IsClaimed(107));
  manager.FinishForTest(107);
}

TEST(SendLifecycleTest, EarlyAcksAreBoundedAndQuarantineReuse) {
  MetadataTransferManager manager(/*start_control_server=*/true);
  manager.SetPendingAckCapacity(2);
  EXPECT_EQ(Ack(&manager, 110).status, 0);
  EXPECT_EQ(Ack(&manager, 111, /*legacy pull op=*/3).status, 0);
  EXPECT_EQ(Ack(&manager, 112).status, 0);
  EXPECT_EQ(manager.PendingAckCount(), 2);
  EXPECT_FALSE(manager.HasPendingAck(110));

  EXPECT_EQ(manager.NotifyForRead("early-ack", 111, {0}), 0);
  EXPECT_EQ(std::get<0>(manager.CompleteReadRaw()),
            std::vector<std::string>({"early-ack"}));
  EXPECT_TRUE(manager.HasTombstone(111));
  EXPECT_THROW(manager.NotifyForRead("reuse", 111, {0}), std::invalid_argument);
}

TEST(LeaseTest, OrderedStatusesCoverEveryLifecycleState) {
  MetadataTransferManager manager(/*start_control_server=*/true);
  manager.NotifyForRead("renew", 201, {0});
  const auto old_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
  manager.SetDeadline(201, old_deadline);
  manager.NotifyForRead("expired", 202, {0});
  manager.SetDeadline(
      202, std::chrono::steady_clock::now() - std::chrono::milliseconds(1));
  manager.NotifyForRead("claimed", 203, {0});
  ASSERT_EQ(Pull(&manager, 203).status, 0);
  manager.NotifyForRead("retained", 204, {0});
  manager.SetRetentionDeadline(
      204, std::chrono::steady_clock::now() - std::chrono::milliseconds(1));

  EXPECT_EQ(manager.Apply(MetadataTransferManager::RenewOp(),
                          {999, 201, 202, 203, 204, 999}),
            std::vector<int32_t>({0, 1, -1, -2, -3, 0}));
  EXPECT_GT(manager.GetDeadline(201), old_deadline);
  manager.FinishForTest(201);
  manager.FinishForTest(203);
}

TEST(LeaseTest, CancelBeforeRegisterIsIdempotentAndPreventsResurrection) {
  MetadataTransferManager manager(/*start_control_server=*/true);
  EXPECT_EQ(manager.Apply(MetadataTransferManager::CancelOp(), {205, 205}),
            std::vector<int32_t>({1, 1}));
  EXPECT_EQ(manager.Apply(MetadataTransferManager::RenewOp(), {205}),
            std::vector<int32_t>({-1}));
  EXPECT_EQ(Pull(&manager, 205).status, -1);
  EXPECT_EQ(manager.NotifyForRead("cancel-before-register", 205, {0}), 0);
  EXPECT_FALSE(manager.HasEntry(205));
  EXPECT_EQ(std::get<0>(manager.CompleteReadRaw()),
            std::vector<std::string>({"cancel-before-register"}));
  EXPECT_THROW(manager.NotifyForRead("duplicate", 205, {0}),
               std::invalid_argument);
}

TEST(LeaseTest, CancelVsPullHasOneLinearizedOwner) {
  MetadataTransferManager manager(/*start_control_server=*/true);
  manager.NotifyForRead("cancel-race", 206, {0});
  std::atomic<bool> go = false;
  auto pull = std::async(std::launch::async, [&]() {
    while (!go.load()) std::this_thread::yield();
    return Pull(&manager, 206);
  });
  auto cancel = std::async(std::launch::async, [&]() {
    while (!go.load()) std::this_thread::yield();
    return manager.Apply(MetadataTransferManager::CancelOp(), {206})[0];
  });
  go.store(true);
  const WireResponse pull_response = pull.get();
  const int32_t cancel_status = cancel.get();
  EXPECT_TRUE((pull_response.status == 0 && cancel_status == -2) ||
              (pull_response.status == -1 && cancel_status == 1));
  if (pull_response.status == 0) manager.FinishForTest(206);
  EXPECT_TRUE(manager.HasTombstone(206));
}

TEST(LeaseProtocolTest, VersionedWireApiPreservesOrderAndChunks) {
  MetadataTransferManager manager(/*start_control_server=*/true);
  manager.NotifyForRead("wire", 207, {0});
  EXPECT_EQ(manager.RenewRemoteLeases(
                "127.0.0.1:" + std::to_string(manager.local_control_port()),
                {999, 207, 999}),
            std::vector<int32_t>({0, 1, 0}));

  std::vector<uint64_t> large(MetadataTransferManager::MaxBatchSize() + 1);
  for (size_t i = 0; i < large.size(); ++i) large[i] = 10000 + i;
  const std::vector<int32_t> results = manager.RenewRemoteLeases(
      "127.0.0.1:" + std::to_string(manager.local_control_port()), large);
  EXPECT_EQ(results.size(), large.size());
  EXPECT_TRUE(std::all_of(results.begin(), results.end(),
                          [](int32_t status) { return status == 0; }));
  manager.FinishForTest(207);
}

TEST(LeaseProtocolTest, RejectsMissingOrWrongRequestVersion) {
  MetadataTransferManager manager(/*start_control_server=*/true);
  MetadataTransferManager::ControlRequestHeader request;
  request.op = MetadataTransferManager::RenewOp();
  request.num_blocks = 1;
  request.ep_idx = 0;
  EXPECT_EQ(RawExchange(&manager, request, {1}).status, -1);
  request.ep_idx = MetadataTransferManager::ProtocolVersion() + 1;
  EXPECT_EQ(RawExchange(&manager, request, {1}).status, -1);
  request.ep_idx = MetadataTransferManager::ProtocolVersion();
  WireResponse response = RawExchange(&manager, request, {1});
  EXPECT_EQ(response.status, 0);
  EXPECT_EQ(response.version, MetadataTransferManager::ProtocolVersion());
  EXPECT_EQ(response.lease_results, std::vector<int32_t>({0}));
}

TEST(LeaseProtocolTest, ClientRejectsOldServerAndMixedResponseVersion) {
  MetadataTransferManager manager;
  manager.SetControlIoTimeout(std::chrono::milliseconds(200));

  auto run_fake_server = [&](int32_t status, uint32_t version,
                             const std::string& error) {
    LoopbackListener listener = OpenLoopbackListener();
    std::thread server([listener, status, version, error]() {
      const int fd = accept(listener.fd, nullptr, nullptr);
      if (fd >= 0) {
        MetadataTransferManager::ControlRequestHeader request;
        if (ReadAll(fd, &request, sizeof(request))) {
          std::vector<uint64_t> body(request.num_blocks);
          if (ReadAll(fd, body.data(), body.size() * sizeof(uint64_t))) {
            MetadataTransferManager::ControlResponseHeader response;
            response.status = status;
            response.num_layers = version;
            response.message_len = status == 0 ? sizeof(int32_t) : error.size();
            (void)SendNoSignal(fd, &response, sizeof(response));
            if (status == 0) {
              const int32_t applied = 1;
              (void)SendNoSignal(fd, &applied, sizeof(applied));
            } else {
              (void)SendNoSignal(fd, error.data(), error.size());
            }
          }
        }
        (void)shutdown(fd, SHUT_RDWR);
        close(fd);
      }
      close(listener.fd);
    });
    bool threw = false;
    try {
      (void)manager.RenewRemoteLeases(
          "127.0.0.1:" + std::to_string(listener.port), {1});
    } catch (const std::runtime_error&) {
      threw = true;
    }
    server.join();
    return threw;
  };

  EXPECT_TRUE(run_fake_server(-1, 0, "unknown control op code: 4"));
  EXPECT_TRUE(
      run_fake_server(0, MetadataTransferManager::ProtocolVersion() + 1, ""));
}

TEST(ControlProtocolTest, TrickledBodyUsesOneAbsoluteDeadline) {
  MetadataTransferManager manager(/*start_control_server=*/true);
  manager.SetControlIoTimeout(std::chrono::milliseconds(40));
  manager.NotifyForRead("trickle", 300, {0});
  const int fd = ConnectLoopback(manager.local_control_port());
  MetadataTransferManager::ControlRequestHeader request;
  request.op = 3;
  request.uuid = 300;
  request.num_blocks = 1;
  request.consumer_data_port = 1;
  ASSERT_TRUE(WriteAll(fd, &request, sizeof(request)));

  const auto start = std::chrono::steady_clock::now();
  const std::vector<uint8_t> body(2 * sizeof(int64_t), 0);
  for (uint8_t byte : body) {
    if (!SendNoSignal(fd, &byte, 1)) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  const WireResponse response = ReadResponse(fd);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  (void)shutdown(fd, SHUT_RDWR);
  close(fd);
  EXPECT_NE(response.status, 0);
  EXPECT_LT(
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
      500);
  EXPECT_TRUE(manager.HasEntry(300));
  EXPECT_EQ(manager.FreeSlotCount(), 2);
  manager.FinishForTest(300);
}

TEST(ControlProtocolTest, TrickledResponseUsesOneAbsoluteClientDeadline) {
  MetadataTransferManager manager;
  manager.SetControlIoTimeout(std::chrono::milliseconds(40));
  LoopbackListener listener = OpenLoopbackListener();
  std::thread server([listener]() {
    const int fd = accept(listener.fd, nullptr, nullptr);
    if (fd >= 0) {
      MetadataTransferManager::ControlRequestHeader request;
      if (ReadAll(fd, &request, sizeof(request))) {
        std::vector<uint64_t> body(request.num_blocks);
        if (ReadAll(fd, body.data(), body.size() * sizeof(uint64_t))) {
          MetadataTransferManager::ControlResponseHeader response;
          response.status = 0;
          response.num_layers = MetadataTransferManager::ProtocolVersion();
          response.message_len = sizeof(int32_t);
          const auto* bytes = reinterpret_cast<const uint8_t*>(&response);
          for (size_t i = 0; i < sizeof(response); ++i) {
            if (!SendNoSignal(fd, bytes + i, 1)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
        }
      }
      (void)shutdown(fd, SHUT_RDWR);
      close(fd);
    }
    close(listener.fd);
  });

  const auto start = std::chrono::steady_clock::now();
  EXPECT_THROW(manager.RenewRemoteLeases(
                   "127.0.0.1:" + std::to_string(listener.port), {1}),
               std::runtime_error);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  server.join();
  EXPECT_LT(
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
      500);
}

TEST(ControlProtocolTest, ShutdownInterruptsPartialHeaderRead) {
  MetadataTransferManager manager(/*start_control_server=*/true);
  manager.SetControlIoTimeout(std::chrono::seconds(30));
  const int fd = ConnectLoopback(manager.local_control_port());
  const uint32_t magic = 0x52414944;
  ASSERT_TRUE(WriteAll(fd, &magic, sizeof(magic)));
  manager.WaitForActiveControlFd();

  auto stop = std::async(std::launch::async, [&]() { manager.StopForTest(); });
  EXPECT_EQ(stop.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  stop.get();
  close(fd);
}

TEST(SendLifecycleTest, TombstonesRemainHardBounded) {
  MetadataTransferManager manager;
  manager.SetTombstoneCapacity(2);
  for (uint64_t uuid : {400, 401, 402}) {
    manager.NotifyForRead("bounded", uuid, {0});
    manager.SetDeadline(
        uuid, std::chrono::steady_clock::now() - std::chrono::milliseconds(1));
    (void)manager.CompleteReadRaw();
  }
  EXPECT_EQ(manager.TombstoneCount(), 2);
  EXPECT_FALSE(manager.HasTombstone(400));
  EXPECT_TRUE(manager.HasTombstone(401));
  EXPECT_TRUE(manager.HasTombstone(402));
}

}  // namespace
}  // namespace tpu_raiden
