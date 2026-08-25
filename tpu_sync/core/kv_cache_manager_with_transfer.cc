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

// Dummy change to force Kokoro retry.
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdlib>
#include <optional>
#include <ratio>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/errors.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/metrics_collector.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/status_macros.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/block_transport.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {

namespace {

bool EncodeIp(const std::string& ip_str, uint8_t* dst) {
  if (inet_pton(AF_INET6, ip_str.c_str(), dst) > 0) {
    return true;
  }
  struct in_addr ipv4_addr;
  if (inet_pton(AF_INET, ip_str.c_str(), &ipv4_addr) > 0) {
    std::memset(dst, 0, 10);
    dst[10] = 0xff;
    dst[11] = 0xff;
    std::memcpy(dst + 12, &ipv4_addr, 4);
    return true;
  }
  return false;
}

constexpr absl::Duration kPendingWorkTimeout = absl::Seconds(30);
constexpr uint64_t kMaxControlMessageBytes = 64 * 1024;
constexpr char kSendUuidNotRegistered[] = "send UUID is not registered yet";

class RemoteControlError : public std::runtime_error {
 public:
  RemoteControlError(int32_t remote_status, std::string remote_message)
      : std::runtime_error("remote Raiden control error: " + remote_message),
        remote_status_(remote_status) {}

  int32_t remote_status() const { return remote_status_; }

 private:
  int32_t remote_status_;
};

const char* LeaseOperationName(uint32_t op) {
  switch (op) {
    case 4:
      return "renew";
    case 5:
      return "cancel";
    default:
      return "unknown";
  }
}

const char* LeaseStatusName(int32_t status) {
  switch (status) {
    case 1:
      return "applied";
    case 0:
      return "unknown";
    case -1:
      return "terminal";
    case -2:
      return "transferring";
    case -3:
      return "max_retention";
    default:
      return "invalid";
  }
}

void RecordLeaseUpdateMetric(uint32_t op, int32_t status) {
  const std::array<telemetry::MetricLabel, 2> labels = {
      telemetry::MetricLabel{"operation", LeaseOperationName(op)},
      telemetry::MetricLabel{"status", LeaseStatusName(status)}};
  telemetry::RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
      "tpu_raiden_legacy_uuid_lease_updates_total", labels);
}

void RecordPullMetric(absl::string_view result) {
  const std::array<telemetry::MetricLabel, 1> labels = {
      telemetry::MetricLabel{"result", result}};
  telemetry::RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
      "tpu_raiden_legacy_uuid_pull_claims_total", labels);
}

[[noreturn]] void ThrowStatus(const std::string& context,
                              const absl::Status& status) {
  throw std::runtime_error(context + ": " + std::string(status.message()));
}

void CheckStatus(const std::string& context, const absl::Status& status) {
  if (!status.ok()) {
    ThrowStatus(context, status);
  }
}

void EmitTimingLog(const std::string& message) { LOG(INFO) << message; }

bool StridedSpanFitsBlock(int64_t offset, int64_t stride, int64_t size,
                          int64_t count, int64_t block_size) {
  if (offset < 0 || stride < 0 || size <= 0 || count <= 0 || block_size <= 0 ||
      offset > block_size || size > block_size - offset) {
    return false;
  }
  if (count == 1) return true;

  // Division avoids overflowing (count - 1) * stride.
  const int64_t remaining = block_size - offset - size;
  return stride <= remaining / (count - 1);
}

bool StridedSpanFitsRegions(int64_t offset, int64_t stride, int64_t size,
                            int64_t count, int64_t block_size,
                            const std::vector<kv_cache::RegionSpec>& regions) {
  if (!StridedSpanFitsBlock(offset, stride, size, count, block_size)) {
    return false;
  }
  for (int64_t repeat = 0; repeat < count; ++repeat) {
    const int64_t start = offset + repeat * stride;
    if (!kv_cache::RegionsCoverRange(regions, static_cast<size_t>(start),
                                     static_cast<size_t>(start + size))) {
      return false;
    }
  }
  return true;
}

template <typename T>
T ValueOrThrow(const std::string& context, absl::StatusOr<T> value_or) {
  if (!value_or.ok()) {
    ThrowStatus(context, value_or.status());
  }
  return std::move(value_or).value();
}

int PollTimeoutMillis(std::chrono::steady_clock::time_point deadline) {
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) return 0;
  const int64_t remaining_us =
      std::chrono::duration_cast<std::chrono::microseconds>(deadline - now)
          .count();
  return static_cast<int>(
      std::min<int64_t>(std::numeric_limits<int>::max(),
                        std::max<int64_t>(1, (remaining_us + 999) / 1000)));
}

absl::Status ReadExactUntil(int fd, void* buffer, size_t length,
                            std::chrono::steady_clock::time_point deadline) {
  uint8_t* ptr = static_cast<uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    const int timeout_ms = PollTimeoutMillis(deadline);
    if (timeout_ms == 0) {
      return absl::DeadlineExceededError(
          "control operation read deadline exceeded");
    }
    pollfd pfd{fd, POLLIN, 0};
    const int ready = poll(&pfd, 1, timeout_ms);
    if (ready < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError("socket read poll failed: " +
                                 std::string(std::strerror(errno)));
    }
    if (ready == 0) {
      return absl::DeadlineExceededError(
          "control operation read deadline exceeded");
    }
    const ssize_t bytes_read = recv(fd, ptr, remaining, MSG_DONTWAIT);
    if (bytes_read < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return absl::InternalError("socket read failed: " +
                                 std::string(std::strerror(errno)));
    }
    if (bytes_read == 0) {
      return absl::InternalError("socket closed during read");
    }
    ptr += bytes_read;
    remaining -= bytes_read;
  }
  return absl::OkStatus();
}

absl::Status WriteExactUntil(int fd, const void* buffer, size_t length,
                             std::chrono::steady_clock::time_point deadline) {
  const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    const int timeout_ms = PollTimeoutMillis(deadline);
    if (timeout_ms == 0) {
      return absl::DeadlineExceededError(
          "control operation write deadline exceeded");
    }
    pollfd pfd{fd, POLLOUT, 0};
    const int ready = poll(&pfd, 1, timeout_ms);
    if (ready < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError("socket write poll failed: " +
                                 std::string(std::strerror(errno)));
    }
    if (ready == 0) {
      return absl::DeadlineExceededError(
          "control operation write deadline exceeded");
    }
    int flags = MSG_DONTWAIT;
#ifdef MSG_NOSIGNAL
    flags |= MSG_NOSIGNAL;
#endif
    const ssize_t written = send(fd, ptr, remaining, flags);
    if (written < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return absl::InternalError("socket write failed: " +
                                 std::string(std::strerror(errno)));
    }
    if (written == 0) {
      return absl::InternalError("socket closed during write");
    }
    ptr += written;
    remaining -= written;
  }
  return absl::OkStatus();
}

void ConfigureSocketIoTimeout(int fd, std::chrono::milliseconds timeout) {
  const int64_t timeout_us = std::max<int64_t>(1, timeout.count() * 1000);
  timeval value;
  value.tv_sec = timeout_us / 1000000;
  value.tv_usec = timeout_us % 1000000;
  if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &value, sizeof(value)) < 0 ||
      setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &value, sizeof(value)) < 0) {
    throw std::runtime_error("setsockopt(control timeout) failed: " +
                             std::string(std::strerror(errno)));
  }
}

std::pair<std::string, int> SplitEndpoint(const std::string& endpoint) {
  std::string host;
  int port = 0;
  if (endpoint.empty()) {
    throw std::invalid_argument("endpoint is empty");
  }
  if (endpoint[0] == '[') {
    size_t closing_bracket = endpoint.find(']');
    if (closing_bracket == std::string::npos ||
        closing_bracket + 2 >= endpoint.size() ||
        endpoint[closing_bracket + 1] != ':') {
      throw std::invalid_argument("invalid IPv6 endpoint: " + endpoint);
    }
    host = endpoint.substr(1, closing_bracket - 1);
    port = std::stoi(endpoint.substr(closing_bracket + 2));
  } else {
    size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos) {
      throw std::invalid_argument("endpoint must be host:port");
    }
    host = endpoint.substr(0, colon);
    port = std::stoi(endpoint.substr(colon + 1));
  }
  return {host, port};
}

int ConnectTcp(const std::string& endpoint,
               std::chrono::steady_clock::time_point deadline) {
  auto [host, port] = SplitEndpoint(endpoint);
  struct addrinfo hints;
  struct addrinfo* res = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  // All production control endpoints are numeric. Avoid the system resolver
  // for those so the supplied deadline covers the full operation. Portable
  // getaddrinfo offers no cancellation API; hostname DNS remains outside the
  // strict socket deadline and is documented as such.
  in_addr numeric_v4;
  in6_addr numeric_v6;
  if (inet_pton(AF_INET, host.c_str(), &numeric_v4) == 1 ||
      inet_pton(AF_INET6, host.c_str(), &numeric_v6) == 1) {
    hints.ai_flags |= AI_NUMERICHOST;
  }

  std::string port_str = std::to_string(port);
  int err = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
  if (err != 0) {
    throw std::runtime_error("Failed to resolve hostname '" + host +
                             "': " + gai_strerror(err));
  }

  int fd = -1;
  std::string last_error = "no resolved address";
  for (addrinfo* candidate = res; candidate != nullptr;
       candidate = candidate->ai_next) {
    if (std::chrono::steady_clock::now() >= deadline) {
      last_error = "deadline exceeded";
      break;
    }
    fd = socket(candidate->ai_family, candidate->ai_socktype,
                candidate->ai_protocol);
    if (fd < 0) {
      last_error = std::strerror(errno);
      continue;
    }
    const int original_flags = fcntl(fd, F_GETFL, 0);
    if (original_flags < 0 ||
        fcntl(fd, F_SETFL, original_flags | O_NONBLOCK) < 0) {
      last_error = std::strerror(errno);
      close(fd);
      fd = -1;
      continue;
    }
    const int opt = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    int connect_result = connect(fd, candidate->ai_addr, candidate->ai_addrlen);
    if (connect_result < 0 && errno == EINPROGRESS) {
      pollfd pfd{fd, POLLOUT, 0};
      int ready;
      do {
        ready = poll(&pfd, 1, PollTimeoutMillis(deadline));
      } while (ready < 0 && errno == EINTR);
      if (ready > 0) {
        int socket_error = 0;
        socklen_t error_length = sizeof(socket_error);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error,
                       &error_length) == 0 &&
            socket_error == 0) {
          connect_result = 0;
        } else {
          last_error = std::strerror(socket_error == 0 ? errno : socket_error);
        }
      } else {
        last_error = ready == 0 ? "deadline exceeded" : std::strerror(errno);
      }
    } else if (connect_result < 0) {
      last_error = std::strerror(errno);
    }
    if (connect_result == 0 && fcntl(fd, F_SETFL, original_flags) == 0) {
      break;
    }
    if (connect_result == 0) last_error = std::strerror(errno);
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd < 0) {
    throw std::runtime_error("connect(" + endpoint + ") failed: " + last_error);
  }
  return fd;
}

static std::string GetPeerIp(int fd) {
  sockaddr_storage addr;
  socklen_t len = sizeof(addr);
  if (getpeername(fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
    throw std::runtime_error("getpeername() failed: " +
                             std::string(std::strerror(errno)));
  }
  char ip_buf[INET6_ADDRSTRLEN];
  if (addr.ss_family == AF_INET) {
    sockaddr_in* s = reinterpret_cast<sockaddr_in*>(&addr);
    if (inet_ntop(AF_INET, &s->sin_addr, ip_buf, sizeof(ip_buf)) == nullptr) {
      throw std::runtime_error("inet_ntop() failed: " +
                               std::string(std::strerror(errno)));
    }
  } else if (addr.ss_family == AF_INET6) {
    sockaddr_in6* s = reinterpret_cast<sockaddr_in6*>(&addr);
    if (inet_ntop(AF_INET6, &s->sin6_addr, ip_buf, sizeof(ip_buf)) == nullptr) {
      throw std::runtime_error("inet_ntop() failed: " +
                               std::string(std::strerror(errno)));
    }
  } else {
    throw std::runtime_error("unknown socket family");
  }
  return std::string(ip_buf);
}
static void WriteBlockIds(
    int fd, const std::vector<int64_t>& block_ids,
    std::chrono::steady_clock::time_point operation_deadline) {
  if (block_ids.empty()) return;
  CheckStatus(
      "control block ids write",
      WriteExactUntil(fd, block_ids.data(), block_ids.size() * sizeof(int64_t),
                      operation_deadline));
}

static std::vector<int64_t> ReadBlockIds(
    int fd, uint64_t num_blocks,
    std::chrono::steady_clock::time_point operation_deadline) {
  if (num_blocks == 0) return {};
  if (num_blocks > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument("num_blocks is too large");
  }
  std::vector<int64_t> block_ids(static_cast<size_t>(num_blocks));
  CheckStatus(
      "control block ids read",
      ReadExactUntil(fd, block_ids.data(), block_ids.size() * sizeof(int64_t),
                     operation_deadline));
  return block_ids;
}

static CopySpec OffsetsImpl(const std::vector<int64_t>& block_ids,
                            bool source_is_compact) {
  const int64_t n = static_cast<int64_t>(block_ids.size());
  CopySpec spec;
  spec.src_offsets.reserve(block_ids.size());
  spec.dst_offsets.reserve(block_ids.size());
  spec.sizes.reserve(block_ids.size());
  for (int64_t start = 0; start < n;) {
    int64_t end = start + 1;
    while (end < n && block_ids[end] == block_ids[end - 1] + 1) {
      ++end;
    }
    const int64_t run_size = end - start;
    if (source_is_compact) {
      spec.src_offsets.push_back(start);
      spec.dst_offsets.push_back(block_ids[start]);
    } else {
      spec.src_offsets.push_back(block_ids[start]);
      spec.dst_offsets.push_back(start);
    }
    spec.sizes.push_back(run_size);
    start = end;
  }
  return spec;
}

static kv_cache::KVCacheCopySpec ToKVCacheCopySpecImpl(const CopySpec& spec) {
  return {.src_offsets = spec.src_offsets,
          .dst_offsets = spec.dst_offsets,
          .sizes = spec.sizes};
}

static CopySpec BuildCoalescedCopySpec(
    const std::vector<int64_t>& src_block_ids,
    const std::vector<int64_t>& dst_block_ids) {
  if (src_block_ids.size() != dst_block_ids.size()) {
    throw std::invalid_argument(
        "src and dst block lists must have same length");
  }
  CopySpec spec;
  if (src_block_ids.empty()) {
    return spec;
  }
  const int64_t n = static_cast<int64_t>(src_block_ids.size());
  spec.src_offsets.reserve(n);
  spec.dst_offsets.reserve(n);
  spec.sizes.reserve(n);

  for (int64_t start = 0; start < n;) {
    int64_t end = start + 1;
    while (end < n && src_block_ids[end] == src_block_ids[end - 1] + 1 &&
           dst_block_ids[end] == dst_block_ids[end - 1] + 1) {
      ++end;
    }
    const int64_t run_size = end - start;
    spec.src_offsets.push_back(src_block_ids[start]);
    spec.dst_offsets.push_back(dst_block_ids[start]);
    spec.sizes.push_back(run_size);
    start = end;
  }
  return spec;
}

static CopyPlan BuildLoadCopyPlan(
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids,
    const std::vector<int64_t>& local_host_block_ids) {
  if (remote_block_ids.size() != local_block_ids.size() ||
      local_block_ids.size() != local_host_block_ids.size()) {
    throw std::invalid_argument(
        "remote_block_ids, local_block_ids, and local_host_block_ids must have "
        "same length");
  }
  CopyPlan plan;
  plan.num_blocks = static_cast<int64_t>(remote_block_ids.size());
  plan.requested_remote_block_ids = remote_block_ids;
  plan.requested_local_block_ids = local_block_ids;
  if (remote_block_ids.empty()) {
    return plan;
  }

  // 1. Determine transport order (sorted by remote_block_ids)
  std::vector<size_t> remote_order(remote_block_ids.size());
  for (size_t i = 0; i < remote_order.size(); ++i) {
    remote_order[i] = i;
  }
  std::stable_sort(remote_order.begin(), remote_order.end(),
                   [&](size_t a, size_t b) {
                     return remote_block_ids[a] < remote_block_ids[b];
                   });

  plan.producer_remote_block_ids.reserve(remote_order.size());
  plan.transport_host_block_ids.reserve(remote_order.size());
  for (size_t i = 0; i < remote_order.size(); ++i) {
    const size_t original_idx = remote_order[i];
    plan.producer_remote_block_ids.push_back(remote_block_ids[original_idx]);
    plan.transport_host_block_ids.push_back(local_host_block_ids[original_idx]);
  }

  // 2. Determine H2D copy plan (sorted by local_block_ids for opt)
  std::vector<size_t> local_order(local_block_ids.size());
  for (size_t i = 0; i < local_order.size(); ++i) {
    local_order[i] = i;
  }
  std::stable_sort(local_order.begin(), local_order.end(),
                   [&](size_t a, size_t b) {
                     return local_block_ids[a] < local_block_ids[b];
                   });

  plan.h2d_local_block_ids.reserve(local_order.size());
  plan.h2d_host_block_ids.reserve(local_order.size());
  for (size_t i = 0; i < local_order.size(); ++i) {
    const size_t original_idx = local_order[i];
    int64_t local_bid = local_block_ids[original_idx];
    int64_t host_bid = local_host_block_ids[original_idx];
    if (plan.h2d_local_block_ids.empty() ||
        plan.h2d_local_block_ids.back() != local_bid) {
      plan.h2d_local_block_ids.push_back(local_bid);
      plan.h2d_host_block_ids.push_back(host_bid);
    } else {
      if (plan.h2d_host_block_ids.back() != host_bid) {
        throw std::invalid_argument(
            "Duplicate local block IDs must map to the same host block ID");
      }
    }
  }

  plan.h2d_copy =
      BuildCoalescedCopySpec(plan.h2d_host_block_ids, plan.h2d_local_block_ids);
  plan.host_dst_to_src.clear();  // No host reordering needed!
  return plan;
}

double DurationMs(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void RecordTransferDuration(double duration_ms) {
  telemetry::RaidenMetricStore::GetGlobalMetricStore().ObserveHistogram(
      telemetry::metric_names::kTransferDurationMs, {}, duration_ms);
}

}  // namespace

CopySpec KVCacheManagerWithTransfer::Offsets(
    const std::vector<int64_t>& block_ids, bool source_is_compact) {
  return OffsetsImpl(block_ids, source_is_compact);
}

kv_cache::KVCacheCopySpec KVCacheManagerWithTransfer::ToKVCacheCopySpec(
    const CopySpec& spec) {
  return ToKVCacheCopySpecImpl(spec);
}

void KVCacheManagerWithTransfer::ValidateRequestedBlocks(
    const SendEntry& entry, const std::vector<int64_t>& requested_block_ids) {
  if (requested_block_ids.empty()) {
    throw std::invalid_argument(
        "pull stream requested no blocks; use ack-only path");
  }
  absl::flat_hash_set<int64_t> seen;
  for (int64_t block_id : requested_block_ids) {
    if (entry.registered_block_set.find(block_id) ==
        entry.registered_block_set.end()) {
      throw std::invalid_argument(
          "pull stream requested block not registered by producer");
    }
    if (!seen.insert(block_id).second) {
      throw std::invalid_argument(
          "pull stream requested duplicate producer block id");
    }
  }
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism,
    HostBufferAllocator host_allocator, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::shared_ptr<MetricsCollector> metrics_collector)
    : KVCacheManagerBase(
          layer_buffers, local_port,
          host_blocks_to_allocate.value_or(num_slots * max_blocks),
          unsafe_skip_buffer_lock, parallelism, host_allocator),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      max_blocks_(max_blocks),
      num_slots_(num_slots),
      timeout_s_(timeout_s),
      unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock),
      metrics_collector_(std::move(metrics_collector)) {
  if (local_control_port_ >= 0) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    dynamic_host_staging_ = DynamicHostStagingEnabled();
    auto status = ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "Failed to configure host staging slots: ", status.message()));
    }
    if (num_layers() > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    // Demand staging allocates per request, so the pool stays whole rather
    // than being carved into fixed slots.
    status = dynamic_host_staging_ ? absl::OkStatus()
                                   : InitializeSlotPool(num_slots_);
    if (!status.ok()) {
      throw std::runtime_error(
          absl::StrCat("Failed to initialize slot pool: ", status.message()));
    }
    StartControlServer();
  }
  if (push_pool_) StartRegistrationRetryScheduler();
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    size_t slice_byte_size, const std::vector<int64_t>& dimensions,
    size_t physical_size, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
    int parallelism, HostBufferAllocator host_allocator, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::optional<int> assigned_numa_node,
    std::shared_ptr<MetricsCollector> metrics_collector)
    : KVCacheManagerBase(
          layer_buffers, local_port,
          host_blocks_to_allocate.value_or(num_slots * max_blocks),
          unsafe_skip_buffer_lock, parallelism, host_allocator,
          /*bind_ip=*/std::nullopt,
          slice_byte_size > 0 ? std::make_optional(slice_byte_size)
                              : std::nullopt,
          dimensions,
          physical_size > 0 ? std::make_optional(physical_size) : std::nullopt,
          assigned_numa_node),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      max_blocks_(max_blocks),
      num_slots_(num_slots),
      timeout_s_(timeout_s),
      unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock),
      metrics_collector_(std::move(metrics_collector)) {
  if (num_layers() == 0 || num_shards() == 0) {
    return;
  }
  if (local_control_port_ >= 0) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    dynamic_host_staging_ = DynamicHostStagingEnabled();
    auto status = ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "Failed to configure host staging slots: ", status.message()));
    }
    if (num_layers() > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    // Demand staging allocates per request, so the pool stays whole rather
    // than being carved into fixed slots.
    status = dynamic_host_staging_ ? absl::OkStatus()
                                   : InitializeSlotPool(num_slots_);
    if (!status.ok()) {
      throw std::runtime_error(
          absl::StrCat("Failed to initialize slot pool: ", status.message()));
    }
    StartControlServer();
  }
  if (push_pool_) StartRegistrationRetryScheduler();
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, int64_t node_id, int64_t local_control_port,
    int64_t max_blocks, int64_t num_slots, double timeout_s,
    std::shared_ptr<MetricsCollector> metrics_collector)
    : KVCacheManagerBase(
          num_layers, num_shards, slice_byte_size, local_port,
          host_blocks_to_allocate.value_or(num_slots * max_blocks), parallelism,
          nullptr),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      max_blocks_(max_blocks),
      num_slots_(num_slots),
      timeout_s_(timeout_s),
      unsafe_skip_buffer_lock_(false),
      metrics_collector_(std::move(metrics_collector)) {
  if (local_control_port_ >= 0) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    dynamic_host_staging_ = DynamicHostStagingEnabled();
    auto status = ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "Failed to configure host staging slots: ", status.message()));
    }
    if (num_layers > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    // Demand staging allocates per request, so the pool stays whole rather
    // than being carved into fixed slots.
    status = dynamic_host_staging_ ? absl::OkStatus()
                                   : InitializeSlotPool(num_slots_);
    if (!status.ok()) {
      throw std::runtime_error(
          absl::StrCat("Failed to initialize slot pool: ", status.message()));
    }
    StartControlServer();
  }
  if (push_pool_) StartRegistrationRetryScheduler();
}

KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer() {
  shutting_down_.store(true, std::memory_order_relaxed);
  StopControlServer();
  StopRegistrationRetryScheduler();
  CancelTransportOperations();
  {
    absl::MutexLock lock(mu_);
    while (!send_entries_.empty()) {
      auto it = send_entries_.begin();
      (void)TerminalizeSendEntryLocked(it->first, it->second,
                                       "manager stopped during producer send",
                                       /*waiting_only=*/false);
    }
    const absl::Time deadline = absl::Now() + absl::Seconds(30);
    while (active_send_callbacks_ != 0 && absl::Now() < deadline) {
      cv_.WaitWithDeadline(&mu_, deadline);
    }
    if (active_send_callbacks_ != 0) {
      LOG(FATAL) << "Legacy send callbacks did not drain after transport "
                    "cancellation; failing closed to prevent callback UAF or "
                    "staging-slot reuse";
    }
  }
  {
    std::unique_lock<std::mutex> lock(async_callbacks_mu_);
    if (!async_callbacks_cv_.wait_for(lock, std::chrono::seconds(30), [this]() {
          return active_async_callbacks_ == 0;
        })) {
      LOG(FATAL) << "Asynchronous manager callbacks did not drain after "
                    "transport cancellation; failing closed to prevent UAF";
    }
  }
  if (!WaitForManagerCallbacks(std::chrono::seconds(30))) {
    LOG(FATAL) << "Base pool-reshard callbacks did not drain before derived "
                  "manager teardown";
  }
  push_pool_.reset();
  pull_pool_.reset();
  if (host_block_manager_ && !all_slots_.empty()) {
    std::vector<int> blocks_to_unlock;
    blocks_to_unlock.reserve(all_slots_.size() * max_blocks_);
    for (const Slot& slot : all_slots_) {
      for (int block_id : slot.block_ids) {
        blocks_to_unlock.push_back(block_id);
      }
    }
    (void)host_block_manager_->Unlock(blocks_to_unlock);
  }
}

std::shared_ptr<void> KVCacheManagerWithTransfer::TrackAsyncCallback() {
  {
    std::lock_guard<std::mutex> lock(async_callbacks_mu_);
    ++active_async_callbacks_;
  }
  return std::shared_ptr<void>(this, [this](void*) {
    {
      std::lock_guard<std::mutex> lock(async_callbacks_mu_);
      if (active_async_callbacks_ == 0) {
        LOG(FATAL) << "Async callback tracker underflow";
      }
      --active_async_callbacks_;
    }
    async_callbacks_cv_.notify_all();
  });
}

int64_t KVCacheManagerWithTransfer::NotifyForRead(
    const std::string& req_id, uint64_t uuid,
    const std::vector<int64_t>& block_ids) {
  const auto register_start = std::chrono::steady_clock::now();
  if (block_ids.empty()) {
    return 0;
  }

  auto entry = std::make_shared<SendEntry>();
  entry->req_id = req_id;
  entry->uuid = uuid;
  entry->registered_num_blocks = static_cast<int64_t>(block_ids.size());
  entry->registered_block_ids = block_ids;
  for (int64_t block_id : block_ids) {
    entry->registered_block_set.insert(block_id);
  }
  const auto lease_duration = std::chrono::milliseconds(
      std::max<int64_t>(1, static_cast<int64_t>(timeout_s_ * 1000.0)));
  entry->deadline = register_start + lease_duration;
  entry->retention_deadline = register_start + lease_duration * 10;
  entry->register_start = register_start;

  bool consumed_early_ack = false;
  bool consumed_early_cancel = false;
  {
    absl::MutexLock lock(mu_);
    const auto now = std::chrono::steady_clock::now();
    PurgeSendTombstonesLocked(now);
    auto tombstone = send_tombstones_.find(uuid);
    if (tombstone != send_tombstones_.end()) {
      if (tombstone->second.kind == SendTombstoneKind::kPreRegistrationCancel) {
        InstallSendTombstoneLocked(uuid, "send cancelled before registration",
                                   now, SendTombstoneKind::kCancelled);
        done_sending_.insert(req_id);
        (void)ConsumePendingAckLocked(uuid, now);
        consumed_early_cancel = true;
      } else {
        throw std::invalid_argument(
            "send UUID is quarantined by a previous terminal outcome");
      }
    }
    if (!consumed_early_cancel &&
        (send_entries_.find(uuid) != send_entries_.end() ||
         draining_send_entries_.find(uuid) != draining_send_entries_.end())) {
      throw std::invalid_argument("send UUID is already registered");
    }
    if (!consumed_early_cancel && ConsumePendingAckLocked(uuid, now)) {
      done_sending_.insert(req_id);
      InstallSendTombstoneLocked(uuid, "send was acknowledged before notify",
                                 now);
      consumed_early_ack = true;
    } else if (!consumed_early_cancel) {
      send_entries_.emplace(uuid, entry);
    }
    UpdateLegacySendGaugesLocked();
  }
  cv_.SignalAll();

  if (consumed_early_ack || consumed_early_cancel) {
    std::ostringstream timing;
    timing << "RAIDEN_TIMING event=producer_registration_suppressed"
           << " req_id=" << req_id << " uuid=" << uuid
           << " node_id=" << node_id_ << " blocks=" << block_ids.size()
           << " reason="
           << (consumed_early_cancel ? "early_cancel" : "early_ack")
           << " enqueue_ms="
           << DurationMs(register_start, std::chrono::steady_clock::now())
           << " failed=0";
    EmitTimingLog(timing.str());
    return 0;
  }

  std::ostringstream timing;
  timing << "RAIDEN_TIMING event=producer_register"
         << " req_id=" << req_id << " uuid=" << uuid << " node_id=" << node_id_
         << " blocks=" << block_ids.size() << " enqueue_ms="
         << DurationMs(register_start, std::chrono::steady_clock::now())
         << " failed=0";
  EmitTimingLog(timing.str());
  return static_cast<int64_t>(uuid);
}

std::vector<int32_t> KVCacheManagerWithTransfer::RenewRemoteLeases(
    const std::string& remote_endpoint, const std::vector<uint64_t>& uuids) {
  return SendLeaseBatch(remote_endpoint, kOpRenewLeases, uuids);
}

std::vector<int32_t> KVCacheManagerWithTransfer::CancelRemoteLeases(
    const std::string& remote_endpoint, const std::vector<uint64_t>& uuids) {
  return SendLeaseBatch(remote_endpoint, kOpCancelLeases, uuids);
}

absl::Status KVCacheManagerWithTransfer::RegisterActivePlan(
    uint64_t uuid, const ::tpu_sync::rpc::StartTransferRequest& request,
    bool is_sender) {
  // Registration is one indivisible step: a concurrent unregister or a
  // second registration of the same uuid waits for it, so a plan is never
  // visible without the staging and receive state that belong to it.
  absl::MutexLock lifecycle(plan_lifecycle_mu_);
  if (kv_cache::KVCacheManagerBase::HasActivePlan(uuid)) {
    return absl::AlreadyExistsError(
        absl::StrCat("Plan with UUID ", uuid, " is already registered!"));
  }
  const uint64_t generation = ++plan_generation_counter_;
  // Under demand staging a plan's device blocks are staged in host blocks
  // allocated for the plan, so the host mirror no longer has to span the
  // device block space. Pool-addressed plans keep their own addressing.
  absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>
      host_block_of;
  std::vector<int> plan_blocks;
  if (dynamic_host_staging_ && request.pool_groups_size() == 0) {
    std::vector<int64_t> device_blocks;
    absl::flat_hash_set<int64_t> seen;
    for (const auto& [src_shard, schedule] : request.shard_push_schedules()) {
      for (const auto& e : schedule.entries()) {
        int64_t id = is_sender ? e.src_block_id() : e.dst_block_id();
        if (seen.insert(id).second) device_blocks.push_back(id);
      }
    }
    if (!device_blocks.empty()) {
      absl::MutexLock lock(mu_);
      auto allocated = host_block_manager_->Allocate(
          static_cast<int>(device_blocks.size()), /*lock=*/true);
      if (!allocated.ok()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "cannot stage ", device_blocks.size(), " blocks for plan ", uuid,
            ": ", allocated.status().message()));
      }
      plan_blocks = *allocated;
      for (size_t i = 0; i < device_blocks.size(); ++i) {
        host_block_of[device_blocks[i]] = plan_blocks[i];
      }
    }
  }


  // Staging ownership is settled before the plan is published. An HBM
  // receiver's blocks belong to its receive entry and return when the
  // upload settles; a sender's blocks, and a host-memory receiver's,
  // belong to the plan and return when it is unregistered.
  const bool hbm_receiver =
      !is_sender && request.dst_mem_type() == ::tpu_sync::rpc::MEMORY_TYPE_HBM;
  if (!plan_blocks.empty() && !hbm_receiver) {
    absl::MutexLock lock(mu_);
    plan_staging_[uuid] = plan_blocks;
  }

  // 2. If we are the receiver and the destination memory type is HBM,
  //    populate active_recv_entries_ to enable automatic H2D copy!
  if (hbm_receiver) {
    absl::MutexLock lock(mu_);
    RecvEntry recv_entry;
    recv_entry.staged_host_blocks = std::move(plan_blocks);
    // Once its receive settles, a demand-staged receiver plan would map
    // host blocks that are already freed; the plan is therefore dropped
    // together with the receive instead of outliving it.
    recv_entry.unregister_on_settle = !recv_entry.staged_host_blocks.empty();
    recv_entry.plan_generation = generation;
    std::string req_id = request.req_id().empty()
                             ? absl::StrCat("resharded_transfer_", uuid)
                             : request.req_id();
    recv_entry.req_id = req_id;

    int64_t total_blocks = 0;
    absl::flat_hash_set<int> unique_dst_blocks;
    for (const auto& [src_replica_idx, schedule] :
         request.shard_push_schedules()) {
      absl::flat_hash_set<std::pair<int, int>>
          unique_transfers_from_this_source;
      for (const auto& push_entry : schedule.entries()) {
        const int64_t dst = push_entry.dst_block_id();
        auto hb = host_block_of.find(dst);
        recv_entry.host_to_chip[hb == host_block_of.end() ? dst : hb->second] =
            dst;
        unique_transfers_from_this_source.insert(
            {push_entry.src_block_id(), push_entry.dst_block_id()});
        unique_dst_blocks.insert(push_entry.dst_block_id());
      }
      total_blocks += unique_transfers_from_this_source.size();
    }
    recv_entry.total_blocks = total_blocks;
    recv_entry.num_completed_blocks = 0;
    recv_entry.deadline = DeadlineFromNow();
    recv_entry.start_time = std::chrono::steady_clock::now();
    recv_entry.phase = RecvEntry::Phase::kTransferring;

    // H2D reads each destination block from wherever it was staged.
    std::vector<int64_t> h2d_local_block_ids(unique_dst_blocks.begin(),
                                             unique_dst_blocks.end());
    std::vector<int64_t> h2d_host_block_ids;
    h2d_host_block_ids.reserve(h2d_local_block_ids.size());
    for (int64_t dst : h2d_local_block_ids) {
      auto hb = host_block_of.find(dst);
      h2d_host_block_ids.push_back(hb == host_block_of.end() ? dst
                                                             : hb->second);
    }
    recv_entry.h2d_copy =
        BuildCoalescedCopySpec(h2d_host_block_ids, h2d_local_block_ids);

    if (total_blocks == 0) {
      ReleaseRecvStagingLocked(&recv_entry);
    }
    if (total_blocks > 0) {
      active_recv_entries_[uuid] = std::move(recv_entry);
      LOG(INFO) << "RegisterActivePlan (Receiver): Populated "
                   "active_recv_entries_ for UUID "
                << uuid << " with " << total_blocks
                << " total physical block-pushes (including duplicates across "
                   "sources) for automatic H2D.";
    }
  }

  // Publish the plan last: pushes resolve through it, so everything they
  // may touch exists by the time it is visible.
  absl::Status registered = kv_cache::KVCacheManagerBase::RegisterActivePlan(
      uuid, request, is_sender, host_block_of, generation);
  if (!registered.ok()) {
    absl::MutexLock lock(mu_);
    auto staged = plan_staging_.find(uuid);
    if (staged != plan_staging_.end()) {
      (void)host_block_manager_->Unlock(staged->second);
      (void)host_block_manager_->Deallocate(staged->second);
      plan_staging_.erase(staged);
    }
    auto recv = active_recv_entries_.find(uuid);
    if (recv != active_recv_entries_.end()) {
      ReleaseRecvStagingLocked(&recv->second);
      active_recv_entries_.erase(recv);
    }
    return registered;
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::RegisterRecv(
    uint64_t uuid, const std::string& req_id, int64_t expected_block_count) {
  absl::MutexLock lock(mu_);
  RecvEntry recv_entry;
  recv_entry.req_id = req_id;
  recv_entry.total_blocks = expected_block_count;
  recv_entry.num_completed_blocks = 0;
  recv_entry.deadline = DeadlineFromNow();
  recv_entry.phase = RecvEntry::Phase::kTransferring;
  // host_to_chip is left empty -> defaults to 1-to-1 mapping in
  // OnBlocksReceived
  active_recv_entries_[uuid] = std::move(recv_entry);
  VLOG(1)
      << "RegisterRecv (Receiver): Registered expected block count for UUID "
      << uuid << " with " << expected_block_count << " expected blocks.";
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::ValidatePoolReshardPlan(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> local_block_ids, bool is_sender) {
  if (plan.req_id().empty()) {
    return absl::InvalidArgumentError("reshard plan req_id must be non-empty");
  }
  if (plan.uuid() <= 0) {
    return absl::InvalidArgumentError("reshard plan uuid must be positive");
  }
  if (!plan.use_block_chunks()) {
    return absl::InvalidArgumentError(
        "pool reshard requires use_block_chunks=true");
  }
  if (plan.transfer_pool_indices().empty()) {
    return absl::InvalidArgumentError(
        "reshard plan must declare transfer_pool_indices");
  }
  if (plan.pool_groups().empty()) {
    return absl::InvalidArgumentError(
        "reshard plans must declare pool_groups (a plan is a list of "
        "groups; entries name their group)");
  }
  for (const auto& group : plan.pool_groups()) {
    if (group.expected_pushes() <= 0) {
      return absl::InvalidArgumentError(
          "every pool group must expect a positive push count");
    }
  }
  if (plan.pool_dtype_tags_size() != static_cast<int>(num_pools())) {
    return absl::InvalidArgumentError(
        absl::StrCat("reshard plan must declare one dtype tag per pool: plan=",
                     plan.pool_dtype_tags_size(), " local=", num_pools()));
  }
  if (local_block_ids.empty()) {
    return absl::InvalidArgumentError("local block ids must not be empty");
  }

  absl::flat_hash_set<int64_t> local_ids(local_block_ids.begin(),
                                         local_block_ids.end());
  // Different groups address different pools, so numerically equal ids
  // across groups are legitimate on both sides. Receiver: the flat list
  // must concatenate the groups' destination runs (uniqueness holds within
  // each group). Sender: the flat list is the union of per-tag source
  // blocks; only bounds are checked here — per-group scoping happens at
  // entry resolution.
  for (int64_t block_id : local_block_ids) {
    if (block_id < 0 || block_id > std::numeric_limits<int>::max()) {
      return absl::InvalidArgumentError(
          "local block ids must be non-negative and fit in int");
    }
  }
  if (!is_sender) {
    size_t cursor = 0;
    for (const auto& group : plan.pool_groups()) {
      absl::flat_hash_set<int64_t> group_ids;
      for (int64_t block_id : group.dst_device_block_ids()) {
        if (!group_ids.insert(block_id).second) {
          return absl::InvalidArgumentError(
              "group destination block ids must be unique");
        }
        if (cursor >= local_block_ids.size() ||
            local_block_ids[cursor] != block_id) {
          return absl::InvalidArgumentError(
              "pool group block ids must concatenate to the plan's "
              "local block ids");
        }
        ++cursor;
      }
    }
    if (cursor != local_block_ids.size()) {
      return absl::InvalidArgumentError(
          "pool group block ids must cover the plan's local block ids");
    }
  }

  // The executor validates the plan's *declared* pool set against this
  // manager's pool table — explicit or implicit — and its geometry. Which
  // tags should move is request data resolved by the controller; no tag name
  // means anything here.
  absl::flat_hash_set<size_t> declared_pools;
  for (int32_t encoded_pool_idx : plan.transfer_pool_indices()) {
    if (encoded_pool_idx < 0) {
      return absl::InvalidArgumentError(
          "transfer pool index must be non-negative");
    }
    const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
    if (!declared_pools.insert(pool_idx).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate transfer pool index ", pool_idx));
    }
    const kv_cache::PoolSpec* spec = pool(pool_idx);
    if (spec == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("transfer pool index out of range: ", pool_idx));
    }
    if (plan.pool_dtype_tags(pool_idx) != spec->dtype_tag) {
      return absl::InvalidArgumentError(
          absl::StrCat("plan dtype tag mismatch for pool ", pool_idx, " (",
                       spec->tag, "): plan=", plan.pool_dtype_tags(pool_idx),
                       " local=", spec->dtype_tag));
    }
    for (int64_t block_id : local_block_ids) {
      if (block_id >= spec->num_blocks) {
        return absl::InvalidArgumentError(
            absl::StrCat("local block id ", block_id,
                         " is out of range for pool ", pool_idx));
      }
    }
  }

  if (plan.shard_push_schedules().empty()) {
    return absl::InvalidArgumentError(
        "reshard plan must contain shard push schedules");
  }

  size_t entry_count = 0;
  absl::flat_hash_set<int64_t> receiver_blocks_with_zero_start;
  for (const auto& [source_rank, schedule] : plan.shard_push_schedules()) {
    if (source_rank < 0) {
      return absl::InvalidArgumentError(
          "reshard schedule source rank must be non-negative");
    }
    for (const auto& entry : schedule.entries()) {
      ++entry_count;
      if (entry.dst_peer().empty()) {
        return absl::InvalidArgumentError(
            "reshard entry dst_peer must be non-empty");
      }
      if (entry.src_block_id() < 0 ||
          entry.src_block_id() > std::numeric_limits<int>::max() ||
          entry.dst_block_id() < 0 ||
          entry.dst_block_id() > std::numeric_limits<int>::max() ||
          entry.dst_shard_idx() < 0 || entry.src_offset_bytes() < 0 ||
          entry.dst_offset_bytes() < 0 || entry.size_bytes() <= 0 ||
          entry.src_stride_bytes() < 0 || entry.dst_stride_bytes() < 0 ||
          entry.count() <= 0 || entry.count() > (1 << 20)) {
        return absl::InvalidArgumentError(
            "reshard entry contains invalid ids, offsets, sizes, or strides");
      }
      if (entry.count() > 1 &&
          (entry.src_stride_bytes() == 0 || entry.dst_stride_bytes() == 0)) {
        return absl::InvalidArgumentError(
            "multi-chunk reshard entries require positive strides");
      }
      if (!is_sender &&
          static_cast<size_t>(entry.dst_shard_idx()) >= num_shards()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "destination shard index ", entry.dst_shard_idx(),
            " is out of range: receiver has ", num_shards(), " shards"));
      }
      const int64_t local_id =
          is_sender ? entry.src_block_id() : entry.dst_block_id();
      if (local_ids.find(local_id) == local_ids.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat(is_sender ? "source" : "destination", " block id ",
                         local_id, " is absent from the local block-id list"));
      }
      const int64_t local_offset =
          is_sender ? entry.src_offset_bytes() : entry.dst_offset_bytes();
      const int64_t local_stride =
          is_sender ? entry.src_stride_bytes() : entry.dst_stride_bytes();
      if (!is_sender && entry.dst_offset_bytes() == 0) {
        receiver_blocks_with_zero_start.insert(entry.dst_block_id());
      }
      const int32_t group_idx = entry.pool_group();
      if (group_idx < 0 || group_idx >= plan.pool_groups_size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "reshard entry declares an unknown pool group ", group_idx));
      }
      std::vector<size_t> entry_pools;
      for (int32_t pool_idx : plan.pool_groups(group_idx).pool_indices()) {
        entry_pools.push_back(static_cast<size_t>(pool_idx));
      }
      for (size_t pool_idx : entry_pools) {
        const kv_cache::PoolSpec* spec = pool(pool_idx);
        if (!StridedSpanFitsRegions(local_offset, local_stride,
                                    entry.size_bytes(), entry.count(),
                                    spec->block_stride_bytes, spec->regions)) {
          return absl::InvalidArgumentError(absl::StrCat(
              is_sender ? "source" : "destination",
              " span exceeds declared pool ", pool_idx,
              " live regions in block ", local_id, ": offset=", local_offset,
              " stride=", local_stride, " size=", entry.size_bytes(), " count=",
              entry.count(), " block_stride_bytes=", spec->block_stride_bytes));
        }
      }
    }
  }
  if (entry_count == 0) {
    return absl::InvalidArgumentError("reshard plan contains no entries");
  }
  if (!is_sender) {
    for (int64_t block_id : local_ids) {
      if (receiver_blocks_with_zero_start.find(block_id) ==
          receiver_blocks_with_zero_start.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "destination block ", block_id,
            " has no transfer entry starting at offset 0; partial-page "
            "destination preservation is not implemented"));
      }
    }
    TF_RETURN_IF_ERROR(ValidatePoolReshardReceiverCoverage(plan));
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::ValidatePoolReshardReceiverCoverage(
    const ::tpu_sync::rpc::StartTransferRequest& plan) {
  constexpr int64_t kMaxExpandedRepeats = 1 << 20;

  struct GroupView {
    std::vector<size_t> pool_indices;
    std::vector<int64_t> dst_ids;
    std::vector<int64_t> extents;
    int64_t expected_pushes = 0;
  };
  std::vector<GroupView> groups;
  absl::flat_hash_set<size_t> grouped_pools;
  for (const auto& group : plan.pool_groups()) {
    GroupView view;
    for (int32_t pool_idx : group.pool_indices()) {
      if (pool_idx < 0 ||
          !grouped_pools.insert(static_cast<size_t>(pool_idx)).second) {
        return absl::InvalidArgumentError(
            "group pool indices must be unique and non-negative");
      }
      view.pool_indices.push_back(static_cast<size_t>(pool_idx));
    }
    view.dst_ids.assign(group.dst_device_block_ids().begin(),
                        group.dst_device_block_ids().end());
    view.extents.assign(group.dst_expected_extent_bytes().begin(),
                        group.dst_expected_extent_bytes().end());
    view.expected_pushes = group.expected_pushes();
    groups.push_back(std::move(view));
  }
  if (grouped_pools.size() !=
      static_cast<size_t>(plan.transfer_pool_indices_size())) {
    return absl::InvalidArgumentError(
        "group pool indices do not partition the plan's transfer pools");
  }
  for (int32_t pool_idx : plan.transfer_pool_indices()) {
    if (!grouped_pools.contains(static_cast<size_t>(pool_idx))) {
      return absl::InvalidArgumentError(
          "group pool indices do not partition the plan's transfer pools");
    }
  }

  const int64_t parallelism = plan.parallelism();
  if (parallelism <= 0) {
    return absl::InvalidArgumentError(
        "receiver plans require positive parallelism for push accounting");
  }

  struct GroupState {
    std::vector<kv_cache::PoolLiveSegment> segments;
    int64_t live_bytes = 0;
    absl::flat_hash_map<int64_t, size_t> ordinals;
    std::vector<std::vector<std::pair<int64_t, int64_t>>> coverage;
    absl::flat_hash_map<
        int32_t, absl::flat_hash_set<std::tuple<std::string, int64_t, int64_t>>>
        pairs_by_sender;
  };
  std::vector<GroupState> states(groups.size());
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const GroupView& view = groups[group_idx];
    GroupState& state = states[group_idx];
    if (view.pool_indices.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("group ", group_idx, " declares no pools"));
    }
    for (size_t pool_idx : view.pool_indices) {
      const kv_cache::PoolSpec* spec = pool(pool_idx);
      if (spec == nullptr) {
        return absl::InvalidArgumentError(
            absl::StrCat("group pool index out of range: ", pool_idx));
      }
      absl::StatusOr<std::vector<kv_cache::PoolLiveSegment>> segments =
          kv_cache::ExpandPoolLiveSegments(*spec);
      if (!segments.ok()) return segments.status();
      if (state.segments.empty()) {
        state.segments = *std::move(segments);
      } else if (state.segments != *segments) {
        return absl::InvalidArgumentError(absl::StrCat(
            "group ", group_idx, " pools must share one live-region map; pool ",
            pool_idx, " disagrees"));
      }
    }
    for (const kv_cache::PoolLiveSegment& segment : state.segments) {
      state.live_bytes += segment.size;
    }
    if (state.live_bytes <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("group ", group_idx, " has no live destination bytes"));
    }
    if (view.extents.empty()) {
      return absl::InvalidArgumentError(
          "receiver plans require dst_expected_extent_bytes");
    }
    if (view.extents.size() != view.dst_ids.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "group ", group_idx,
          " extents do not match its destination block count: got ",
          view.extents.size(), ", expected ", view.dst_ids.size()));
    }
    for (size_t ordinal = 0; ordinal < view.extents.size(); ++ordinal) {
      const int64_t extent = view.extents[ordinal];
      if (extent <= 0 || extent > state.live_bytes) {
        return absl::InvalidArgumentError(
            absl::StrCat("extent ", extent, " for destination block ordinal ",
                         ordinal, " of group ", group_idx, " is outside (0, ",
                         state.live_bytes, "]"));
      }
      if (ordinal != view.extents.size() - 1 && extent != state.live_bytes) {
        return absl::InvalidArgumentError(
            "extents must cover every destination block fully except the "
            "final one");
      }
    }
    for (size_t ordinal = 0; ordinal < view.dst_ids.size(); ++ordinal) {
      state.ordinals[view.dst_ids[ordinal]] = ordinal;
    }
    state.coverage.resize(view.dst_ids.size());
  }

  int64_t expanded_repeats = 0;
  for (const auto& [source_rank, schedule] : plan.shard_push_schedules()) {
    for (const auto& entry : schedule.entries()) {
      const size_t group_idx = static_cast<size_t>(entry.pool_group());
      if (group_idx >= states.size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "reshard entry declares an unknown pool group ", group_idx));
      }
      GroupState& state = states[group_idx];
      const auto ordinal_it = state.ordinals.find(entry.dst_block_id());
      if (ordinal_it == state.ordinals.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "reshard entry targets destination block ", entry.dst_block_id(),
            " outside its group ", group_idx, " destination set"));
      }
      const int64_t extent = groups[group_idx].extents[ordinal_it->second];
      expanded_repeats += entry.count();
      if (expanded_repeats > kMaxExpandedRepeats) {
        return absl::InvalidArgumentError(
            "receiver plan exceeds the repeat expansion bound");
      }
      for (int64_t repeat = 0; repeat < entry.count(); ++repeat) {
        const int64_t physical =
            entry.dst_offset_bytes() + repeat * entry.dst_stride_bytes();
        absl::StatusOr<std::pair<int64_t, int64_t>> range =
            kv_cache::PhysicalLiveRangeToLogical(state.segments, physical,
                                                 entry.size_bytes());
        if (!range.ok()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "reshard entry for destination block ", entry.dst_block_id(),
              " crosses padding or lies outside declared live regions: ",
              range.status().message()));
        }
        if (range->second > extent) {
          return absl::InvalidArgumentError(absl::StrCat(
              "reshard entry exceeds destination block ", entry.dst_block_id(),
              " declared live tail: end=", range->second, " extent=", extent));
        }
        state.coverage[ordinal_it->second].push_back(*range);
      }
      state.pairs_by_sender[source_rank].insert(std::make_tuple(
          entry.dst_peer(), static_cast<int64_t>(entry.src_block_id()),
          static_cast<int64_t>(entry.dst_block_id())));
    }
  }

  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const GroupView& view = groups[group_idx];
    GroupState& state = states[group_idx];
    int64_t calculated_pushes = 0;
    for (const auto& [source_rank, pairs] : state.pairs_by_sender) {
      calculated_pushes +=
          std::min(parallelism, static_cast<int64_t>(pairs.size()));
    }
    if (calculated_pushes != view.expected_pushes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "expected pushes for group ", group_idx,
          " do not match the received schedules: declared=",
          view.expected_pushes, " recomputed=", calculated_pushes));
    }
    for (size_t ordinal = 0; ordinal < view.dst_ids.size(); ++ordinal) {
      std::vector<std::pair<int64_t, int64_t>>& intervals =
          state.coverage[ordinal];
      std::sort(intervals.begin(), intervals.end());
      int64_t covered_until = 0;
      for (const auto& [start_bytes, end_bytes] : intervals) {
        if (start_bytes != covered_until) {
          return absl::InvalidArgumentError(absl::StrCat(
              "receiver schedule has a destination coverage ",
              start_bytes < covered_until ? "overlap" : "gap", " for block ",
              view.dst_ids[ordinal], " at byte ", start_bytes));
        }
        covered_until = end_bytes;
      }
      if (covered_until != view.extents[ordinal]) {
        return absl::InvalidArgumentError(absl::StrCat(
            "receiver schedule does not cover the exact live bytes for "
            "destination block ",
            view.dst_ids[ordinal], ": covered=", covered_until,
            " expected=", view.extents[ordinal]));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::PoolReshardPush(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> src_block_ids, int parallelism) {
  TF_RETURN_IF_ERROR(
      ValidatePoolReshardPlan(plan, src_block_ids, /*is_sender=*/true));
  // Device-only executor: without device attachments there are no bytes this
  // path could legitimately move; host-only managers fail closed with no
  // host-mode branch to mask device bugs.
  if (buffer_holds_.empty()) {
    return absl::FailedPreconditionError(
        "pool reshard push requires a device-attached manager; host-only "
        "managers are not supported");
  }
  if (parallelism <= 0) {
    return absl::InvalidArgumentError("parallelism must be positive");
  }

  auto schedule_it = plan.shard_push_schedules().find(0);
  if (schedule_it == plan.shard_push_schedules().end()) {
    if (plan.shard_push_schedules().size() != 1) {
      return absl::InvalidArgumentError(
          "sender plan must use local schedule key 0");
    }
    schedule_it = plan.shard_push_schedules().begin();
  }
  std::set<std::string> peers;
  for (const auto& entry : schedule_it->second.entries()) {
    peers.insert(entry.dst_peer());
  }
  if (peers.empty()) {
    return absl::InvalidArgumentError("sender plan contains no peers");
  }

  InitTransportServer();
  TF_RETURN_IF_ERROR(kv_cache::KVCacheManagerBase::RegisterActivePlan(
      plan.uuid(), plan, /*is_sender=*/true));

  auto state = std::make_shared<PoolReshardSendEntry>();
  state->req_id = plan.req_id();
  state->uuid = plan.uuid();
  state->parallelism = parallelism;
  state->remaining_pool_peer_pushes =
      plan.transfer_pool_indices_size() * peers.size();
  state->plan = plan;
  state->deadline = DeadlineFromNow();
  {
    absl::MutexLock lock(mu_);
    if (active_pool_reshard_sends_.contains(plan.uuid())) {
      (void)kv_cache::KVCacheManagerBase::UnregisterActivePlan(plan.uuid());
      return absl::AlreadyExistsError(
          absl::StrCat("pool reshard send UUID already active: ", plan.uuid()));
    }
    active_pool_reshard_sends_[plan.uuid()] = state;
  }

  // Multi-tag plans scope each pool's staging and pushes to its group's
  // entries; the flat src_block_ids argument is the legacy single-tag
  // whole-plan block list.
  const auto pool_group_index = [&plan](size_t pool_idx) -> int {
    for (int group_idx = 0; group_idx < plan.pool_groups_size(); ++group_idx) {
      const auto& indices = plan.pool_groups(group_idx).pool_indices();
      if (std::find(indices.begin(), indices.end(),
                    static_cast<int32_t>(pool_idx)) != indices.end()) {
        return group_idx;
      }
    }
    return -1;
  };
  auto local_schedule_it = plan.shard_push_schedules().find(0);
  if (local_schedule_it == plan.shard_push_schedules().end() &&
      plan.shard_push_schedules().size() == 1) {
    local_schedule_it = plan.shard_push_schedules().begin();
  }

  for (int32_t encoded_pool_idx : plan.transfer_pool_indices()) {
    const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
    std::vector<int64_t> pool_src_block_ids(src_block_ids.begin(),
                                            src_block_ids.end());
    if (local_schedule_it != plan.shard_push_schedules().end()) {
      const int group_idx = pool_group_index(pool_idx);
      std::set<int64_t> group_src_ids;
      for (const auto& entry : local_schedule_it->second.entries()) {
        if (entry.pool_group() == group_idx) {
          group_src_ids.insert(static_cast<int64_t>(entry.src_block_id()));
        }
      }
      pool_src_block_ids.assign(group_src_ids.begin(), group_src_ids.end());
      if (pool_src_block_ids.empty()) {
        // This sender owns none of the group's bytes (e.g. a PCP rank whose
        // interleave slices all fall past a short prefix): release the pool's
        // per-peer completion slots instead of failing the plan. The
        // receiver's expected pushes count only senders with scheduled pairs.
        for (size_t peer_idx = 0; peer_idx < peers.size(); ++peer_idx) {
          FinishPoolReshardSend(plan.uuid(), absl::OkStatus());
        }
        continue;
      }
    }
    // Bounded host staging: lease one arena slot per source page of this
    // pool's storage for the transfer before staging its bytes (no-op on
    // full-mirror storages). Released in FinishPoolReshardSend.
    if (const kv_cache::PoolSpec* pool_spec = pool(pool_idx);
        pool_spec != nullptr) {
      absl::Status lease_status = AcquirePoolStagingLease(
          plan.uuid(), pool_spec->storage_index, pool_src_block_ids,
          pool_staging_lease_timeout());
      if (!lease_status.ok()) {
        FinishPoolReshardSend(plan.uuid(), lease_status);
        return lease_status;
      }
    }
    auto future_or = D2hPoolBlocks(pool_idx, pool_src_block_ids,
                                   /*shard_idx=*/std::nullopt,
                                   static_cast<uint64_t>(plan.uuid()));
    if (!future_or.ok()) {
      FinishPoolReshardSend(plan.uuid(), future_or.status());
      return future_or.status();
    }
    raiden::PjRtCopyFuture future = std::move(future_or).value();
    state->d2h_futures.push_back(future);
    std::shared_ptr<void> callback_guard = TrackAsyncCallback();
    future.OnReady([this, uuid = static_cast<uint64_t>(plan.uuid()), pool_idx,
                    callback_guard](auto status_or) mutable {
      std::shared_ptr<void> callback_lifetime = std::move(callback_guard);
      (void)callback_lifetime;
      if (!status_or.ok()) {
        FinishPoolReshardSend(uuid, status_or.status());
        return;
      }
      if (stopping_.load(std::memory_order_acquire)) {
        FinishPoolReshardSend(
            uuid, absl::CancelledError("manager stopped before pool push"));
        return;
      }
      StartPoolReshardPush(uuid, pool_idx);
    });
  }
  return absl::OkStatus();
}

void KVCacheManagerWithTransfer::StartPoolReshardPush(uint64_t uuid,
                                                      size_t pool_idx) {
  std::shared_ptr<PoolReshardSendEntry> state;
  {
    absl::MutexLock lock(mu_);
    auto it = active_pool_reshard_sends_.find(uuid);
    if (it == active_pool_reshard_sends_.end()) return;
    state = it->second;
  }

  auto schedule_it = state->plan.shard_push_schedules().find(0);
  if (schedule_it == state->plan.shard_push_schedules().end()) {
    schedule_it = state->plan.shard_push_schedules().begin();
  }
  // A pool pushes only its own group's (src, dst) pairs.
  int pool_group_idx = -1;
  for (int group_idx = 0; group_idx < state->plan.pool_groups_size();
       ++group_idx) {
    const auto& indices = state->plan.pool_groups(group_idx).pool_indices();
    if (std::find(indices.begin(), indices.end(),
                  static_cast<int32_t>(pool_idx)) != indices.end()) {
      pool_group_idx = group_idx;
      break;
    }
  }
  std::map<std::string, std::vector<std::pair<int, int>>> transfers_by_peer;
  std::map<std::string, std::set<std::pair<int, int>>> seen_by_peer;
  for (const auto& entry : schedule_it->second.entries()) {
    if (entry.pool_group() != pool_group_idx) {
      continue;
    }
    const std::pair<int, int> pair{static_cast<int>(entry.src_block_id()),
                                   static_cast<int>(entry.dst_block_id())};
    if (seen_by_peer[entry.dst_peer()].insert(pair).second) {
      transfers_by_peer[entry.dst_peer()].push_back(pair);
    }
  }

  std::shared_ptr<transport::BlockTransport> transport_server =
      GetTransportServer();
  if (transport_server == nullptr) {
    FinishPoolReshardSend(
        uuid, absl::FailedPreconditionError("transport server is not running"));
    return;
  }

  for (const auto& [peer, transfers] : transfers_by_peer) {
    std::vector<int> src_ids;
    std::vector<int> dst_ids;
    src_ids.reserve(transfers.size());
    dst_ids.reserve(transfers.size());
    for (const auto& [src_id, dst_id] : transfers) {
      src_ids.push_back(src_id);
      dst_ids.push_back(dst_id);
    }
    std::shared_ptr<void> callback_guard = TrackAsyncCallback();
    transport_server->AsyncPush(
        {peer}, src_ids, dst_ids, state->parallelism,
        transport::MajorOrder::kLayerMajor, uuid, static_cast<int>(pool_idx),
        [this, uuid,
         callback_guard](absl::StatusOr<std::vector<int>> result) mutable {
          std::shared_ptr<void> callback_lifetime = std::move(callback_guard);
          (void)callback_lifetime;
          FinishPoolReshardSend(
              uuid, result.ok() ? absl::OkStatus() : result.status());
        });
  }
}

void KVCacheManagerWithTransfer::FinishPoolReshardSend(
    uint64_t uuid, const absl::Status& status) {
  bool finished = false;
  {
    absl::MutexLock lock(mu_);
    auto it = active_pool_reshard_sends_.find(uuid);
    if (it == active_pool_reshard_sends_.end()) return;
    auto& state = *it->second;
    if (state.finalizing) return;
    if (!status.ok()) {
      LOG(ERROR) << "Pool reshard send failed uuid=" << uuid
                 << " req_id=" << state.req_id << ": " << status;
      state.failed = true;
      state.finalizing = true;
      finished = true;
    } else if (--state.remaining_pool_peer_pushes == 0) {
      state.finalizing = true;
      finished = true;
    }
  }
  if (finished) {
    absl::Status unregister = UnregisterActivePlan(uuid);
    if (!unregister.ok() && !absl::IsNotFound(unregister)) {
      LOG(ERROR) << "Failed to unregister pool reshard sender plan " << uuid
                 << ": " << unregister;
    }
    // Every push of every pool has completed (or the send failed): the host
    // staging bytes are no longer read, so the arena slots go back.
    ReleasePoolStagingLeases(uuid);
    absl::MutexLock lock(mu_);
    auto it = active_pool_reshard_sends_.find(uuid);
    if (it == active_pool_reshard_sends_.end()) return;
    if (it->second->failed ||
        (!unregister.ok() && !absl::IsNotFound(unregister))) {
      failed_recving_.insert(it->second->req_id);
    } else {
      done_sending_.insert(it->second->req_id);
    }
    active_pool_reshard_sends_.erase(it);
  }
}

absl::Status KVCacheManagerWithTransfer::PoolReshardRegisterRecv(
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> chip_block_ids) {
  TF_RETURN_IF_ERROR(
      ValidatePoolReshardPlan(plan, chip_block_ids, /*is_sender=*/false));
  // Device-only executor (see PoolReshardPush): arming a receive on a
  // host-only manager is refused rather than silently landing in mirrors.
  if (buffer_holds_.empty()) {
    return absl::FailedPreconditionError(
        "pool reshard receive requires a device-attached manager; host-only "
        "managers are not supported");
  }
  if (plan.dst_mem_type() != ::tpu_sync::rpc::MEMORY_TYPE_HBM) {
    return absl::InvalidArgumentError(
        "pool reshard receiver requires dst_mem_type=HBM");
  }
  {
    absl::MutexLock lock(mu_);
    if (active_recv_entries_.contains(plan.uuid())) {
      return absl::AlreadyExistsError(
          absl::StrCat("pool reshard recv UUID already active: ", plan.uuid()));
    }
  }

  // Bounded host staging: the wire still lands at device (chip) block ids,
  // but on a bounded storage those ids are remapped to arena slots leased to
  // this uuid. Lease the union of every pool's destination ids per storage
  // before arming; a failure here refuses the arm cleanly (the coordinator
  // abandons the claim and no sender is dispatched). Full-mirror storages are
  // no-ops. Released in FinishPoolReshardRecvPool / the deadline sweep.
  {
    std::map<size_t, std::set<int64_t>> dst_ids_by_storage;
    std::map<size_t, std::vector<int64_t>> group_dst_by_pool;
    for (const auto& group : plan.pool_groups()) {
      std::vector<int64_t> group_dst_ids(group.dst_device_block_ids().begin(),
                                         group.dst_device_block_ids().end());
      for (int32_t pool_idx : group.pool_indices()) {
        group_dst_by_pool[static_cast<size_t>(pool_idx)] = group_dst_ids;
      }
    }
    for (int32_t encoded_pool_idx : plan.transfer_pool_indices()) {
      const size_t pool_idx = static_cast<size_t>(encoded_pool_idx);
      const kv_cache::PoolSpec* pool_spec = pool(pool_idx);
      if (pool_spec == nullptr ||
          !PoolStorageStagingBounded(pool_spec->storage_index)) {
        continue;
      }
      auto ids_it = group_dst_by_pool.find(pool_idx);
      const std::vector<int64_t> fallback(chip_block_ids.begin(),
                                          chip_block_ids.end());
      const std::vector<int64_t>& ids =
          ids_it == group_dst_by_pool.end() ? fallback : ids_it->second;
      dst_ids_by_storage[pool_spec->storage_index].insert(ids.begin(),
                                                          ids.end());
    }
    for (const auto& [storage_idx, ids] : dst_ids_by_storage) {
      std::vector<int64_t> id_list(ids.begin(), ids.end());
      absl::Status lease_status = AcquirePoolStagingLease(
          plan.uuid(), storage_idx, id_list, pool_staging_lease_timeout());
      if (!lease_status.ok()) {
        ReleasePoolStagingLeases(plan.uuid());
        return lease_status;
      }
    }
  }

  absl::Status register_status =
      kv_cache::KVCacheManagerBase::RegisterActivePlan(plan.uuid(), plan,
                                                       /*is_sender=*/false);
  if (!register_status.ok()) {
    ReleasePoolStagingLeases(plan.uuid());
    return register_status;
  }
  RecvEntry recv_entry;
  recv_entry.req_id = plan.req_id();
  recv_entry.is_pool_reshard = true;
  recv_entry.deadline = DeadlineFromNow();
  recv_entry.start_time = std::chrono::steady_clock::now();
  recv_entry.phase = RecvEntry::Phase::kTransferring;
  recv_entry.chip_block_ids.assign(chip_block_ids.begin(),
                                   chip_block_ids.end());
  // The wire lands at chip block ids (full mirrors address them directly;
  // bounded storages through the lease acquired above).
  for (int32_t pool_idx : plan.transfer_pool_indices()) {
    recv_entry.expected_pool_indices.insert(static_cast<size_t>(pool_idx));
    recv_entry.pool_order_ranks[static_cast<size_t>(pool_idx)] = 0;
  }
  for (const auto& group : plan.pool_groups()) {
    std::vector<int64_t> group_dst_ids(group.dst_device_block_ids().begin(),
                                       group.dst_device_block_ids().end());
    for (int32_t pool_idx : group.pool_indices()) {
      recv_entry.pool_order_ranks[static_cast<size_t>(pool_idx)] =
          group.order_rank();
      recv_entry.pool_dst_block_ids[static_cast<size_t>(pool_idx)] =
          group_dst_ids;
    }
  }
  {
    absl::MutexLock lock(mu_);
    active_recv_entries_[plan.uuid()] = std::move(recv_entry);
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::UnregisterActivePlan(uint64_t uuid) {
  absl::MutexLock lifecycle(plan_lifecycle_mu_);
  bool deferred = false;
  {
    absl::MutexLock lock(mu_);
    auto it = plan_staging_.find(uuid);
    if (it != plan_staging_.end()) {
      (void)host_block_manager_->Unlock(it->second);
      (void)host_block_manager_->Deallocate(it->second);
      plan_staging_.erase(it);
    }
    // A receive still in flight keeps its plan: pushes the transport has
    // already accepted must keep resolving into the plan's staging blocks.
    // The plan is dropped when the receive completes, fails, or times out.
    auto recv = active_recv_entries_.find(uuid);
    if (recv != active_recv_entries_.end() && !recv->second.is_pool_reshard) {
      recv->second.unregister_on_settle = true;
      deferred = true;
    }
  }
  if (deferred) return absl::OkStatus();
  return kv_cache::KVCacheManagerBase::UnregisterActivePlan(uuid);
}

void KVCacheManagerWithTransfer::UnregisterSettledPlan(uint64_t uuid,
                                                       uint64_t generation) {
  // Serialized with registration so cleanup for one registration can never
  // remove a newer one reusing the uuid, or race its progress counters.
  absl::MutexLock lifecycle(plan_lifecycle_mu_);
  if (generation != 0) {
    std::optional<uint64_t> current = ActivePlanGeneration(uuid);
    if (!current.has_value() || *current != generation) {
      return;  // the plan is gone, or the uuid already belongs to a newer one
    }
  }
  absl::Status status =
      kv_cache::KVCacheManagerBase::UnregisterActivePlan(uuid);
  if (!status.ok() && !absl::IsNotFound(status)) {
    LOG(ERROR) << "Failed to unregister settled transfer plan " << uuid << ": "
               << status;
  }
}

std::vector<RaidenTransferEndpoint>
KVCacheManagerWithTransfer::get_local_endpoints() const {
  // NOTE: prefers the CONTROL port when a control server is running, because
  // StartRead speaks the control protocol. Callers that need the block-
  // transport data protocol (H2hRead/H2dRead pulls, H2hWrite pushes, and
  // therefore worker registration) must use get_local_data_endpoints()
  // instead: aiming a data-protocol connection at the control port hangs both
  // sides with no error, since each waits for the other's framing.
  return BuildEndpoints(local_control_port_ > 0 ? local_control_port_
                                                : local_port().value_or(0));
}

std::vector<RaidenTransferEndpoint>
KVCacheManagerWithTransfer::get_local_data_endpoints() const {
  return BuildEndpoints(local_port().value_or(0));
}

std::vector<RaidenTransferEndpoint> KVCacheManagerWithTransfer::BuildEndpoints(
    int64_t port) const {
  std::vector<int64_t> all_shards(num_shards_);
  for (size_t i = 0; i < num_shards_; ++i) {
    all_shards[i] = static_cast<int64_t>(i);
  }
  std::vector<RaidenTransferEndpoint> eps;
  for (const auto& ip : local_ips()) {
    std::string endpoint = absl::StrContains(ip, ':')
                               ? absl::StrCat("[", ip, "]:", port)
                               : absl::StrCat(ip, ":", port);
    eps.push_back({endpoint, all_shards});
  }
  return eps;
}

bool KVCacheManagerWithTransfer::EncodeIpToIpv6Bytes(const std::string& ip,
                                                     uint8_t out[16]) {
  // An IPv4 address must be sent as IPv4-mapped IPv6 ("::ffff:a.b.c.d") --
  // inet_pton(AF_INET6, "<ipv4>") fails on a bare IPv4 string. If it still
  // fails to parse, zero the field.
  const std::string mapped = absl::StrContains(ip, ':') ? ip : "::ffff:" + ip;
  if (inet_pton(AF_INET6, mapped.c_str(), out) <= 0) {
    std::memset(out, 0, 16);
    return false;
  }
  return true;
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::vector<std::string>& remote_endpoints,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  std::string target_ep;
  if (!remote_endpoints.empty()) {
    target_ep = remote_endpoints[0];
  }
  StartRead(req_id, uuid, target_ep, remote_block_ids, local_block_ids,
            parallelism, local_host_block_ids);
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::vector<RaidenTransferEndpoint>& remote_descriptors,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  if (remote_descriptors.empty()) {
    return;
  }
  // TODO: Deal with the case where the shards on both sides don't
  // perfectly match. KVCacheManagerWithTransfer is bound to a single NUMA node
  // / single endpoint. Multi-endpoint routing across sockets is orchestrated by
  // the JAX facade.
  if (remote_descriptors.size() != 1) {
    VLOG(1) << "KVCacheManagerWithTransfer::StartRead received "
            << remote_descriptors.size()
            << " descriptors, selecting first endpoint.";
  }
  StartRead(req_id, uuid, remote_descriptors[0].endpoint, remote_block_ids,
            local_block_ids, parallelism, local_host_block_ids);
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::string& remote_endpoint,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  LOG(INFO) << "StartRead (initiate): req_id=" << req_id << ", uuid=" << uuid
            << ", numa=" << assigned_numa_node().value_or(-1);
  VLOG(1) << "KVCacheManagerWithTransfer::StartRead (Hybrid Bridge) called. "
             "req_id: "
          << req_id << ", uuid: " << uuid << ", remote: " << remote_endpoint
          << ", Thread: " << std::this_thread::get_id();
  // local_block_ids index the consumer's DEVICE KV cache, not the host staging
  // pool; reusing them as host indices overflows the host buffer once a device
  // block id exceeds num_host_blocks. If the caller didn't supply explicit host
  // indices, borrow a staging slot and stage into its reserved host blocks
  // (slot.block_ids -- the real, possibly non-contiguous host blocks).
  std::vector<int64_t> host_block_ids;
  int64_t recv_slot = -1;
  std::vector<int> staged_host_blocks;
  if (local_host_block_ids.has_value()) {
    host_block_ids = *local_host_block_ids;
  } else if (!local_block_ids.empty()) {
    absl::MutexLock lock(mu_);
    absl::flat_hash_set<int64_t> unique_local_bids(local_block_ids.begin(),
                                                   local_block_ids.end());
    RecvEntry staging;
    auto staged = AcquireRecvStagingLocked(
        static_cast<int64_t>(unique_local_bids.size()), &staging);
    if (!staged.has_value()) {
      // Request larger than the staging pool can seat: surface as a recv
      // failure (the connector can recompute) rather than throwing.
      LOG(ERROR) << "StartRead: cannot stage " << unique_local_bids.size()
                 << " blocks for req_id=" << req_id
                 << " (dynamic=" << dynamic_host_staging_
                 << ", free_host_blocks="
                 << host_block_manager_->num_free_blocks()
                 << ", free_slots=" << free_slots_.size()
                 << ", max_blocks=" << max_blocks_ << ")";
      failed_recving_.insert(req_id);
      return;
    }
    recv_slot = staging.slot_idx;
    staged_host_blocks = std::move(staging.staged_host_blocks);
    absl::flat_hash_map<kv_cache::DeviceBlockId, kv_cache::HostBlockId>
        local_to_host;
    size_t host_block_idx = 0;
    host_block_ids.reserve(local_block_ids.size());
    for (size_t k = 0; k < local_block_ids.size(); ++k) {
      int64_t local_bid = local_block_ids[k];
      auto it = local_to_host.find(local_bid);
      if (it == local_to_host.end()) {
        int64_t host_bid = (*staged)[host_block_idx++];
        local_to_host[local_bid] = host_bid;
        host_block_ids.push_back(host_bid);
      } else {
        host_block_ids.push_back(it->second);
      }
    }
  }
  CopyPlan load_plan =
      BuildLoadCopyPlan(remote_block_ids, local_block_ids, host_block_ids);

  const auto registration_deadline = DeadlineFromNow();
  {
    absl::MutexLock lock(mu_);
    if (active_recv_entries_.find(uuid) != active_recv_entries_.end()) {
      failed_recving_.insert(req_id);
      ReleaseStagingLocked(recv_slot, &staged_host_blocks);
      return;
    }
    RecvEntry entry;
    entry.req_id = req_id;
    entry.slot_idx = recv_slot;
    entry.staged_host_blocks = std::move(staged_host_blocks);
    entry.deadline = registration_deadline;
    entry.start_time = std::chrono::steady_clock::now();
    entry.chip_block_ids = load_plan.h2d_local_block_ids;
    entry.total_blocks = load_plan.num_blocks;
    entry.num_completed_blocks = 0;
    entry.num_completed_layers = 0;
    // Read the H2D source from the actual staged host blocks (coalesced), not a
    // compact 0..n-1 region -- the producer writes into host_block_ids, so the
    // consumer must read back from the same blocks.
    entry.h2d_copy = load_plan.h2d_copy;
    for (size_t i = 0; i < load_plan.transport_host_block_ids.size(); ++i) {
      entry.host_to_chip[load_plan.transport_host_block_ids[i]] =
          load_plan.h2d_local_block_ids[i];
    }
    entry.h2d_dispatch_futures.reserve(load_plan.h2d_local_block_ids.size());
    active_recv_entries_.emplace(uuid, std::move(entry));
  }

  if (metrics_collector_) {
    uint64_t total_bytes = static_cast<uint64_t>(load_plan.num_blocks) *
                           num_layers() * num_shards_ * slice_byte_size_;
    metrics_collector_->RecordStart(uuid, req_id, load_plan.num_blocks,
                                    total_bytes);
  }

  if (load_plan.num_blocks == 0) {
    absl::MutexLock lock(mu_);
    done_recving_.insert(req_id);
    auto empty_it = active_recv_entries_.find(uuid);
    if (empty_it != active_recv_entries_.end()) {
      ReleaseRecvStagingLocked(&empty_it->second);
      active_recv_entries_.erase(empty_it);
    }
    return;
  }

  const std::optional<int> target_node = assigned_numa_node();
  NumaThreadPool* const registration_pool = push_pool_.get();
  const auto registration_start = std::chrono::steady_clock::now();
  auto shared_load_plan = std::make_shared<CopyPlan>(std::move(load_plan));
  auto attempts = std::make_shared<size_t>(0);
  auto retryable_unknowns = std::make_shared<size_t>(0);

  // A dedicated manager-owned timer paces retries, then submits each attempt
  // as a separate FIFO pool task. No push/pull worker sleeps while the
  // producer is still finishing prefill.
  auto registration_attempt = std::make_shared<std::function<void()>>();
  std::weak_ptr<std::function<void()>> weak_registration_attempt =
      registration_attempt;
  *registration_attempt = [this, req_id, uuid, remote_endpoint,
                           registration_deadline, registration_start,
                           target_node, shared_load_plan, attempts,
                           retryable_unknowns, weak_registration_attempt]() {
    auto fail_receive = [this, req_id, uuid]() {
      absl::MutexLock lock(mu_);
      auto it = active_recv_entries_.find(uuid);
      if (it == active_recv_entries_.end() || it->second.req_id != req_id) {
        return;
      }
      failed_recving_.insert(req_id);
      ReleaseRecvStagingLocked(&it->second);
      active_recv_entries_.erase(it);
    };

    try {
      const auto now = std::chrono::steady_clock::now();
      if (stopping_) {
        throw std::runtime_error(
            "manager stopped while registering remote KV pull");
      }
      if (now >= registration_deadline) {
        throw std::runtime_error(
            "timed out retrying remote KV pull registration");
      }
      {
        absl::MutexLock lock(mu_);
        auto it = active_recv_entries_.find(uuid);
        if (it == active_recv_entries_.end() || it->second.req_id != req_id ||
            it->second.phase != RecvEntry::Phase::kRegistering) {
          return;
        }
      }

      ++*attempts;
      const auto attempt_deadline =
          std::min(registration_deadline, now + control_io_timeout_);
      try {
        LOG(INFO) << "StartRead (connecting): req_id=" << req_id
                  << ", uuid=" << uuid << ", attempt=" << *attempts
                  << ", numa=" << assigned_numa_node().value_or(-1);
        int control_fd = ConnectTcp(remote_endpoint, attempt_deadline);
        auto control_cleanup =
            std::unique_ptr<int, void (*)(int*)>(&control_fd, [](int* p) {
              if (p && *p >= 0) close(*p);
            });
        ConfigureSocketIoTimeout(control_fd, control_io_timeout_);

        ControlRequestHeader stream_request;
        stream_request.magic = kControlMagic;
        stream_request.op = kOpPullStream;
        stream_request.uuid = uuid;
        stream_request.ep_idx = 0;
        stream_request.num_blocks =
            static_cast<uint64_t>(shared_load_plan->num_blocks);
        stream_request.consumer_data_port =
            static_cast<uint32_t>(local_data_port_);

        std::vector<std::string> ips = local_ips();
        stream_request.num_ips =
            std::min(ips.size(), static_cast<size_t>(kMaxNics));
        for (size_t i = 0; i < stream_request.num_ips; ++i) {
          if (!EncodeIp(ips[i], stream_request.consumer_ips[i])) {
            std::memset(stream_request.consumer_ips[i], 0, 16);
          }
        }
        CheckStatus("control pull stream write",
                    WriteExactUntil(control_fd, &stream_request,
                                    sizeof(stream_request), attempt_deadline));
        WriteBlockIds(control_fd, shared_load_plan->producer_remote_block_ids,
                      attempt_deadline);
        WriteBlockIds(control_fd, shared_load_plan->transport_host_block_ids,
                      attempt_deadline);
        (void)ReadControlResponseHeader(control_fd, attempt_deadline);
      } catch (const RemoteControlError& e) {
        if (e.remote_status() != kControlRetryableUnknown) throw;
        ++*retryable_unknowns;
        if (std::chrono::steady_clock::now() >= registration_deadline) {
          throw std::runtime_error(
              "timed out waiting for remote send UUID registration");
        }
        auto next_attempt = weak_registration_attempt.lock();
        if (!next_attempt) {
          throw std::runtime_error(
              "remote KV registration retry task was released");
        }
        const auto retry_now = std::chrono::steady_clock::now();
        const auto retry_delay =
            ComputeRegistrationRetryDelay(uuid, *retryable_unknowns);
        const auto retry_due =
            std::min(registration_deadline, retry_now + retry_delay);
        ScheduleRegistrationRetry(retry_due, target_node,
                                  [next_attempt]() { (*next_attempt)(); });
        return;
      }

      {
        absl::MutexLock lock(mu_);
        auto it = active_recv_entries_.find(uuid);
        if (it == active_recv_entries_.end() || it->second.req_id != req_id ||
            it->second.phase != RecvEntry::Phase::kRegistering) {
          return;
        }
        it->second.phase = RecvEntry::Phase::kTransferring;
        it->second.deadline = DeadlineFromNow();
      }

      std::ostringstream timing;
      timing << "RAIDEN_TIMING event=consumer_pull_registered"
             << " req_id=" << req_id << " uuid=" << uuid
             << " node_id=" << node_id_
             << " blocks=" << shared_load_plan->num_blocks
             << " attempts=" << *attempts
             << " registration_retries=" << *retryable_unknowns
             << " registration_wait_ms="
             << DurationMs(registration_start, std::chrono::steady_clock::now())
             << " failed=0";
      EmitTimingLog(timing.str());
    } catch (const std::exception& e) {
      LOG(ERROR) << "Raiden consumer error during Hybrid Bridge StartRead "
                    "connect: "
                 << e.what() << " req_id=" << req_id << " uuid=" << uuid
                 << " attempts=" << *attempts
                 << " registration_retries=" << *retryable_unknowns;
      fail_receive();
    }
  };

  try {
    registration_pool->Schedule(
        target_node, [registration_attempt]() { (*registration_attempt)(); });
  } catch (...) {
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it != active_recv_entries_.end() && it->second.req_id == req_id) {
      failed_recving_.insert(req_id);
      ReleaseRecvStagingLocked(&it->second);
      active_recv_entries_.erase(it);
    }
    throw;
  }
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheManagerWithTransfer::CompleteReadRaw() {
  std::vector<std::string> done_sending;
  std::vector<std::string> done_recving;
  std::vector<std::string> failed_recving;
  std::vector<std::pair<uint64_t, uint64_t>> settled_plans;
  {
    absl::MutexLock lock(mu_);
    const auto now = std::chrono::steady_clock::now();
    for (auto it = send_entries_.begin(); it != send_entries_.end();) {
      const uint64_t uuid = it->first;
      const std::shared_ptr<SendEntry> entry = it->second;
      const bool waiting_expired =
          entry->phase == SendEntry::Phase::kWaitingForPull &&
          entry->deadline <= now;
      const bool transfer_expired =
          entry->phase == SendEntry::Phase::kTransferring &&
          entry->transfer_deadline <= now;
      ++it;
      if (waiting_expired || transfer_expired) {
        if (transfer_expired) entry->failed = true;
        settled_plans.emplace_back(uuid, 0);
        (void)TerminalizeSendEntryLocked(
            uuid, entry,
            waiting_expired ? "send expired while waiting for pull"
                            : "send transfer exceeded its deadline",
            /*waiting_only=*/waiting_expired);
      }
    }
    for (auto it = active_pool_reshard_sends_.begin();
         it != active_pool_reshard_sends_.end();) {
      const auto& entry = it->second;
      if (entry->deadline <= now) {
        failed_recving_.insert(entry->req_id);
        settled_plans.emplace_back(it->first, 0);
        auto erase_it = it++;
        active_pool_reshard_sends_.erase(erase_it);
      } else {
        ++it;
      }
    }
    // Reclaim recv entries whose transfer never completed (e.g. the producer
    // died or never finished pushing). Without this the entry and its host
    // staging slot leak forever, eventually exhausting the slot pool. Surface
    // the timeout as a recv failure so the connector can recompute the blocks.
    for (auto it = active_recv_entries_.begin();
         it != active_recv_entries_.end();) {
      auto& entry = it->second;
      if (entry.phase == RecvEntry::Phase::kTransferring &&
          (entry.network_completed ||
           entry.num_completed_layers == num_layers())) {
        bool all_h2d_done = true;
        for (auto& f : entry.h2d_futures) {
          if (!f.IsReady()) {
            all_h2d_done = false;
            break;
          }
        }
        if (all_h2d_done) {
          LOG(INFO) << "CompleteReadRaw (polling completion): req_id="
                    << entry.req_id;
          done_recving_.insert(entry.req_id);
          ReleaseRecvStagingLocked(&entry);
          if (entry.unregister_on_settle) {
            settled_plans.emplace_back(it->first, entry.plan_generation);
          }
          active_recv_entries_.erase(it++);
          continue;
        }
      }

      if (entry.phase == RecvEntry::Phase::kTransferring &&
          entry.deadline <= now) {
        failed_recving_.insert(entry.req_id);
        ReleaseRecvStagingLocked(&entry);
        settled_plans.emplace_back(it->first, entry.plan_generation);
        active_recv_entries_.erase(it++);
      } else {
        ++it;
      }
    }
    done_sending.assign(done_sending_.begin(), done_sending_.end());
    done_recving.assign(done_recving_.begin(), done_recving_.end());
    failed_recving.assign(failed_recving_.begin(), failed_recving_.end());
    done_sending_.clear();
    done_recving_.clear();
    failed_recving_.clear();
  }
  // Unregistering drops the plan and its transport receive-progress counters
  // (ForgetPushProgress), so a settled uuid is reusable.
  for (const auto& [uuid, generation] : settled_plans) {
    UnregisterSettledPlan(uuid, generation);
    // A settled (completed or timed-out) pool-reshard sender/receiver may
    // still hold bounded-staging arena slots.
    ReleasePoolStagingLeases(uuid);
  }
  return {done_sending, done_recving, failed_recving};
}

StageResult KVCacheManagerWithTransfer::IssueH2D(
    int64_t slot_idx, int64_t num_blocks,
    const std::vector<int64_t>& local_block_ids) {
  if (num_layers() == 0) {
    throw std::runtime_error("KV cache manager is not registered");
  }
  if (slot_idx < 0 || slot_idx >= num_slots_) {
    throw std::out_of_range("slot_idx out of range");
  }
  if (num_blocks < 0 || num_blocks > max_blocks_) {
    throw std::out_of_range("num_blocks out of range");
  }
  if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
    throw std::invalid_argument("num_blocks must match len(local_block_ids)");
  }

  // Get the actual host block IDs for the first num_blocks in the slot
  const Slot& slot = all_slots_[slot_idx];
  std::vector<int64_t> host_block_ids;
  host_block_ids.reserve(num_blocks);
  for (int64_t i = 0; i < num_blocks; ++i) {
    host_block_ids.push_back(slot.block_ids[i]);
  }

  // Coalesce contiguous (host, device) block runs
  CopySpec copy_spec = BuildCoalescedCopySpec(host_block_ids, local_block_ids);
  kv_cache::KVCacheCopySpec transfer_spec = ToKVCacheCopySpec(copy_spec);

  // We still calculate host_spans for the result, but we don't use slot_idx
  // in H2d call to avoid slot-based double offsetting in the base class.
  std::vector<kv_cache::KVCacheHostSpan> host_spans =
      LayerSpans(slot_idx, num_blocks);

  auto future = std::make_shared<TransferFuture>();
  int64_t total_bytes = 0;
  for (const kv_cache::KVCacheHostSpan& span : host_spans) {
    total_bytes += static_cast<int64_t>(span.nbytes);
  }

  // Call H2dSyncDispatch with slot_idx = std::nullopt to use actual host block
  // IDs
  auto fut_or = H2dSyncDispatch(transfer_spec.src_offsets,
                                transfer_spec.dst_offsets, transfer_spec.sizes,
                                /*slot_idx=*/std::nullopt);
  if (!fut_or.ok()) {
    throw std::runtime_error("Failed to issue H2D transfer: " +
                             std::string(fut_or.status().message()));
  }
  future->Add(std::move(fut_or.value()));

  return {.future = std::move(future),
          .host_spans = std::move(host_spans),
          .total_bytes = total_bytes,
          .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
}

std::vector<kv_cache::KVCacheHostSpan> KVCacheManagerWithTransfer::LayerSpans(
    int64_t slot_idx, int64_t num_blocks) {
  if (num_layers() == 0) {
    throw std::runtime_error("KV cache manager is not registered");
  }
  if (num_blocks < 0 || num_blocks > max_blocks_) {
    throw std::out_of_range("num_blocks out of range");
  }
  std::vector<kv_cache::KVCacheHostSpan> spans;
  const Slot& slot = all_slots_[slot_idx];

  // Coalesce contiguous runs of block IDs in the slot
  struct Run {
    int64_t start_block_id;
    int64_t size;
  };
  std::vector<Run> runs;
  for (int64_t start = 0; start < num_blocks;) {
    int64_t end = start + 1;
    while (end < num_blocks &&
           slot.block_ids[end] == slot.block_ids[end - 1] + 1) {
      ++end;
    }
    runs.push_back({slot.block_ids[start], end - start});
    start = end;
  }

  spans.reserve(num_layers() * num_shards() * runs.size());
  for (size_t layer_idx = 0; layer_idx < num_layers(); ++layer_idx) {
    const int64_t per_layer = LayerBlockByteSize(layer_idx);
    const size_t layer_bytes =
        per_layer > 0 ? static_cast<size_t>(per_layer) : slice_byte_size_;
    for (size_t shard_idx = 0; shard_idx < num_shards(); ++shard_idx) {
      const auto& shard_info = layers_[layer_idx].shards[shard_idx];
      for (const auto& run : runs) {
        const size_t byte_offset =
            static_cast<size_t>(run.start_block_id) * layer_bytes;
        const size_t nbytes = static_cast<size_t>(run.size) * layer_bytes;
        spans.push_back(kv_cache::KVCacheHostSpan{
            .ptr = const_cast<uint8_t*>(shard_info.host_ptr) + byte_offset,
            .nbytes = nbytes,
            .slot_idx = slot_idx,
            .base_major = run.start_block_id,
            .num_major = run.size,
            .layer_idx = layer_idx,
            .shard_idx = shard_idx});
      }
    }
  }
  return spans;
}

absl::Status KVCacheManagerWithTransfer::InitializeSlotPool(int64_t num_slots) {
  if (host_block_manager_->num_free_blocks() < num_slots * max_blocks_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Insufficient free host blocks to initialize slot pool. Required: ",
        num_slots * max_blocks_,
        ", Available: ", host_block_manager_->num_free_blocks()));
  }
  free_slots_.clear();
  all_slots_.clear();
  all_slots_.reserve(num_slots);
  for (int64_t i = 0; i < num_slots; ++i) {
    ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                     host_block_manager_->Allocate(max_blocks_,
                                                   /*lock=*/true));
    if (allocated_ids.size() != max_blocks_) {
      return absl::InternalError(absl::StrCat(
          "Slot pool allocation returned incorrect number of blocks: ",
          allocated_ids.size(), ", expected: ", max_blocks_));
    }
    Slot slot{/*slot_idx=*/i, /*block_ids=*/allocated_ids};
    all_slots_.push_back(slot);
    free_slots_.push_back(slot);
  }
  return absl::OkStatus();
}

bool KVCacheManagerWithTransfer::DynamicHostStagingEnabled() {
  const char* raw = std::getenv("TPU_RAIDEN_DYNAMIC_HOST_STAGING");
  return raw != nullptr && std::string(raw) == "1";
}

std::optional<std::vector<int64_t>>
KVCacheManagerWithTransfer::AcquireRecvStagingLocked(int64_t num_blocks,
                                                     RecvEntry* entry) {
  if (num_blocks <= 0) return std::vector<int64_t>();
  if (!dynamic_host_staging_) {
    if (num_blocks > max_blocks_ || free_slots_.empty()) return std::nullopt;
    Slot slot = AcquireSlotLocked();
    entry->slot_idx = slot.slot_idx;
    std::vector<int64_t> blocks;
    blocks.reserve(num_blocks);
    for (int64_t i = 0; i < num_blocks; ++i) {
      blocks.push_back(slot.block_ids[i]);
    }
    return blocks;
  }
  // Lock the pages so the pool's LRU cannot evict staging that is mid-flight.
  // An allocation miss is only reported back; this helper neither retries
  // nor logs. The producer retries it until its deadline, the consumer
  // fails it at once, and each logs its own verdict.
  auto allocated = host_block_manager_->Allocate(static_cast<int>(num_blocks),
                                                 /*lock=*/true);
  if (!allocated.ok()) {
    return std::nullopt;
  }
  entry->staged_host_blocks = *allocated;
  std::vector<int64_t> blocks;
  blocks.reserve(allocated->size());
  for (int id : *allocated) blocks.push_back(static_cast<int64_t>(id));
  return blocks;
}

void KVCacheManagerWithTransfer::ReleaseRecvStagingLocked(RecvEntry* entry) {
  if (entry == nullptr) return;
  ReleaseStagingLocked(entry->slot_idx, &entry->staged_host_blocks);
}

void KVCacheManagerWithTransfer::ReleaseStagingLocked(
    int64_t slot_idx, std::vector<int>* staged_host_blocks) {
  if (staged_host_blocks != nullptr && !staged_host_blocks->empty()) {
    (void)host_block_manager_->Unlock(*staged_host_blocks);
    (void)host_block_manager_->Deallocate(*staged_host_blocks);
    staged_host_blocks->clear();
    return;
  }
  ReleaseSlotLocked(slot_idx);
}

KVCacheManagerWithTransfer::Slot KVCacheManagerWithTransfer::AcquireSlot() {
  absl::MutexLock lock(mu_);
  return AcquireSlotLocked();
}

KVCacheManagerWithTransfer::Slot
KVCacheManagerWithTransfer::AcquireSlotLocked() {
  if (free_slots_.empty()) {
    throw std::runtime_error("Raiden host slot pool exhausted");
  }
  Slot slot = free_slots_.front();
  free_slots_.pop_front();
  return slot;
}

void KVCacheManagerWithTransfer::ReleaseSlotLocked(int64_t slot_idx) {
  if (slot_idx < 0 || slot_idx >= num_slots_) {
    return;
  }
  free_slots_.push_back(all_slots_[slot_idx]);
}

void KVCacheManagerWithTransfer::ReleaseEntrySlotLocked(
    const std::shared_ptr<SendEntry>& entry) {
  if (!entry || entry->slot_released) return;
  if (entry->slot_idx < 0 && entry->staged_host_blocks.empty()) return;
  if (entry->slot_idx >= 0) {
    RemoveStagingReadinessLocked(entry->slot_idx);
  }
  for (int64_t block_id : entry->registered_block_ids) {
    active_producer_blocks_.erase(block_id);
  }
  if (!entry->staged_host_blocks.empty()) {
    (void)host_block_manager_->Unlock(entry->staged_host_blocks);
    (void)host_block_manager_->Deallocate(entry->staged_host_blocks);
    entry->staged_host_blocks.clear();
  } else {
    ReleaseSlotLocked(entry->slot_idx);
  }
  entry->slot_released = true;
}

void KVCacheManagerWithTransfer::PurgeSendTombstonesLocked(
    std::chrono::steady_clock::time_point now) {
  while (!send_tombstone_order_.empty()) {
    const uint64_t uuid = send_tombstone_order_.front();
    auto it = send_tombstones_.find(uuid);
    if (it == send_tombstones_.end()) {
      send_tombstone_order_.pop_front();
      continue;
    }
    if (it->second.expires_at > now) break;
    send_tombstone_order_.erase(it->second.order_it);
    send_tombstones_.erase(it);
  }
  while (send_tombstones_.size() > max_send_tombstones_ &&
         !send_tombstone_order_.empty()) {
    const uint64_t uuid = send_tombstone_order_.front();
    send_tombstone_order_.pop_front();
    send_tombstones_.erase(uuid);
  }
}

void KVCacheManagerWithTransfer::InstallSendTombstoneLocked(
    uint64_t uuid, std::string message,
    std::chrono::steady_clock::time_point now, SendTombstoneKind kind) {
  auto existing = send_tombstones_.find(uuid);
  if (existing != send_tombstones_.end()) {
    send_tombstone_order_.erase(existing->second.order_it);
    send_tombstones_.erase(existing);
  }
  if (max_send_tombstones_ == 0) return;
  while (send_tombstones_.size() >= max_send_tombstones_ &&
         !send_tombstone_order_.empty()) {
    const uint64_t oldest = send_tombstone_order_.front();
    send_tombstone_order_.pop_front();
    send_tombstones_.erase(oldest);
  }
  send_tombstone_order_.push_back(uuid);
  SendTombstone tombstone;
  tombstone.message = std::move(message);
  tombstone.kind = kind;
  tombstone.expires_at = now + send_tombstone_ttl_;
  tombstone.order_it = std::prev(send_tombstone_order_.end());
  send_tombstones_.emplace(uuid, std::move(tombstone));
  cv_.SignalAll();
}

void KVCacheManagerWithTransfer::PurgePendingAcksLocked(
    std::chrono::steady_clock::time_point now) {
  while (!pending_ack_order_.empty()) {
    const uint64_t uuid = pending_ack_order_.front();
    auto it = pending_acks_.find(uuid);
    if (it == pending_acks_.end()) {
      pending_ack_order_.pop_front();
      continue;
    }
    if (it->second.expires_at > now) break;
    pending_ack_order_.erase(it->second.order_it);
    pending_acks_.erase(it);
  }
  while (pending_acks_.size() > max_pending_acks_ &&
         !pending_ack_order_.empty()) {
    const uint64_t uuid = pending_ack_order_.front();
    pending_ack_order_.pop_front();
    pending_acks_.erase(uuid);
  }
}

void KVCacheManagerWithTransfer::InstallPendingAckLocked(
    uint64_t uuid, std::chrono::steady_clock::time_point now) {
  PurgePendingAcksLocked(now);
  auto existing = pending_acks_.find(uuid);
  if (existing != pending_acks_.end()) {
    pending_ack_order_.erase(existing->second.order_it);
    pending_acks_.erase(existing);
  }
  if (max_pending_acks_ == 0) return;
  while (pending_acks_.size() >= max_pending_acks_ &&
         !pending_ack_order_.empty()) {
    const uint64_t oldest = pending_ack_order_.front();
    pending_ack_order_.pop_front();
    pending_acks_.erase(oldest);
  }
  pending_ack_order_.push_back(uuid);
  PendingAck pending_ack;
  pending_ack.expires_at = now + pending_ack_ttl_;
  pending_ack.order_it = std::prev(pending_ack_order_.end());
  pending_acks_.emplace(uuid, std::move(pending_ack));
}

bool KVCacheManagerWithTransfer::ConsumePendingAckLocked(
    uint64_t uuid, std::chrono::steady_clock::time_point now) {
  PurgePendingAcksLocked(now);
  auto it = pending_acks_.find(uuid);
  if (it == pending_acks_.end()) return false;
  pending_ack_order_.erase(it->second.order_it);
  pending_acks_.erase(it);
  return true;
}

void KVCacheManagerWithTransfer::UpdateLegacySendGaugesLocked() const {
  size_t waiting = 0;
  size_t transferring = draining_send_entries_.size();
  for (const auto& [uuid, entry] : send_entries_) {
    (void)uuid;
    if (entry->phase == SendEntry::Phase::kWaitingForPull) {
      ++waiting;
    } else {
      ++transferring;
    }
  }
  auto& store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  const std::string node_id = std::to_string(node_id_);
  const std::string control_port = std::to_string(local_control_port_);
  const std::array<telemetry::MetricLabel, 3> waiting_label = {
      telemetry::MetricLabel{"state", "waiting"},
      telemetry::MetricLabel{"node_id", node_id},
      telemetry::MetricLabel{"control_port", control_port}};
  const std::array<telemetry::MetricLabel, 3> transferring_label = {
      telemetry::MetricLabel{"state", "transferring"},
      telemetry::MetricLabel{"node_id", node_id},
      telemetry::MetricLabel{"control_port", control_port}};
  const std::array<telemetry::MetricLabel, 3> tombstone_label = {
      telemetry::MetricLabel{"state", "tombstone"},
      telemetry::MetricLabel{"node_id", node_id},
      telemetry::MetricLabel{"control_port", control_port}};
  store.SetGauge("tpu_raiden_legacy_uuid_entries", waiting_label, waiting);
  store.SetGauge("tpu_raiden_legacy_uuid_entries", transferring_label,
                 transferring);
  store.SetGauge("tpu_raiden_legacy_uuid_entries", tombstone_label,
                 send_tombstones_.size());
}

bool KVCacheManagerWithTransfer::TerminalizeSendEntryLocked(
    uint64_t uuid, const std::shared_ptr<SendEntry>& expected,
    const std::string& message, bool waiting_only, SendTombstoneKind kind) {
  auto it = send_entries_.find(uuid);
  if (it == send_entries_.end() || it->second != expected) return false;
  if (waiting_only && expected->phase != SendEntry::Phase::kWaitingForPull) {
    return false;
  }

  expected->terminal_requested = true;
  expected->terminal_message = message;
  if (!expected->completion_reported) {
    done_sending_.insert(expected->req_id);
    expected->completion_reported = true;
  }
  send_entries_.erase(it);
  InstallSendTombstoneLocked(uuid, message, std::chrono::steady_clock::now(),
                             kind);
  if (expected->phase == SendEntry::Phase::kTransferring &&
      expected->pending_layer_callbacks != 0) {
    draining_send_entries_[uuid] = expected;
  } else {
    ReleaseEntrySlotLocked(expected);
  }
  UpdateLegacySendGaugesLocked();
  cv_.SignalAll();
  return true;
}

void KVCacheManagerWithTransfer::FinalizeDrainedSendLocked(
    const std::shared_ptr<SendEntry>& entry) {
  if (!entry || entry->pending_layer_callbacks != 0) return;
  auto draining = draining_send_entries_.find(entry->uuid);
  if (draining != draining_send_entries_.end() && draining->second == entry) {
    ReleaseEntrySlotLocked(entry);
    draining_send_entries_.erase(draining);
  }
  UpdateLegacySendGaugesLocked();
  cv_.SignalAll();
}

void KVCacheManagerWithTransfer::FinishSendLayer(
    const std::shared_ptr<SendEntry>& entry, const absl::Status& status,
    const std::string& message) {
  absl::MutexLock lock(mu_);
  if (!entry) return;
  if (!status.ok()) {
    entry->failed = true;
    entry->terminal_requested = true;
    entry->terminal_message = message;
    (void)TerminalizeSendEntryLocked(entry->uuid, entry, message,
                                     /*waiting_only=*/false);
  }
  if (entry->pending_layer_callbacks > 0) {
    --entry->pending_layer_callbacks;
  }
  if (active_send_callbacks_ > 0) --active_send_callbacks_;
  if (entry->pending_layer_callbacks == 0) {
    if (send_entries_.find(entry->uuid) != send_entries_.end()) {
      (void)TerminalizeSendEntryLocked(
          entry->uuid, entry, status.ok() ? "send transfer completed" : message,
          /*waiting_only=*/false);
    }
    FinalizeDrainedSendLocked(entry);
  }
  cv_.SignalAll();
}

std::vector<int32_t> KVCacheManagerWithTransfer::ApplyLeaseBatchLocked(
    uint32_t op, const std::vector<uint64_t>& uuids,
    std::chrono::steady_clock::time_point now) {
  if (op != kOpRenewLeases && op != kOpCancelLeases) {
    throw std::invalid_argument("invalid lease batch operation");
  }
  PurgeSendTombstonesLocked(now);
  const auto lease_duration = std::chrono::milliseconds(
      std::max<int64_t>(1, static_cast<int64_t>(timeout_s_ * 1000.0)));
  std::map<uint64_t, int32_t> deduplicated;
  std::vector<int32_t> results;
  results.reserve(uuids.size());

  for (uint64_t uuid : uuids) {
    auto duplicate = deduplicated.find(uuid);
    if (duplicate != deduplicated.end()) {
      results.push_back(duplicate->second);
      RecordLeaseUpdateMetric(op, duplicate->second);
      continue;
    }

    LeaseUpdateStatus status = LeaseUpdateStatus::kUnknown;
    auto tombstone = send_tombstones_.find(uuid);
    if (tombstone != send_tombstones_.end()) {
      if (op == kOpCancelLeases &&
          (tombstone->second.kind ==
               SendTombstoneKind::kPreRegistrationCancel ||
           tombstone->second.kind == SendTombstoneKind::kCancelled)) {
        const SendTombstoneKind kind = tombstone->second.kind;
        const std::string message = tombstone->second.message;
        InstallSendTombstoneLocked(uuid, message, now, kind);
        status = LeaseUpdateStatus::kApplied;
      } else {
        status = LeaseUpdateStatus::kTerminal;
      }
    } else {
      auto entry_it = send_entries_.find(uuid);
      if (entry_it != send_entries_.end()) {
        const std::shared_ptr<SendEntry> entry = entry_it->second;
        if (entry->phase != SendEntry::Phase::kWaitingForPull) {
          status = LeaseUpdateStatus::kTransferring;
        } else if (now >= entry->retention_deadline) {
          (void)TerminalizeSendEntryLocked(
              uuid, entry, "send reached maximum heartbeat retention",
              /*waiting_only=*/true);
          status = LeaseUpdateStatus::kMaxRetentionReached;
        } else if (now >= entry->deadline) {
          (void)TerminalizeSendEntryLocked(uuid, entry,
                                           "send expired before lease update",
                                           /*waiting_only=*/true);
          status = LeaseUpdateStatus::kTerminal;
        } else if (op == kOpCancelLeases) {
          (void)TerminalizeSendEntryLocked(
              uuid, entry, "send released by remote decoder",
              /*waiting_only=*/true, SendTombstoneKind::kCancelled);
          status = LeaseUpdateStatus::kApplied;
        } else {
          entry->deadline =
              std::min(entry->retention_deadline, now + lease_duration);
          status = LeaseUpdateStatus::kApplied;
        }
      } else if (draining_send_entries_.find(uuid) !=
                 draining_send_entries_.end()) {
        status = LeaseUpdateStatus::kTransferring;
      } else if (op == kOpCancelLeases) {
        InstallSendTombstoneLocked(uuid, "send cancelled before registration",
                                   now,
                                   SendTombstoneKind::kPreRegistrationCancel);
        (void)ConsumePendingAckLocked(uuid, now);
        status = LeaseUpdateStatus::kApplied;
      }
    }

    const int32_t wire_status = static_cast<int32_t>(status);
    deduplicated.emplace(uuid, wire_status);
    results.push_back(wire_status);
    RecordLeaseUpdateMetric(op, wire_status);
  }
  UpdateLegacySendGaugesLocked();
  return results;
}

std::vector<int32_t> KVCacheManagerWithTransfer::SendLeaseBatch(
    const std::string& remote_endpoint, uint32_t op,
    const std::vector<uint64_t>& uuids) {
  if (uuids.empty()) return {};
  if (op != kOpRenewLeases && op != kOpCancelLeases) {
    throw std::invalid_argument("invalid lease control operation");
  }

  // Deduplicate before chunking so repeated cancellation remains idempotent
  // even when duplicate positions straddle two bounded wire requests.
  std::map<uint64_t, size_t> unique_index;
  std::vector<uint64_t> unique_uuids;
  std::vector<size_t> result_indices;
  unique_uuids.reserve(uuids.size());
  result_indices.reserve(uuids.size());
  for (uint64_t uuid : uuids) {
    auto [it, inserted] = unique_index.emplace(uuid, unique_uuids.size());
    if (inserted) unique_uuids.push_back(uuid);
    result_indices.push_back(it->second);
  }
  if (unique_uuids.size() != uuids.size()) {
    const std::vector<int32_t> unique_results =
        SendLeaseBatch(remote_endpoint, op, unique_uuids);
    std::vector<int32_t> results;
    results.reserve(uuids.size());
    for (size_t index : result_indices)
      results.push_back(unique_results[index]);
    return results;
  }

  if (uuids.size() > kMaxLeaseBatchSize) {
    std::vector<int32_t> combined;
    combined.reserve(uuids.size());
    for (size_t begin = 0; begin < uuids.size(); begin += kMaxLeaseBatchSize) {
      const size_t end = std::min(uuids.size(), begin + kMaxLeaseBatchSize);
      std::vector<uint64_t> chunk(uuids.begin() + begin, uuids.begin() + end);
      std::vector<int32_t> chunk_results =
          SendLeaseBatch(remote_endpoint, op, chunk);
      combined.insert(combined.end(), chunk_results.begin(),
                      chunk_results.end());
    }
    return combined;
  }

  const auto operation_deadline =
      std::chrono::steady_clock::now() + control_io_timeout_;
  int control_fd = ConnectTcp(remote_endpoint, operation_deadline);
  auto control_cleanup =
      std::unique_ptr<int, void (*)(int*)>(&control_fd, [](int* p) {
        if (p && *p >= 0) close(*p);
      });
  ConfigureSocketIoTimeout(control_fd, control_io_timeout_);

  ControlRequestHeader request;
  request.magic = kControlMagic;
  request.op = op;
  request.ep_idx = kLeaseProtocolVersion;
  request.num_blocks = static_cast<uint64_t>(uuids.size());
  CheckStatus("lease control header write",
              WriteExactUntil(control_fd, &request, sizeof(request),
                              operation_deadline));
  CheckStatus(
      "lease control UUID body write",
      WriteExactUntil(control_fd, uuids.data(), uuids.size() * sizeof(uint64_t),
                      operation_deadline));

  ControlResponseHeader response;
  CheckStatus("lease control response read",
              ReadExactUntil(control_fd, &response, sizeof(response),
                             operation_deadline));
  if (response.magic != kResponseMagic) {
    throw std::runtime_error("bad lease control response magic");
  }
  if (response.message_len > kMaxControlMessageBytes) {
    throw std::runtime_error("lease control response body is too large");
  }
  if (response.status != 0) {
    std::string message(response.message_len, '\0');
    if (!message.empty()) {
      CheckStatus("lease control error body read",
                  ReadExactUntil(control_fd, message.data(), message.size(),
                                 operation_deadline));
    }
    throw RemoteControlError(response.status, std::move(message));
  }
  if (response.num_layers != kLeaseProtocolVersion) {
    throw std::runtime_error("lease control protocol version mismatch");
  }
  const size_t expected_bytes = uuids.size() * sizeof(int32_t);
  if (response.message_len != expected_bytes) {
    throw std::runtime_error("lease control response length mismatch");
  }
  std::vector<int32_t> results(uuids.size());
  CheckStatus("lease control result body read",
              ReadExactUntil(control_fd, results.data(), expected_bytes,
                             operation_deadline));
  for (int32_t status : results) {
    if (status <
            static_cast<int32_t>(LeaseUpdateStatus::kMaxRetentionReached) ||
        status > static_cast<int32_t>(LeaseUpdateStatus::kApplied)) {
      throw std::runtime_error("lease control returned an invalid status");
    }
  }
  return results;
}

std::shared_ptr<KVCacheManagerWithTransfer::StagingReadinessState>
KVCacheManagerWithTransfer::CreateStagingReadiness(int64_t slot_idx,
                                                   int64_t num_blocks) {
  auto state = std::make_shared<StagingReadinessState>();
  state->slot_idx = slot_idx;
  state->num_blocks = num_blocks;
  state->num_layers = num_layers();
  state->num_shards = num_shards();
  state->layers.resize(state->num_layers * state->num_shards);
  {
    absl::MutexLock lock(mu_);
    staging_readiness_[slot_idx] = state;
  }
  return state;
}

void KVCacheManagerWithTransfer::MarkStagingLayerReady(
    const std::shared_ptr<StagingReadinessState>& state, size_t layer_idx,
    size_t shard_idx, absl::Status status) {
  if (!state) return;
  const size_t layer_state_idx = layer_idx * state->num_shards + shard_idx;
  if (layer_state_idx >= state->layers.size()) return;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    state->layers[layer_state_idx].done = true;
    state->layers[layer_state_idx].status = std::move(status);
  }
  state->cv.notify_all();
}

void KVCacheManagerWithTransfer::RemoveStagingReadinessLocked(
    int64_t slot_idx) {
  auto it = staging_readiness_.find(slot_idx);
  if (it == staging_readiness_.end()) return;
  std::shared_ptr<StagingReadinessState> state = it->second;
  staging_readiness_.erase(it);
  {
    std::lock_guard<std::mutex> state_lock(state->mu);
    for (StagingLayerReady& layer : state->layers) {
      if (!layer.done) {
        layer.done = true;
        layer.status = absl::CancelledError("staging slot was released");
      }
    }
  }
  state->cv.notify_all();
}

void KVCacheManagerWithTransfer::RegisterBlockReadinessCallback(
    size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
    transport::BlockTransportDelegate::HostBlockReadyCallback cb) {
  if (block_id < 0 || max_blocks_ <= 0) {
    cb(absl::OkStatus());
    return;
  }
  if (uuid == kLeaseAuthorizedPullUuid) {
    // A lease-authorised pull. The reader holds a lease over blocks this node
    // already published as host-resident, so their device-to-host copy
    // provably completed before the lease was granted, and the pin keeps them
    // from being reused for the duration of the read. Gating here would add
    // nothing -- and would be actively wrong, because the fallback scan below
    // can only match some OTHER transfer's entry, making this read wait on a
    // future that has nothing to do with it.
    cb(absl::OkStatus());
    return;
  }
  std::shared_ptr<SendEntry> entry;
  {
    absl::MutexLock lock(mu_);
    // Exact match: the transfer named itself, so gate on its own D2H. Reached
    // by uuid-carrying senders; a pull that did not identify itself falls
    // through to the scan below.
    auto it = send_entries_.find(uuid);
    if (it != send_entries_.end()) {
      entry = it->second;
    } else {
      // Fallback: no entry owns this uuid, so look for any live transfer that
      // registered this block id and wait on ITS copy. Conservative and
      // imprecise -- the match is by block id alone, so an unrelated transfer
      // can gate this one, and a stale entry whose future never resolves would
      // stall it until the reader's own deadline fires.
      for (const auto& [u, e] : send_entries_) {
        if (e->registered_block_set.find(block_id) !=
                e->registered_block_set.end() &&
            layer_idx < e->d2h_layer_futures.size()) {
          entry = e;
          break;
        }
      }
    }
  }
  if (!entry || layer_idx >= entry->d2h_layer_futures.size()) {
    cb(absl::OkStatus());
    return;
  }
  entry->d2h_layer_futures[layer_idx].OnReady(
      [cb = std::move(cb)](auto status_or) { cb(status_or.status()); });
}

void KVCacheManagerWithTransfer::ScheduleAsyncTask(std::function<void()> task) {
  push_pool_->Schedule(std::move(task));
}

void KVCacheManagerWithTransfer::StartRegistrationRetryScheduler() {
  std::lock_guard<std::mutex> lock(registration_retry_mu_);
  if (registration_retry_thread_.joinable()) return;
  registration_retry_stopping_ = false;
  registration_retry_thread_ =
      std::thread([this]() { RegistrationRetryLoop(); });
}

void KVCacheManagerWithTransfer::StopRegistrationRetryScheduler() {
  {
    std::lock_guard<std::mutex> lock(registration_retry_mu_);
    registration_retry_stopping_ = true;
    registration_retry_tasks_.clear();
  }
  registration_retry_cv_.notify_all();
  if (registration_retry_thread_.joinable()) {
    registration_retry_thread_.join();
  }
}

void KVCacheManagerWithTransfer::RegistrationRetryLoop() {
  while (true) {
    RegistrationRetryTask retry;
    {
      std::unique_lock<std::mutex> lock(registration_retry_mu_);
      while (true) {
        if (registration_retry_stopping_) return;
        if (registration_retry_tasks_.empty()) {
          registration_retry_cv_.wait(lock, [this]() {
            return registration_retry_stopping_ ||
                   !registration_retry_tasks_.empty();
          });
          continue;
        }

        const auto due = registration_retry_tasks_.begin()->first;
        if (std::chrono::steady_clock::now() < due) {
          registration_retry_cv_.wait_until(lock, due, [this, due]() {
            return registration_retry_stopping_ ||
                   registration_retry_tasks_.empty() ||
                   registration_retry_tasks_.begin()->first < due;
          });
          continue;
        }

        auto it = registration_retry_tasks_.begin();
        retry = std::move(it->second);
        registration_retry_tasks_.erase(it);
        break;
      }
    }

    if (stopping_.load(std::memory_order_acquire)) continue;
    try {
      push_pool_->Schedule(retry.target_node, std::move(retry.task));
    } catch (const std::exception& e) {
      if (!stopping_.load(std::memory_order_acquire)) {
        LOG(ERROR) << "Failed to schedule remote KV registration retry: "
                   << e.what();
      }
    }
  }
}

void KVCacheManagerWithTransfer::ScheduleRegistrationRetry(
    std::chrono::steady_clock::time_point due, std::optional<int> target_node,
    std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(registration_retry_mu_);
    if (registration_retry_stopping_ ||
        stopping_.load(std::memory_order_acquire)) {
      throw std::runtime_error("registration retry scheduler is stopping");
    }
    registration_retry_tasks_.emplace(
        due, RegistrationRetryTask{target_node, std::move(task)});
  }
  registration_retry_cv_.notify_one();
}

std::chrono::milliseconds
KVCacheManagerWithTransfer::ComputeRegistrationRetryDelay(
    uint64_t uuid, size_t retry_number) const {
  retry_number = std::max<size_t>(1, retry_number);
  const uint64_t exponent = std::min<size_t>(retry_number - 1, 20);
  const int64_t base_ms =
      std::max<int64_t>(1, registration_retry_base_delay_.count());
  const int64_t max_ms =
      std::max<int64_t>(base_ms, registration_retry_max_delay_.count());
  const uint64_t factor = uint64_t{1} << exponent;
  const int64_t exponential_ms =
      static_cast<uint64_t>(base_ms) > static_cast<uint64_t>(max_ms) / factor
          ? max_ms
          : std::min<int64_t>(max_ms, base_ms * static_cast<int64_t>(factor));

  // Stable per-(uuid, attempt) jitter makes behavior deterministic in tests
  // while avoiding synchronized retry bursts across queued decoders.
  uint64_t mixed = uuid + 0x9e3779b97f4a7c15ULL * retry_number;
  mixed ^= mixed >> 30;
  mixed *= 0xbf58476d1ce4e5b9ULL;
  mixed ^= mixed >> 27;
  mixed *= 0x94d049bb133111ebULL;
  mixed ^= mixed >> 31;
  const int64_t radius = std::max<int64_t>(1, exponential_ms / 4);
  const int64_t jitter =
      static_cast<int64_t>(mixed % static_cast<uint64_t>(2 * radius + 1)) -
      radius;
  return std::chrono::milliseconds(
      std::clamp<int64_t>(exponential_ms + jitter, 1, max_ms));
}

void KVCacheManagerWithTransfer::StartControlServer() {
  control_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
  if (control_fd_ < 0) {
    throw std::runtime_error("control socket() failed: " +
                             std::string(std::strerror(errno)));
  }
  int opt = 1;
  setsockopt(control_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  setsockopt(control_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  int ipv6only = 0;
  if (setsockopt(control_fd_, IPPROTO_IPV6, IPV6_V6ONLY, &ipv6only,
                 sizeof(ipv6only)) < 0) {
    LOG(WARNING) << "setsockopt IPV6_V6ONLY=0 failed: " << std::strerror(errno);
  }

  sockaddr_in6 addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin6_family = AF_INET6;
  addr.sin6_addr = in6addr_any;
  addr.sin6_port = htons(local_control_port_);

  if (bind(control_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::string err = std::strerror(errno);
    close(control_fd_);
    control_fd_ = -1;
    throw std::runtime_error("control bind(" +
                             std::to_string(local_control_port_) +
                             ") failed: " + err);
  }
  socklen_t len = sizeof(addr);
  if (getsockname(control_fd_, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
    close(control_fd_);
    control_fd_ = -1;
    throw std::runtime_error("getsockname() failed: " +
                             std::string(std::strerror(errno)));
  }
  local_control_port_ = ntohs(addr.sin6_port);
  if (listen(control_fd_, 128) < 0) {
    close(control_fd_);
    control_fd_ = -1;
    throw std::runtime_error("listen() failed: " +
                             std::string(std::strerror(errno)));
  }
  stopping_ = false;
  control_thread_ = std::thread([this]() { ControlServerLoop(); });
}

void KVCacheManagerWithTransfer::StopControlServer() {
  stopping_ = true;
  {
    absl::MutexLock lock(mu_);
    cv_.SignalAll();
    for (int fd : active_control_fds_) {
      // The worker owns close(). shutdown() interrupts a partial read without
      // racing descriptor reuse.
      (void)shutdown(fd, SHUT_RDWR);
    }
    while (!staging_readiness_.empty()) {
      RemoveStagingReadinessLocked(staging_readiness_.begin()->first);
    }
  }
  // Wake workers parked in ProcessPullStream waiting for a send entry that
  // will never arrive, so their loops observe stopping_ and the pools can
  // join them.
  cv_.SignalAll();
  const int control_fd = control_fd_;
  if (control_fd >= 0) (void)shutdown(control_fd, SHUT_RDWR);
  if (control_thread_.joinable()) {
    control_thread_.join();
  }
  {
    // Every handler may schedule producer work on push_pool_. Keep both the
    // manager and that pool alive until shutdown has interrupted and drained
    // all accepted control connections.
    absl::MutexLock lock(mu_);
    while (!active_control_fds_.empty()) {
      cv_.Wait(&mu_);
    }
  }
  if (control_fd >= 0) close(control_fd);
  control_fd_ = -1;
}

void KVCacheManagerWithTransfer::ControlServerLoop() {
  while (!stopping_) {
    pollfd pfd;
    pfd.fd = control_fd_;
    pfd.events = POLLIN;
    int r = poll(&pfd, 1, 200);
    if (r < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (r == 0) continue;
    int client_fd = accept(control_fd_, nullptr, nullptr);
    if (client_fd < 0) {
      if (errno == EINTR) continue;
      break;
    }
    {
      absl::MutexLock lock(mu_);
      if (stopping_) {
        (void)shutdown(client_fd, SHUT_RDWR);
        close(client_fd);
        break;
      }
      active_control_fds_.insert(client_fd);
    }
    const std::optional<int> source_node = assigned_numa_node();

    try {
      pull_pool_->Schedule(source_node, [this, client_fd]() {
        try {
          HandleControlConnection(client_fd);
        } catch (...) {
          LOG(ERROR) << "Unexpected non-standard control handler exception";
        }
        {
          absl::MutexLock lock(mu_);
          active_control_fds_.erase(client_fd);
          cv_.SignalAll();
        }
        close(client_fd);
      });
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to schedule control connection: " << e.what();
      {
        absl::MutexLock lock(mu_);
        active_control_fds_.erase(client_fd);
        cv_.SignalAll();
      }
      (void)shutdown(client_fd, SHUT_RDWR);
      close(client_fd);
    }
  }
}

void KVCacheManagerWithTransfer::HandleControlConnection(int fd) {
  const auto operation_deadline =
      std::chrono::steady_clock::now() + control_io_timeout_;
  try {
    ConfigureSocketIoTimeout(fd, control_io_timeout_);
    ControlRequestHeader req;
    CheckStatus("control request header read",
                ReadExactUntil(fd, &req, sizeof(req), operation_deadline));
    if (req.magic != kControlMagic) {
      throw std::runtime_error("bad control request magic");
    }
    if (req.op == kOpAck || (req.op == kOpPullStream && req.num_blocks == 0)) {
      if (req.num_blocks != 0) {
        throw std::runtime_error("ack control request included a body");
      }
      AckSend(req.uuid);
      ControlResponseHeader response;
      response.magic = kResponseMagic;
      response.status = 0;
      CheckStatus(
          "control ack response write",
          WriteExactUntil(fd, &response, sizeof(response), operation_deadline));
    } else if (req.op == kOpPullStream) {
      ProcessPullStream(fd, req, operation_deadline);
    } else if (req.op == kOpRenewLeases || req.op == kOpCancelLeases) {
      ProcessLeaseBatch(fd, req, operation_deadline);
    } else {
      throw std::runtime_error("unknown control op code: " +
                               std::to_string(req.op));
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Raiden producer error in control connection handler: "
               << e.what();
    if (stopping_) return;
    ControlResponseHeader response;
    response.magic = kResponseMagic;
    response.status = -1;
    std::string message = e.what();
    if (message.size() > kMaxControlMessageBytes) {
      message.resize(kMaxControlMessageBytes);
    }
    response.message_len = message.size();
    (void)WriteExactUntil(fd, &response, sizeof(response), operation_deadline);
    if (response.message_len > 0) {
      (void)WriteExactUntil(fd, message.data(), message.size(),
                            operation_deadline);
    }
  }
}

void KVCacheManagerWithTransfer::ProcessPullStream(
    int fd, const ControlRequestHeader& req,
    std::chrono::steady_clock::time_point operation_deadline) {
  if (req.num_blocks == 0 ||
      req.num_blocks > static_cast<uint64_t>(max_blocks_)) {
    throw std::invalid_argument("pull stream num_blocks is out of range");
  }
  if (req.consumer_data_port == 0 ||
      req.consumer_data_port > std::numeric_limits<uint16_t>::max()) {
    throw std::invalid_argument("pull stream consumer data port is invalid");
  }
  if (req.num_ips > kMaxNics) {
    throw std::invalid_argument("pull stream has too many consumer IPs");
  }

  // Always consume the bounded body before replying. Closing a socket with
  // unread request bytes can reset TCP and discard a retryable response.
  std::vector<int64_t> src_block_ids =
      ReadBlockIds(fd, req.num_blocks, operation_deadline);
  std::vector<int64_t> dst_block_ids =
      ReadBlockIds(fd, req.num_blocks, operation_deadline);

  std::shared_ptr<SendEntry> entry;
  std::string terminal_error;
  {
    absl::MutexLock lock(mu_);
    const auto now = std::chrono::steady_clock::now();
    PurgeSendTombstonesLocked(now);
    auto tombstone = send_tombstones_.find(req.uuid);
    if (tombstone != send_tombstones_.end()) {
      terminal_error = "terminal send UUID: " + tombstone->second.message;
    } else {
      auto it = send_entries_.find(req.uuid);
      if (it != send_entries_.end()) {
        entry = it->second;
        if (entry->phase != SendEntry::Phase::kWaitingForPull) {
          terminal_error = "pull stream UUID was already claimed";
        } else if (entry->deadline <= now) {
          (void)TerminalizeSendEntryLocked(req.uuid, entry,
                                           "send expired before pull claim",
                                           /*waiting_only=*/true);
          terminal_error = "send expired before pull claim";
          entry.reset();
        }
      }
    }
  }
  if (!terminal_error.empty()) {
    RecordPullMetric("rejected");
    throw std::runtime_error(terminal_error);
  }
  if (!entry) {
    ControlResponseHeader response;
    response.magic = kResponseMagic;
    response.status = kControlRetryableUnknown;
    const std::string message = kSendUuidNotRegistered;
    response.message_len = message.size();
    CheckStatus(
        "control retryable response header write",
        WriteExactUntil(fd, &response, sizeof(response), operation_deadline));
    CheckStatus("control retryable response body write",
                WriteExactUntil(fd, message.data(), message.size(),
                                operation_deadline));
    retryable_unknown_pull_responses_.fetch_add(1, std::memory_order_relaxed);
    RecordPullMetric("retryable_unknown");
    return;
  }

  ValidateRequestedBlocks(*entry, src_block_ids);

  std::vector<std::string> peer_ips;
  if (req.num_ips > 0) {
    for (uint32_t i = 0;
         i < std::min(req.num_ips, static_cast<uint32_t>(kMaxNics)); ++i) {
      char ip_str[INET6_ADDRSTRLEN];
      bool is_ipv4_mapped = true;
      for (int j = 0; j < 10; ++j) {
        if (req.consumer_ips[i][j] != 0) {
          is_ipv4_mapped = false;
          break;
        }
      }
      if (req.consumer_ips[i][10] != 0xff || req.consumer_ips[i][11] != 0xff) {
        is_ipv4_mapped = false;
      }

      if (is_ipv4_mapped) {
        struct in_addr ipv4_addr;
        std::memcpy(&ipv4_addr, req.consumer_ips[i] + 12, 4);
        if (inet_ntop(AF_INET, &ipv4_addr, ip_str, sizeof(ip_str)) != nullptr) {
          peer_ips.push_back(ip_str);
        }
      } else {
        if (inet_ntop(AF_INET6, req.consumer_ips[i], ip_str, sizeof(ip_str)) !=
            nullptr) {
          peer_ips.push_back(ip_str);
        }
      }
    }
  }

  if (peer_ips.empty() && req.num_ips == 0) {
    LOG(WARNING) << "No consumer IPs specified in ControlRequestHeader.";
  }

  if (peer_ips.empty()) {
    std::string peer_ip = GetPeerIp(fd);
    if (!peer_ip.empty()) {
      peer_ips.push_back(peer_ip);
    }
  }

  std::vector<std::string> remote_data_endpoints;
  for (const auto& peer_ip : peer_ips) {
    if (absl::StrContains(peer_ip, ':')) {
      remote_data_endpoints.push_back(
          absl::StrCat("[", peer_ip, "]:", req.consumer_data_port));
    } else {
      remote_data_endpoints.push_back(
          absl::StrCat(peer_ip, ":", req.consumer_data_port));
    }
  }

  if (remote_data_endpoints.empty()) {
    RecordPullMetric("rejected");
    throw std::runtime_error("pull stream has no usable consumer endpoint");
  }

  std::string preaccept_error;
  {
    // Linearization point: identity, phase, deadline, and staging capacity are
    // checked and claimed atomically before success is observable.
    absl::MutexLock lock(mu_);
    const auto now = std::chrono::steady_clock::now();
    PurgeSendTombstonesLocked(now);
    auto it = send_entries_.find(req.uuid);
    if (stopping_) {
      preaccept_error = "manager stopped before pull claim";
    } else if (it == send_entries_.end() || it->second != entry) {
      preaccept_error = "send UUID disappeared before pull claim";
    } else if (entry->phase != SendEntry::Phase::kWaitingForPull) {
      preaccept_error = "pull stream UUID was already claimed";
    } else if (now >= operation_deadline) {
      preaccept_error = "control operation expired before pull claim";
    } else if (entry->deadline <= now) {
      preaccept_error = "send expired before pull claim";
    } else {
      entry->phase = SendEntry::Phase::kTransferring;
      entry->transfer_deadline = DeadlineFromNow();
      UpdateLegacySendGaugesLocked();
    }
    if (!preaccept_error.empty() && it != send_entries_.end() &&
        it->second == entry &&
        entry->phase == SendEntry::Phase::kWaitingForPull) {
      (void)TerminalizeSendEntryLocked(req.uuid, entry, preaccept_error,
                                       /*waiting_only=*/true);
    }
  }
  if (!preaccept_error.empty()) {
    RecordPullMetric("rejected");
    throw std::runtime_error(preaccept_error);
  }

  std::vector<int64_t> host_block_ids;
  if (!AcquireSendStagingWithRetry(req.uuid, src_block_ids,
                                   operation_deadline, &host_block_ids)) {
    RecordPullMetric("rejected");
    throw std::runtime_error(
        "producer staging unavailable before pull acceptance");
  }
  {
    absl::MutexLock lock(mu_);
    auto it = send_entries_.find(req.uuid);
    if (stopping_ || it == send_entries_.end() || it->second != entry ||
        entry->phase != SendEntry::Phase::kTransferring) {
      preaccept_error = "send UUID disappeared before pull acceptance";
    } else {
      entry->remote_data_endpoints = remote_data_endpoints;
      entry->src_ints.assign(host_block_ids.begin(), host_block_ids.end());
      entry->dst_ints.assign(dst_block_ids.begin(), dst_block_ids.end());
      // This startup guard prevents timeout/cancellation from releasing the
      // staging allocation before StartPushInternal has attached callbacks.
      entry->pending_layer_callbacks = 1;
      ++active_send_callbacks_;
    }
  }
  if (!preaccept_error.empty()) {
    RecordPullMetric("rejected");
    throw std::runtime_error(preaccept_error);
  }

  // A positive response means the producer irrevocably owns staging.
  ControlResponseHeader response;
  response.magic = kResponseMagic;
  response.status = 0;
  response.num_layers = static_cast<uint32_t>(num_layers() * num_shards());
  response.data_port = static_cast<uint32_t>(local_data_port_);
  const absl::Status response_status =
      WriteExactUntil(fd, &response, sizeof(response), operation_deadline);
  if (!response_status.ok()) {
    FinishSendLayer(entry, response_status,
                    "failed to send positive pull response");
    RecordPullMetric("response_failed");
    ThrowStatus("control stream response header write", response_status);
  }
  RecordPullMetric("accepted");

  VLOG(1) << "ProcessPullStream (Hybrid Bridge) successfully acknowledged "
             "consumer. Intercepting and launching StartPushInternal to "
          << (remote_data_endpoints.empty() ? "" : remote_data_endpoints[0])
          << (remote_data_endpoints.size() > 1 ? " and others" : "");

  try {
    push_pool_->Schedule(
        assigned_numa_node(),
        [this, uuid = req.uuid, entry,
         remote_data_endpoints = std::move(remote_data_endpoints),
         src_block_ids = std::move(src_block_ids),
         dst_block_ids = std::move(dst_block_ids)]() {
          try {
            StartPushInternal(uuid, remote_data_endpoints, src_block_ids,
                              dst_block_ids);
          } catch (const std::exception& e) {
            LOG(ERROR) << "Failed to start accepted send UUID " << uuid << ": "
                       << e.what();
            FinishSendLayer(entry, absl::InternalError(e.what()),
                            "accepted send failed during startup");
          } catch (...) {
            LOG(ERROR) << "Failed to start accepted send UUID " << uuid
                       << ": unknown exception";
            FinishSendLayer(entry,
                            absl::InternalError("unknown startup exception"),
                            "accepted send failed during startup");
          }
        });
  } catch (const std::exception& e) {
    FinishSendLayer(entry, absl::InternalError(e.what()),
                    "accepted send could not be scheduled");
    // The positive response is already on the wire. Do not make the outer
    // handler attempt to append a contradictory second response.
    LOG(ERROR) << "Failed to schedule accepted send UUID " << req.uuid << ": "
               << e.what();
  } catch (...) {
    FinishSendLayer(entry, absl::InternalError("unknown scheduling exception"),
                    "accepted send could not be scheduled");
    // The positive response is already on the wire. Do not make the outer
    // handler attempt to append a contradictory second response.
    LOG(ERROR) << "Failed to schedule accepted send UUID " << req.uuid
               << ": unknown exception";
  }
}

void KVCacheManagerWithTransfer::ProcessLeaseBatch(
    int fd, const ControlRequestHeader& req,
    std::chrono::steady_clock::time_point operation_deadline) {
  if (req.op != kOpRenewLeases && req.op != kOpCancelLeases) {
    throw std::invalid_argument("invalid lease batch operation");
  }
  if (req.num_blocks == 0 || req.num_blocks > kMaxLeaseBatchSize) {
    throw std::invalid_argument("lease batch size is out of range");
  }

  // Consume the bounded body before returning a version/field error. Closing
  // with unread request bytes can reset TCP and hide the useful response.
  std::vector<uint64_t> uuids(static_cast<size_t>(req.num_blocks));
  CheckStatus("lease batch UUID body read",
              ReadExactUntil(fd, uuids.data(), uuids.size() * sizeof(uint64_t),
                             operation_deadline));
  if (req.ep_idx != kLeaseProtocolVersion) {
    throw std::invalid_argument("unsupported lease control protocol version");
  }
  if (req.uuid != 0 || req.consumer_data_port != 0 || req.num_ips != 0) {
    throw std::invalid_argument("lease batch mixed incompatible fields");
  }
  std::vector<int32_t> results;
  {
    absl::MutexLock lock(mu_);
    if (stopping_) {
      throw std::runtime_error("manager stopped during lease update");
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= operation_deadline) {
      throw std::runtime_error("control operation expired before lease update");
    }
    results = ApplyLeaseBatchLocked(req.op, uuids, now);
  }

  ControlResponseHeader response;
  response.magic = kResponseMagic;
  response.status = 0;
  response.num_layers = kLeaseProtocolVersion;
  response.message_len = results.size() * sizeof(int32_t);
  CheckStatus(
      "lease batch response header write",
      WriteExactUntil(fd, &response, sizeof(response), operation_deadline));
  CheckStatus("lease batch response body write",
              WriteExactUntil(fd, results.data(), response.message_len,
                              operation_deadline));
}

bool KVCacheManagerWithTransfer::AcquireSendStagingWithRetry(
    uint64_t uuid, const std::vector<int64_t>& src_block_ids,
    std::chrono::steady_clock::time_point operation_deadline,
    std::vector<int64_t>* host_block_ids) {
  // Both the producer lease and this control request bound the staging wait.
  std::chrono::steady_clock::time_point stage_deadline;
  while (true) {
    if (shutting_down_.load(std::memory_order_relaxed)) {
      return false;  // the manager is being destroyed; its state goes with it
    }
    {
      absl::MutexLock lock(mu_);
      auto it = send_entries_.find(uuid);
      if (it == send_entries_.end()) {
        return false;  // request cancelled while waiting for staging
      }
      stage_deadline = std::min(it->second->deadline, operation_deadline);
      // Staging that can never seat this request fails it now rather than
      // after the deadline: a fixed slot holds max_blocks_ pages, the
      // per-transfer pool holds total_blocks() pages.
      const int64_t capacity = dynamic_host_staging_
                                   ? host_block_manager_->total_blocks()
                                   : max_blocks_;
      if (static_cast<int64_t>(src_block_ids.size()) > capacity) {
        LOG(ERROR) << "StartPushInternal: request " << it->second->req_id
                   << " needs " << src_block_ids.size() << " blocks but "
                   << (dynamic_host_staging_ ? "the host staging pool holds "
                                             : "a staging slot holds ")
                   << capacity;
        it->second->failed = true;
        (void)TerminalizeSendEntryLocked(
            uuid, it->second, "send exceeds producer staging capacity",
            /*waiting_only=*/false);
        return false;
      }
      RecvEntry staging;
      auto staged = AcquireRecvStagingLocked(
          static_cast<int64_t>(src_block_ids.size()), &staging);
      if (staged.has_value()) {
        it->second->slot_idx = staging.slot_idx;
        it->second->staged_host_blocks = std::move(staging.staged_host_blocks);
        *host_block_ids = std::move(*staged);
        return true;
      }
    }
    // Staging exhausted: wait for in-flight sends to hand blocks back
    // instead of reporting a send that never happened. The consumer's own
    // deadline still bounds the total wait.
    if (std::chrono::steady_clock::now() >= stage_deadline) {
      absl::MutexLock lock(mu_);
      auto it = send_entries_.find(uuid);
      if (it != send_entries_.end()) {
        LOG(ERROR) << "StartPushInternal: staging exhausted serving "
                   << it->second->req_id << " (" << src_block_ids.size()
                   << " blocks; free_host_blocks="
                   << host_block_manager_->num_free_blocks()
                   << ", total_host_blocks="
                   << host_block_manager_->total_blocks()
                   << ", free_slots=" << free_slots_.size()
                   << "); reporting transfer failure";
        it->second->failed = true;
        (void)TerminalizeSendEntryLocked(
            uuid, it->second, "producer staging wait exceeded its deadline",
            /*waiting_only=*/false);
      }
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void KVCacheManagerWithTransfer::StartPushInternal(
    uint64_t uuid, const std::vector<std::string>& remote_data_endpoints,
    const std::vector<int64_t>& src_block_ids,
    const std::vector<int64_t>& dst_block_ids) {
  // Stage the producer's device KV into a host slot (slot.block_ids) and send
  // those host blocks to the consumer, keeping host offsets within the staging
  // pool. Writing D2H straight to host[src_block_id] overflows the host buffer
  // once a device block id exceeds num_host_blocks.
  std::vector<int64_t> host_block_ids;
  std::shared_ptr<SendEntry> entry;
  {
    absl::MutexLock lock(mu_);
    auto it = send_entries_.find(uuid);
    if (it == send_entries_.end()) {
      throw std::runtime_error("send entry disappeared after pull acceptance");
    }
    entry = it->second;
    if (entry->phase != SendEntry::Phase::kTransferring ||
        entry->pending_layer_callbacks == 0) {
      throw std::runtime_error(
          "accepted send does not own its startup callback guard");
    }
    host_block_ids.assign(entry->src_ints.begin(), entry->src_ints.end());
    if (host_block_ids.size() != src_block_ids.size() ||
        entry->dst_ints.size() != dst_block_ids.size()) {
      throw std::runtime_error("accepted send staging metadata is incomplete");
    }
    entry->remote_data_endpoints = remote_data_endpoints;
    entry->dst_ints.assign(dst_block_ids.begin(), dst_block_ids.end());
  }

  // Coalesce contiguous (device,host) block runs into a few large copies. With
  // per-block segments (sizes=1) a contiguous KV range becomes n device copies
  // that flood the command queue with small ops and serialize against prefill
  // GEMMs on the shared TensorCore; coalescing collapses a contiguous range to
  // one copy, matching the pre-Hybrid-Push pull path.
  std::vector<raiden::PjRtCopyFuture> futures;
  absl::Status issue_status = absl::OkStatus();
  try {
    CopySpec d2h_copy = BuildCoalescedCopySpec(src_block_ids, host_block_ids);
    futures.reserve(num_layers());
    for (size_t layer_idx = 0; layer_idx < num_layers(); ++layer_idx) {
      LOG(INFO) << "StartPushInternal (D2H start) layer " << layer_idx
                << ": uuid=" << uuid
                << ", numa=" << assigned_numa_node().value_or(-1);
      auto future_or = D2hSyncDispatch(
          d2h_copy.src_offsets, d2h_copy.dst_offsets, d2h_copy.sizes,
          /*slot_idx=*/std::nullopt, /*layer_idx=*/layer_idx);
      if (!future_or.ok()) {
        issue_status = future_or.status();
        break;
      }
      futures.push_back(std::move(future_or.value()));
    }
  } catch (const std::exception& e) {
    issue_status = absl::InternalError(e.what());
  } catch (...) {
    issue_status =
        absl::InternalError("unknown exception while issuing producer D2H");
  }

  {
    absl::MutexLock lock(mu_);
    entry->d2h_layer_futures = std::move(futures);
    entry->pending_layer_callbacks += entry->d2h_layer_futures.size();
    active_send_callbacks_ += entry->d2h_layer_futures.size();
    --active_send_callbacks_;
    --entry->pending_layer_callbacks;
    if (!issue_status.ok()) {
      entry->failed = true;
      (void)TerminalizeSendEntryLocked(uuid, entry, "producer D2H issue failed",
                                       /*waiting_only=*/false);
    }
    if (entry->pending_layer_callbacks == 0) {
      if (send_entries_.find(uuid) != send_entries_.end()) {
        (void)TerminalizeSendEntryLocked(uuid, entry,
                                         issue_status.ok()
                                             ? "send transfer completed"
                                             : "producer D2H issue failed",
                                         /*waiting_only=*/false);
      }
      FinalizeDrainedSendLocked(entry);
    }
    cv_.SignalAll();
  }

  if (entry->d2h_layer_futures.empty()) {
    return;
  }

  for (size_t layer_idx = 0; layer_idx < entry->d2h_layer_futures.size();
       ++layer_idx) {
    auto completed = std::make_shared<std::atomic<bool>>(false);
    auto finish_once = [this, entry, completed](const absl::Status& status,
                                                const std::string& message) {
      if (!completed->exchange(true)) {
        FinishSendLayer(entry, status, message);
      }
    };
    try {
      entry->d2h_layer_futures[layer_idx].OnReady(
          [this, entry, layer_idx, completed, finish_once](auto status_or) {
            if (!status_or.ok()) {
              LOG(ERROR) << "StartPushInternal: D2H copy failed for layer "
                         << layer_idx
                         << ", status: " << status_or.status().ToString();
              finish_once(status_or.status(), "producer D2H completion failed");
              return;
            }
            try {
              push_pool_->Schedule(assigned_numa_node(),
                                   [this, entry, layer_idx, completed]() {
                                     SendNextLayer(entry, layer_idx, completed);
                                   });
            } catch (const std::exception& e) {
              finish_once(absl::InternalError(e.what()),
                          "failed to schedule producer H2H transfer");
            } catch (...) {
              finish_once(
                  absl::InternalError(
                      "unknown exception scheduling producer H2H transfer"),
                  "failed to schedule producer H2H transfer");
            }
          });
    } catch (const std::exception& e) {
      finish_once(absl::InternalError(e.what()),
                  "producer D2H callback registration failed");
    } catch (...) {
      finish_once(absl::InternalError(
                      "unknown producer D2H callback registration exception"),
                  "producer D2H callback registration failed");
    }
  }
}

void KVCacheManagerWithTransfer::SendNextLayer(
    const std::shared_ptr<SendEntry>& entry, size_t layer_idx,
    const std::shared_ptr<std::atomic<bool>>& completed) {
  auto finish_once = [this, entry, completed](const absl::Status& status,
                                              const std::string& message) {
    if (!completed->exchange(true)) {
      FinishSendLayer(entry, status, message);
    }
  };
  bool skip = false;
  {
    absl::MutexLock lock(mu_);
    auto it = send_entries_.find(entry->uuid);
    skip = stopping_ || entry->terminal_requested ||
           it == send_entries_.end() || it->second != entry;
  }
  if (skip) {
    finish_once(absl::OkStatus(), "send transfer was terminalized");
    return;
  }

  LOG(INFO) << "StartPushInternal (H2H start layer " << layer_idx
            << "): uuid=" << entry->uuid
            << ", numa=" << assigned_numa_node().value_or(-1);
  const int numa_node = assigned_numa_node().value_or(-1);
  try {
    H2hWriteDirectAsync(
        entry->remote_data_endpoints, entry->src_ints, entry->dst_ints,
        entry->uuid, layer_idx,
        [entry, layer_idx, numa_node,
         finish_once](absl::StatusOr<std::vector<int>> push_res) {
          if (!push_res.ok()) {
            LOG(ERROR) << "H2hWrite failed for layer " << layer_idx << ": "
                       << push_res.status().ToString();
            finish_once(push_res.status(), "producer H2H transfer failed");
            return;
          }
          LOG(INFO) << "StartPushInternal (H2H complete layer " << layer_idx
                    << "): uuid=" << entry->uuid << ", numa=" << numa_node;
          finish_once(absl::OkStatus(), "");
        });
  } catch (const std::exception& e) {
    finish_once(absl::InternalError(e.what()), "producer H2H issue threw");
  } catch (...) {
    finish_once(absl::InternalError("unknown producer H2H issue exception"),
                "producer H2H issue threw");
  }
}

absl::Status KVCacheManagerWithTransfer::WaitForPendingWork() {
  LOG(INFO) << "Waiting for pending transfer work to complete...";
  const absl::Time start = absl::Now();
  while (true) {
    {
      absl::MutexLock lock(mu_);
      bool recv_pending = false;
      for (const auto& [uuid, entry] : active_recv_entries_) {
        (void)uuid;
        if (!entry.is_pool_reshard || !entry.network_completed) {
          recv_pending = true;
          break;
        }
        for (const auto& future : entry.h2d_futures) {
          if (!future.IsReady()) {
            recv_pending = true;
            break;
          }
        }
        if (recv_pending) break;
      }
      bool legacy_send_pending =
          active_send_callbacks_ != 0 || !draining_send_entries_.empty();
      if (!legacy_send_pending) {
        for (const auto& [uuid, entry] : send_entries_) {
          (void)uuid;
          if (entry->phase == SendEntry::Phase::kTransferring) {
            legacy_send_pending = true;
            break;
          }
        }
      }
      if (!recv_pending && !legacy_send_pending &&
          active_pool_reshard_sends_.empty()) {
        break;
      }
      const absl::Duration elapsed = absl::Now() - start;
      if (elapsed > kPendingWorkTimeout) {
        return absl::DeadlineExceededError(
            "Timeout waiting for pending transfer work");
      }
    }
    absl::SleepFor(absl::Milliseconds(100));
  }
  LOG(INFO) << "All pending transfer work completed.";
  return absl::OkStatus();
}

std::string KVCacheManagerWithTransfer::EndpointWithPort(
    const std::string& endpoint, int port) const {
  auto [host, ignored_port] = SplitEndpoint(endpoint);
  (void)ignored_port;
  return host + ":" + std::to_string(port);
}

void KVCacheManagerWithTransfer::AckRemote(const std::string& remote_endpoint,
                                           uint64_t uuid) {
  const auto operation_deadline =
      std::chrono::steady_clock::now() + control_io_timeout_;
  int control_fd = ConnectTcp(remote_endpoint, operation_deadline);
  auto control_cleanup =
      std::unique_ptr<int, void (*)(int*)>(&control_fd, [](int* p) {
        if (p && *p >= 0) close(*p);
      });
  ConfigureSocketIoTimeout(control_fd, control_io_timeout_);
  ControlRequestHeader stream_request;
  stream_request.magic = kControlMagic;
  stream_request.op = kOpPullStream;
  stream_request.uuid = uuid;
  stream_request.ep_idx = 0;
  stream_request.num_blocks = 0;
  CheckStatus("control pull stream write (empty)",
              WriteExactUntil(control_fd, &stream_request,
                              sizeof(stream_request), operation_deadline));
  (void)ReadControlResponseHeader(control_fd, operation_deadline);
}

KVCacheManagerWithTransfer::ControlResponseHeader
KVCacheManagerWithTransfer::ReadControlResponseHeader(
    int fd, std::chrono::steady_clock::time_point operation_deadline) {
  ControlResponseHeader response;
  CheckStatus(
      "control response read",
      ReadExactUntil(fd, &response, sizeof(response), operation_deadline));
  if (response.magic != kResponseMagic) {
    throw std::runtime_error("bad control response magic");
  }
  if (response.message_len > kMaxControlMessageBytes) {
    throw std::runtime_error("control response body is too large");
  }
  if (response.status != 0) {
    std::string message(response.message_len, '\0');
    if (response.message_len > 0) {
      CheckStatus("control error body read",
                  ReadExactUntil(fd, message.data(), message.size(),
                                 operation_deadline));
    }
    throw RemoteControlError(response.status, std::move(message));
  }
  return response;
}

void KVCacheManagerWithTransfer::AckSend(uint64_t uuid) {
  std::shared_ptr<SendEntry> entry;
  bool terminalized = false;
  {
    absl::MutexLock lock(mu_);
    const auto now = std::chrono::steady_clock::now();
    PurgeSendTombstonesLocked(now);
    if (send_tombstones_.find(uuid) != send_tombstones_.end()) return;
    auto draining = draining_send_entries_.find(uuid);
    if (draining != draining_send_entries_.end()) {
      draining->second->ack_received = true;
      return;
    }
    auto it = send_entries_.find(uuid);
    if (it == send_entries_.end()) {
      InstallPendingAckLocked(uuid, now);
      return;
    }
    entry = it->second;
    if (entry->phase == SendEntry::Phase::kTransferring) {
      // Async D2H/H2H still owns the staging slot after acceptance.
      entry->ack_received = true;
      return;
    }
    terminalized = TerminalizeSendEntryLocked(
        uuid, entry, "send was acknowledged", /*waiting_only=*/false);
  }
  if (!terminalized) return;
  const auto ack_done = std::chrono::steady_clock::now();
  std::ostringstream timing;
  timing << "RAIDEN_TIMING event=producer_ack"
         << " req_id=" << entry->req_id << " uuid=" << entry->uuid
         << " node_id=" << node_id_ << " blocks=" << entry->num_blocks
         << " bytes=" << entry->total_bytes
         << " stage_to_ack_ms=" << DurationMs(entry->d2h_done, ack_done)
         << " register_to_ack_ms="
         << DurationMs(entry->register_start, ack_done)
         << " failed=" << (entry->failed ? 1 : 0);
  EmitTimingLog(timing.str());
}

std::chrono::steady_clock::time_point
KVCacheManagerWithTransfer::DeadlineFromNow() const {
  return std::chrono::steady_clock::now() +
         std::chrono::milliseconds(static_cast<int64_t>(timeout_s_ * 1000.0));
}

void KVCacheManagerWithTransfer::ConfigureDataPortFromKvTransfer() {
  if (num_layers() == 0) {
    local_data_port_ = 0;
    return;
  }
  std::optional<int> data_port = local_port();
  if (!data_port.has_value()) {
    throw std::runtime_error("KVCacheManager BlockTransport is not running");
  }
  local_data_port_ = *data_port;
}

std::vector<int> KVCacheManagerWithTransfer::ContiguousBlockIds(
    uint64_t base, uint64_t count) const {
  if (count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::out_of_range("block count exceeds int range");
  }
  if (base > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      count >
          static_cast<uint64_t>(std::numeric_limits<int>::max()) - base + 1) {
    throw std::out_of_range("block id range exceeds int range");
  }
  std::vector<int> ids;
  ids.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    ids.push_back(static_cast<int>(base + i));
  }
  return ids;
}

std::optional<int> KVCacheManagerWithTransfer::GetLocalTpuNumaNode(
    xla::PjRtBuffer* buf) const {
  if (buf && buf->device()) {
    int node = GetPjRtDeviceNumaNode(buf->device());
    if (node >= 0) {
      return node;
    }
  }
  return std::nullopt;
}

absl::Status KVCacheManagerWithTransfer::OnBlocksReceived(
    const std::vector<int>& block_ids, uint64_t uuid) {
  VLOG(1) << "KVCacheManagerWithTransfer::OnBlocksReceived called. uuid: "
          << uuid << ", received blocks count: " << block_ids.size();

  std::string req_id;
  int64_t recv_slot = -1;
  std::vector<int> recv_staged;
  CopySpec h2d_copy;
  absl::flat_hash_map<kv_cache::HostBlockId, kv_cache::DeviceBlockId>
      host_to_chip;
  bool found = false;
  bool unregister_plan = false;
  uint64_t plan_generation = 0;
  std::vector<int> accumulated_host_blocks;

  std::chrono::steady_clock::time_point start_time;
  bool should_record_duration = false;
  {
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it != active_recv_entries_.end()) {
      if (it->second.is_pool_reshard) {
        return absl::OkStatus();
      }
      it->second.num_completed_blocks += block_ids.size();
      if (it->second.num_completed_blocks == block_ids.size()) {
        if (metrics_collector_) {
          metrics_collector_->RecordFirstPacket(uuid);
        }
      }
      it->second.accumulated_host_block_ids.insert(
          it->second.accumulated_host_block_ids.end(), block_ids.begin(),
          block_ids.end());

      if (it->second.num_completed_blocks >=
          it->second.total_blocks * num_layers()) {
        it->second.network_completed = true;
        req_id = it->second.req_id;
        recv_slot = it->second.slot_idx;
        if (metrics_collector_) {
          metrics_collector_->RecordLastPacket(uuid);
        }
        if (it->second.num_completed_layers == num_layers()) {
          start_time = it->second.start_time;
          should_record_duration = true;

          if (metrics_collector_) {
            metrics_collector_->RecordEnd(uuid);
          }
          found = true;
          recv_staged = std::move(it->second.staged_host_blocks);
          unregister_plan = it->second.unregister_on_settle;
          plan_generation = it->second.plan_generation;
          active_recv_entries_.erase(it);
        }
      } else {
        VLOG(1) << "OnBlocksReceived: Partial blocks received for uuid " << uuid
                << ", completed: " << it->second.num_completed_blocks << " / "
                << it->second.total_blocks * num_layers();
        return absl::OkStatus();
      }
    }
  }

  if (should_record_duration) {
    RecordTransferDuration(
        DurationMs(start_time, std::chrono::steady_clock::now()));
  }

  if (!found) {
    // Forward to base class for direct pull operations
    return RaidenManagerBase::OnBlocksReceived(block_ids, uuid);
  }

  {
    absl::MutexLock lock(mu_);
    done_recving_.insert(req_id);
    ReleaseStagingLocked(recv_slot, &recv_staged);
  }
  if (unregister_plan) UnregisterSettledPlan(uuid, plan_generation);

  LOG(INFO) << "OnBlocksReceived (Network + H2D complete): req_id=" << req_id
            << ", uuid=" << uuid
            << ", numa=" << assigned_numa_node().value_or(-1);
  return absl::OkStatus();
}

absl::Status KVCacheManagerWithTransfer::OnPoolReceived(size_t pool_idx,
                                                        uint64_t uuid) {
  {
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it == active_recv_entries_.end()) {
      return absl::NotFoundError(
          absl::StrCat("no active receiver for UUID ", uuid));
    }
    RecvEntry& entry = it->second;
    if (!entry.is_pool_reshard) {
      // Plan-declared accounting fired for a receiver armed on the legacy
      // path: the plan and the arm disagree. Fail closed instead of guessing.
      return absl::FailedPreconditionError(
          absl::StrCat("pool completion for UUID ", uuid,
                       " but the receiver was armed on the legacy path"));
    }
    if (entry.expected_pool_indices.find(pool_idx) ==
        entry.expected_pool_indices.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "received undeclared pool ", pool_idx, " for UUID ", uuid));
    }
    if (entry.started_pool_indices.find(pool_idx) !=
        entry.started_pool_indices.end()) {
      return absl::AlreadyExistsError(
          absl::StrCat("pool completed more than once: ", pool_idx));
    }
    entry.started_pool_indices.insert(pool_idx);
  }

  LaunchEligiblePoolH2ds(uuid);
  return absl::OkStatus();
}

void KVCacheManagerWithTransfer::LaunchEligiblePoolH2ds(uint64_t uuid) {
  std::vector<std::pair<size_t, std::vector<int64_t>>> to_launch;
  {
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it == active_recv_entries_.end()) return;
    RecvEntry& entry = it->second;
    if (entry.reshard_finalizing) return;
    for (size_t pool_idx : entry.started_pool_indices) {
      if (entry.h2d_launched_pools.count(pool_idx)) continue;
      const auto rank_it = entry.pool_order_ranks.find(pool_idx);
      const int rank =
          rank_it == entry.pool_order_ranks.end() ? 0 : rank_it->second;
      bool prerequisites_uploaded = true;
      for (size_t other : entry.expected_pool_indices) {
        const auto other_it = entry.pool_order_ranks.find(other);
        const int other_rank =
            other_it == entry.pool_order_ranks.end() ? 0 : other_it->second;
        if (other_rank < rank && entry.completed_pool_indices.find(other) ==
                                     entry.completed_pool_indices.end()) {
          prerequisites_uploaded = false;
          break;
        }
      }
      if (!prerequisites_uploaded) continue;
      entry.h2d_launched_pools.insert(pool_idx);
      const auto ids_it = entry.pool_dst_block_ids.find(pool_idx);
      to_launch.emplace_back(pool_idx, ids_it == entry.pool_dst_block_ids.end()
                                           ? entry.chip_block_ids
                                           : ids_it->second);
    }
  }
  for (auto& [pool_idx, chip_block_ids] : to_launch) {
    auto future_or = H2dPoolBlocks(pool_idx, chip_block_ids,
                                   /*shard_idx=*/std::nullopt, uuid);
    if (!future_or.ok()) {
      FinishPoolReshardRecvPool(uuid, pool_idx, future_or.status());
      continue;
    }
    raiden::PjRtCopyFuture future = std::move(future_or).value();
    std::shared_ptr<void> callback_guard = TrackAsyncCallback();
    future.OnReady([this, uuid, pool_idx = pool_idx,
                    callback_guard](auto status_or) mutable {
      std::shared_ptr<void> callback_lifetime = std::move(callback_guard);
      (void)callback_lifetime;
      FinishPoolReshardRecvPool(
          uuid, pool_idx,
          status_or.ok() ? absl::OkStatus() : status_or.status());
    });
    {
      absl::MutexLock lock(mu_);
      auto it = active_recv_entries_.find(uuid);
      if (it != active_recv_entries_.end()) {
        it->second.h2d_futures.push_back(future);
      }
    }
  }
}

void KVCacheManagerWithTransfer::FinishPoolReshardRecvPool(
    uint64_t uuid, size_t pool_idx, const absl::Status& status) {
  bool finished = false;
  {
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it == active_recv_entries_.end()) return;
    RecvEntry& entry = it->second;
    if (entry.reshard_finalizing) return;
    if (!status.ok()) {
      entry.reshard_finalizing = true;
      finished = true;
    } else {
      entry.completed_pool_indices.insert(pool_idx);
      if (entry.completed_pool_indices == entry.expected_pool_indices) {
        entry.reshard_finalizing = true;
        finished = true;
      }
    }
  }
  if (!finished && status.ok()) {
    // A completed upload may unblock deferred higher-order-rank pools.
    LaunchEligiblePoolH2ds(uuid);
  }
  std::chrono::steady_clock::time_point start_time;
  bool should_record_duration = false;
  if (finished) {
    absl::Status unregister = UnregisterActivePlan(uuid);
    if (!unregister.ok() && !absl::IsNotFound(unregister)) {
      LOG(ERROR) << "Failed to unregister pool reshard receiver plan " << uuid
                 << ": " << unregister;
    }
    // All pools uploaded (or the receive failed): the staging bytes have been
    // consumed by the H2D, so the arena slots go back to the pool.
    ReleasePoolStagingLeases(uuid);
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it == active_recv_entries_.end()) return;
    if (!status.ok() || (!unregister.ok() && !absl::IsNotFound(unregister))) {
      failed_recving_.insert(it->second.req_id);
      active_recv_entries_.erase(it);
    } else {
      start_time = it->second.start_time;
      should_record_duration = true;

      it->second.network_completed = true;
      done_recving_.insert(it->second.req_id);
    }
  }
  if (should_record_duration) {
    RecordTransferDuration(
        DurationMs(start_time, std::chrono::steady_clock::now()));
  }
}

absl::Status KVCacheManagerWithTransfer::OnLayerReceived(size_t layer_idx,
                                                         uint64_t uuid) {
  CopySpec h2d_copy;
  std::string req_id;
  int64_t recv_slot;
  bool trigger_enqueue = false;
  {
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it == active_recv_entries_.end()) {
      return absl::OkStatus();
    }
    auto& entry = it->second;
    h2d_copy = entry.h2d_copy;
    req_id = entry.req_id;
    recv_slot = entry.slot_idx;
    if (!entry.h2d_started) {
      entry.h2d_started = true;
      trigger_enqueue = true;
    }
  }
  if (trigger_enqueue && metrics_collector_) {
    metrics_collector_->RecordH2dEnqueue(uuid);
  }

  LOG(INFO) << "OnLayerReceived (H2D copy start) layer " << layer_idx
            << ": req_id=" << req_id << ", uuid=" << uuid
            << ", numa=" << assigned_numa_node().value_or(-1);

  auto future_or = H2dSyncDispatch(h2d_copy.src_offsets, h2d_copy.dst_offsets,
                                   h2d_copy.sizes, /*slot_idx=*/std::nullopt,
                                   /*layer_idx=*/layer_idx);
  if (!future_or.ok()) {
    bool unregister_plan = false;
    uint64_t plan_generation = 0;
    {
      absl::MutexLock lock(mu_);
      failed_recving_.insert(req_id);
      auto it = active_recv_entries_.find(uuid);
      if (it != active_recv_entries_.end()) {
        ReleaseRecvStagingLocked(&it->second);
        unregister_plan = it->second.unregister_on_settle;
        plan_generation = it->second.plan_generation;
        active_recv_entries_.erase(it);
      }
    }
    if (unregister_plan) UnregisterSettledPlan(uuid, plan_generation);
    return future_or.status();
  }

  auto future = future_or.value();
  std::shared_ptr<void> callback_guard = TrackAsyncCallback();
  future.OnReady([this, uuid, layer_idx, recv_slot, req_id,
                  metrics_collector = metrics_collector_,
                  callback_guard](auto status_or) mutable {
    std::shared_ptr<void> callback_lifetime = std::move(callback_guard);
    (void)callback_lifetime;
    bool unregister_plan = false;
    uint64_t plan_generation = 0;
    {
      absl::MutexLock lock(mu_);
      auto it = active_recv_entries_.find(uuid);
      if (it == active_recv_entries_.end()) {
        return;
      }
      auto& entry = it->second;
      if (status_or.ok()) {
        LOG(INFO) << "OnLayerReceived (H2D copy complete) layer " << layer_idx
                  << ": req_id=" << req_id
                  << ", numa=" << assigned_numa_node().value_or(-1);
        entry.num_completed_layers++;
        if (entry.num_completed_layers == num_layers()) {
          // TODO: Find a way to optimize this by moving out of the mutex.
          RecordTransferDuration(
              DurationMs(entry.start_time, std::chrono::steady_clock::now()));

          if (metrics_collector) {
            metrics_collector->RecordH2dComplete(uuid);
          }
          LOG(INFO) << "All layers H2D copy complete: req_id=" << req_id;
          if (metrics_collector) {
            metrics_collector->RecordEnd(uuid);
          }
          done_recving_.insert(req_id);
          ReleaseRecvStagingLocked(&entry);
          unregister_plan = entry.unregister_on_settle;
          plan_generation = entry.plan_generation;
          active_recv_entries_.erase(uuid);
        }
      } else {
        LOG(ERROR) << "OnLayerReceived (H2D copy failed) layer " << layer_idx
                   << " for req_id: " << req_id
                   << ", error: " << status_or.status().ToString();
        failed_recving_.insert(req_id);
        ReleaseRecvStagingLocked(&entry);
        unregister_plan = entry.unregister_on_settle;
        plan_generation = entry.plan_generation;
        active_recv_entries_.erase(uuid);
      }
    }
    if (unregister_plan) UnregisterSettledPlan(uuid, plan_generation);
  });

  {
    absl::MutexLock lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it != active_recv_entries_.end()) {
      it->second.h2d_futures.push_back(future);
    }
  }

  return absl::OkStatus();
}

}  // namespace tpu_raiden
