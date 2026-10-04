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

#include "tpu_sync/weight_sync/manager/v3/test_pull_wire_codec.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/weight_sync/manager/v3/dynamic_pull_engine.h"
#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

constexpr int64_t kInfinite = std::numeric_limits<int64_t>::max();

class Writer {
 public:
  template <typename T>
  void Fixed(T value) {
    out_.append(reinterpret_cast<const char*>(&value), sizeof(value));
  }
  void String(absl::string_view s) {
    Fixed<uint32_t>(static_cast<uint32_t>(s.size()));
    out_.append(s.data(), s.size());
  }
  void Unit(const RaidenId& unit) {
    String(unit.job_name);
    String(unit.job_replica_id);
    String(unit.data_name);
    Fixed<int32_t>(unit.data_replica_idx);
  }
  void Status(const absl::Status& status) {
    Fixed<int32_t>(static_cast<int32_t>(status.code()));
    String(status.message());
  }
  std::string Take() { return std::move(out_); }

 private:
  std::string out_;
};

class Reader {
 public:
  explicit Reader(absl::string_view in) : in_(in) {}

  template <typename T>
  bool Fixed(T& value) {
    if (in_.size() < sizeof(T)) return false;
    std::memcpy(&value, in_.data(), sizeof(T));
    in_.remove_prefix(sizeof(T));
    return true;
  }
  bool String(std::string& s) {
    uint32_t size = 0;
    if (!Fixed(size) || in_.size() < size) return false;
    s.assign(in_.data(), size);
    in_.remove_prefix(size);
    return true;
  }
  bool Unit(RaidenId& unit) {
    return String(unit.job_name) && String(unit.job_replica_id) &&
           String(unit.data_name) && Fixed(unit.data_replica_idx);
  }
  bool Status(absl::Status& status) {
    int32_t code = 0;
    std::string message;
    if (!Fixed(code) || !String(message)) return false;
    status = absl::Status(static_cast<absl::StatusCode>(code), message);
    return true;
  }
  bool done() const { return in_.empty(); }

 private:
  absl::string_view in_;
};

absl::Status Malformed(absl::string_view what) {
  return absl::InvalidArgumentError(absl::StrCat("Malformed test pull ", what));
}

}  // namespace

std::string TestPullWireCodec::EncodeRequest(
    const PullServiceRequest& request) {
  Writer w;
  w.String(request.req_id);
  w.Fixed<uint64_t>(request.uuid);
  w.Unit(request.unit);
  w.Fixed<int32_t>(request.host_idx);
  w.Fixed<int32_t>(request.max_grants);
  w.Fixed<int64_t>(request.long_poll == absl::InfiniteDuration()
                       ? kInfinite
                       : absl::ToInt64Nanoseconds(request.long_poll));
  w.Fixed<uint8_t>(request.seeded ? 1 : 0);
  w.Fixed<uint8_t>(request.lost_data ? 1 : 0);
  w.Fixed<uint32_t>(static_cast<uint32_t>(request.completed.size()));
  for (uint64_t lease : request.completed) w.Fixed<uint64_t>(lease);
  w.Fixed<uint32_t>(static_cast<uint32_t>(request.failed.size()));
  for (const FailedPull& failed : request.failed) {
    w.Fixed<uint64_t>(failed.lease_id);
    w.String(failed.reason);
  }
  return w.Take();
}

absl::StatusOr<PullServiceRequest> TestPullWireCodec::DecodeRequest(
    absl::string_view payload) const {
  Reader r(payload);
  PullServiceRequest request;
  int64_t long_poll_ns = 0;
  uint8_t seeded = 0;
  uint8_t lost_data = 0;
  uint32_t num_completed = 0;
  if (!r.String(request.req_id) || !r.Fixed(request.uuid) ||
      !r.Unit(request.unit) || !r.Fixed(request.host_idx) ||
      !r.Fixed(request.max_grants) || !r.Fixed(long_poll_ns) ||
      !r.Fixed(seeded) || !r.Fixed(lost_data) || !r.Fixed(num_completed)) {
    return Malformed("request");
  }
  request.long_poll = long_poll_ns == kInfinite
                          ? absl::InfiniteDuration()
                          : absl::Nanoseconds(long_poll_ns);
  request.seeded = seeded != 0;
  request.lost_data = lost_data != 0;
  for (uint32_t i = 0; i < num_completed; ++i) {
    uint64_t lease = 0;
    if (!r.Fixed(lease)) return Malformed("request");
    request.completed.push_back(lease);
  }
  uint32_t num_failed = 0;
  if (!r.Fixed(num_failed)) return Malformed("request");
  for (uint32_t i = 0; i < num_failed; ++i) {
    FailedPull failed;
    if (!r.Fixed(failed.lease_id) || !r.String(failed.reason)) {
      return Malformed("request");
    }
    request.failed.push_back(std::move(failed));
  }
  if (!r.done()) return Malformed("request");
  return request;
}

std::string TestPullWireCodec::EncodeReply(
    const PullServiceReply& reply) const {
  Writer w;
  w.Fixed<int32_t>(static_cast<int32_t>(reply.kind));
  w.Status(reply.status);
  w.Fixed<uint32_t>(static_cast<uint32_t>(reply.grants.size()));
  for (const PhysicalPullGrant& grant : reply.grants) {
    w.Fixed<int32_t>(grant.bundle_id);
    w.Fixed<int32_t>(grant.source_replica);
    w.Unit(grant.source_unit);
    w.String(grant.source_data_endpoint);
    w.Fixed<uint64_t>(grant.lease_id);
    w.Fixed<int64_t>(absl::ToUnixNanos(grant.expiry));
  }
  return w.Take();
}

absl::StatusOr<PullServiceReply> TestPullWireCodec::DecodeReply(
    absl::string_view payload) {
  Reader r(payload);
  PullServiceReply reply;
  int32_t kind = 0;
  uint32_t num_grants = 0;
  if (!r.Fixed(kind) || !r.Status(reply.status) || !r.Fixed(num_grants)) {
    return Malformed("reply");
  }
  reply.kind = static_cast<PullReplyKind>(kind);
  reply.grants.resize(num_grants);
  for (PhysicalPullGrant& grant : reply.grants) {
    int64_t expiry_ns = 0;
    if (!r.Fixed(grant.bundle_id) || !r.Fixed(grant.source_replica) ||
        !r.Unit(grant.source_unit) || !r.String(grant.source_data_endpoint) ||
        !r.Fixed(grant.lease_id) || !r.Fixed(expiry_ns)) {
      return Malformed("reply");
    }
    grant.expiry = absl::FromUnixNanos(expiry_ns);
  }
  if (!r.done()) return Malformed("reply");
  return reply;
}

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden
