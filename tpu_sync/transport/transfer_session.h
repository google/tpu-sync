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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_TRANSFER_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_TRANSFER_SESSION_H_

#include <cstddef>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"

namespace tpu_raiden {

class TransferPlan;

// Base interface for producer and consumer KV-cache transfer sessions.
//
// Implementations are thread-safe and synchronize session lifecycle state
// internally.
class TransferSession {
 public:
  virtual ~TransferSession() = default;

  // Begins the work this side initiates, once the session is registered with
  // its manager. Sessions that only wait for a peer (passive receives, and
  // send sessions driven by the consumer's pull) return OkStatus(). Call at
  // most once, without holding the manager's lock. Thread-safe.
  virtual absl::Status Start() = 0;

  // Returns true once the session's outcome has been decided and all in-flight
  // operations have drained.
  virtual bool Done() const = 0;

  // Records the session's outcome with |status| and transitions the session
  // into draining (or marks it done immediately if no operations are in
  // flight).
  virtual void Finish(const absl::Status& status = absl::OkStatus()) = 0;

  // Attempts to admit an incoming push on this session, returning true if the
  // session accepts it and false if it is draining or done, or never accepts
  // pushes (send sessions). Each successful TryBeginOp() must be paired with a
  // matching EndOp().
  virtual bool TryBeginOp() = 0;

  // Marks one in-flight operation started via TryBeginOp() as complete,
  // settling the session if it is draining and no operations remain in flight.
  virtual void EndOp() = 0;

  // Returns the session's status (OkStatus() unless the session has failed).
  virtual absl::Status GetStatus() const = 0;

  // Blocks until Done() becomes true and returns GetStatus().
  virtual absl::Status AwaitForDone() = 0;

  // Returns true if the session's outcome has been decided and it is draining
  // in-flight operations (or has finished draining).
  virtual bool IsDraining() const = 0;

  // Returns the host staging buffer whose bytes chunk resolution addresses for
  // |unit| (a layer index on the layer transfer path, a pool index on the pool
  // reshard path). Send sessions return the buffer their device-to-host copy
  // already leased; receive sessions lease one on first use, waiting until the
  // session deadline while every buffer is leased. kNotFound means the unit
  // has no staging buffer and is addressed in place. Thread-safe.
  virtual absl::StatusOr<size_t> ResolveStagingSlot(size_t unit) = 0;

  // Reports received data for this session, as routed by its manager. With
  // empty |minor_dimension_indices|, unit |major_dimension_idx| (a layer index
  // on the layer transfer path, a pool index on the pool reshard path,
  // matching ResolveStagingSlot()) has fully landed in host staging, and
  // receive sessions start its host-to-device or pool work. With non-empty
  // |minor_dimension_indices|, it is a block report that |shard_ids| of those
  // destination blocks have landed; it completes no unit and
  // |major_dimension_idx| is ignored. Send sessions return
  // kFailedPrecondition. Thread-safe.
  virtual absl::Status OnDataReceived(
      size_t major_dimension_idx, absl::Span<const int> minor_dimension_indices,
      absl::Span<const int> shard_ids) = 0;

  // Returns a future that resolves once the data of unit |major_dimension_idx|
  // (a layer index on the layer transfer path, a pool index on the pool
  // reshard path, matching ResolveStagingSlot()) is ready in host memory for
  // the transport to read. Send sessions that stage the unit with a
  // device-to-host copy resolve it with that copy's status, or with the
  // send's failure if the copy is never issued. A session holding no pending
  // copy for the unit (receive sessions, a send whose copy has not been set
  // up or whose staging was released) returns an already-OK future.
  // Thread-safe.
  virtual tsl::Future<> GetDataReadyFuture(size_t major_dimension_idx) = 0;

  // Returns the session's active transfer plan, or absl::nullopt if this
  // session is planless or done.
  virtual absl::optional<std::shared_ptr<const TransferPlan>> plan() const = 0;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_TRANSFER_SESSION_H_
