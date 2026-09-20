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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_SESSION_H_

#include "absl/status/status.h"

namespace tpu_raiden {

// Base interface for producer and consumer KV-cache transfer sessions.
//
// Implementations are thread-safe and synchronize session lifecycle state
// internally.
class TransferSession {
 public:
  virtual ~TransferSession() = default;

  // Returns true once the session's outcome has been decided and all in-flight
  // operations have drained.
  virtual bool Done() const = 0;

  // Records the session's outcome with |status| and transitions the session
  // into draining (or marks it done immediately if no operations are in
  // flight).
  virtual void Finish(const absl::Status& status = absl::OkStatus()) = 0;

  // Returns the session's status (OkStatus() unless the session has failed).
  virtual absl::Status GetStatus() const = 0;

  // Blocks until Done() becomes true and returns GetStatus().
  virtual absl::Status AwaitForDone() = 0;

  // Returns true if the session's outcome has been decided and it is draining
  // in-flight operations (or has finished draining).
  virtual bool IsDraining() const = 0;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_SESSION_H_
