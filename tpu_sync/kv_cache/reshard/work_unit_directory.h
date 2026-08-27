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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_WORK_UNIT_DIRECTORY_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_WORK_UNIT_DIRECTORY_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

// Decoded register_work_unit arguments: std::nullopt reproduces Python's
// None (the framed server maps absent/zero wire fields to None before
// calling the controller; the same mapping feeds this struct).
struct WorkUnitRegistration {
  RaidenId unit;
  std::vector<std::string> shards;
  std::optional<std::string> control_plane_rpc_address;
  std::optional<std::vector<int64_t>> mesh_shape;
  std::optional<std::vector<int32_t>> layout;
  std::optional<std::vector<int64_t>> global_shape;
  std::optional<int32_t> itemsize;
  std::optional<std::vector<tpu_sync::rpc::PoolSpecProto>> pool_manifest;
  std::optional<std::string> layout_fingerprint;
  std::optional<int64_t> page_tokens;
  std::optional<int32_t> transfer_parallelism;
  std::optional<int32_t> transfer_rank;
  std::optional<std::vector<tpu_sync::rpc::VariableMetadataProto>> variables;
};

// Port of RaidenController's work-unit registration surface. All state
// transitions run under an injected controller-wide mutex, reproducing the
// Python controller's single-lock atomicity between registration and the
// request-block registry (a worker restart invalidates that unit's request
// blocks inside the same critical section — the service wires the two
// modules under one lock).
class WorkUnitDirectory {
 public:
  // `mu` is the shared controller mutex owned by ReshardService.
  explicit WorkUnitDirectory(absl::Mutex* mu) : mu_(mu) {}

  // Pre-lock validation half of register_work_unit: everything Python
  // checks before taking self._lock, byte-identical error strings.
  static absl::Status Validate(const WorkUnitRegistration& reg);

  // The locked half: replacement-not-patch registration. Caller must hold
  // the shared mutex and must have run Validate().
  void RegisterLocked(const WorkUnitRegistration& reg)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // _metadata_proto_locked: rebuilds the wire metadata proto field-for-field
  // in Python's field order. Caller must hold the shared mutex.
  absl::StatusOr<tpu_sync::rpc::RegisterWorkUnitRequest> MetadataProtoLocked(
      const RaidenId& unit) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // get_all_metadata (registration insertion order).
  std::vector<tpu_sync::rpc::RegisterWorkUnitRequest> AllMetadata() const;

  // _get_local_metadata: metadata for exactly these units.
  absl::StatusOr<std::vector<tpu_sync::rpc::RegisterWorkUnitRequest>>
  LocalMetadata(const std::vector<RaidenId>& units) const;

  bool IsRegisteredLocked(const RaidenId& unit) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Live pool manifest view for register_request_blocks' per-tag live-byte
  // check. Empty when the unit has no manifest.
  std::vector<tpu_sync::rpc::PoolSpecProto> PoolManifestLocked(
      const RaidenId& unit) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::Mutex* mu() const ABSL_LOCK_RETURNED(mu_) { return mu_; }

 private:
  struct Record {
    std::vector<std::string> shards;
    // Mirrors worker_rpc_client._endpoints for this unit ("" = absent).
    std::string control_plane_rpc_address;
    std::optional<std::vector<int64_t>> mesh_shape;
    std::optional<std::vector<int32_t>> layout;
    std::optional<std::vector<int64_t>> global_shape;
    std::optional<int32_t> itemsize;
    std::optional<std::vector<tpu_sync::rpc::PoolSpecProto>> pools;
    std::optional<std::string> layout_fingerprint;
    std::optional<int64_t> page_tokens;
    std::optional<int32_t> transfer_parallelism;
    std::optional<int32_t> transfer_rank;
    std::optional<std::vector<tpu_sync::rpc::VariableMetadataProto>> variables;
  };

  int FindLocked(const RaidenId& unit) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::Mutex* mu_;
  // Insertion-ordered registry (Python dict semantics; replacement keeps
  // position).
  std::vector<RaidenId> order_ ABSL_GUARDED_BY(mu_);
  std::vector<Record> records_ ABSL_GUARDED_BY(mu_);
};

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_WORK_UNIT_DIRECTORY_H_
