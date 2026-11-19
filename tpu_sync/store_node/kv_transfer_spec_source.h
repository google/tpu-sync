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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_STORE_NODE_KV_TRANSFER_SPEC_SOURCE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_STORE_NODE_KV_TRANSFER_SPEC_SOURCE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace tpu_raiden {
namespace store_node {

// The subset of the global registry's KVTransferSpec that a host store node
// consumes; see global_registry.proto for the field semantics. The authority
// for these values is the serving hosts' manager runtime; the node only
// receives them, it never derives them.
struct KVTransferSpec {
  // Bytes one block occupies in each block array on one shard, in
  // registration order. Empty when the workers' block arrays differ.
  std::vector<uint64_t> block_array_bytes;
  // Device shards each block array is split across.
  size_t num_kv_shards = 0;
  // Transfer workers on the serving hosts, node ids densely [0, num_workers).
  size_t num_workers = 0;
  // Set instead of block_array_bytes when the workers' block arrays differ:
  // entry i is the block_array_bytes of the worker with node id i.
  std::vector<std::vector<uint64_t>> worker_block_array_bytes;
};

// The block arrays of the worker with node id `node_id`. Requires a spec that
// passed ValidateSpec and node_id < spec.num_workers.
inline const std::vector<uint64_t>& WorkerBlockArrayBytes(
    const KVTransferSpec& spec, size_t node_id) {
  if (spec.worker_block_array_bytes.empty()) return spec.block_array_bytes;
  CHECK_LT(node_id, spec.worker_block_array_bytes.size());
  return spec.worker_block_array_bytes[node_id];
}

// Returns InvalidArgument unless the spec describes at least one block array
// for every worker, in exactly one of its two forms, and every quantity is
// positive.
inline absl::Status ValidateSpec(const KVTransferSpec& spec) {
  if (spec.block_array_bytes.empty() == spec.worker_block_array_bytes.empty()) {
    return absl::InvalidArgumentError(
        "KVTransferSpec must set exactly one of block_array_bytes and "
        "worker_block_array_bytes");
  }
  if (!spec.worker_block_array_bytes.empty() &&
      spec.worker_block_array_bytes.size() != spec.num_workers) {
    return absl::InvalidArgumentError(absl::StrCat(
        "KVTransferSpec has ", spec.worker_block_array_bytes.size(),
        " worker geometries for ", spec.num_workers, " workers"));
  }
  const size_t geometries = spec.worker_block_array_bytes.empty()
                                ? 1
                                : spec.worker_block_array_bytes.size();
  for (size_t w = 0; w < geometries; ++w) {
    const std::vector<uint64_t>& arrays = WorkerBlockArrayBytes(spec, w);
    if (arrays.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("KVTransferSpec worker ", w, " has no block arrays"));
    }
    for (size_t i = 0; i < arrays.size(); ++i) {
      if (arrays[i] == 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("KVTransferSpec worker ", w, " block_array_bytes[", i,
                         "] must be positive"));
      }
    }
  }
  if (spec.num_kv_shards == 0) {
    return absl::InvalidArgumentError(
        "KVTransferSpec num_kv_shards must be positive");
  }
  if (spec.num_workers == 0) {
    return absl::InvalidArgumentError(
        "KVTransferSpec num_workers must be positive");
  }
  return absl::OkStatus();
}

// Where a booting host store node obtains the deployment's KVTransferSpec.
//
// Get() contract:
//  - OK: the spec is known. The value is fixed for the lifetime of the
//    deployment; callers read it once at boot.
//  - NotFound: nothing published yet. Expected during turnup, when the store
//    node can come up before any serving host has published its spec; the
//    caller retries.
//  - Unavailable: the source itself is not reachable yet (also expected
//    during turnup); the caller retries.
//  - anything else: fatal.
class KVTransferSpecSource {
 public:
  virtual ~KVTransferSpecSource() = default;

  virtual absl::StatusOr<KVTransferSpec> Get() = 0;
};

// KVTransferSpec fixed at construction. The production source is
// GrsKVTransferSpecSource, which polls the spec the serving hosts publish
// in the global registry; this one serves tests and registry-less runs.
class StaticKVTransferSpecSource : public KVTransferSpecSource {
 public:
  explicit StaticKVTransferSpecSource(KVTransferSpec spec) : spec_(spec) {}

  absl::StatusOr<KVTransferSpec> Get() override { return spec_; }

 private:
  KVTransferSpec spec_;
};

}  // namespace store_node
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_STORE_NODE_KV_TRANSFER_SPEC_SOURCE_H_
