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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_DECLARATION_TYPES_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_DECLARATION_TYPES_H_

#include <cstdint>
#include <string>
#include <vector>

#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

// Value-type mirrors of rpc/raiden_controller.py's declaration IR
// dataclasses. Wire schema lives in rpc/controller_service.proto; these are
// the planner-friendly PODs (converted to protos only at emission).

// One contiguous (or uniformly strided) byte copy between one source and
// one destination block of the same pool. Ordinals index the owning
// registration's block_ids; the destination index addresses the transfer's
// destination id list — never physical ids.
struct PoolByteSpan {
  int64_t src_block_ordinal = 0;
  int64_t src_offset_bytes = 0;
  int64_t dst_block_index = 0;
  int64_t dst_offset_bytes = 0;
  int64_t size_bytes = 0;
  int64_t src_stride_bytes = 0;
  int64_t dst_stride_bytes = 0;
  int32_t count = 1;

  friend bool operator==(const PoolByteSpan& a, const PoolByteSpan& b) {
    return a.src_block_ordinal == b.src_block_ordinal &&
           a.src_offset_bytes == b.src_offset_bytes &&
           a.dst_block_index == b.dst_block_index &&
           a.dst_offset_bytes == b.dst_offset_bytes &&
           a.size_bytes == b.size_bytes &&
           a.src_stride_bytes == b.src_stride_bytes &&
           a.dst_stride_bytes == b.dst_stride_bytes && a.count == b.count;
  }
};

// One rank's declared byte-span contribution to one pool tag.
struct PoolSpanRegistration {
  std::string tag;
  std::vector<int64_t> block_ids;
  std::vector<PoolByteSpan> spans;
  int64_t declared_bytes = 0;
  // 0 (legacy): page-indexed destination spans. 1: request-global compact
  // destination byte space; the controller performs the page split.
  int32_t dst_space_version = 0;

  friend bool operator==(const PoolSpanRegistration& a,
                         const PoolSpanRegistration& b) {
    return a.tag == b.tag && a.block_ids == b.block_ids && a.spans == b.spans &&
           a.declared_bytes == b.declared_bytes &&
           a.dst_space_version == b.dst_space_version;
  }
};

// Producer-owned block IDs and byte-span declarations for one request and
// unit.
struct RequestBlockRegistration {
  int64_t uuid = 0;
  std::vector<int64_t> block_ids;
  double expires_at = 0.0;
  std::vector<PoolSpanRegistration> pool_spans;
};

// Formats a RaidenId exactly as Python's frozen dataclass repr renders it:
// RaidenId(job_name='p', job_replica_id='0', data_name='kv',
// data_replica_idx=0). Several wire-contract error strings embed this.
std::string PythonRepr(const RaidenId& id);

// Python str() of a list of Python-repr'd values, e.g. "[1, 2, 3]".
std::string PythonReprIntList(const std::vector<int64_t>& values);

RaidenId RaidenIdFromProto(const tpu_sync::rpc::RaidenIdProto& proto);
tpu_sync::rpc::RaidenIdProto RaidenIdToProto(const RaidenId& id);

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_DECLARATION_TYPES_H_
