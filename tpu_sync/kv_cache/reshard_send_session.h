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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_SEND_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_SEND_SESSION_H_

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/transfer_session.h"

namespace tpu_raiden {

namespace kv_cache {
class KVCacheManagerBase;
}  // namespace kv_cache

class StagingBlockAllocator;

// Creates the producer side of a single push-driven pool-reshard send transfer
// (identified by |plan|'s uuid): once started, it copies each transferred
// pool's |src_block_ids| from device to host staging and pushes them to the
// plan's peers. Validates the sender schedule in |plan|, resolves the data
// connection policy (TPU_RAIDEN_DATA_CONN_REUSE) and initializes the transport
// server. The returned session is thread-safe.
absl::StatusOr<std::shared_ptr<TransferSession>> CreateReshardSendSession(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator,
    absl::Span<const int64_t> src_block_ids, int parallelism,
    std::chrono::steady_clock::time_point deadline,
    ::tpu_sync::rpc::StartTransferRequest plan);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_SEND_SESSION_H_
