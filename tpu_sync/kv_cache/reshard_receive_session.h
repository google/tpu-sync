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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_RECEIVE_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_RECEIVE_SESSION_H_

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

// Creates the consumer side of a multi-tag pool-reshard receive operation for
// |plan|: it receives each pool's pushes into bounded host staging and uploads
// completed pools to |chip_blocks| on device in order-rank order. Validates
// |plan| and acquires bounded pool staging leases per storage via
// |staging_allocator|. The returned session is thread-safe.
absl::StatusOr<std::shared_ptr<TransferSession>> CreateReshardReceiveSession(
    kv_cache::KVCacheManagerBase* base,
    StagingBlockAllocator* absl_nullable staging_allocator,
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    absl::Span<const int64_t> chip_blocks,
    std::chrono::steady_clock::time_point deadline);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_RESHARD_RECEIVE_SESSION_H_
