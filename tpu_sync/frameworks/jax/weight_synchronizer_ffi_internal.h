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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_WEIGHT_SYNCHRONIZER_FFI_INTERNAL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_WEIGHT_SYNCHRONIZER_FFI_INTERNAL_H_

#include <cstdint>
#include <memory>

#include "absl/types/span.h"
#include "xla/ffi/api/ffi.h"
#include "xla/stream_executor/stream.h"
#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi.h"

namespace tpu_raiden {
namespace weight_sync {

extern std::unique_ptr<stream_executor::Stream> g_streams[kMaxShards];
void ClearSharedWsMap();

xla::ffi::Error TriggerWeightSynchronizerInitImpl(
    xla::ffi::AnyBuffer x, xla::ffi::AnyBuffer shard_idx_buf,
    xla::ffi::AnyBuffer slice_byte_sizes_buf, int32_t local_port,
    int32_t parallelism, int32_t num_layers, int32_t listener_port,
    int32_t num_shards, xla::ffi::Result<xla::ffi::AnyBuffer> out);

xla::ffi::Error TriggerWeightSynchronizerInitAndD2hHelper(
    xla::ffi::AnyBuffer shard_idx_buf, xla::ffi::AnyBuffer slice_byte_sizes_buf,
    absl::Span<const xla::ffi::AnyBuffer> jax_arrays, int32_t local_port,
    int32_t parallelism, int32_t num_layers, int32_t listener_port,
    int32_t num_shards, xla::ffi::Result<xla::ffi::AnyBuffer> out);

xla::ffi::Error TriggerMultiH2DImpl(xla::ffi::AnyBuffer shard_idx_buf,
                                    xla::ffi::RemainingRets rets);

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_WEIGHT_SYNCHRONIZER_FFI_INTERNAL_H_
