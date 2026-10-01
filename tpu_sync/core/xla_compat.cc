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

#include "tpu_sync/core/xla_compat.h"

#include <memory>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_external.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace raiden {
namespace {

// Converts the raw buffer pointer from a tracked device buffer hold to
// RawBuffer*.
#if RAIDEN_JAX >= 1002 && RAIDEN_JAX < 1100
template <typename T>
RawBuffer* ToRawBuffer(T* buffer) {
  // jax 0.10.2 only: PjRtRawBufferInterface and PjRtRawBuffer are siblings
  // deriving from PJRT_RawBuffer. Cast through the shared base with
  // static_assert.
  static_assert(sizeof(RawBuffer) == sizeof(PJRT_RawBuffer),
                "PjRtRawBufferInterface adds state to PJRT_RawBuffer at this "
                "revision, so it can no longer be reached by reinterpreting "
                "the shared base.");
  return reinterpret_cast<RawBuffer*>(static_cast<PJRT_RawBuffer*>(buffer));
}
#else
template <typename T>
RawBuffer* ToRawBuffer(T* buffer) {
  static_assert(std::is_convertible_v<T*, RawBuffer*>,
                "Buffer type is not convertible to RawBuffer*");
  return buffer;
}
#endif

// In JAX 0.11.0, CommonPjRtBuffer::ScopedHold is 40 bytes (a raw
// AbstractTrackedDeviceBuffer* at offset 24 plus a unique_ptr at offset 32,
// which is nullptr for kUsage holds). In JAX 0.11.1+, those two fields were
// merged into an 8-byte tsl::MaybeOwning pointer at offset 24, shrinking
// ScopedHold to 32 bytes. Constructing in-place inside PaddedScopedHold via
// C++17 guaranteed copy elision provides 8 bytes of trailing padding for 0.11.0
// RVO and avoids calling ScopedHold's move constructor across ABI boundaries.
struct PaddedScopedHold {
  xla::CommonPjRtBuffer::ScopedHold hold;
  void* abi_pad_0_11_0 = nullptr;

  explicit PaddedScopedHold(xla::CommonPjRtBuffer* common_buffer)
      : hold(common_buffer->GetBufferWithHold(
            xla::CommonPjRtBuffer::ScopedHold::kUsage)) {}
};

}  // namespace

BufferKind ClassifyBuffer(const xla::PjRtBuffer* buffer) {
  if (dynamic_cast<const xla::CommonPjRtBuffer*>(buffer) != nullptr) {
    return BufferKind::kCommon;
  }
  if (dynamic_cast<const xla::PjRtCApiBuffer*>(buffer) != nullptr) {
    return BufferKind::kCApi;
  }
  return BufferKind::kUnsupported;
}

absl::StatusOr<CommonBufferAcquisition> AcquireCommonRawBuffer(
    xla::PjRtBuffer* buffer, bool unsafe_skip_buffer_lock) {
  auto* common_buffer = dynamic_cast<xla::CommonPjRtBuffer*>(buffer);
  if (common_buffer == nullptr) {
    return absl::InvalidArgumentError("Not a CommonPjRtBuffer");
  }

  auto hold_wrapper = std::make_shared<PaddedScopedHold>(common_buffer);
  if (!hold_wrapper->hold.ok()) {
    return hold_wrapper->hold.status();
  }

  CommonBufferAcquisition result;
  result.raw_buffer = tsl::FormRef(
      ToRawBuffer(hold_wrapper->hold.buffer()->raw_buffer().get()));
  if (!unsafe_skip_buffer_lock) {
    result.hold = ScopedHold(std::move(hold_wrapper));
  }
  return result;
}

absl::StatusOr<PJRT_RawBuffer*> CreateCApiRawAlias(
    xla::PjRtBuffer* buffer, const PJRT_Api* c_api,
    const PJRT_RawBuffer_Extension* extension) {
  auto* capi_buffer = dynamic_cast<xla::PjRtCApiBuffer*>(buffer);
  if (capi_buffer == nullptr) {
    return absl::InvalidArgumentError("Not a PjRtCApiBuffer");
  }
  return pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(c_api, extension,
                                                     capi_buffer->c_buffer());
}

const PJRT_RawBuffer_Extension* GetRawBufferExtension(
    const xla::PjRtBuffer* buffer, const PJRT_Api** out_c_api) {
  auto* capi_buffer = dynamic_cast<const xla::PjRtCApiBuffer*>(buffer);
  if (capi_buffer == nullptr) return nullptr;
  if (out_c_api != nullptr) *out_c_api = capi_buffer->pjrt_c_api();
  auto* capi_client = dynamic_cast<xla::PjRtCApiClient*>(
      const_cast<xla::PjRtClient*>(capi_buffer->client()));
  if (capi_client == nullptr) return nullptr;
  return capi_client->FindExtension<PJRT_RawBuffer_Extension>(
      PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
}

}  // namespace raiden
