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

// Converts the raw buffer pointer from the tracked device buffer hold into
// a RawBuffer*. The static_assert ensures compile-time type safety.
template <typename T>
RawBuffer* ToRawBuffer(T* buffer) {
  static_assert(std::is_convertible_v<T*, RawBuffer*>,
                "Tracked device buffer hold return type is not convertible to "
                "RawBuffer*; a compatibility specialization is required.");
  return buffer;
}

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

  auto hold = common_buffer->GetBufferWithHold(
      xla::CommonPjRtBuffer::ScopedHold::kUsage);
  if (!hold.ok()) {
    return hold.status();
  }

  CommonBufferAcquisition result;
  result.raw_buffer =
      tsl::FormRef(ToRawBuffer(hold.buffer()->raw_buffer().get()));
  if (!unsafe_skip_buffer_lock) {
    result.hold = ScopedHold(
        std::make_shared<xla::CommonPjRtBuffer::ScopedHold>(std::move(hold)));
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
