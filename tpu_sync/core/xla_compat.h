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

// Encapsulates version-fragile XLA PJRT types for TPU Raiden.
//
// XLA's raw-buffer hierarchy, tracked-device-buffer holds, and C-API client
// classes evolve across JAX/XLA releases. To prevent leaking these fragile
// headers across the codebase, all access is confined to this compatibility
// layer.
//
// Only xla_compat.{h,cc} may include:
//   - xla/pjrt/raw_buffer.h
//   - xla/pjrt/abstract_tracked_device_buffer.h
//   - xla/pjrt/c_api_client/pjrt_c_api_client.h

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_XLA_COMPAT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_XLA_COMPAT_H_

#include <cstddef>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
class PjRtBuffer;
}  // namespace xla

namespace raiden {

// The stable raw-buffer type for Raiden transfers.
//
// XLA's internal PjRtRawBufferRef alias varies across revisions (resolving to
// RCReference<CommonPjRtRawBuffer> vs. RCReference<PjRtRawBufferInterface>).
// Raiden uses RawBuffer to provide a uniform alias across supported versions.
// The transfer methods (memory_space, GetHostPointer, GetOnDeviceSizeInBytes,
// CopyRawHostToDevice, CopyRawDeviceToHost) share identical names and
// signatures.
using RawBuffer = xla::PjRtRawBufferInterface;
using RawBufferRef = tsl::RCReference<RawBuffer>;

// Type-erased wrapper for CommonPjRtBuffer::ScopedHold.
//
// CommonPjRtBuffer::ScopedHold is a nested class and cannot be forward-declared
// in C++. Holding it directly would force abstract_tracked_device_buffer.h into
// all headers including raw_transfer_core.h. Because callers only need to
// manage the hold's lifetime during transfer execution, ScopedHold wraps an
// opaque shared pointer to preserve RAII semantics without header leakage.
class ScopedHold {
 public:
  ScopedHold() = default;
  // Allows callers and default arguments to pass nullptr for an empty hold.
  ScopedHold(std::nullptr_t) {}  // NOLINT(google-explicit-constructor)
  explicit ScopedHold(std::shared_ptr<void> impl) : impl_(std::move(impl)) {}

  bool is_valid() const { return impl_ != nullptr; }
  explicit operator bool() const { return is_valid(); }
  void reset() { impl_.reset(); }

 private:
  std::shared_ptr<void> impl_;
};

// Transfer path supported by a PjRtBuffer.
enum class BufferKind {
  kUnsupported,
  kCommon,  // In-process C++ path (xla::CommonPjRtBuffer).
  kCApi,    // PJRT C-API plugin path (xla::PjRtCApiBuffer).
};

// Identifies whether buffer is CommonPjRtBuffer or PjRtCApiBuffer.
BufferKind ClassifyBuffer(const xla::PjRtBuffer* buffer);

struct CommonBufferAcquisition {
  RawBufferRef raw_buffer;
  ScopedHold hold;
};

// Acquires a usage hold and raw buffer from a CommonPjRtBuffer.
// If unsafe_skip_buffer_lock is true, the hold is dropped before returning.
absl::StatusOr<CommonBufferAcquisition> AcquireCommonRawBuffer(
    xla::PjRtBuffer* buffer, bool unsafe_skip_buffer_lock = false);

// Creates a raw alias of a C-API buffer. Caller owns the returned
// PJRT_RawBuffer* and must release it via pjrt::PjRtCApiRawBuffer_Destroy.
absl::StatusOr<PJRT_RawBuffer*> CreateCApiRawAlias(
    xla::PjRtBuffer* buffer, const PJRT_Api* c_api,
    const PJRT_RawBuffer_Extension* extension);

// Returns the RawBuffer C-API extension for the buffer's client, or nullptr
// if unsupported. If out_c_api is provided, populates the PJRT_Api pointer.
const PJRT_RawBuffer_Extension* GetRawBufferExtension(
    const xla::PjRtBuffer* buffer, const PJRT_Api** out_c_api = nullptr);

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_XLA_COMPAT_H_
