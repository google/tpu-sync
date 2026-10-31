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

// Isolates version-fragile XLA PJRT types (raw buffers, holds, and C-API
// client) to this compatibility layer to prevent header leakage across callers.

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_XLA_COMPAT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_XLA_COMPAT_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/ref_count.h"

// Which JAX this is being built against, as major*10000 + minor*100 + patch.
#ifndef RAIDEN_JAX
#define RAIDEN_JAX 1102
#endif

namespace xla {
class PjRtBuffer;
class PjRtClient;
}  // namespace xla

namespace raiden {

// Uniform RawBuffer alias across supported JAX versions.
#if RAIDEN_JAX < 1002
// jax 0.10.0 and 0.10.1: PjRtRawBuffer is the root.
using RawBuffer = xla::PjRtRawBuffer;
#else
// jax 0.10.2 introduced PjRtRawBufferInterface; 0.11.0 reparented PjRtRawBuffer
// under it.
using RawBuffer = xla::PjRtRawBufferInterface;
#endif
using RawBufferRef = tsl::RCReference<RawBuffer>;

// Type-erased wrapper for CommonPjRtBuffer::ScopedHold to avoid header leakage.
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

// C-API-only PjRtClient helpers that dispatch strictly through the PJRT_Api C
// function table on xla::PjRtCApiClient with no C++ virtual fallback on
// xla::PjRtClient, avoiding vtable layout skew across XLA builds.
absl::string_view PjRtClientPlatformName(const xla::PjRtClient* client);

// Returns true iff `client` is a non-null `xla::PjRtCApiClient` whose C-API
// platform name is not "cpu". Non-C-API clients (e.g. `TfrtCpuClient` in unit
// tests) are treated as not DMA-capable by design so that no virtual
// `xla::PjRtClient` dispatch ever happens across potentially skewed XLA
// vtables.
bool PjRtClientSupportsDma(const xla::PjRtClient* client);

absl::Status PjRtClientDmaMap(xla::PjRtClient* client, void* data, size_t size);
absl::Status PjRtClientDmaUnmap(xla::PjRtClient* client, void* data);

// Returns the device layout that `client` gives an `element_type` array of
// `dims` by default, i.e. the layout of a buffer of that shape created without
// asking for one. Answers through `xla::PjRtCApiClient`'s own implementation,
// which asks the plugin through the PJRT_Api function table. Returns
// UNIMPLEMENTED for any other client, and for a plugin without the PJRT layouts
// extension.
absl::StatusOr<xla::Layout> PjRtClientDefaultLayout(
    xla::PjRtClient* client, xla::PrimitiveType element_type,
    absl::Span<const int64_t> dims);

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_XLA_COMPAT_H_
