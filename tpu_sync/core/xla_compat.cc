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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
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

void DestroyPjrtError(const PJRT_Api* c_api, PJRT_Error* error) {
  if (error == nullptr || c_api == nullptr ||
      c_api->PJRT_Error_Destroy == nullptr) {
    return;
  }
  PJRT_Error_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.extension_start = nullptr;
  destroy_args.error = error;
  c_api->PJRT_Error_Destroy(&destroy_args);
}

absl::Status ConsumePjrtError(const PJRT_Api* c_api, PJRT_Error* error) {
  if (error == nullptr) {
    return absl::OkStatus();
  }
  std::string message;
  if (c_api != nullptr && c_api->PJRT_Error_Message != nullptr) {
    PJRT_Error_Message_Args msg_args;
    msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    msg_args.extension_start = nullptr;
    msg_args.error = error;
    msg_args.message = nullptr;
    msg_args.message_size = 0;
    c_api->PJRT_Error_Message(&msg_args);
    if (msg_args.message != nullptr) {
      message.assign(msg_args.message, msg_args.message_size);
    }
  }

  absl::StatusCode status_code = absl::StatusCode::kInternal;
  if (c_api != nullptr && c_api->PJRT_Error_GetCode != nullptr) {
    PJRT_Error_GetCode_Args code_args;
    code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
    code_args.extension_start = nullptr;
    code_args.error = error;
    code_args.code = PJRT_Error_Code_INTERNAL;
    PJRT_Error* code_err = c_api->PJRT_Error_GetCode(&code_args);
    if (code_err != nullptr) {
      DestroyPjrtError(c_api, code_err);
    } else if (code_args.code != PJRT_Error_Code_OK) {
      status_code = static_cast<absl::StatusCode>(code_args.code);
    }
  }

  DestroyPjrtError(c_api, error);
  return absl::Status(status_code, message);
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

absl::string_view PjRtClientPlatformName(const xla::PjRtClient* client) {
  const auto* capi_client = dynamic_cast<const xla::PjRtCApiClient*>(client);
  if (capi_client == nullptr) {
    return "";
  }
  const PJRT_Api* c_api = capi_client->pjrt_c_api();
  PJRT_Client* c_client =
      const_cast<xla::PjRtCApiClient*>(capi_client)->pjrt_c_client();
  if (c_api == nullptr || c_api->PJRT_Client_PlatformName == nullptr ||
      c_client == nullptr) {
    return "";
  }
  PJRT_Client_PlatformName_Args args;
  args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client;
  args.platform_name = nullptr;
  args.platform_name_size = 0;
  PJRT_Error* err = c_api->PJRT_Client_PlatformName(&args);
  if (err != nullptr) {
    DestroyPjrtError(c_api, err);
    return "";
  }
  if (args.platform_name == nullptr) {
    return "";
  }
  return absl::string_view(args.platform_name, args.platform_name_size);
}

bool PjRtClientSupportsDma(const xla::PjRtClient* client) {
  return client != nullptr &&
         dynamic_cast<const xla::PjRtCApiClient*>(client) != nullptr &&
         PjRtClientPlatformName(client) != "cpu";
}

absl::Status PjRtClientDmaMap(xla::PjRtClient* client, void* data,
                              size_t size) {
  if (client == nullptr) {
    return absl::InvalidArgumentError("PjRtClient is null in PjRtClientDmaMap");
  }
  auto* capi_client = dynamic_cast<xla::PjRtCApiClient*>(client);
  if (capi_client == nullptr) {
    return absl::UnimplementedError("DMA mapping requires a PJRT C-API client");
  }
  const PJRT_Api* c_api = capi_client->pjrt_c_api();
  PJRT_Client* c_client = capi_client->pjrt_c_client();
  if (c_api == nullptr || c_api->PJRT_Client_DmaMap == nullptr ||
      c_client == nullptr) {
    return absl::InternalError(
        "PJRT_Client_DmaMap is unavailable on PjRtCApiClient");
  }
  PJRT_Client_DmaMap_Args args;
  args.struct_size = PJRT_Client_DmaMap_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client;
  args.data = data;
  args.size = size;
  return ConsumePjrtError(c_api, c_api->PJRT_Client_DmaMap(&args));
}

absl::Status PjRtClientDmaUnmap(xla::PjRtClient* client, void* data) {
  if (client == nullptr) {
    return absl::InvalidArgumentError(
        "PjRtClient is null in PjRtClientDmaUnmap");
  }
  auto* capi_client = dynamic_cast<xla::PjRtCApiClient*>(client);
  if (capi_client == nullptr) {
    return absl::UnimplementedError(
        "DMA unmapping requires a PJRT C-API client");
  }
  const PJRT_Api* c_api = capi_client->pjrt_c_api();
  PJRT_Client* c_client = capi_client->pjrt_c_client();
  if (c_api == nullptr || c_api->PJRT_Client_DmaUnmap == nullptr ||
      c_client == nullptr) {
    return absl::InternalError(
        "PJRT_Client_DmaUnmap is unavailable on PjRtCApiClient");
  }
  PJRT_Client_DmaUnmap_Args args;
  args.struct_size = PJRT_Client_DmaUnmap_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = c_client;
  args.data = data;
  return ConsumePjrtError(c_api, c_api->PJRT_Client_DmaUnmap(&args));
}

absl::StatusOr<xla::Layout> PjRtClientDefaultLayout(
    xla::PjRtClient* client, xla::PrimitiveType element_type,
    absl::Span<const int64_t> dims) {
  if (client == nullptr) {
    return absl::InvalidArgumentError(
        "PjRtClient is null in PjRtClientDefaultLayout");
  }
  auto* capi_client = dynamic_cast<xla::PjRtCApiClient*>(client);
  if (capi_client == nullptr) {
    return absl::UnimplementedError(
        "Default device layouts require a PJRT C-API client");
  }
  // Without the extension, PjRtCApiClient answers with an untiled layout rather
  // than the one the device uses.
  if (capi_client->FindExtension<PJRT_Extension_Base>(
          PJRT_Extension_Type::PJRT_Extension_Type_Layouts) == nullptr) {
    return absl::UnimplementedError(
        "The PJRT plugin has no layouts extension to report default device "
        "layouts");
  }
  // Qualified, so that it does not dispatch through the xla::PjRtClient vtable
  // of a client that may come from another XLA build.
  return capi_client->xla::PjRtCApiClient::GetDefaultLayout(element_type, dims);
}

}  // namespace raiden
