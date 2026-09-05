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

#ifndef THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_CORE_H_
#define THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_CORE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_external.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace raiden {

// Version-agnostic name for XLA's common raw-buffer type.
//
// XLA renamed this class from `xla::PjRtRawBufferInterface` to
// `xla::CommonPjRtRawBuffer` (see xla/pjrt/raw_buffer.h). Naming the concrete
// class directly makes this header fail to compile against XLA revisions on the
// other side of that rename. The `xla::PjRtRawBufferRef` alias
// (`tsl::RCReference<...>`) is stable across the rename, so we derive the
// pointee type from it. This keeps `raw_transfer_core.h` buildable against both
// pre- and post-rename XLA without a preprocessor version gate.
using RaidenRawBuffer =
    std::remove_reference_t<decltype(*std::declval<xla::PjRtRawBufferRef>())>;

struct RawBufferHolder {
  const PJRT_Api* c_api;
  const PJRT_RawBuffer_Extension* extension;
  PJRT_RawBuffer* buffer;

  RawBufferHolder(const PJRT_Api* api, const PJRT_RawBuffer_Extension* ext,
                  PJRT_RawBuffer* buf)
      : c_api(api), extension(ext), buffer(buf) {}

  ~RawBufferHolder() {
    if (buffer) {
      pjrt::PjRtCApiRawBuffer_Destroy(c_api, extension, buffer);
    }
  }
};

// Self-contained PJRT_Error -> absl::Status (consumes/destroys the error).
inline absl::Status PjrtErrorToStatusLocal(const PJRT_Api* c_api,
                                           PJRT_Error* error) {
  if (error == nullptr) return absl::OkStatus();
  PJRT_Error_Message_Args ma;
  ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  ma.extension_start = nullptr;
  ma.error = error;
  ma.message = nullptr;
  ma.message_size = 0;
  c_api->PJRT_Error_Message(&ma);
  std::string msg(ma.message, ma.message_size);
  PJRT_Error_Destroy_Args da;
  da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  da.extension_start = nullptr;
  da.error = error;
  c_api->PJRT_Error_Destroy(&da);
  return absl::InternalError(msg);
}

inline const PJRT_RawBuffer_Extension* GetRawBufferExtension(
    const xla::PjRtBuffer* buffer, const PJRT_Api** out_c_api = nullptr) {
  auto* capi_buffer = dynamic_cast<const xla::PjRtCApiBuffer*>(buffer);
  if (!capi_buffer) return nullptr;
  if (out_c_api) *out_c_api = capi_buffer->pjrt_c_api();
  auto* capi_client = dynamic_cast<xla::PjRtCApiClient*>(
      const_cast<xla::PjRtClient*>(capi_buffer->client()));
  if (!capi_client) return nullptr;
  return capi_client->FindExtension<PJRT_RawBuffer_Extension>(
      PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
}

inline int64_t GetMajorSliceByteSize(const xla::Shape& shape) {
  if (shape.dimensions_size() == 0) return 0;

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t stride = 1;
  for (int i = 1; i < shape.dimensions_size(); ++i) {
    stride *= shape.dimensions(i);
  }

  const xla::Layout* xla_layout = nullptr;
  if (shape.has_layout()) {
    xla_layout = &shape.layout();
  }

  if (xla_layout && !xla_layout->tiles().empty() &&
      shape.dimensions_size() >= 3) {
    const xla::Tile& tile = xla_layout->tiles()[0];
    auto tile_dims = tile.dimensions();
    if (tile_dims.size() != 2) {
      throw std::runtime_error("Only 2D tiling supported for now");
    }
    int64_t tH = tile_dims[0];
    int64_t tW = tile_dims[1];
    int64_t rank = shape.dimensions_size();

    // Find the two most minor logical dimensions physically.
    int64_t logical_minor_0 = xla_layout->minor_to_major(0);
    int64_t logical_minor_1 = xla_layout->minor_to_major(1);

    int64_t num_tiles_0 = (shape.dimensions(logical_minor_0) + tW - 1) / tW;
    int64_t num_tiles_1 = (shape.dimensions(logical_minor_1) + tH - 1) / tH;
    int64_t tiled_2d_block_size =
        num_tiles_0 * num_tiles_1 * tH * tW * itemsize;

    int64_t multiplier = 1;
    for (int i = 1; i < rank; ++i) {
      if (i != logical_minor_0 && i != logical_minor_1) {
        multiplier *= shape.dimensions(i);
      }
    }

    int64_t size_per_major_dim = tiled_2d_block_size * multiplier;

    return size_per_major_dim;
  }

  return stride * itemsize;
}

struct RaidenBufferHandle {
  xla::PjRtBuffer* buffer =
      nullptr;  // ABSL_DEPRECATED("Use shape/device or raw handles instead")
  xla::PjRtDevice* device = nullptr;

  xla::Shape shape;
  bool is_common_buffer = false;

  // For CommonPjRtBuffer:
  tsl::RCReference<RaidenRawBuffer> common_raw_buffer;
  std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold> common_hold;

  // For PjRtCApiBuffer:
  PJRT_RawBuffer* c_raw_buffer = nullptr;
  std::shared_ptr<RawBufferHolder> c_hold;

  static absl::StatusOr<RaidenBufferHandle> Acquire(
      xla::PjRtBuffer* buf, const PJRT_Api* c_api = nullptr,
      const PJRT_RawBuffer_Extension* extension = nullptr,
      bool unsafe_skip_buffer_lock = false) {
    RaidenBufferHandle result;
    result.buffer = buf;
    if (buf) {
      result.shape = buf->on_device_shape();
      result.device = buf->device();
    }

    auto* common_buf = dynamic_cast<xla::CommonPjRtBuffer*>(buf);
    auto* capi_buf = dynamic_cast<xla::PjRtCApiBuffer*>(buf);

    if (common_buf) {
      result.is_common_buffer = true;
      auto hold = common_buf->GetBufferWithHold(
          xla::CommonPjRtBuffer::ScopedHold::kUsage);
      if (!hold.ok()) {
        return hold.status();
      }
      // Workaround for OSS type discrepancies.
      result.common_raw_buffer = tsl::FormRef<RaidenRawBuffer>(
          reinterpret_cast<RaidenRawBuffer*>(
              hold.buffer()->raw_buffer().get()));
      if (!unsafe_skip_buffer_lock) {
        result.common_hold =
            std::make_shared<xla::CommonPjRtBuffer::ScopedHold>(
                std::move(hold));
      }
      return result;
    }

    if (capi_buf) {
      result.is_common_buffer = false;
      if (!extension) {
        extension = GetRawBufferExtension(buf, &c_api);
        if (!extension) {
          return absl::InternalError("RawBuffer extension missing");
        }
      }
      auto status_or_raw = pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(
          c_api, extension, capi_buf->c_buffer());
      if (!status_or_raw.ok()) {
        return status_or_raw.status();
      }
      result.c_raw_buffer = status_or_raw.value();
      result.c_hold = std::make_shared<RawBufferHolder>(c_api, extension,
                                                        result.c_raw_buffer);
      return result;
    }

    return absl::InvalidArgumentError("Unsupported PjRtBuffer type");
  }

  static absl::StatusOr<RaidenBufferHandle> AcquireFromRaw(
      RaidenRawBuffer* raw_buf, const xla::Shape& shape,
      bool unsafe_skip_buffer_lock = false) {
    RaidenBufferHandle result;
    result.shape = shape;
    if (raw_buf && raw_buf->memory_space() &&
        !raw_buf->memory_space()->devices().empty()) {
      result.device = raw_buf->memory_space()->devices()[0];
    }

    result.is_common_buffer = true;
    result.common_raw_buffer = tsl::FormRef(raw_buf);
    return result;
  }

  size_t GetOnDeviceSizeInBytes() const {
    if (is_common_buffer) {
      return common_raw_buffer->GetOnDeviceSizeInBytes();
    }
    return pjrt::PjRtCApiRawBuffer_GetOnDeviceSizeInBytes(
        c_hold->c_api, c_hold->extension, c_raw_buffer);
  }

  void* GetHostPointer() const {
    if (is_common_buffer) {
      return common_raw_buffer->GetHostPointer();
    }
    if (common_raw_buffer) {
      return common_raw_buffer->GetHostPointer();
    }
    throw std::runtime_error(
        "GetHostPointer not implemented for legacy C-API buffer handle");
  }

  xla::Future<> CopyRawHostToDevice(const void* src, int64_t device_offset,
                                    int64_t size) const {
    if (is_common_buffer) {
      return common_raw_buffer->CopyRawHostToDevice(src, device_offset, size);
    }
    return pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
        c_hold->c_api, c_hold->extension, c_raw_buffer, src, device_offset,
        size);
  }

  xla::Future<> CopyRawDeviceToHost(void* host_ptr, int64_t device_offset,
                                    int64_t size) const {
    if (is_common_buffer) {
      return common_raw_buffer->CopyRawDeviceToHost(host_ptr, device_offset,
                                                    size);
    }
    return pjrt::PjRtCApiRawBuffer_CopyRawDeviceToHost(
        c_hold->c_api, c_hold->extension, c_raw_buffer, host_ptr, device_offset,
        size);
  }

  // --- PJRT C-API completion path (raw PJRT_Event*, NOT xla::Future) ---
  // Routes completion through the stable C ABI, avoiding xla::Future /
  // tsl::AsyncValue / JoinFutures, whose C++ ABI mismatches when raiden and
  // the framework (torch_tpu libpywrap) are independent XLA builds. Valid
  // only for C-API buffers (TPU); common buffers use the xla::Future path.
  bool supports_event() const { return !is_common_buffer && c_hold != nullptr; }
  const PJRT_Api* c_api() const { return c_hold ? c_hold->c_api : nullptr; }

  absl::StatusOr<PJRT_Event*> CopyRawHostToDeviceEvent(const void* src,
                                                       int64_t device_offset,
                                                       int64_t size) const {
    PJRT_RawBuffer_CopyRawHostToDevice_Args args;
    args.struct_size = PJRT_RawBuffer_CopyRawHostToDevice_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = c_raw_buffer;
    args.src = src;
    args.offset = device_offset;
    args.transfer_size = size;
    args.event = nullptr;
    PJRT_Error* err =
        c_hold->extension->PJRT_RawBuffer_CopyRawHostToDevice(&args);
    if (err) return PjrtErrorToStatusLocal(c_hold->c_api, err);
    return args.event;
  }

  absl::StatusOr<PJRT_Event*> CopyRawDeviceToHostEvent(void* host_ptr,
                                                       int64_t device_offset,
                                                       int64_t size) const {
    PJRT_RawBuffer_CopyRawDeviceToHost_Args args;
    args.struct_size = PJRT_RawBuffer_CopyRawDeviceToHost_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.buffer = c_raw_buffer;
    args.dst = host_ptr;
    args.offset = device_offset;
    args.transfer_size = size;
    args.event = nullptr;
    PJRT_Error* err =
        c_hold->extension->PJRT_RawBuffer_CopyRawDeviceToHost(&args);
    if (err) return PjrtErrorToStatusLocal(c_hold->c_api, err);
    return args.event;
  }
};

using BufferHoldAndAlias = RaidenBufferHandle;

struct BufferHolder {
  std::shared_ptr<RawBufferHolder> c_api_hold;
  std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold> hold;
  std::shared_ptr<xla::PjRtBuffer::ExternalReference> ext_hold;
  std::shared_ptr<void> user_hold;
};

using BufferHolders = std::vector<BufferHolder>;

inline xla::Future<BufferHolder> CreateBufferFuture(
    std::vector<xla::Future<>> futures,
    std::shared_ptr<RawBufferHolder> c_api_hold = nullptr,
    std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold> hold = nullptr,
    std::shared_ptr<xla::PjRtBuffer::ExternalReference> ext_hold = nullptr,
    std::shared_ptr<void> user_hold = nullptr) {
  auto join_future = xla::JoinFutures(futures);
  if (!join_future.IsValid()) {
    return xla::Future<BufferHolder>(
        BufferHolder{std::move(c_api_hold), std::move(hold),
                     std::move(ext_hold), std::move(user_hold)});
  }
  return join_future.Map([c_api_hold = std::move(c_api_hold),
                          hold = std::move(hold),
                          ext_hold = std::move(ext_hold),
                          user_hold = std::move(user_hold)]() mutable {
    return BufferHolder{std::move(c_api_hold), std::move(hold),
                        std::move(ext_hold), std::move(user_hold)};
  });
}

inline xla::Future<BufferHolders> FlattenPjRtFutures(
    xla::Future<std::vector<BufferHolders>> futures) {
  if (!futures.IsValid()) {
    return xla::Future<BufferHolders>(BufferHolders{});
  }
  return futures.Map([](std::vector<BufferHolders> vecs) {
    std::vector<BufferHolder> result;
    for (auto& vec : vecs) {
      for (auto& h : vec) {
        result.push_back(std::move(h));
      }
    }
    return result;
  });
}

// Shared bundle of PJRT_Events; destroys them once when the last copy of the
// owning PjRtCopyFuture drops. Copy-safe (shared ownership).
struct PjRtEventBundle {
  const PJRT_Api* c_api = nullptr;
  std::vector<PJRT_Event*> events;
  ~PjRtEventBundle() {
    if (!c_api) return;
    for (PJRT_Event* e : events) {
      if (!e) continue;
      PJRT_Event_Destroy_Args a;
      a.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      a.extension_start = nullptr;
      a.event = e;
      CHECK_OK(PjrtErrorToStatusLocal(c_api, c_api->PJRT_Event_Destroy(&a)));
    }
  }
};

struct PjRtCopyFuture {
  xla::Future<> future;
  BufferHolders holds;
  std::shared_ptr<void> keep_alive;
  // PJRT C-API completion (stable C ABI; offload/TPU path). When set, IsReady/
  // Await use PJRT_Event_* instead of xla::Future/AsyncValue. A vector so that
  // JoinPjRtCopyFutures can aggregate bundles without re-owning/freeing events.
  std::vector<std::shared_ptr<PjRtEventBundle>> event_bundles;

  PjRtCopyFuture() = default;
  PjRtCopyFuture(xla::Future<> f, BufferHolders h,
                 std::shared_ptr<void> k = nullptr)
      : future(std::move(f)), holds(std::move(h)), keep_alive(std::move(k)) {}

  explicit PjRtCopyFuture(BufferHolders h)
      : future(xla::Future<>()), holds(std::move(h)) {}

  template <typename T>
  static PjRtCopyFuture FromFuture(xla::Future<T> f) {
    auto ready_future = f.GetReadyFuture();
    auto keep_alive = std::make_shared<xla::Future<T>>(std::move(f));
    return PjRtCopyFuture(std::move(ready_future), {}, std::move(keep_alive));
  }

  // Build from raw PJRT_Events (C-API completion path).
  static PjRtCopyFuture FromEvents(const PJRT_Api* c_api,
                                   std::vector<PJRT_Event*> evs,
                                   BufferHolders h) {
    PjRtCopyFuture out;
    out.holds = std::move(h);
    auto bundle = std::make_shared<PjRtEventBundle>();
    bundle->c_api = c_api;
    bundle->events = std::move(evs);
    out.event_bundles.push_back(std::move(bundle));
    return out;
  }

  bool IsValid() const {
    if (future.IsValid()) return true;
    for (const auto& b : event_bundles) {
      if (b && !b->events.empty()) return true;
    }
    return false;
  }

  bool IsReady() const {
    for (const auto& b : event_bundles) {
      if (!b || !b->c_api) continue;
      for (PJRT_Event* e : b->events) {
        PJRT_Event_IsReady_Args a;
        a.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
        a.extension_start = nullptr;
        a.event = e;
        a.is_ready = false;
        PJRT_Error* err = b->c_api->PJRT_Event_IsReady(&a);
        if (err) {
          (void)PjrtErrorToStatusLocal(b->c_api, err);
          return false;
        }
        if (!a.is_ready) return false;
      }
    }
    if (future.IsValid() && !future.IsReady()) return false;
    return true;
  }

  // Non-blocking error probe. Returns the failure status if a *ready* event or
  // the future already carries an error, and OkStatus otherwise (still pending
  // or completed successfully). Unlike Await() it never blocks -- it issues no
  // PJRT_Event_Await / BlockUntilReady -- so it is safe to call from a poll
  // loop racing the live model. Pair it with IsReady() to distinguish a
  // successful completion from a failed one.
  absl::Status PollError() {
    absl::Status status = absl::OkStatus();
    for (const auto& b : event_bundles) {
      if (!b || !b->c_api) continue;
      for (PJRT_Event* e : b->events) {
        PJRT_Event_IsReady_Args ra;
        ra.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
        ra.extension_start = nullptr;
        ra.event = e;
        ra.is_ready = false;
        PJRT_Error* rerr = b->c_api->PJRT_Event_IsReady(&ra);
        if (rerr) {
          status = PjrtErrorToStatusLocal(b->c_api, rerr);
          continue;
        }
        if (!ra.is_ready) continue;  // pending: no error to report yet
        PJRT_Event_Error_Args ee;
        ee.struct_size = PJRT_Event_Error_Args_STRUCT_SIZE;
        ee.extension_start = nullptr;
        ee.event = e;
        PJRT_Error* eerr = b->c_api->PJRT_Event_Error(&ee);
        if (eerr) status = PjrtErrorToStatusLocal(b->c_api, eerr);
      }
    }
    if (future.IsValid() && future.IsReady()) {
      tsl::AsyncValue* av = future.async_value();
      if (av && av->IsError()) status = av->GetError();
    }
    return status;
  }

  // Carries one PJRT event's completion into its own promise. The bundle
  // reference keeps the event alive -- the bundle destructor destroys it --
  // until the callback has run.
  struct EventPromiseCtx {
    xla::Promise<> promise;
    std::shared_ptr<PjRtEventBundle> bundle;
  };

  template <typename F>
  void OnReady(F&& f) {
    if (!future.IsValid() && !event_bundles.empty()) {
      // One PJRT completion callback per event, joined into a single future.
      // Callbacks run on PJRT completion threads, so no thread is held per
      // in-flight copy.
      std::vector<xla::Future<>> event_futures;
      for (const auto& b : event_bundles) {
        if (!b || !b->c_api) continue;
        for (PJRT_Event* e : b->events) {
          if (e == nullptr) continue;
          auto [promise, fut] = xla::MakePromise();
          event_futures.push_back(std::move(fut));
          auto* ctx = new EventPromiseCtx{std::move(promise), b};
          PJRT_Event_OnReady_Args oa;
          oa.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
          oa.extension_start = nullptr;
          oa.event = e;
          oa.user_arg = ctx;
          oa.callback = +[](PJRT_Error* error, void* user_arg) {
            std::unique_ptr<EventPromiseCtx> cb_ctx(
                static_cast<EventPromiseCtx*>(user_arg));
            if (error != nullptr) {
              cb_ctx->promise.Set(
                  PjrtErrorToStatusLocal(cb_ctx->bundle->c_api, error));
            } else {
              cb_ctx->promise.Set();
            }
          };
          PJRT_Error* rerr = b->c_api->PJRT_Event_OnReady(&oa);
          if (rerr != nullptr) {
            // The callback will never fire for this event.
            ctx->promise.Set(PjrtErrorToStatusLocal(b->c_api, rerr));
            delete ctx;
          }
        }
      }
      // With no live events this is an already-successful future.
      future = xla::JoinFutures(event_futures);
    }

    if (future.IsValid()) {
      future.OnReady([holds = holds, keep_alive = keep_alive,
                      f = std::forward<F>(f)](absl::Status status) mutable {
        if (status.ok()) {
          std::forward<F>(f)(absl::StatusOr<BufferHolders>(holds));
        } else {
          std::forward<F>(f)(absl::StatusOr<BufferHolders>(status));
        }
      });
      return;
    }
    // Reached only with no future and no events, i.e. nothing to complete
    // against: run the callback with the buffers the caller already holds.
    std::forward<F>(f)(absl::StatusOr<BufferHolders>(holds));
  }

  absl::Status Await() {
    absl::Status status = absl::OkStatus();
    for (const auto& b : event_bundles) {
      if (!b || !b->c_api) continue;
      for (PJRT_Event* e : b->events) {
        PJRT_Event_Await_Args aw;
        aw.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        aw.extension_start = nullptr;
        aw.event = e;
        PJRT_Error* err = b->c_api->PJRT_Event_Await(&aw);
        if (err) {
          status = PjrtErrorToStatusLocal(b->c_api, err);
          continue;
        }
        PJRT_Event_Error_Args ee;
        ee.struct_size = PJRT_Event_Error_Args_STRUCT_SIZE;
        ee.extension_start = nullptr;
        ee.event = e;
        PJRT_Error* eerr = b->c_api->PJRT_Event_Error(&ee);
        if (eerr) status = PjrtErrorToStatusLocal(b->c_api, eerr);
      }
    }
    if (future.IsValid()) {
      future.BlockUntilReady(
          static_cast<void (*)(tsl::AsyncValue*)>(tsl::BlockUntilReady));
      tsl::AsyncValue* av = future.async_value();
      if (av->IsError()) {
        status = av->GetError();
      }
    }
    return status;
  }

  void AddKeepAlive(std::shared_ptr<void> k) {
    if (!k) return;
    if (!keep_alive) {
      keep_alive = std::move(k);
    } else {
      struct CombinedKeepAlive {
        std::shared_ptr<void> old_ka;
        std::shared_ptr<void> new_ka;
      };
      keep_alive = std::make_shared<CombinedKeepAlive>(
          CombinedKeepAlive{std::move(keep_alive), std::move(k)});
    }
  }
};

inline PjRtCopyFuture JoinPjRtCopyFutures(
    absl::Span<const PjRtCopyFuture> futures) {
  std::vector<xla::Future<>> sub_futures;
  BufferHolders combined_holds;
  PjRtCopyFuture joined;
  for (const auto& f : futures) {
    if (f.future.IsValid()) {
      sub_futures.push_back(f.future);
    }
    for (const auto& h : f.holds) {
      combined_holds.push_back(h);
    }
    for (const auto& b : f.event_bundles) {
      if (b) joined.event_bundles.push_back(b);
    }
    if (f.keep_alive) {
      joined.AddKeepAlive(f.keep_alive);
    }
  }
  // Only build an xla::Future join if there is at least one (avoids
  // instantiating the JoinFutures/AsyncValue path on the pure-event path).
  if (!sub_futures.empty()) {
    joined.future = xla::JoinFutures(sub_futures);
  }
  joined.holds = std::move(combined_holds);
  return joined;
}

// One shard's worth of D2h copies (device->host). offsets in BYTES.
struct D2hCopy {
  void* dst;
  int64_t src_off;
  int64_t size;
};
struct H2dCopy {
  const void* src;
  int64_t dst_off;
  int64_t size;
};

// Issue a shard's copies and return one completion future. Uses the PJRT
// C-API event path when the buffer supports it (TPU); otherwise the legacy
// xla::Future path. `copies` offsets are already in bytes.
inline absl::StatusOr<PjRtCopyFuture> IssueD2hShard(
    const BufferHoldAndAlias& hold, const std::vector<D2hCopy>& copies) {
  BufferHolders holds{
      BufferHolder{hold.c_hold, hold.common_hold, nullptr, nullptr}};
  if (!hold.is_common_buffer && hold.c_hold == nullptr) {
    return PjRtCopyFuture(std::move(holds));
  }

  if (hold.supports_event()) {
    std::vector<PJRT_Event*> evs;
    evs.reserve(copies.size());
    for (const auto& c : copies) {
      auto ev = hold.CopyRawDeviceToHostEvent(c.dst, c.src_off, c.size);
      if (!ev.ok()) return ev.status();
      evs.push_back(ev.value());
    }
    return PjRtCopyFuture::FromEvents(hold.c_api(), std::move(evs),
                                      std::move(holds));
  }
  std::vector<xla::Future<>> fs;
  fs.reserve(copies.size());
  for (const auto& c : copies) {
    fs.push_back(hold.CopyRawDeviceToHost(c.dst, c.src_off, c.size));
  }
  return PjRtCopyFuture::FromFuture(
      CreateBufferFuture(std::move(fs), hold.c_hold, hold.common_hold));
}

inline absl::StatusOr<PjRtCopyFuture> IssueH2dShard(
    const BufferHoldAndAlias& hold, const std::vector<H2dCopy>& copies) {
  BufferHolders holds{
      BufferHolder{hold.c_hold, hold.common_hold, nullptr, nullptr}};
  if (!hold.is_common_buffer && hold.c_hold == nullptr) {
    return PjRtCopyFuture(std::move(holds));
  }

  if (hold.supports_event()) {
    std::vector<PJRT_Event*> evs;
    evs.reserve(copies.size());
    for (const auto& c : copies) {
      auto ev = hold.CopyRawHostToDeviceEvent(c.src, c.dst_off, c.size);
      if (!ev.ok()) return ev.status();
      evs.push_back(ev.value());
    }
    return PjRtCopyFuture::FromEvents(hold.c_api(), std::move(evs),
                                      std::move(holds));
  }
  std::vector<xla::Future<>> fs;
  fs.reserve(copies.size());
  for (const auto& c : copies) {
    fs.push_back(hold.CopyRawHostToDevice(c.src, c.dst_off, c.size));
  }
  return PjRtCopyFuture::FromFuture(
      CreateBufferFuture(std::move(fs), hold.c_hold, hold.common_hold));
}

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_CORE_H_
