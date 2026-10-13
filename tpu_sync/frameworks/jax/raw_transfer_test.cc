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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/xla_compat.h"
#include "tpu_sync/frameworks/jax/mock_nanobind.h"
#include "tpu_sync/frameworks/jax/raw_transfer_internal.h"

namespace raiden {
namespace {

// Helper to construct a mock JAX Array for Host pointer (Destination in D2H,
// Source in H2D)
nb::object CreateMockHostArray(uint8_t* raw_ptr, size_t size) {
  nb::object arr;
  nb::list shards;
  nb::object shard;
  nb::object shard_data;

  nb::object unsafe_buffer_pointer_fn;
  unsafe_buffer_pointer_fn.set_callable(
      [raw_ptr]() { return nb::object(reinterpret_cast<size_t>(raw_ptr)); });
  shard_data.set_attr("unsafe_buffer_pointer", unsafe_buffer_pointer_fn);

  nb::object on_device_size_in_bytes_fn;
  on_device_size_in_bytes_fn.set_callable(
      [size]() { return nb::object(size); });
  shard_data.set_attr("on_device_size_in_bytes", on_device_size_in_bytes_fn);

  shard.set_attr("data", shard_data);
  shards.add_element(shard);
  arr.set_attr("addressable_shards", shards);
  return arr;
}

// Helper to construct a mock JAX Array for PJRT Buffer (Source in D2H,
// Destination in H2D)
nb::object CreateMockDeviceArray(xla::PjRtBuffer* pjrt_buffer) {
  nb::object arr;
  nb::list shards;
  nb::object shard;
  nb::object shard_data;

  RawBuffer* raw_buf = nullptr;
  auto acquisition = AcquireCommonRawBuffer(pjrt_buffer);
  if (acquisition.ok()) {
    raw_buf = acquisition->raw_buffer.get();
  }

  // We must have a valid raw buffer for this test path.
  CHECK_NE(raw_buf, nullptr) << "Failed to get raw buffer from pjrt_buffer";

  nb::object unsafe_raw_buffer_fn;
  unsafe_raw_buffer_fn.set_callable([raw_buf]() {
    nb::object raw_buf_obj(0);  // Dummy value to not be None
    raw_buf_obj.set_attr("ptr", nb::object(reinterpret_cast<size_t>(raw_buf)));
    return raw_buf_obj;
  });
  shard_data.set_attr("unsafe_raw_buffer", unsafe_raw_buffer_fn);

  nb::list shape_list;
  for (int64_t dim : pjrt_buffer->on_device_shape().dimensions()) {
    shape_list.add_element(nb::object(dim));
  }
  shard_data.set_attr("shape", shape_list);

  shard.set_attr("data", shard_data);
  shards.add_element(shard);
  arr.set_attr("addressable_shards", shards);
  return arr;
}

class RawTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize Host PJRT Client
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  }

  std::unique_ptr<xla::PjRtClient> client_;
};

TEST_F(RawTransferTest, TestTransferD2HAsync) {
  constexpr size_t kSize = 1024;
  std::vector<uint8_t> src_data(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    src_data[i] = static_cast<uint8_t>(i % 256);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());

  // Create Host PjRtBuffer
  TF_ASSERT_OK_AND_ASSIGN(
      auto src_buffer,
      client_->BufferFromHostBuffer(
          src_data.data(), xla::U8, {static_cast<int64_t>(kSize)},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          /*device_layout=*/nullptr));

  // Destination Host buffer
  std::vector<uint8_t> dst_data(kSize, 0);

  // Wrap in mock JAX arrays
  nb::object src_arr = CreateMockDeviceArray(src_buffer.get());
  nb::object dst_arr = CreateMockHostArray(dst_data.data(), kSize);

  // Trigger async transfer
  TF_ASSERT_OK_AND_ASSIGN(PjRtCopyFuture future,
                          transfer_d2h_async(src_arr, dst_arr));

  // Wait
  TF_ASSERT_OK(future.Await());

  // Verify data
  EXPECT_EQ(dst_data, src_data);
}

TEST_F(RawTransferTest, TestTransferH2DAsync) {
  constexpr size_t kSize = 1024;
  std::vector<uint8_t> src_data(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    src_data[i] = static_cast<uint8_t>(i % 256);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());

  // Create Destination Host PjRtBuffer (initially zero)
  std::vector<uint8_t> initial_dst_data(kSize, 0);
  TF_ASSERT_OK_AND_ASSIGN(
      auto dst_buffer,
      client_->BufferFromHostBuffer(
          initial_dst_data.data(), xla::U8, {static_cast<int64_t>(kSize)},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          /*device_layout=*/nullptr));

  // Wrap in mock JAX arrays
  nb::object src_arr = CreateMockHostArray(src_data.data(), kSize);
  nb::object dst_arr = CreateMockDeviceArray(dst_buffer.get());

  // Trigger async transfer
  TF_ASSERT_OK_AND_ASSIGN(PjRtCopyFuture future,
                          transfer_h2d_async(src_arr, dst_arr));

  // Wait
  TF_ASSERT_OK(future.Await());

  // Verify data by reading back from PJRT buffer
  std::vector<uint8_t> dst_data(kSize, 0);
  TF_ASSERT_OK(dst_buffer->CopyRawToHost(dst_data.data(), 0, kSize).Await());
  EXPECT_EQ(dst_data, src_data);
}

TEST_F(RawTransferTest, TestTransferD2HBatchAsync) {
  constexpr size_t kSize = 512;
  constexpr size_t kBatch = 3;

  std::vector<std::vector<uint8_t>> src_datas(kBatch,
                                              std::vector<uint8_t>(kSize));
  std::vector<std::unique_ptr<xla::PjRtBuffer>> src_buffers;
  src_buffers.reserve(kBatch);

  nb::list src_arrs;
  nb::list dst_arrs;

  std::vector<std::vector<uint8_t>> dst_datas(kBatch,
                                              std::vector<uint8_t>(kSize, 0));

  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());

  for (size_t b = 0; b < kBatch; ++b) {
    for (size_t i = 0; i < kSize; ++i) {
      src_datas[b][i] = static_cast<uint8_t>((b + i) % 256);
    }

    TF_ASSERT_OK_AND_ASSIGN(
        auto src_buffer,
        client_->BufferFromHostBuffer(
            src_datas[b].data(), xla::U8, {static_cast<int64_t>(kSize)},
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));

    src_arrs.add_element(CreateMockDeviceArray(src_buffer.get()));
    dst_arrs.add_element(CreateMockHostArray(dst_datas[b].data(), kSize));

    src_buffers.push_back(std::move(src_buffer));
  }

  // Trigger batch transfer
  TF_ASSERT_OK_AND_ASSIGN(PjRtCopyFuture future,
                          transfer_d2h_batch_async(src_arrs, dst_arrs));

  // Wait
  TF_ASSERT_OK(future.Await());

  // Verify all
  for (size_t b = 0; b < kBatch; ++b) {
    EXPECT_EQ(dst_datas[b], src_datas[b]);
  }
}

TEST_F(RawTransferTest, TestTransferH2DBatchAsync) {
  constexpr size_t kSize = 512;
  constexpr size_t kBatch = 3;

  std::vector<std::vector<uint8_t>> src_datas(kBatch,
                                              std::vector<uint8_t>(kSize));
  std::vector<std::unique_ptr<xla::PjRtBuffer>> dst_buffers;
  dst_buffers.reserve(kBatch);

  nb::list src_arrs;
  nb::list dst_arrs;

  std::vector<uint8_t> initial_dst_data(kSize, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());

  for (size_t b = 0; b < kBatch; ++b) {
    for (size_t i = 0; i < kSize; ++i) {
      src_datas[b][i] = static_cast<uint8_t>((b + i) % 256);
    }

    TF_ASSERT_OK_AND_ASSIGN(
        auto dst_buffer,
        client_->BufferFromHostBuffer(
            initial_dst_data.data(), xla::U8, {static_cast<int64_t>(kSize)},
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));

    src_arrs.add_element(CreateMockHostArray(src_datas[b].data(), kSize));
    dst_arrs.add_element(CreateMockDeviceArray(dst_buffer.get()));

    dst_buffers.push_back(std::move(dst_buffer));
  }

  // Trigger batch transfer
  TF_ASSERT_OK_AND_ASSIGN(PjRtCopyFuture future,
                          transfer_h2d_batch_async(src_arrs, dst_arrs));

  // Wait
  TF_ASSERT_OK(future.Await());

  // Verify all by reading back
  for (size_t b = 0; b < kBatch; ++b) {
    std::vector<uint8_t> dst_data(kSize, 0);
    TF_ASSERT_OK(
        dst_buffers[b]->CopyRawToHost(dst_data.data(), 0, kSize).Await());
    EXPECT_EQ(dst_data, src_datas[b]);
  }
}

struct FakePjRtEvent {
  bool ready = false;
  bool awaited = false;
  bool destroyed = false;
  int on_ready_registrations = 0;
  PJRT_Event_OnReadyCallback callback = nullptr;
  void* user_arg = nullptr;
};

struct FakePjRtError {
  std::string msg;
};

struct FakeRawBufferEnv {
  PJRT_Api c_api{};
  PJRT_RawBuffer_Extension ext{};
  int copy_calls = 0;
  int fail_on_call = -1;
  std::vector<std::unique_ptr<FakePjRtEvent>> events;

  FakeRawBufferEnv() {
    c_api.PJRT_Error_Message = +[](PJRT_Error_Message_Args* args) {
      auto* err = reinterpret_cast<const FakePjRtError*>(args->error);
      args->message = err->msg.data();
      args->message_size = err->msg.size();
    };
    c_api.PJRT_Error_Destroy = +[](PJRT_Error_Destroy_Args* args) {
      delete reinterpret_cast<FakePjRtError*>(args->error);
    };
    c_api.PJRT_Event_Await = +[](PJRT_Event_Await_Args* args) -> PJRT_Error* {
      auto* ev = reinterpret_cast<FakePjRtEvent*>(args->event);
      ev->awaited = true;
      ev->ready = true;
      return nullptr;
    };
    c_api.PJRT_Event_Error =
        +[](PJRT_Event_Error_Args*) -> PJRT_Error* { return nullptr; };
    c_api.PJRT_Event_OnReady =
        +[](PJRT_Event_OnReady_Args* args) -> PJRT_Error* {
      auto* ev = reinterpret_cast<FakePjRtEvent*>(args->event);
      ev->on_ready_registrations++;
      ev->callback = args->callback;
      ev->user_arg = args->user_arg;
      if (ev->ready) {
        args->callback(nullptr, args->user_arg);
      }
      return nullptr;
    };
    c_api.PJRT_Event_Destroy =
        +[](PJRT_Event_Destroy_Args* args) -> PJRT_Error* {
      auto* ev = reinterpret_cast<FakePjRtEvent*>(args->event);
      ev->destroyed = true;
      return nullptr;
    };
    ext.PJRT_RawBuffer_Destroy =
        +[](PJRT_RawBuffer_Destroy_Args*) -> PJRT_Error* { return nullptr; };
    ext.PJRT_RawBuffer_CopyRawDeviceToHost =
        +[](PJRT_RawBuffer_CopyRawDeviceToHost_Args* args) -> PJRT_Error* {
      auto* env = reinterpret_cast<FakeRawBufferEnv*>(args->buffer);
      if (env->copy_calls++ == env->fail_on_call) {
        return reinterpret_cast<PJRT_Error*>(
            new FakePjRtError{"simulated d2h copy error"});
      }
      auto ev = std::make_unique<FakePjRtEvent>();
      args->event = reinterpret_cast<PJRT_Event*>(ev.get());
      env->events.push_back(std::move(ev));
      return nullptr;
    };
    ext.PJRT_RawBuffer_CopyRawHostToDevice =
        +[](PJRT_RawBuffer_CopyRawHostToDevice_Args* args) -> PJRT_Error* {
      auto* env = reinterpret_cast<FakeRawBufferEnv*>(args->buffer);
      if (env->copy_calls++ == env->fail_on_call) {
        return reinterpret_cast<PJRT_Error*>(
            new FakePjRtError{"simulated h2d copy error"});
      }
      auto ev = std::make_unique<FakePjRtEvent>();
      args->event = reinterpret_cast<PJRT_Event*>(ev.get());
      env->events.push_back(std::move(ev));
      return nullptr;
    };
  }

  BufferHoldAndAlias MakeHold() {
    BufferHoldAndAlias hold;
    hold.is_common_buffer = false;
    hold.c_raw_buffer = reinterpret_cast<PJRT_RawBuffer*>(this);
    hold.c_hold = std::make_shared<RawBufferHolder>(&c_api, &ext, nullptr);
    return hold;
  }
};

TEST(JoinBeforeErrorRawTransferTest,
     IssueD2hShardAwaitsAndDestroysEventsOnPartialFailure) {
  FakeRawBufferEnv env;
  env.fail_on_call = 1;
  BufferHoldAndAlias hold = env.MakeHold();

  uint8_t dst[64] = {};
  std::vector<D2hCopy> copies = {{dst, 0, 32}, {dst + 32, 32, 32}};
  absl::StatusOr<PjRtCopyFuture> res = IssueD2hShard(hold, copies);
  ASSERT_FALSE(res.ok());
  EXPECT_EQ(res.status().message(), "simulated d2h copy error");
  ASSERT_EQ(env.events.size(), 1u);
  EXPECT_TRUE(env.events[0]->awaited);
  EXPECT_TRUE(env.events[0]->destroyed);
}

TEST(JoinBeforeErrorRawTransferTest,
     IssueH2dShardAwaitsAndDestroysEventsOnPartialFailure) {
  FakeRawBufferEnv env;
  env.fail_on_call = 1;
  BufferHoldAndAlias hold = env.MakeHold();

  const uint8_t src[64] = {};
  std::vector<H2dCopy> copies = {{src, 0, 32}, {src + 32, 32, 32}};
  absl::StatusOr<PjRtCopyFuture> res = IssueH2dShard(hold, copies);
  ASSERT_FALSE(res.ok());
  EXPECT_EQ(res.status().message(), "simulated h2d copy error");
  ASSERT_EQ(env.events.size(), 1u);
  EXPECT_TRUE(env.events[0]->awaited);
  EXPECT_TRUE(env.events[0]->destroyed);
}

TEST(JoinBeforeErrorRawTransferTest, DrainPjRtCopyFuturesAwaitsAllFutures) {
  FakeRawBufferEnv env;
  FakePjRtEvent ev1;
  FakePjRtEvent ev2;
  std::vector<PjRtCopyFuture> futures = {
      PjRtCopyFuture::FromEvents(&env.c_api,
                                 {reinterpret_cast<PJRT_Event*>(&ev1)}, {}),
      PjRtCopyFuture::FromEvents(&env.c_api,
                                 {reinterpret_cast<PJRT_Event*>(&ev2)}, {}),
  };
  DrainPjRtCopyFutures(futures);
  EXPECT_TRUE(ev1.awaited);
  EXPECT_TRUE(ev2.awaited);
}

TEST(JoinBeforeErrorRawTransferTest,
     OnReadyWaitsForBothFutureAndEventsWithoutDuplicateRegistration) {
  FakeRawBufferEnv env;
  FakePjRtEvent ev1;
  FakePjRtEvent ev2;
  PjRtCopyFuture ev_fut1 = PjRtCopyFuture::FromEvents(
      &env.c_api, {reinterpret_cast<PJRT_Event*>(&ev1)}, {});
  PjRtCopyFuture ev_fut2 = PjRtCopyFuture::FromEvents(
      &env.c_api, {reinterpret_cast<PJRT_Event*>(&ev2)}, {});

  bool first_cb_called = false;
  ev_fut1.OnReady([&](absl::StatusOr<BufferHolders> s) {
    EXPECT_TRUE(s.ok());
    first_cb_called = true;
  });
  EXPECT_EQ(ev1.on_ready_registrations, 1);
  EXPECT_FALSE(ev_fut1.future.IsValid());

  auto [promise, xla_f] = xla::MakePromise();
  PjRtCopyFuture xla_fut(std::move(xla_f), {});
  PjRtCopyFuture joined = JoinPjRtCopyFutures({ev_fut1, ev_fut2, xla_fut});

  bool joined_cb_called = false;
  joined.OnReady([&](absl::StatusOr<BufferHolders> s) {
    EXPECT_TRUE(s.ok());
    joined_cb_called = true;
  });
  EXPECT_EQ(ev1.on_ready_registrations, 1);
  EXPECT_EQ(ev2.on_ready_registrations, 1);
  EXPECT_FALSE(joined_cb_called);

  // Fulfill xla_fut and ev1; joined must still wait for ev2.
  promise.Set();
  ev1.ready = true;
  ASSERT_NE(ev1.callback, nullptr);
  ev1.callback(nullptr, ev1.user_arg);
  EXPECT_TRUE(first_cb_called);
  EXPECT_FALSE(joined_cb_called);

  // Fulfill ev2; now joined completes.
  ev2.ready = true;
  ASSERT_NE(ev2.callback, nullptr);
  ev2.callback(nullptr, ev2.user_arg);
  EXPECT_TRUE(joined_cb_called);
}

}  // namespace
}  // namespace raiden
