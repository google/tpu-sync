// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include "tpu_sync/core/recv_nan_check.h"

#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>

#include "tpu_sync/core/recv_nan_check_worker.h"
namespace tpu_raiden {
namespace {
TEST(RecvNanCheckTest, BatchesSharedStorageAndPreservesEveryPoolResultIndex) {
  const auto bf16 = *NanFormatForDtype("bf16");
  const auto f16 = *NanFormatForDtype("f16");
  const auto f32 = *NanFormatForDtype("f32");
  std::vector<recv_nan_internal::BatchInput> inputs = {
      {3, 1, 2, bf16}, {3, 1, 1, f32}, {3, 1, 2, bf16},
      {3, 1, 2, f16}, {3, 2, 2, bf16}, {4, 1, 2, bf16}};
  const auto batch = recv_nan_internal::BuildNanCheckBatch(inputs);
  EXPECT_EQ(batch.raw_inputs.size(), 3);
  EXPECT_EQ(batch.groups.size(), 5);
  std::vector<size_t> seen(inputs.size());
  for (const auto& group : batch.groups) {
    const auto& raw = inputs[batch.raw_inputs[group.raw_argument]];
    const auto& first = inputs[group.inputs.front()];
    for (size_t i : group.inputs) {
      ++seen[i];
      EXPECT_EQ(inputs[i].storage, raw.storage);
      EXPECT_EQ(inputs[i].shard, raw.shard);
      EXPECT_EQ(inputs[i].bucket, first.bucket);
      EXPECT_EQ(inputs[i].format.element_bytes, first.format.element_bytes);
      EXPECT_EQ(inputs[i].format.exponent_mask, first.format.exponent_mask);
    }
  }
  EXPECT_EQ(seen, std::vector<size_t>(inputs.size(), 1));
}
TEST(RecvNanCheckTest, ScanSignatureDoesNotDependOnPoolOrder) {
  const auto bf16 = *NanFormatForDtype("bf16");
  std::vector<recv_nan_internal::BatchInput> inputs;
  for (int layer = 0; layer < 45; ++layer) {
    inputs.push_back({0, 0, 2, bf16});
    inputs.push_back({0, 0, 32, bf16});
  }
  const auto forward = recv_nan_internal::BuildNanCheckBatch(inputs);
  std::reverse(inputs.begin(), inputs.end());
  const auto reverse = recv_nan_internal::BuildNanCheckBatch(inputs);
  ASSERT_EQ(forward.raw_inputs.size(), 1);
  ASSERT_EQ(reverse.raw_inputs.size(), 1);
  ASSERT_EQ(forward.groups.size(), 2);
  ASSERT_EQ(reverse.groups.size(), 2);
  for (size_t group = 0; group < 2; ++group) {
    EXPECT_EQ(reverse.groups[group].raw_argument,
              forward.groups[group].raw_argument);
    EXPECT_EQ(reverse.groups[group].inputs.size(), 45);
    EXPECT_EQ(inputs[reverse.groups[group].inputs.front()].bucket,
              group == 0 ? 2 : 32);
  }
}
TEST(RecvNanCheckTest, ClipsCompactPrefixWithoutReadingRegionGapsOrTail) {
  const std::array<NanCheckRange, 2> ranges = {{{0, 16}, {64, 16}}};
  auto chunks = BuildNanCheckChunks(256, ranges, 24);
  ASSERT_EQ(chunks.size(), 2);
  EXPECT_EQ(chunks[0].word_begin, 64);
  EXPECT_EQ(chunks[0].byte_begin, 256);
  EXPECT_EQ(chunks[0].byte_end, 272);
  EXPECT_EQ(chunks[1].word_begin, 80);
  EXPECT_EQ(chunks[1].byte_begin, 320);
  EXPECT_EQ(chunks[1].byte_end, 328);
  EXPECT_TRUE(BuildNanCheckChunks(0, ranges, 0).empty());
}
TEST(RecvNanCheckTest, KeepsHalfWordEndpointsAndBoundsGatherChunks) {
  const std::array<NanCheckRange, 1> ranges = {{{2, 65536}}};
  auto chunks = BuildNanCheckChunks(0, ranges, 65536);
  ASSERT_EQ(chunks.size(), 3);
  EXPECT_EQ(chunks[0].word_begin, 0);
  EXPECT_EQ(chunks[0].byte_begin, 2);
  EXPECT_EQ(chunks[0].byte_end, 32768);
  EXPECT_EQ(chunks[1].byte_begin, 32768);
  EXPECT_EQ(chunks[1].byte_end, 65536);
  EXPECT_EQ(chunks[2].byte_begin, 65536);
  EXPECT_EQ(chunks[2].byte_end, 65538);
}
TEST(RecvNanCheckTest, KeepsByteEndpointsAndBoundsGatherChunks) {
  const std::array<NanCheckRange, 1> ranges = {{{1, 65536}}};
  auto chunks = BuildNanCheckChunks(0, ranges, 65536);
  ASSERT_EQ(chunks.size(), 3);
  EXPECT_EQ(chunks[0].word_begin, 0);
  EXPECT_EQ(chunks[0].byte_begin, 1);
  EXPECT_EQ(chunks[0].byte_end, 32768);
  EXPECT_EQ(chunks[1].byte_begin, 32768);
  EXPECT_EQ(chunks[1].byte_end, 65536);
  EXPECT_EQ(chunks[2].byte_begin, 65536);
  EXPECT_EQ(chunks[2].byte_end, 65537);
}
TEST(RecvNanCheckTest, DumpsFirstThreeNanHitsPerHundredChecksPerPool) {
  NanDumpWindow a, b;
  for (int i = 1; i <= 303; ++i) {
    EXPECT_EQ(a.OnCheck(true), i % 100 >= 1 && i % 100 <= 3) << i;
  }
  EXPECT_TRUE(b.OnCheck(true));
  NanDumpWindow mixed;
  for (int i = 1; i <= 200; ++i) {
    const bool nan = i >= 98 && i <= 104;
    EXPECT_EQ(mixed.OnCheck(nan), nan && i != 104) << i;
  }
  EXPECT_TRUE(mixed.OnCheck(true));
}
TEST(RecvNanCheckTest, SupportsNumericFormatsWithoutGuessingIntegerCarriers) {
  EXPECT_EQ(NanFormatForDtype("bf16")->exponent_mask, 0x7f80);
  EXPECT_EQ(NanFormatForDtype("f16")->fraction_mask, 0x03ff);
  EXPECT_EQ(NanFormatForDtype("float32")->element_bytes, 4);
  EXPECT_FALSE(NanFormatForDtype("uint8"));
  EXPECT_FALSE(NanFormatForDtype("fp8"));
  for (const auto dtype : {"float8_e4m3fn", "float8_e5m2",
                           "float8_e4m3fnuz", "float8_e5m2fnuz"}) {
    const auto format = NanFormatForDtype(dtype);
    ASSERT_TRUE(format) << dtype;
    EXPECT_EQ(format->element_bytes, 1) << dtype;
    EXPECT_TRUE(NanFormatForPool("fa.l3", dtype)) << dtype;
    EXPECT_TRUE(NanFormatForPool("gdn.ssm.g0.l3", dtype)) << dtype;
    EXPECT_FALSE(NanFormatForPool("other", dtype)) << dtype;
  }
}
TEST(RecvNanCheckTest, SeparatesFp8ClassifiersSharingTheSameStorage) {
  std::vector<recv_nan_internal::BatchInput> inputs;
  for (const auto dtype : {"float8_e4m3fn", "float8_e5m2",
                           "float8_e4m3fnuz", "float8_e5m2fnuz", "bf16"}) {
    const auto format = NanFormatForDtype(dtype);
    ASSERT_TRUE(format) << dtype;
    inputs.push_back({0, 0, 1, *format});
  }
  const auto batch = recv_nan_internal::BuildNanCheckBatch(inputs);
  ASSERT_EQ(batch.raw_inputs.size(), 1);
  ASSERT_EQ(batch.groups.size(), 4);
  std::vector<std::vector<size_t>> actual;
  for (const auto& group : batch.groups) actual.push_back(group.inputs);
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(actual, (std::vector<std::vector<size_t>>{{0}, {1}, {2, 3}, {4}}));
}
TEST(RecvNanCheckTest, ChecksOnlyDeclaredFaAndGdnNumericPools) {
  for (const auto tag : {"fa", "fa.l3", "gdn.conv", "gdn.ssm",
                         "gdn.conv.g0", "gdn.ssm.g12.l45"}) {
    EXPECT_TRUE(NanFormatForPool(tag, "bf16")) << tag;
    EXPECT_EQ(NanFormatForPool(tag, "f32")->element_bytes, 4) << tag;
    EXPECT_FALSE(NanFormatForPool(tag, "uint8")) << tag;
  }
  for (const auto tag : {"", "fa.l", "fa.l3x", "other", "gdn", "gdn.ssmx",
                         "gdn.conv.g", "gdn.conv.g0.l", "gdn.ssm.g0.l1.extra"}) {
    EXPECT_FALSE(NanFormatForPool(tag, "bf16")) << tag;
  }
}
TEST(RecvNanCheckTest, PublishesAfterSubmissionWithoutWaitingForBool) {
  recv_nan_internal::Worker worker;
  std::atomic<bool> ready = false;
  std::atomic<bool> consumed = false;
  std::promise<void> submitted, next_submitted;
  auto first = submitted.get_future();
  auto next = next_submitted.get_future();
  ASSERT_TRUE(worker.Schedule(
      [&] {
        return [&] {
          if (!ready.load()) return false;
          consumed = true;
          return true;
        };
      },
      [&] { submitted.set_value(); }));
  EXPECT_EQ(first.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  EXPECT_FALSE(consumed.load());
  EXPECT_TRUE(worker.Schedule([] { return [] { return true; }; },
                              [&] { next_submitted.set_value(); }));
  EXPECT_EQ(next.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  ready = true;
  worker.Drain();
  EXPECT_TRUE(consumed.load());
}
TEST(RecvNanCheckTest, DrainRetainsPendingReadbackAndRejectsNewWork) {
  recv_nan_internal::Worker worker;
  auto owner = std::make_shared<int>(7);
  std::weak_ptr<int> weak = owner;
  std::atomic<bool> ready = false;
  std::promise<void> submitted;
  auto notified = submitted.get_future();
  ASSERT_TRUE(worker.Schedule(
      [&, owner] { return [&, owner] { return ready.load(); }; },
      [&] { submitted.set_value(); }));
  owner.reset();
  EXPECT_EQ(notified.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  auto drain = std::async(std::launch::async, [&] { worker.Drain(); });
  EXPECT_EQ(drain.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_FALSE(weak.expired());
  ready = true;
  drain.get();
  EXPECT_TRUE(weak.expired());
  EXPECT_FALSE(worker.Schedule([] { return [] { return true; }; }, [] {}));
}
TEST(RecvNanCheckTest, BoundsOutstandingChecksWithoutBlockingSubmission) {
  recv_nan_internal::Worker worker;
  std::atomic<bool> ready = false;
  for (int i = 0; i < 128; ++i) {
    EXPECT_TRUE(worker.Schedule([&] { return [&] { return ready.load(); }; },
                                [] {}));
  }
  EXPECT_FALSE(worker.Schedule([] { return [] { return true; }; }, [] {}));
  ready = true;
  worker.Drain();
}
TEST(RecvNanCheckTest, ConsumesOutOfOrderCompletionsInSubmissionOrder) {
  recv_nan_internal::Worker worker;
  std::atomic<bool> first_ready = false;
  std::promise<void> third_submitted;
  auto third = third_submitted.get_future();
  std::vector<int> consumed;
  EXPECT_TRUE(worker.Schedule(
      [&] {
        return [&] {
          if (!first_ready.load()) return false;
          consumed.push_back(1);
          return true;
        };
      }, [] {}));
  EXPECT_TRUE(worker.Schedule(
      [&] { return [&] { consumed.push_back(2); return true; }; },
      [] {}));
  EXPECT_TRUE(worker.Schedule([] { return [] { return true; }; },
                              [&] { third_submitted.set_value(); }));
  // Reaching the third submission proves the worker has already polled the
  // first two checks while the first result was still pending.
  EXPECT_EQ(third.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  // The second result is ready, but it must not overtake the first check.
  first_ready = true;
  worker.Drain();
  EXPECT_EQ(consumed, (std::vector<int>{1, 2}));
}
}  // namespace
}  // namespace tpu_raiden
