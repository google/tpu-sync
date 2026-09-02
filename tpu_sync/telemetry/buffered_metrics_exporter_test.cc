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

#include "tpu_sync/telemetry/buffered_metrics_exporter.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/label_util.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

// -----------------------------------------------------------------------------
// QueueBuffer Tests
// -----------------------------------------------------------------------------

TEST(QueueBufferTest, EmptyBufferExtractAndResetReturnsEmpty) {
  QueueBuffer<16> buffer;
  EXPECT_THAT(buffer.ExtractAndReset(), IsEmpty());
}

TEST(QueueBufferTest, SequentialPushAndExtractAndReset) {
  QueueBuffer<8> buffer;
  buffer.Push(1.5);
  buffer.Push(2.5);
  buffer.Push(3.5);

  EXPECT_THAT(buffer.ExtractAndReset(), ElementsAre(1.5, 2.5, 3.5));
  EXPECT_THAT(buffer.ExtractAndReset(), IsEmpty());

  // Verify buffer can be reused after reset
  buffer.Push(4.5);
  buffer.Push(5.5);
  EXPECT_THAT(buffer.ExtractAndReset(), ElementsAre(4.5, 5.5));
  EXPECT_THAT(buffer.ExtractAndReset(), IsEmpty());
}

TEST(QueueBufferTest, CapacityBoundaryAndOverflowTruncation) {
  QueueBuffer<4> buffer;
  for (int i = 1; i <= 10; ++i) {
    buffer.Push(static_cast<double>(i));
  }

  // Exactly the first 4 elements should be stored; the remaining 6 dropped.
  EXPECT_THAT(buffer.ExtractAndReset(), ElementsAre(1.0, 2.0, 3.0, 4.0));
  EXPECT_THAT(buffer.ExtractAndReset(), IsEmpty());
}

TEST(QueueBufferTest, FloatingPointBitwisePreservation) {
  QueueBuffer<8> buffer;
  const double kInf = std::numeric_limits<double>::infinity();
  const double kSubnormal = 1e-308;

  buffer.Push(0.0);
  buffer.Push(-0.0);
  buffer.Push(-123.456789);
  buffer.Push(kInf);
  buffer.Push(kSubnormal);

  std::vector<double> extracted = buffer.ExtractAndReset();
  ASSERT_EQ(extracted.size(), 5);
  EXPECT_EQ(extracted[0], 0.0);
  EXPECT_TRUE(std::signbit(extracted[1]));
  EXPECT_DOUBLE_EQ(extracted[2], -123.456789);
  EXPECT_EQ(extracted[3], kInf);
  EXPECT_DOUBLE_EQ(extracted[4], kSubnormal);
}

TEST(QueueBufferTest, ConcurrentMultiThreadedPush) {
  constexpr size_t kCapacity = 1000;
  QueueBuffer<kCapacity> buffer;

  constexpr int kNumThreads = 8;
  constexpr int kPushesPerThread = 100;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&buffer, t] {
      for (int i = 0; i < kPushesPerThread; ++i) {
        buffer.Push(static_cast<double>(t * 1000 + i));
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::vector<double> extracted = buffer.ExtractAndReset();
  EXPECT_EQ(extracted.size(), kNumThreads * kPushesPerThread);
  EXPECT_THAT(buffer.ExtractAndReset(), IsEmpty());
}

TEST(QueueBufferTest, BufferResetAndReuseCycle) {
  QueueBuffer<100> buffer(/*max_capacity=*/100);
  for (int i = 0; i < 30; ++i) {
    buffer.Push(static_cast<double>(i));
  }
  std::vector<double> first_extract = buffer.ExtractAndReset();
  EXPECT_EQ(first_extract.size(), 30);

  // Subsequent push cycle after reset
  for (int i = 0; i < 40; ++i) {
    buffer.Push(static_cast<double>(i * 2));
  }
  std::vector<double> second_extract = buffer.ExtractAndReset();
  EXPECT_EQ(second_extract.size(), 40);
}

TEST(QueueBufferTest, ConcurrentPushAndExtract) {
  constexpr size_t kCapacity = 50000;
  QueueBuffer<kCapacity> buffer;

  constexpr int kNumThreads = 4;
  constexpr int kPushesPerThread = 5000;
  std::atomic<bool> stop_consumer{false};
  std::atomic<size_t> total_extracted{0};

  std::thread consumer([&] {
    while (!stop_consumer.load(std::memory_order_relaxed)) {
      std::vector<double> batch = buffer.ExtractAndReset();
      total_extracted.fetch_add(batch.size(), std::memory_order_relaxed);
      std::this_thread::yield();
    }
    std::vector<double> final_batch = buffer.ExtractAndReset();
    total_extracted.fetch_add(final_batch.size(), std::memory_order_relaxed);
  });

  std::vector<std::thread> producers;
  producers.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    producers.emplace_back([&buffer, t] {
      for (int i = 0; i < kPushesPerThread; ++i) {
        buffer.Push(static_cast<double>(t * 10000 + i));
      }
    });
  }

  for (auto& producer : producers) {
    producer.join();
  }
  stop_consumer.store(true, std::memory_order_relaxed);
  consumer.join();

  EXPECT_EQ(total_extracted.load(), kNumThreads * kPushesPerThread);
}

// -----------------------------------------------------------------------------
// LockFreeCounterAccumulator Tests
// -----------------------------------------------------------------------------

TEST(LockFreeCounterAccumulatorTest, InitialValueZero) {
  LockFreeCounterAccumulator counter;
  EXPECT_EQ(counter.ExchangeAndReset(), 0);
}

TEST(LockFreeCounterAccumulatorTest, BasicAddAndReset) {
  LockFreeCounterAccumulator counter;
  counter.Add(100);
  counter.Add(50);
  EXPECT_EQ(counter.ExchangeAndReset(), 150);
  EXPECT_EQ(counter.ExchangeAndReset(), 0);

  // Verify counter can be reused after reset
  counter.Add(25);
  EXPECT_EQ(counter.ExchangeAndReset(), 25);
  EXPECT_EQ(counter.ExchangeAndReset(), 0);
}

TEST(LockFreeCounterAccumulatorTest, ConcurrentAddStress) {
  LockFreeCounterAccumulator counter;
  constexpr int kNumThreads = 8;
  constexpr int kIterations = 10000;
  constexpr uint64_t kDeltaPerIter = 5;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&counter] {
      for (int i = 0; i < kIterations; ++i) {
        counter.Add(kDeltaPerIter);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  constexpr uint64_t kExpectedTotal =
      static_cast<uint64_t>(kNumThreads) * kIterations * kDeltaPerIter;
  EXPECT_EQ(counter.ExchangeAndReset(), kExpectedTotal);
  EXPECT_EQ(counter.ExchangeAndReset(), 0);
}

// -----------------------------------------------------------------------------
// MetricFamilyBuffer Tests
// -----------------------------------------------------------------------------

TEST(MetricFamilyBufferTest, EmptyLabelsReturnsUnlabeledAccumulator) {
  MetricFamilyBuffer<LockFreeCounterAccumulator> family_buffer;
  auto* acc1 = family_buffer.GetOrCreate({});
  ASSERT_NE(acc1, nullptr);

  acc1->Add(42);

  auto* acc2 = family_buffer.GetOrCreate({});
  EXPECT_EQ(acc1, acc2);
  EXPECT_EQ(acc2->ExchangeAndReset(), 42);
}

TEST(MetricFamilyBufferTest, DistinctLabelsReturnDistinctAccumulators) {
  MetricFamilyBuffer<LockFreeCounterAccumulator> family_buffer;
  MetricLabel label_a[] = {{"type", "a"}};
  MetricLabel label_b[] = {{"type", "b"}};

  auto* acc_unlabeled = family_buffer.GetOrCreate({});
  auto* acc_a = family_buffer.GetOrCreate(label_a);
  auto* acc_b = family_buffer.GetOrCreate(label_b);

  ASSERT_NE(acc_unlabeled, nullptr);
  ASSERT_NE(acc_a, nullptr);
  ASSERT_NE(acc_b, nullptr);
  EXPECT_NE(acc_unlabeled, acc_a);
  EXPECT_NE(acc_unlabeled, acc_b);
  EXPECT_NE(acc_a, acc_b);

  // Calling again with same labels returns same pointer
  EXPECT_EQ(family_buffer.GetOrCreate(label_a), acc_a);
}

TEST(MetricFamilyBufferTest, SameLabelsInDifferentOrderReturnSameAccumulator) {
  MetricFamilyBuffer<LockFreeCounterAccumulator> family_buffer;
  MetricLabel labels_order1[] = {{"k1", "v1"}, {"k2", "v2"}};
  MetricLabel labels_order2[] = {{"k2", "v2"}, {"k1", "v1"}};

  auto* acc1 = family_buffer.GetOrCreate(labels_order1);
  auto* acc2 = family_buffer.GetOrCreate(labels_order2);

  ASSERT_NE(acc1, nullptr);
  EXPECT_EQ(acc1, acc2);
}

TEST(MetricFamilyBufferTest, CapacityLimitEnforced) {
  MetricFamilyBuffer<LockFreeCounterAccumulator> family_buffer;
  for (size_t i = 0; i < kMaxLabeledSeries; ++i) {
    std::string val = absl::StrCat("val_", i);
    MetricLabel label[] = {{"key", val}};
    EXPECT_NE(family_buffer.GetOrCreate(label), nullptr);
  }

  MetricLabel overflow_label[] = {{"key", "overflow"}};
  EXPECT_EQ(family_buffer.GetOrCreate(overflow_label), nullptr);
}

TEST(MetricFamilyBufferTest, ForEachAccumulatorTraversesUnlabeledAndLabeled) {
  MetricFamilyBuffer<LockFreeCounterAccumulator> family_buffer;
  family_buffer.GetOrCreate({})->Add(10);

  MetricLabel label_a[] = {{"type", "a"}};
  MetricLabel label_b[] = {{"type", "b"}};
  family_buffer.GetOrCreate(label_a)->Add(20);
  family_buffer.GetOrCreate(label_b)->Add(30);

  std::map<std::string, uint64_t> visited;
  family_buffer.ForEachAccumulator(
      [&](absl::string_view canonical_labels, LockFreeCounterAccumulator* acc) {
        visited[std::string(canonical_labels)] = acc->ExchangeAndReset();
      });

  EXPECT_EQ(visited[""], 10);
  EXPECT_EQ(visited["{type=\"a\"}"], 20);
  EXPECT_EQ(visited["{type=\"b\"}"], 30);
}

TEST(MetricFamilyBufferTest, ConcurrentGetOrCreateThreadSafe) {
  MetricFamilyBuffer<LockFreeCounterAccumulator> family_buffer;
  constexpr int kNumThreads = 8;
  constexpr int kIterations = 1000;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&family_buffer, t] {
      for (int i = 0; i < kIterations; ++i) {
        if (i % 2 == 0) {
          auto* acc = family_buffer.GetOrCreate({});
          ASSERT_NE(acc, nullptr);
          acc->Add(1);
        } else {
          std::string label_val = absl::StrCat("val_", (t * 10 + i) % 10);
          MetricLabel label[] = {{"shard", label_val}};
          auto* acc = family_buffer.GetOrCreate(label);
          ASSERT_NE(acc, nullptr);
          acc->Add(1);
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  uint64_t total = 0;
  family_buffer.ForEachAccumulator(
      [&](absl::string_view, LockFreeCounterAccumulator* acc) {
        total += acc->ExchangeAndReset();
      });

  constexpr uint64_t kExpectedTotal =
      static_cast<uint64_t>(kNumThreads) * kIterations;
  EXPECT_EQ(total, kExpectedTotal);
}

TEST(MetricFamilyBufferTest, LargeLabelSetExceedingStackBuffer) {
  MetricFamilyBuffer<LockFreeCounterAccumulator> family_buffer;
  const std::string long_value(kDefaultPrometheusStackBufferSize + 64, 'x');
  const MetricLabel label[] = {{"huge_key", long_value}};
  LockFreeCounterAccumulator* first_accumulator =
      family_buffer.GetOrCreate(label);
  ASSERT_NE(first_accumulator, nullptr);
  first_accumulator->Add(99);

  // Subsequent call finds existing accumulator via heap fallback path.
  LockFreeCounterAccumulator* second_accumulator =
      family_buffer.GetOrCreate(label);
  EXPECT_EQ(first_accumulator, second_accumulator);
  EXPECT_EQ(second_accumulator->ExchangeAndReset(), 99);
}

TEST(MetricFamilyBufferTest, ExactStackBufferCapacityBoundaries) {
  MetricFamilyBuffer<LockFreeCounterAccumulator> family_buffer;
  // 255 bytes (kDefaultPrometheusStackBufferSize - 1)
  const std::string val_255(kDefaultPrometheusStackBufferSize - 7, 'a');
  const MetricLabel label_255[] = {{"k", val_255}};
  // 256 bytes (kDefaultPrometheusStackBufferSize, exact stack fit)
  const std::string val_256(kDefaultPrometheusStackBufferSize - 6, 'b');
  const MetricLabel label_256[] = {{"k", val_256}};
  // 257 bytes (kDefaultPrometheusStackBufferSize + 1, minimal heap fallback)
  const std::string val_257(kDefaultPrometheusStackBufferSize - 5, 'c');
  const MetricLabel label_257[] = {{"k", val_257}};

  LockFreeCounterAccumulator* acc_255 = family_buffer.GetOrCreate(label_255);
  LockFreeCounterAccumulator* acc_256 = family_buffer.GetOrCreate(label_256);
  LockFreeCounterAccumulator* acc_257 = family_buffer.GetOrCreate(label_257);

  ASSERT_NE(acc_255, nullptr);
  ASSERT_NE(acc_256, nullptr);
  ASSERT_NE(acc_257, nullptr);
  EXPECT_NE(acc_255, acc_256);
  EXPECT_NE(acc_256, acc_257);
  EXPECT_NE(acc_255, acc_257);

  EXPECT_EQ(family_buffer.GetOrCreate(label_255), acc_255);
  EXPECT_EQ(family_buffer.GetOrCreate(label_256), acc_256);
  EXPECT_EQ(family_buffer.GetOrCreate(label_257), acc_257);
}

TEST(MetricFamilyBufferTest,
     UnsortedMultiLabelSetExceedingStackBufferReturnsSameAccumulator) {
  MetricFamilyBuffer<LockFreeCounterAccumulator> family_buffer;
  const std::string long_value(kDefaultPrometheusStackBufferSize + 64, 'x');
  const MetricLabel first_order[] = {{"z_key", long_value}, {"a_key", "val"}};
  const MetricLabel second_order[] = {{"a_key", "val"}, {"z_key", long_value}};

  LockFreeCounterAccumulator* first_accumulator =
      family_buffer.GetOrCreate(first_order);
  LockFreeCounterAccumulator* second_accumulator =
      family_buffer.GetOrCreate(second_order);
  ASSERT_NE(first_accumulator, nullptr);
  EXPECT_EQ(first_accumulator, second_accumulator);
}

// -----------------------------------------------------------------------------
// BufferedMetricsExporter Tests
// -----------------------------------------------------------------------------

TEST(BufferedMetricsExporterTest, EmptySamplesWhenNoMetricsUpdated) {
  BufferedMetricsExporter exporter;
  EXPECT_THAT(exporter.GetAndResetMetricSamples(), IsEmpty());
}

TEST(BufferedMetricsExporterTest, GetTextSnapshotReturnsEmpty) {
  BufferedMetricsExporter exporter;
  exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 100);
  EXPECT_EQ(exporter.GetTextSnapshot(), "");
}

TEST(BufferedMetricsExporterTest, CounterAccumulationAndReset) {
  BufferedMetricsExporter exporter;
  exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 100);
  exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 50);
  exporter.IncrementCounter(metric_names::kReceivedBytesTotal, {}, 200);
  exporter.IncrementCounter(metric_names::kTransferFailuresTotal, {}, 3);

  auto samples = exporter.GetAndResetMetricSamples();
  EXPECT_EQ(samples["tpu_raiden_sent_bytes_total"],
            (std::vector<double>{150.0}));
  EXPECT_EQ(samples["tpu_raiden_received_bytes_total"],
            (std::vector<double>{200.0}));
  EXPECT_EQ(samples["tpu_raiden_transfer_failures_total"],
            (std::vector<double>{3.0}));

  auto reset_samples = exporter.GetAndResetMetricSamples();
  EXPECT_THAT(reset_samples, IsEmpty());
}

TEST(BufferedMetricsExporterTest, GaugeAndHistogramMetricsExport) {
  MetricMetadata custom_metrics[] = {
      MetricMetadata{
          .name = "active_connections",
          .description = "Number of active connections.",
          .type = MetricType::kGauge,
      },
      MetricMetadata{
          .name = "transfer_latency_seconds",
          .description = "Transfer latency in seconds.",
          .type = MetricType::kHistogram,
      },
  };

  BufferedMetricsExporter exporter(custom_metrics);

  exporter.SetGauge("active_connections", {}, 12.0);
  exporter.SetGauge("active_connections", {}, 15.0);
  exporter.ObserveHistogram("transfer_latency_seconds", {}, 0.005);
  exporter.ObserveHistogram("transfer_latency_seconds", {}, 0.015);

  auto samples = exporter.GetAndResetMetricSamples();
  EXPECT_EQ(samples["tpu_raiden_active_connections"],
            (std::vector<double>{12.0, 15.0}));
  EXPECT_EQ(samples["tpu_raiden_transfer_latency_seconds"],
            (std::vector<double>{0.005, 0.015}));

  EXPECT_THAT(exporter.GetAndResetMetricSamples(), IsEmpty());
}

TEST(BufferedMetricsExporterTest, MetricLabelsPassedAndAccumulatedSeparately) {
  BufferedMetricsExporter exporter;
  MetricLabel push_labels[] = {
      {"direction", "push"},
  };
  MetricLabel pull_labels[] = {
      {"direction", "pull"},
  };

  exporter.IncrementCounter(metric_names::kSentBytesTotal, push_labels, 1024);
  exporter.IncrementCounter(metric_names::kSentBytesTotal, push_labels, 512);
  exporter.IncrementCounter(metric_names::kSentBytesTotal, pull_labels, 2048);
  exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 100);

  auto samples = exporter.GetAndResetMetricSamples();
  EXPECT_EQ(samples["tpu_raiden_sent_bytes_total{direction=\"push\"}"],
            (std::vector<double>{1536.0}));
  EXPECT_EQ(samples["tpu_raiden_sent_bytes_total{direction=\"pull\"}"],
            (std::vector<double>{2048.0}));
  EXPECT_EQ(samples["tpu_raiden_sent_bytes_total"],
            (std::vector<double>{100.0}));
}

TEST(BufferedMetricsExporterTest, MultiDimensionalLabelsSortingInvariance) {
  BufferedMetricsExporter exporter;
  MetricLabel labels1[] = {
      {"error_code", "DEADLINE_EXCEEDED"},
      {"direction", "pull"},
  };
  MetricLabel labels2[] = {
      {"direction", "pull"},
      {"error_code", "DEADLINE_EXCEEDED"},
  };

  exporter.IncrementCounter(metric_names::kTransferFailuresTotal, labels1, 1);
  exporter.IncrementCounter(metric_names::kTransferFailuresTotal, labels2, 2);

  auto samples = exporter.GetAndResetMetricSamples();
  EXPECT_EQ(samples["tpu_raiden_transfer_failures_total{direction=\"pull\","
                    "error_code=\"DEADLINE_EXCEEDED\"}"],
            (std::vector<double>{3.0}));
}

TEST(BufferedMetricsExporterTest, UnlabeledGaugesAndHistograms) {
  BufferedMetricsExporter exporter;
  exporter.SetGauge(metric_names::kBufferAllocatedBytes, {}, 1024.0 * 1024.0);
  exporter.ObserveHistogram(metric_names::kTransferDurationMs, {}, 15.5);
  exporter.ObserveHistogram(metric_names::kH2dTransferTimeMs, {}, 2.5);
  exporter.ObserveHistogram(metric_names::kD2hTransferTimeMs, {}, 3.5);

  auto samples = exporter.GetAndResetMetricSamples();
  EXPECT_EQ(samples["tpu_raiden_buffer_allocated_bytes"],
            (std::vector<double>{1024.0 * 1024.0}));
  EXPECT_EQ(samples["tpu_raiden_transfer_duration_ms"],
            (std::vector<double>{15.5}));
  EXPECT_EQ(samples["tpu_raiden_h2d_transfer_time_ms"],
            (std::vector<double>{2.5}));
  EXPECT_EQ(samples["tpu_raiden_d2h_transfer_time_ms"],
            (std::vector<double>{3.5}));
}

TEST(BufferedMetricsExporterTest, LabeledGaugesAndHistograms) {
  MetricMetadata custom_metrics[] = {
      MetricMetadata{
          .name = "custom_gauge",
          .description = "Custom labeled gauge.",
          .type = MetricType::kGauge,
      },
      MetricMetadata{
          .name = "custom_histogram",
          .description = "Custom labeled histogram.",
          .type = MetricType::kHistogram,
      },
  };
  BufferedMetricsExporter exporter(custom_metrics);

  MetricLabel label_a[] = {{"type", "a"}};
  MetricLabel label_b[] = {{"type", "b"}};

  exporter.SetGauge("custom_gauge", label_a, 1024.0 * 1024.0);
  exporter.SetGauge("custom_gauge", label_b, 2048.0 * 1024.0);

  MetricLabel direct_pull[] = {{"direction", "pull"}, {"mode", "direct"}};
  MetricLabel reshard_pull[] = {{"direction", "pull"}, {"mode", "reshard"}};

  exporter.ObserveHistogram("custom_histogram", direct_pull, 15.5);
  exporter.ObserveHistogram("custom_histogram", reshard_pull, 42.0);

  auto samples = exporter.GetAndResetMetricSamples();
  EXPECT_EQ(samples["tpu_raiden_custom_gauge{type=\"a\"}"],
            (std::vector<double>{1024.0 * 1024.0}));
  EXPECT_EQ(samples["tpu_raiden_custom_gauge{type=\"b\"}"],
            (std::vector<double>{2048.0 * 1024.0}));
  EXPECT_EQ(
      samples
          ["tpu_raiden_custom_histogram{direction=\"pull\",mode=\"direct\"}"],
      (std::vector<double>{15.5}));
  EXPECT_EQ(
      samples
          ["tpu_raiden_custom_histogram{direction=\"pull\",mode=\"reshard\"}"],
      (std::vector<double>{42.0}));
}

TEST(BufferedMetricsExporterTest, CardinalitySafeguardCap) {
  BufferedMetricsExporter exporter;
  for (int i = 0; i < 600; ++i) {
    std::string val = absl::StrCat("tag_", i);
    MetricLabel label[] = {{"series", val}};
    exporter.IncrementCounter(metric_names::kSentBytesTotal, label, 1);
  }

  auto samples = exporter.GetAndResetMetricSamples();
  // Bounded to at most kMaxLabeledSeries (512)
  EXPECT_LE(samples.size(), kMaxLabeledSeries);
}

TEST(BufferedMetricsExporterTest, UnknownMetricNameIsIgnoredSafely) {
  BufferedMetricsExporter exporter;
  exporter.IncrementCounter("nonexistent_counter", {}, 100);
  exporter.SetGauge("nonexistent_gauge", {}, 42.0);
  exporter.ObserveHistogram("nonexistent_histogram", {}, 1.23);

  EXPECT_THAT(exporter.GetAndResetMetricSamples(), IsEmpty());
}

TEST(BufferedMetricsExporterTest,
     InterleavedConcurrentEmissionsAndExtractions) {
  BufferedMetricsExporter exporter;
  constexpr int kNumWriters = 6;
  constexpr int kIncrementsPerWriter = 2000;
  constexpr uint64_t kDelta = 1;

  std::atomic<bool> writers_done{false};
  std::atomic<double> total_extracted{0.0};

  // Reader thread performing continuous extractions concurrently with writers
  std::thread reader([&] {
    while (!writers_done.load(std::memory_order_relaxed)) {
      auto samples = exporter.GetAndResetMetricSamples();
      if (auto it = samples.find("tpu_raiden_sent_bytes_total");
          it != samples.end()) {
        for (double v : it->second) {
          total_extracted.fetch_add(v, std::memory_order_relaxed);
        }
      }
    }
    // Final drain after writers finish
    auto final_samples = exporter.GetAndResetMetricSamples();
    if (auto it = final_samples.find("tpu_raiden_sent_bytes_total");
        it != final_samples.end()) {
      for (double v : it->second) {
        total_extracted.fetch_add(v, std::memory_order_relaxed);
      }
    }
  });

  std::vector<std::thread> writers;
  writers.reserve(kNumWriters);
  for (int w = 0; w < kNumWriters; ++w) {
    writers.emplace_back([&] {
      for (int i = 0; i < kIncrementsPerWriter; ++i) {
        exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, kDelta);
      }
    });
  }

  for (auto& writer : writers) {
    writer.join();
  }
  writers_done.store(true, std::memory_order_relaxed);
  reader.join();

  constexpr double kExpectedTotal =
      static_cast<double>(kNumWriters * kIncrementsPerWriter * kDelta);
  EXPECT_DOUBLE_EQ(total_extracted.load(), kExpectedTotal);
  EXPECT_THAT(exporter.GetAndResetMetricSamples(), IsEmpty());
}

TEST(BufferedMetricsExporterTest, ConcurrentLabeledEmissions) {
  BufferedMetricsExporter exporter;
  constexpr int kNumWriters = 4;
  constexpr int kIncrementsPerWriter = 1000;

  std::vector<std::thread> writers;
  writers.reserve(kNumWriters);
  for (int w = 0; w < kNumWriters; ++w) {
    writers.emplace_back([&exporter, w] {
      std::string dir = (w % 2 == 0) ? "push" : "pull";
      MetricLabel labels[] = {{"direction", dir}};
      for (int i = 0; i < kIncrementsPerWriter; ++i) {
        exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 10);
      }
    });
  }

  for (auto& writer : writers) {
    writer.join();
  }

  auto samples = exporter.GetAndResetMetricSamples();
  EXPECT_EQ(samples["tpu_raiden_sent_bytes_total{direction=\"push\"}"],
            (std::vector<double>{20000.0}));
  EXPECT_EQ(samples["tpu_raiden_sent_bytes_total{direction=\"pull\"}"],
            (std::vector<double>{20000.0}));
}

TEST(BufferedMetricsExporterTest, LargeLabelsExceedingStackBufferExport) {
  BufferedMetricsExporter exporter;
  const std::string long_value(kDefaultPrometheusStackBufferSize + 64, 'a');
  const MetricLabel label[] = {{"long_param", long_value}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, label, 500);

  EXPECT_THAT(
      exporter.GetAndResetMetricSamples(),
      ElementsAre(Pair(absl::StrCat("tpu_raiden_sent_bytes_total{long_param=\"",
                                    long_value, "\"}"),
                       ElementsAre(500.0))));
}

TEST(BufferedMetricsExporterTest, SpecialCharactersEscapedInExportedLabels) {
  BufferedMetricsExporter exporter;
  const MetricLabel label[] = {{"query", "line1\nline2\"quoted\"with\\slash"}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, label, 1);

  EXPECT_THAT(
      exporter.GetAndResetMetricSamples(),
      ElementsAre(Pair("tpu_raiden_sent_bytes_total{"
                       "query=\"line1\\nline2\\\"quoted\\\"with\\\\slash\"}",
                       ElementsAre(1.0))));
}

TEST(BufferedMetricsExporterTest, EmptyLabelValueExport) {
  BufferedMetricsExporter exporter;
  const MetricLabel label[] = {{"empty_key", ""}};
  exporter.IncrementCounter(metric_names::kSentBytesTotal, label, 10);

  EXPECT_THAT(exporter.GetAndResetMetricSamples(),
              ElementsAre(Pair("tpu_raiden_sent_bytes_total{empty_key=\"\"}",
                               ElementsAre(10.0))));
}

TEST(BufferedMetricsExporterTest,
     UnsortedMultiLabelOverflowAggregatesIntoSingleSeries) {
  BufferedMetricsExporter exporter;
  const std::string long_value(kDefaultPrometheusStackBufferSize + 64, 'x');
  const MetricLabel first_order[] = {{"z_key", long_value}, {"a_key", "val"}};
  const MetricLabel second_order[] = {{"a_key", "val"}, {"z_key", long_value}};

  exporter.IncrementCounter(metric_names::kSentBytesTotal, first_order, 100);
  exporter.IncrementCounter(metric_names::kSentBytesTotal, second_order, 200);

  EXPECT_THAT(
      exporter.GetAndResetMetricSamples(),
      ElementsAre(Pair(
          absl::StrCat("tpu_raiden_sent_bytes_total{a_key=\"val\",z_key=\"",
                       long_value, "\"}"),
          ElementsAre(300.0))));
}

TEST(BufferedMetricsExporterTest,
     UnsortedLabelsExceedingInlinedCapacitySortedAndAggregated) {
  BufferedMetricsExporter exporter;
  const MetricLabel descending[] = {
      {"k09", "v9"}, {"k08", "v8"}, {"k07", "v7"}, {"k06", "v6"}, {"k05", "v5"},
      {"k04", "v4"}, {"k03", "v3"}, {"k02", "v2"}, {"k01", "v1"}, {"k00", "v0"},
  };
  const MetricLabel ascending[] = {
      {"k00", "v0"}, {"k01", "v1"}, {"k02", "v2"}, {"k03", "v3"}, {"k04", "v4"},
      {"k05", "v5"}, {"k06", "v6"}, {"k07", "v7"}, {"k08", "v8"}, {"k09", "v9"},
  };

  exporter.IncrementCounter(metric_names::kSentBytesTotal, descending, 40);
  exporter.IncrementCounter(metric_names::kSentBytesTotal, ascending, 60);

  EXPECT_THAT(
      exporter.GetAndResetMetricSamples(),
      ElementsAre(Pair(
          "tpu_raiden_sent_bytes_total{k00=\"v0\",k01=\"v1\",k02=\"v2\","
          "k03=\"v3\",k04=\"v4\",k05=\"v5\",k06=\"v6\",k07=\"v7\",k08=\"v8\","
          "k09=\"v9\"}",
          ElementsAre(100.0))));
}

}  // namespace
}  // namespace tpu_raiden::telemetry
