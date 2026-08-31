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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BUFFERED_METRICS_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BUFFERED_METRICS_EXPORTER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

inline constexpr size_t kDefaultQueueBufferSize = 4096;

// Fixed-capacity sample buffer for step-level metric observations.
// Uses an absl::Mutex to ensure thread safety with O(1) buffer swapping
// on extraction.
template <size_t N = kDefaultQueueBufferSize>
class QueueBuffer {
 public:
  explicit QueueBuffer(size_t max_capacity = N) : max_capacity_(max_capacity) {}

  // Adds a value to the buffer. If the buffer is full, the value is dropped and
  // a rate-limited warning is logged.
  void Push(double val) {
    absl::MutexLock lock(mutex_);
    if (samples_.size() < max_capacity_) {
      samples_.push_back(val);
    } else {
      LOG_EVERY_N_SEC(WARNING, 5)
          << "QueueBuffer capacity (" << max_capacity_
          << ") exceeded; dropping metric sample.";
    }
  }

  // Returns the values in the buffer and resets the buffer to empty.
  // The returned vector will contain at most max_capacity_ values.
  std::vector<double> ExtractAndReset() {
    std::vector<double> extracted;
    {
      absl::MutexLock lock(mutex_);
      extracted.swap(samples_);
    }
    return extracted;
  }

 private:
  const size_t max_capacity_;
  mutable absl::Mutex mutex_;
  std::vector<double> samples_ ABSL_GUARDED_BY(mutex_);
};

// Accumulator for counter deltas.
class LockFreeCounterAccumulator {
 public:
  // Adds a value to the counter.
  void Add(uint64_t val) { value_.fetch_add(val, std::memory_order_relaxed); }

  // Returns the current value of the counter and resets it to 0.
  uint64_t ExchangeAndReset() {
    return value_.exchange(0, std::memory_order_relaxed);
  }

 private:
  std::atomic<uint64_t> value_{0};
};

inline constexpr size_t kMaxLabeledSeries = 512;
// Formats canonical Prometheus label string for in-memory series
// identification. Returns "{key1=\"val1\",key2=\"val2\"}" sorted by key, with
// Prometheus character escaping. Returns "" if labels are empty.
std::string FormatCanonicalLabels(LabelSpan labels);

// Encapsulates all time-series buffers for a single metric family.
template <typename AccumulatorType>
class MetricFamilyBuffer {
 public:
  MetricFamilyBuffer() = default;

  MetricFamilyBuffer(const MetricFamilyBuffer&) = delete;
  MetricFamilyBuffer& operator=(const MetricFamilyBuffer&) = delete;
  MetricFamilyBuffer(MetricFamilyBuffer&&) = delete;
  MetricFamilyBuffer& operator=(MetricFamilyBuffer&&) = delete;

  // Returns the accumulator for the given label span, creating one if it does
  // not yet exist. When labels are empty, returns the unlabeled_ fast path
  // without locks.
  AccumulatorType* GetOrCreate(LabelSpan labels) const {
    if (labels.empty()) {
      return &unlabeled_;
    }
    std::string canonical_labels = FormatCanonicalLabels(labels);
    {
      absl::ReaderMutexLock lock(labeled_mu_);
      if (auto it = labeled_.find(canonical_labels); it != labeled_.end()) {
        return it->second.get();
      }
    }
    absl::MutexLock lock(labeled_mu_);
    auto it = labeled_.find(canonical_labels);
    if (it != labeled_.end()) {
      return it->second.get();
    }
    if (labeled_.size() >= kMaxLabeledSeries) {
      LOG_EVERY_N_SEC(WARNING, 5)
          << "Max labeled series capacity (" << kMaxLabeledSeries
          << ") exceeded; dropping metric series for labels: "
          << canonical_labels;
      return nullptr;
    }
    auto [insert_it, _] = labeled_.try_emplace(
        std::move(canonical_labels), std::make_unique<AccumulatorType>());
    return insert_it->second.get();
  }

  template <typename Fn>
  void ForEachAccumulator(Fn&& fn) const {
    fn(/*canonical_labels=*/"", &unlabeled_);
    absl::ReaderMutexLock lock(labeled_mu_);
    for (const auto& [canonical_labels, acc] : labeled_) {
      fn(canonical_labels, acc.get());
    }
  }

 private:
  mutable AccumulatorType unlabeled_;
  mutable absl::Mutex labeled_mu_;
  mutable absl::flat_hash_map<std::string, std::unique_ptr<AccumulatorType>>
      labeled_ ABSL_GUARDED_BY(labeled_mu_);
};

// MetricsBackend for step-level sample buffering.
//
// Thread Safety:
// IncrementCounter(), SetGauge(), and ObserveHistogram() are thread-safe.
// Counters use atomic fetch-add, while gauges and histograms use fine-grained
// mutex-protected sample buffers.
class BufferedMetricsExporter : public MetricsBackend {
 public:
  explicit BufferedMetricsExporter(
      absl::Span<const MetricMetadata> metrics = metric_metadata::kAllMetrics);
  ~BufferedMetricsExporter() override = default;

  BufferedMetricsExporter(const BufferedMetricsExporter&) = delete;
  BufferedMetricsExporter& operator=(const BufferedMetricsExporter&) = delete;
  BufferedMetricsExporter(BufferedMetricsExporter&&) = delete;
  BufferedMetricsExporter& operator=(BufferedMetricsExporter&&) = delete;

  void IncrementCounter(absl::string_view name, LabelSpan labels,
                        uint64_t val) const override;

  void SetGauge(absl::string_view name, LabelSpan labels,
                double val) const override;

  void ObserveHistogram(absl::string_view name, LabelSpan labels,
                        double val) const override;

  std::string GetTextSnapshot() const override { return ""; }

  std::map<std::string, std::vector<double>> GetAndResetMetricSamples()
      override;

 private:
  absl::flat_hash_map<
      std::string,
      std::unique_ptr<MetricFamilyBuffer<LockFreeCounterAccumulator>>>
      counters_;
  absl::flat_hash_map<
      std::string,
      std::unique_ptr<MetricFamilyBuffer<QueueBuffer<>>>>
      gauges_;
  absl::flat_hash_map<
      std::string,
      std::unique_ptr<MetricFamilyBuffer<QueueBuffer<>>>>
      histograms_;
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BUFFERED_METRICS_EXPORTER_H_
