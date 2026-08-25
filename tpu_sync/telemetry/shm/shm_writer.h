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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_SHM_SHM_WRITER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_SHM_SHM_WRITER_H_

#include <sys/types.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/shm/shm_layout.h"

namespace tpu_raiden::telemetry {

// Configuration options for the shared-memory telemetry writer.
struct ShmWriterOptions {
  // Directory where shared-memory segment files (.mmap) are stored (e.g.
  // "/dev/shm" or "/tmp").
  std::string shm_dir;
  // Worker local rank identifier in distributed multi-rank environments.
  std::string local_rank;
  // POSIX file creation mode/permissions for segment files. Defaults to 0644 so
  // out-of-process metric collectors can read shared-memory segments.
  mode_t file_mode = 0644;
};

// Thread-safe shared-memory telemetry writer using memory-mapped segment files.
// Uses dynamic on-demand Table of Contents, multi-chunk expansion, and
// reader-lock protected pointer caching. Intended for high-throughput,
// low-overhead metric recording for out-of-process collection and aggregation.
//
// NOTE: ShmWriter is intentionally standalone (not derived from MetricsBackend)
// to avoid virtual dispatch overhead and because shared-memory telemetry does
// not maintain in-process textual snapshots or stateful sample reset buffers.
class ShmWriter {
 public:
  static constexpr size_t kMaxChunks = 16;

  explicit ShmWriter(const ShmWriterOptions& options = {});
  ~ShmWriter();

  ShmWriter(const ShmWriter&) = delete;
  ShmWriter& operator=(const ShmWriter&) = delete;
  ShmWriter(ShmWriter&&) = delete;
  ShmWriter& operator=(ShmWriter&&) = delete;

  // Metric recording methods.
  void IncrementCounter(absl::string_view name, LabelSpan labels,
                        uint64_t val = 1) const;
  void SetGauge(absl::string_view name, LabelSpan labels, double val) const;
  void ObserveHistogram(absl::string_view name, LabelSpan labels,
                        double val) const;

 private:
  friend class ShmWriterTest;

  struct MetricKey {
    std::string name;
    std::string encoded_labels;

    template <typename H>
    friend H AbslHashValue(H h, const MetricKey& k) {
      return H::combine(std::move(h), k.name, k.encoded_labels);
    }

    bool operator==(const MetricKey& other) const = default;
  };

  struct MetricKeyView {
    absl::string_view name;
    absl::string_view encoded_labels;

    template <typename H>
    friend H AbslHashValue(H h, const MetricKeyView& k) {
      return H::combine(std::move(h), k.name, k.encoded_labels);
    }

    bool operator==(const MetricKeyView& other) const = default;
  };

  struct MetricKeyHash {
    using is_transparent = void;
    template <typename T>
    size_t operator()(const T& k) const {
      return absl::HashOf(k.name, k.encoded_labels);
    }
  };

  struct MetricKeyEq {
    using is_transparent = void;
    template <typename T, typename U>
    bool operator()(const T& a, const U& b) const {
      return a.name == b.name && a.encoded_labels == b.encoded_labels;
    }
  };

  struct Chunk {
    int fd = -1;
    ShmSegmentLayout* segment = nullptr;
    std::string file_path;
    uint32_t chunk_index = 0;
  };

  bool AllocateNewChunk() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  template <typename SlotT>
  auto& GetCache() const ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
    if constexpr (std::is_same_v<SlotT, std::atomic<uint64_t>>) {
      return counter_cache_;
    } else if constexpr (std::is_same_v<SlotT, std::atomic<double>>) {
      return gauge_cache_;
    } else if constexpr (std::is_same_v<SlotT, ShmHistogramSlot>) {
      return histogram_cache_;
    }
  }

  template <typename SlotT, MetricType Type>
  SlotT* GetOrCreateSlot(absl::string_view name, LabelSpan labels) const;

  ShmWriterOptions options_;
  std::string uuid_;

  mutable absl::Mutex mutex_;
  mutable std::vector<Chunk> chunks_ ABSL_GUARDED_BY(mutex_);
  mutable absl::flat_hash_map<MetricKey, std::atomic<uint64_t>*, MetricKeyHash,
                              MetricKeyEq>
      counter_cache_ ABSL_GUARDED_BY(mutex_);
  mutable absl::flat_hash_map<MetricKey, std::atomic<double>*, MetricKeyHash,
                              MetricKeyEq>
      gauge_cache_ ABSL_GUARDED_BY(mutex_);
  mutable absl::flat_hash_map<MetricKey, ShmHistogramSlot*, MetricKeyHash,
                              MetricKeyEq>
      histogram_cache_ ABSL_GUARDED_BY(mutex_);
};

// Encodes a span of metric labels into a canonical semicolon-delimited string
// representation (e.g., "key1=val1;key2=val2").
//
// Labels are sorted lexicographically by key to guarantee deterministic output
// regardless of caller-specified label order. Delimiter characters ('\', '=',
// ';') in keys and values are escaped with backslashes (e.g. '\=', '\;', '\\')
// to prevent collision and enable unambiguous round-trip parsing by collectors.
std::string EncodeLabels(LabelSpan labels);

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_SHM_SHM_WRITER_H_
