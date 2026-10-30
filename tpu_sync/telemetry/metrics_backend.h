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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_METRICS_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_METRICS_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tpu_raiden::telemetry {

enum class MetricType : uint8_t {
  kCounter,
  kGauge,
  kHistogram,
};

inline constexpr double kDefaultHistogramBuckets[] = {
    0.1,    0.25,   0.5,    1.0,     2.5,     5.0,    10.0,
    25.0,   50.0,   100.0,  250.0,   500.0,   750.0,  1000.0,
    2500.0, 5000.0, 7500.0, 10000.0, 25000.0, 50000.0};

inline constexpr int kDefaultExporterPort = 0;
inline constexpr absl::string_view kDefaultExporterHost = "0.0.0.0";
inline constexpr int kMinPort = 1;
inline constexpr int kMaxPort = 65535;

// Configuration options for metric exporters.
struct ExporterOptions {
  // Bind address for HTTP metric exporter. Defaults to "0.0.0.0".
  std::string bind_address{kDefaultExporterHost};
  // TCP port for HTTP metric exporter. If <= 0, HTTP serving is disabled.
  int port = kDefaultExporterPort;
  // Non-owning view of histogram bucket boundaries. Defaults to
  // kDefaultHistogramBuckets and is copied by the exporter during construction.
  // Note: Shared-memory backends (e.g. BaseShmExporter and its derived
  // exporters like PrometheusShmExporter) ignore custom bucket spans and use
  // fixed compile-time bucket layouts (kDefaultHistogramBuckets).
  absl::Span<const double> custom_buckets = kDefaultHistogramBuckets;
  // Worker local rank identifier in distributed multi-rank environments. Note:
  // Fallback to the LOCAL_RANK environment variable is handled at the
  // metrics_api initialization layer; may be std::nullopt for non-worker or
  // coordinator processes that do not set LOCAL_RANK.
  std::optional<std::string> local_rank;
  // Absolute path of the directory for POSIX shared-memory segments used for
  // inter-worker telemetry aggregation via BaseShmExporter and its derived
  // exporters. Trailing slashes are stripped; the root path "/" is prohibited.
  std::optional<std::string> shm_dir;
};

// Structure defining centralized metadata for a Raiden metric.
struct MetricMetadata {
  absl::string_view name;
  absl::string_view description;
  MetricType type;
  absl::Span<const double> buckets = kDefaultHistogramBuckets;
  absl::Span<const absl::string_view> label_names = {};

  // Enables compile-time member-wise value equality comparisons and test
  // assertions with zero runtime overhead.
  bool operator==(const MetricMetadata& other) const = default;
};

namespace metric_names {

inline constexpr absl::string_view kSentBytesTotal = "sent_bytes_total";
inline constexpr absl::string_view kReceivedBytesTotal = "received_bytes_total";
inline constexpr absl::string_view kTransferFailuresTotal =
    "transfer_failures_total";
inline constexpr absl::string_view kTransferDurationMs = "transfer_duration_ms";
inline constexpr absl::string_view kP2pTransferTimeMs = "p2p_transfer_time_ms";

inline constexpr absl::string_view kH2dBytesTotal = "h2d_bytes_total";
inline constexpr absl::string_view kH2dTransferTimeMs = "h2d_transfer_time_ms";
inline constexpr absl::string_view kH2dDmaTimeMs = "h2d_dma_time_ms";
inline constexpr absl::string_view kD2hBytesTotal = "d2h_bytes_total";
inline constexpr absl::string_view kD2hTransferTimeMs = "d2h_transfer_time_ms";
inline constexpr absl::string_view kD2hDmaTimeMs = "d2h_dma_time_ms";
inline constexpr absl::string_view kBufferAllocatedBytes =
    "buffer_allocated_bytes";
inline constexpr absl::string_view kStorageBytesTotal = "storage_bytes_total";
inline constexpr absl::string_view kHostMemoryBreakdownBytes =
    "host_memory_breakdown_bytes";
inline constexpr absl::string_view kHostNumaMemoryBytes =
    "host_numa_memory_bytes";
inline constexpr absl::string_view kStorageFilesystemBytes =
    "storage_filesystem_bytes";

inline constexpr absl::string_view kWeightSyncSentBytesTotal =
    "weight_sync_sent_bytes_total";
inline constexpr absl::string_view kWeightSyncReceivedBytesTotal =
    "weight_sync_received_bytes_total";
inline constexpr absl::string_view kWeightSyncTransferFailuresTotal =
    "weight_sync_transfer_failures_total";
inline constexpr absl::string_view kWeightSyncP2pTransferTimeMs =
    "weight_sync_p2p_transfer_time_ms";
inline constexpr absl::string_view kWeightSyncD2hTransferTimeMs =
    "weight_sync_d2h_transfer_time_ms";
inline constexpr absl::string_view kWeightSyncH2dTransferTimeMs =
    "weight_sync_h2d_transfer_time_ms";
inline constexpr absl::string_view kWeightSyncPushDurationMs =
    "weight_sync_push_duration_ms";
inline constexpr absl::string_view kWeightSyncE2eBroadcastDurationMs =
    "weight_sync_e2e_broadcast_duration_ms";
inline constexpr absl::string_view kWeightSyncBufferAllocatedBytes =
    "weight_sync_buffer_allocated_bytes";
inline constexpr absl::string_view kWeightSyncTilingTimeMs =
    "weight_sync_tiling_time_ms";
inline constexpr absl::string_view kWeightSyncDetilingTimeMs =
    "weight_sync_detiling_time_ms";
inline constexpr absl::string_view kWeightSyncScheduleGenerationTimeMs =
    "weight_sync_schedule_generation_time_ms";

inline constexpr absl::string_view kPeregrineE2eLatencyUs =
    "peregrine_e2e_latency_us";
inline constexpr absl::string_view kPeregrineRequestSizeBytes =
    "peregrine_request_size_bytes";
inline constexpr absl::string_view kPeregrineBytesTotal =
    "peregrine_bytes_total";
inline constexpr absl::string_view kPeregrineErrorsTotal =
    "peregrine_errors_total";

}  // namespace metric_names

namespace metric_descriptions {

inline constexpr absl::string_view kSentBytesTotal =
    "Total count of bytes sent over TPU Raiden interfaces.";
inline constexpr absl::string_view kReceivedBytesTotal =
    "Total count of bytes received over TPU Raiden interfaces.";
inline constexpr absl::string_view kTransferFailuresTotal =
    "Cumulative total count of transfer failures across all interfaces.";
inline constexpr absl::string_view kTransferDurationMs =
    "Measures End-to-End (E2E) latency bound around the entire request in "
    "milliseconds, including setup delays.";
inline constexpr absl::string_view kP2pTransferTimeMs =
    "Peer-to-Peer network batch transfer latency in milliseconds.";
inline constexpr absl::string_view kBufferAllocatedBytes =
    "Current host DRAM buffer capacity allocated in bytes for KV cache staging "
    "across all layers and shards.";
inline constexpr absl::string_view kHostMemoryBreakdownBytes =
    "Host DRAM memory breakdown in bytes by component (kv_pool, process_heap, "
    "page_cache, free)";
inline constexpr absl::string_view kHostNumaMemoryBytes =
    "Per-NUMA-node host DRAM memory in bytes by state (used, free, total)";

inline constexpr absl::string_view kH2dBytesTotal =
    "Cumulative bytes requested for Host DRAM to Device HBM transfers that "
    "completed successfully.";
inline constexpr absl::string_view kH2dTransferTimeMs =
    "Host-to-Device transfer latency in milliseconds.";
inline constexpr absl::string_view kH2dDmaTimeMs =
    "Non-overlapping Host-to-Device PCIe DMA wire transfer duration in "
    "milliseconds.";
inline constexpr absl::string_view kD2hBytesTotal =
    "Cumulative bytes requested for Device HBM to Host DRAM transfers that "
    "completed successfully.";
inline constexpr absl::string_view kD2hTransferTimeMs =
    "Device-to-Host transfer latency in milliseconds.";
inline constexpr absl::string_view kD2hDmaTimeMs =
    "Non-overlapping Device-to-Host PCIe DMA wire transfer duration in "
    "milliseconds.";
inline constexpr absl::string_view kStorageBytesTotal =
    "Total bytes read from or written to secondary storage";
inline constexpr absl::string_view kStorageFilesystemBytes =
    "Secondary storage filesystem capacity in bytes by mount and state (total, "
    "used, free)";

inline constexpr absl::string_view kWeightSyncSentBytesTotal =
    "Total count of bytes sent over TPU Raiden weight sync interfaces.";
inline constexpr absl::string_view kWeightSyncReceivedBytesTotal =
    "Total count of bytes received over TPU Raiden weight sync interfaces.";
inline constexpr absl::string_view kWeightSyncTransferFailuresTotal =
    "Cumulative total count of transfer failures during weight "
    "synchronization.";
inline constexpr absl::string_view kWeightSyncP2pTransferTimeMs =
    "Peer-to-Peer network transfer latency for weight sync in milliseconds.";
inline constexpr absl::string_view kWeightSyncD2hTransferTimeMs =
    "Device-to-Host transfer latency for weight sync in milliseconds.";
inline constexpr absl::string_view kWeightSyncH2dTransferTimeMs =
    "Host-to-Device transfer latency for weight sync in milliseconds.";
inline constexpr absl::string_view kWeightSyncPushDurationMs =
    "Duration of weight sync push operations in milliseconds.";
inline constexpr absl::string_view kWeightSyncE2eBroadcastDurationMs =
    "End-to-end broadcast transfer duration for weight sync in milliseconds.";
inline constexpr absl::string_view kWeightSyncBufferAllocatedBytes =
    "Current host DRAM buffer capacity allocated in bytes for weight sync "
    "staging across all layers and shards.";
inline constexpr absl::string_view kWeightSyncTilingTimeMs =
    "Time spent tiling weight buffers in milliseconds.";
inline constexpr absl::string_view kWeightSyncDetilingTimeMs =
    "Time spent detiling weight buffers in milliseconds.";
inline constexpr absl::string_view kWeightSyncScheduleGenerationTimeMs =
    "Time spent generating weight sync transfer schedule in milliseconds.";

inline constexpr absl::string_view kPeregrineE2eLatencyUs =
    "Peregrine end-to-end transfer duration in microseconds.";
inline constexpr absl::string_view kPeregrineRequestSizeBytes =
    "Peregrine per-operation transfer request size in bytes.";
inline constexpr absl::string_view kPeregrineBytesTotal =
    "Total payload bytes transferred across Peregrine data channels.";
inline constexpr absl::string_view kPeregrineErrorsTotal =
    "Cumulative total count of Peregrine transfer failures.";

}  // namespace metric_descriptions

namespace metric_labels {

inline constexpr absl::string_view kDirection = "direction";
inline constexpr absl::string_view kDirectionPush = "push";
inline constexpr absl::string_view kDirectionPull = "pull";
inline constexpr absl::string_view kDirectionPullResponse = "pull_response";
inline constexpr absl::string_view kDirectionWrite = "write";
inline constexpr absl::string_view kDirectionRead = "read";

inline constexpr absl::string_view kErrorCode = "error_code";
inline constexpr absl::string_view kLocalRank = "local_rank";
inline constexpr absl::string_view kSrcIp = "src_ip";
inline constexpr absl::string_view kDstIp = "dst_ip";
inline constexpr absl::string_view kUnknownIp = "unknown";
inline constexpr absl::string_view kHostIp = "host_ip";
inline constexpr absl::string_view kComponent = "component";
inline constexpr absl::string_view kComponentKvPool = "kv_pool";
inline constexpr absl::string_view kComponentProcessHeap = "process_heap";
inline constexpr absl::string_view kComponentPageCache = "page_cache";
inline constexpr absl::string_view kComponentFree = "free";
inline constexpr absl::string_view kNumaNode = "numa_node";
inline constexpr absl::string_view kMount = "mount";
inline constexpr absl::string_view kState = "state";
inline constexpr absl::string_view kStateUsed = "used";
inline constexpr absl::string_view kStateFree = "free";
inline constexpr absl::string_view kStateTotal = "total";

inline constexpr absl::string_view kDirectionLabels[] = {kDirection};
inline constexpr absl::string_view kSentBytesLabels[] = {kDirection, kSrcIp,
                                                         kDstIp};
inline constexpr absl::string_view kTransferFailuresLabels[] = {kDirection,
                                                                kErrorCode};
inline constexpr absl::string_view kP2pTransferTimeLabels[] = {kSrcIp, kDstIp,
                                                               kLocalRank};
inline constexpr absl::string_view kPcieTransferLabels[] = {kHostIp,
                                                            kLocalRank};
inline constexpr absl::string_view kStorageBytesLabels[] = {kLocalRank,
                                                            kDirection};
inline constexpr absl::string_view kHostMemoryBreakdownLabels[] = {kComponent};
inline constexpr absl::string_view kHostNumaMemoryLabels[] = {kNumaNode,
                                                              kState};
inline constexpr absl::string_view kStorageFilesystemBytesLabels[] = {kMount,
                                                                      kState};

}  // namespace metric_labels

namespace metric_metadata {

using metric_labels::kDirectionLabels;
using metric_labels::kHostMemoryBreakdownLabels;
using metric_labels::kHostNumaMemoryLabels;
using metric_labels::kP2pTransferTimeLabels;
using metric_labels::kPcieTransferLabels;
using metric_labels::kSentBytesLabels;
using metric_labels::kStorageBytesLabels;
using metric_labels::kStorageFilesystemBytesLabels;
using metric_labels::kTransferFailuresLabels;

inline constexpr MetricMetadata kSentBytesTotal{
    .name = metric_names::kSentBytesTotal,
    .description = metric_descriptions::kSentBytesTotal,
    .type = MetricType::kCounter,
    .label_names = kSentBytesLabels};

inline constexpr MetricMetadata kReceivedBytesTotal{
    .name = metric_names::kReceivedBytesTotal,
    .description = metric_descriptions::kReceivedBytesTotal,
    .type = MetricType::kCounter,
    .label_names = kDirectionLabels};

inline constexpr MetricMetadata kTransferFailuresTotal{
    .name = metric_names::kTransferFailuresTotal,
    .description = metric_descriptions::kTransferFailuresTotal,
    .type = MetricType::kCounter,
    .label_names = kTransferFailuresLabels};

inline constexpr MetricMetadata kTransferDurationMs{
    .name = metric_names::kTransferDurationMs,
    .description = metric_descriptions::kTransferDurationMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kP2pTransferTimeMs{
    .name = metric_names::kP2pTransferTimeMs,
    .description = metric_descriptions::kP2pTransferTimeMs,
    .type = MetricType::kHistogram,
    .label_names = kP2pTransferTimeLabels};

inline constexpr MetricMetadata kH2dBytesTotal{
    .name = metric_names::kH2dBytesTotal,
    .description = metric_descriptions::kH2dBytesTotal,
    .type = MetricType::kCounter,
    .label_names = kPcieTransferLabels};

inline constexpr MetricMetadata kH2dTransferTimeMs{
    .name = metric_names::kH2dTransferTimeMs,
    .description = metric_descriptions::kH2dTransferTimeMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kH2dDmaTimeMs{
    .name = metric_names::kH2dDmaTimeMs,
    .description = metric_descriptions::kH2dDmaTimeMs,
    .type = MetricType::kHistogram,
    .label_names = kPcieTransferLabels};

inline constexpr MetricMetadata kD2hBytesTotal{
    .name = metric_names::kD2hBytesTotal,
    .description = metric_descriptions::kD2hBytesTotal,
    .type = MetricType::kCounter,
    .label_names = kPcieTransferLabels};

inline constexpr MetricMetadata kD2hTransferTimeMs{
    .name = metric_names::kD2hTransferTimeMs,
    .description = metric_descriptions::kD2hTransferTimeMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kD2hDmaTimeMs{
    .name = metric_names::kD2hDmaTimeMs,
    .description = metric_descriptions::kD2hDmaTimeMs,
    .type = MetricType::kHistogram,
    .label_names = kPcieTransferLabels};

inline constexpr MetricMetadata kBufferAllocatedBytes{
    .name = metric_names::kBufferAllocatedBytes,
    .description = metric_descriptions::kBufferAllocatedBytes,
    .type = MetricType::kGauge};

inline constexpr MetricMetadata kStorageBytesTotal{
    .name = metric_names::kStorageBytesTotal,
    .description = metric_descriptions::kStorageBytesTotal,
    .type = MetricType::kCounter,
    .label_names = kStorageBytesLabels};

inline constexpr MetricMetadata kHostMemoryBreakdownBytes{
    .name = metric_names::kHostMemoryBreakdownBytes,
    .description = metric_descriptions::kHostMemoryBreakdownBytes,
    .type = MetricType::kGauge,
    .label_names = kHostMemoryBreakdownLabels};

inline constexpr MetricMetadata kHostNumaMemoryBytes{
    .name = metric_names::kHostNumaMemoryBytes,
    .description = metric_descriptions::kHostNumaMemoryBytes,
    .type = MetricType::kGauge,
    .label_names = kHostNumaMemoryLabels};

inline constexpr MetricMetadata kStorageFilesystemBytes{
    .name = metric_names::kStorageFilesystemBytes,
    .description = metric_descriptions::kStorageFilesystemBytes,
    .type = MetricType::kGauge,
    .label_names = kStorageFilesystemBytesLabels};
inline constexpr MetricMetadata kWeightSyncSentBytesTotal{
    .name = metric_names::kWeightSyncSentBytesTotal,
    .description = metric_descriptions::kWeightSyncSentBytesTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kWeightSyncReceivedBytesTotal{
    .name = metric_names::kWeightSyncReceivedBytesTotal,
    .description = metric_descriptions::kWeightSyncReceivedBytesTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kWeightSyncTransferFailuresTotal{
    .name = metric_names::kWeightSyncTransferFailuresTotal,
    .description = metric_descriptions::kWeightSyncTransferFailuresTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kWeightSyncP2pTransferTimeMs{
    .name = metric_names::kWeightSyncP2pTransferTimeMs,
    .description = metric_descriptions::kWeightSyncP2pTransferTimeMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kWeightSyncD2hTransferTimeMs{
    .name = metric_names::kWeightSyncD2hTransferTimeMs,
    .description = metric_descriptions::kWeightSyncD2hTransferTimeMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kWeightSyncH2dTransferTimeMs{
    .name = metric_names::kWeightSyncH2dTransferTimeMs,
    .description = metric_descriptions::kWeightSyncH2dTransferTimeMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kWeightSyncPushDurationMs{
    .name = metric_names::kWeightSyncPushDurationMs,
    .description = metric_descriptions::kWeightSyncPushDurationMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kWeightSyncE2eBroadcastDurationMs{
    .name = metric_names::kWeightSyncE2eBroadcastDurationMs,
    .description = metric_descriptions::kWeightSyncE2eBroadcastDurationMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kWeightSyncBufferAllocatedBytes{
    .name = metric_names::kWeightSyncBufferAllocatedBytes,
    .description = metric_descriptions::kWeightSyncBufferAllocatedBytes,
    .type = MetricType::kGauge};

inline constexpr MetricMetadata kWeightSyncTilingTimeMs{
    .name = metric_names::kWeightSyncTilingTimeMs,
    .description = metric_descriptions::kWeightSyncTilingTimeMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kWeightSyncDetilingTimeMs{
    .name = metric_names::kWeightSyncDetilingTimeMs,
    .description = metric_descriptions::kWeightSyncDetilingTimeMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kWeightSyncScheduleGenerationTimeMs{
    .name = metric_names::kWeightSyncScheduleGenerationTimeMs,
    .description = metric_descriptions::kWeightSyncScheduleGenerationTimeMs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kPeregrineE2eLatencyUs{
    .name = metric_names::kPeregrineE2eLatencyUs,
    .description = metric_descriptions::kPeregrineE2eLatencyUs,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kPeregrineRequestSizeBytes{
    .name = metric_names::kPeregrineRequestSizeBytes,
    .description = metric_descriptions::kPeregrineRequestSizeBytes,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kPeregrineBytesTotal{
    .name = metric_names::kPeregrineBytesTotal,
    .description = metric_descriptions::kPeregrineBytesTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kPeregrineErrorsTotal{
    .name = metric_names::kPeregrineErrorsTotal,
    .description = metric_descriptions::kPeregrineErrorsTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kAllMetrics[] = {
    kSentBytesTotal,
    kReceivedBytesTotal,
    kTransferFailuresTotal,
    kTransferDurationMs,
    kP2pTransferTimeMs,
    kH2dBytesTotal,
    kH2dTransferTimeMs,
    kH2dDmaTimeMs,
    kD2hBytesTotal,
    kD2hTransferTimeMs,
    kD2hDmaTimeMs,
    kBufferAllocatedBytes,
    kWeightSyncSentBytesTotal,
    kWeightSyncReceivedBytesTotal,
    kWeightSyncTransferFailuresTotal,
    kWeightSyncP2pTransferTimeMs,
    kWeightSyncD2hTransferTimeMs,
    kWeightSyncH2dTransferTimeMs,
    kWeightSyncPushDurationMs,
    kWeightSyncE2eBroadcastDurationMs,
    kWeightSyncBufferAllocatedBytes,
    kWeightSyncTilingTimeMs,
    kWeightSyncDetilingTimeMs,
    kWeightSyncScheduleGenerationTimeMs,
    kPeregrineE2eLatencyUs,
    kPeregrineRequestSizeBytes,
    kPeregrineBytesTotal,
    kPeregrineErrorsTotal,
    kStorageBytesTotal,
    kHostMemoryBreakdownBytes,
    kHostNumaMemoryBytes,
    kStorageFilesystemBytes,
};
}  // namespace metric_metadata

// Structure defining a metric key-value label pair.
struct MetricLabel {
  absl::string_view key;
  absl::string_view value;

  auto operator<=>(const MetricLabel& other) const = default;
};

// Allocation-free label view span type definition
using LabelSpan = absl::Span<const MetricLabel>;

// Standard inline capacity for label containers across telemetry pipelines.
// Guarantees zero heap allocation for label sets up to this size.
inline constexpr size_t kDefaultInlinedLabelCapacity = 8;

// Abstract Dual-Backend Interface
class MetricsBackend {
 public:
  MetricsBackend() = default;
  MetricsBackend(const MetricsBackend&) = delete;
  MetricsBackend& operator=(const MetricsBackend&) = delete;
  MetricsBackend(MetricsBackend&&) = delete;
  MetricsBackend& operator=(MetricsBackend&&) = delete;

  virtual ~MetricsBackend() = default;

  virtual void IncrementCounter(absl::string_view name, LabelSpan labels,
                                uint64_t val) const = 0;

  virtual void SetGauge(absl::string_view name, LabelSpan labels,
                        double val) const = 0;

  virtual void ObserveHistogram(absl::string_view name, LabelSpan labels,
                                double val) const = 0;

  virtual std::string GetTextSnapshot() const = 0;

  virtual std::map<std::string, std::vector<double>>
  GetAndResetMetricSamples() {
    return {};
  }
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_METRICS_BACKEND_H_
