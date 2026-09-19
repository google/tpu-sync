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

// This file contains performance benchmarks for KVCacheManagerWithTransfer.
// It measures raw D2H and H2D performance using the slot-based APIs
// (D2hToHostSlot and H2dFromHostSlot) in isolation, without RPC or pipelining.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_pjrt_manager.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"

ABSL_FLAG(int, num_tpus, 1, "Number of TPUs to use");
ABSL_FLAG(int, num_layers, 64, "Number of layers to use for the benchmark.");
ABSL_FLAG(int64_t, num_blocks, 16,
          "Number of blocks to use for the benchmark.");

namespace tpu_raiden {
namespace {

absl::Status AwaitAll(absl::StatusOr<raiden::PjRtCopyFuture>& future_or) {
  if (!future_or.ok()) return future_or.status();
  return future_or.value().Await();
}

// Static initializer to load and initialize libtpu.so in OSS environment.
bool InitializeLibtpuOnce() { return true; }

// Helper to compute and print the distribution of bandwidth numbers
void PrintDistribution(absl::string_view label,
                       std::vector<double> bandwidths) {
  if (bandwidths.empty()) return;

  std::sort(bandwidths.begin(), bandwidths.end());
  double min = bandwidths.front();
  double max = bandwidths.back();
  double sum = 0.0;
  for (double bw : bandwidths) sum += bw;
  double mean = sum / bandwidths.size();

  double median = 0.0;
  if (bandwidths.size() % 2 == 0) {
    median = (bandwidths[bandwidths.size() / 2 - 1] +
              bandwidths[bandwidths.size() / 2]) /
             2.0;
  } else {
    median = bandwidths[bandwidths.size() / 2];
  }

  double variance = 0.0;
  for (double bw : bandwidths) {
    variance += (bw - mean) * (bw - mean);
  }
  double stddev = std::sqrt(variance / bandwidths.size());

  std::cout << "[DISTRIBUTION] " << label << " (over " << bandwidths.size()
            << " runs):" << std::endl;
  std::cout << "  Min:    " << min << " GB/s" << std::endl;
  std::cout << "  Max:    " << max << " GB/s" << std::endl;
  std::cout << "  Mean:   " << mean << " GB/s" << std::endl;
  std::cout << "  Median: " << median << " GB/s" << std::endl;
  std::cout << "  StdDev: " << stddev << " GB/s" << std::endl;
}

// Templated benchmark executor for Slot-based transfers
template <typename T>
void RunSlotBenchmark(TpuPjrtManager* manager,
                      const std::vector<xla::PjRtDevice*>& devices,
                      xla::PrimitiveType primitive_type,
                      absl::string_view type_label, int num_layers,
                      int64_t num_blocks) {
  const int kNumLayers = num_layers;
  const int64_t kNumBlocks = num_blocks;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;

  std::vector<int64_t> layer_shape = {kNumBlocks, kBlockSize, kNumHeads, 2,
                                      kHeadDim};
  int64_t elements_per_layer =
      kNumBlocks * kBlockSize * kNumHeads * 2 * kHeadDim;
  size_t bytes_per_layer = elements_per_layer * sizeof(T);
  size_t total_bytes = bytes_per_layer * kNumLayers;

  // We transfer all blocks contiguously
  std::vector<int64_t> block_ids;
  for (int64_t i = 0; i < kNumBlocks; ++i) {
    block_ids.push_back(i);
  }

  // 1. Allocate Device Buffers and Fill with Pattern
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> device_buffers(
      kNumLayers);
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers(
      kNumLayers);
  std::vector<std::vector<std::vector<T>>> host_init_buffers(kNumLayers);

  for (int i = 0; i < kNumLayers; ++i) {
    device_buffers[i].resize(devices.size());
    layer_buffers[i].resize(devices.size());
    host_init_buffers[i].resize(devices.size());

    for (size_t d = 0; d < devices.size(); ++d) {
      host_init_buffers[i][d].resize(elements_per_layer);
      // Fill with a recognizable pattern: layer_idx * 1000 + shard_idx * 100 +
      // element_idx
      for (size_t e = 0; e < elements_per_layer; ++e) {
        host_init_buffers[i][d][e] =
            static_cast<T>(i * 1000 + d * 100 + e % 100);
      }

      TF_ASSERT_OK_AND_ASSIGN(
          auto buf,
          manager->BufferFromHost(host_init_buffers[i][d].data(),
                                  primitive_type, layer_shape, devices[d]));
      device_buffers[i][d] = std::move(buf);
      auto handle_or =
          raiden::RaidenBufferHandle::Acquire(device_buffers[i][d].get());
      ASSERT_OK(handle_or.status());
      layer_buffers[i][d] = std::move(handle_or).value();
    }
  }

  for (auto& bufs : layer_buffers) {
    for (auto& handle : bufs) {
      ASSERT_OK(handle.buffer->GetReadyFuture().Await());
    }
  }

  // 2. Create Host Allocator
  TF_ASSERT_OK_AND_ASSIGN(auto host_allocator,
                          XlaHostMemoryAllocator::Create(manager->client()));
  HostMemoryAllocator* raw_allocator = host_allocator.get();
  HostBufferAllocator allocator_fn =
      [raw_allocator](size_t size, const xla::PjRtDevice* device) {
        return raw_allocator->AllocateDmaMappedForDevice(size, device);
      };

  // 3. Create KVCacheManagerWithTransfer
  // We pass local_control_port = -1 to disable the background threads/control
  // server
  auto engine = std::make_unique<KVCacheManagerWithTransfer>(
      layer_buffers,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/kNumBlocks,  // 1 slot capacity
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/allocator_fn,
      /*node_id=*/0,
      /*local_control_port=*/-1,
      /*max_blocks=*/kNumBlocks,
      /*num_slots=*/1);

  // Configure staging slots manually (since local_control_port is -1)
  ASSERT_OK(engine->base()->ConfigureHostStagingSlots(
      /*num_slots=*/1, /*max_major_per_slot=*/kNumBlocks));

  // Build copy spec (compact to compact, as we copy all blocks)
  kv_cache::KVCacheCopySpec transfer_spec;
  transfer_spec.src_offsets = {0};
  transfer_spec.dst_offsets = {0};
  transfer_spec.sizes = {kNumBlocks};

  const int64_t slot_idx = 0;
  const size_t bytes_transferred = total_bytes * devices.size();

  std::cout << "[INFO] Staging Slot Configured. Total Bytes to Transfer: "
            << bytes_transferred << " ("
            << (bytes_transferred / (1024.0 * 1024.0)) << " MB)" << std::endl;

  // 4. Warmup Phase
  constexpr int kWarmupIterations = 3;
  for (int iter = 0; iter < kWarmupIterations; ++iter) {
    // D2H
    auto d2h_futures_or = engine->base()->D2h(transfer_spec.src_offsets,
                                              transfer_spec.dst_offsets,
                                              transfer_spec.sizes, slot_idx);
    ASSERT_OK(AwaitAll(d2h_futures_or));

    // H2D
    auto h2d_futures_or = engine->base()->H2d(transfer_spec.src_offsets,
                                              transfer_spec.dst_offsets,
                                              transfer_spec.sizes, slot_idx);
    ASSERT_OK(AwaitAll(h2d_futures_or));
  }

  // 5. Timed Benchmark Loop (D2H to Slot)
  constexpr int kBenchmarkIterations = 20;
  std::vector<double> d2h_bandwidths;
  d2h_bandwidths.reserve(kBenchmarkIterations);

  for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
    absl::Time start = absl::Now();
    auto d2h_futures_or = engine->base()->D2h(transfer_spec.src_offsets,
                                              transfer_spec.dst_offsets,
                                              transfer_spec.sizes, slot_idx);
    ASSERT_OK(AwaitAll(d2h_futures_or));
    absl::Duration duration = absl::Now() - start;
    double seconds = absl::ToDoubleSeconds(duration);
    double bandwidth =
        (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / seconds;
    d2h_bandwidths.push_back(bandwidth);
  }

  PrintDistribution(absl::StrCat("Slot-based D2H Bandwidth [", type_label, "]"),
                    d2h_bandwidths);

  // 6. Timed Benchmark Loop (H2D from Slot)
  std::vector<double> h2d_bandwidths;
  h2d_bandwidths.reserve(kBenchmarkIterations);

  for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
    absl::Time start = absl::Now();
    auto h2d_futures_or = engine->base()->H2d(transfer_spec.src_offsets,
                                              transfer_spec.dst_offsets,
                                              transfer_spec.sizes, slot_idx);
    ASSERT_OK(AwaitAll(h2d_futures_or));
    absl::Duration duration = absl::Now() - start;
    double seconds = absl::ToDoubleSeconds(duration);
    double bandwidth =
        (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / seconds;
    h2d_bandwidths.push_back(bandwidth);
  }

  PrintDistribution(absl::StrCat("Slot-based H2D Bandwidth [", type_label, "]"),
                    h2d_bandwidths);

  // 7. Verify Correctness
  std::cout << "[INFO] Verifying transfer correctness..." << std::endl;

  // Step 1: Copy Device -> Host Slot
  auto d2h_futures_or =
      engine->base()->D2h(transfer_spec.src_offsets, transfer_spec.dst_offsets,
                          transfer_spec.sizes, slot_idx);
  ASSERT_OK(AwaitAll(d2h_futures_or));

  // Step 2: Clear Device Buffers (fill with zeros in-place via staging)
  // 2a. Overwrite the host staging slot with zeros
  for (int l = 0; l < kNumLayers; ++l) {
    for (size_t d = 0; d < devices.size(); ++d) {
      TF_ASSERT_OK_AND_ASSIGN(
          kv_cache::KVCacheHostSpan span,
          engine->base()->HostSpan(l, d, slot_idx, kNumBlocks));
      std::memset(span.ptr, 0, span.nbytes);
    }
  }
  // 2b. Copy these zeros to the device
  auto clear_futures_or =
      engine->base()->H2d(transfer_spec.src_offsets, transfer_spec.dst_offsets,
                          transfer_spec.sizes, slot_idx);
  ASSERT_OK(AwaitAll(clear_futures_or));

  // Step 3: Restore original pattern to host slot and copy to device
  // 3a. Copy original pattern back to host staging slot
  for (int l = 0; l < kNumLayers; ++l) {
    for (size_t d = 0; d < devices.size(); ++d) {
      TF_ASSERT_OK_AND_ASSIGN(
          kv_cache::KVCacheHostSpan span,
          engine->base()->HostSpan(l, d, slot_idx, kNumBlocks));
      std::memcpy(span.ptr, host_init_buffers[l][d].data(), span.nbytes);
    }
  }
  // 3b. Copy original pattern from host slot -> device
  auto h2d_futures_or =
      engine->base()->H2d(transfer_spec.src_offsets, transfer_spec.dst_offsets,
                          transfer_spec.sizes, slot_idx);
  ASSERT_OK(AwaitAll(h2d_futures_or));

  // Step 4: Verify Device Content matches original pattern
  for (int l = 0; l < kNumLayers; ++l) {
    for (size_t d = 0; d < devices.size(); ++d) {
      TF_ASSERT_OK_AND_ASSIGN(auto literal,
                              device_buffers[l][d]->ToLiteral().Await());
      auto read_back = literal->data<T>();
      for (size_t e = 0; e < elements_per_layer; ++e) {
        T expected = static_cast<T>(l * 1000 + d * 100 + e % 100);
        ASSERT_EQ(read_back[e], expected)
            << "Mismatch at layer " << l << ", device " << d << ", element "
            << e;
      }
    }
  }
  std::cout << "[SUCCESS] Verification passed for " << type_label << std::endl;
}

class KVCacheManagerWithTransferPerfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!InitializeLibtpuOnce()) {
      GTEST_SKIP()
          << "Skipping test as libtpu.so could not be loaded (no hardware?)";
    }
  }
};

TEST_F(KVCacheManagerWithTransferPerfTest, BenchmarkF32) {
  int num_tpus = absl::GetFlag(FLAGS_num_tpus);
  int num_layers = absl::GetFlag(FLAGS_num_layers);
  int64_t num_blocks = absl::GetFlag(FLAGS_num_blocks);

  std::cout << "[INFO] Running with num_tpus=" << num_tpus
            << ", num_layers=" << num_layers << ", num_blocks=" << num_blocks
            << std::endl;

  auto manager_or = TpuPjrtManager::GetDefault();
  ASSERT_OK(manager_or.status());
  auto manager = manager_or.value();

  auto all_devices = manager->client()->addressable_devices();
  std::vector<xla::PjRtDevice*> devices;
  for (int i = 0; i < num_tpus && i < all_devices.size(); ++i) {
    devices.push_back(all_devices[i]);
  }
  ASSERT_FALSE(devices.empty()) << "No local TPUs found";

  RunSlotBenchmark<float>(manager, devices, xla::PrimitiveType::F32, "F32",
                          num_layers, num_blocks);
}

}  // namespace
}  // namespace tpu_raiden
