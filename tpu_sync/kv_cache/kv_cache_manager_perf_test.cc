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

// This file contains performance benchmarks for KVCacheManagerBase.
// It measures raw D2H and H2D performance, similar to
// raw_transfer_perf_test.cc, but utilizing the KVCacheManagerBase class.
//
// By default, the test suite runs a lightweight configuration (4 data types,
// 64 layers, 16 blocks) to keep presubmit runs fast (~1 minute).
//
// To manually run deeper scenarios (e.g. on a host with 8 TPU v7x chips):
//   bazel test --config=oss -c opt \
//     --test_arg=--num_tpus=8 \
//     --test_arg=--num_layers=1024 \
//     --test_arg=--num_blocks=8 \
//     //tpu_sync/core:kv_cache_manager_perf_test

#include <dlfcn.h>

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
#include "xla/types.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_pjrt_manager.h"
#include "tpu_sync/core/tpu_utils.h"
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
bool InitializeLibtpuOnce() {
  static bool initialized = false;
  if (initialized) return true;

  const char* libtpu_path = std::getenv("TPU_LIBRARY_PATH");
  if (!libtpu_path || std::string(libtpu_path).empty()) {
    libtpu_path = "libtpu.so";
  }
  std::cout << "[INFO] Dynamically loading libtpu from: " << libtpu_path
            << std::endl;
  void* library = dlopen(libtpu_path, RTLD_NOW | RTLD_GLOBAL);
  if (!library) {
    std::cerr << "[WARNING] Failed to dlopen libtpu.so: " << dlerror()
              << ". This might be expected if not running on TPU hardware."
              << std::endl;
    return false;
  }
  auto initialize_fn = reinterpret_cast<void (*)(bool, int, const char**)>(
      dlsym(library, "TfTpu_Initialize"));
  if (!initialize_fn) {
    std::cerr << "[WARNING] TfTpu_Initialize symbol not found in libtpu.so"
              << std::endl;
    dlclose(library);
    return false;
  }
  std::cout << "[INFO] Calling TfTpu_Initialize(true, 0, nullptr)."
            << std::endl;
  initialize_fn(/*init_library=*/true, 0, nullptr);
  std::cout << "[INFO] libtpu.so initialized successfully." << std::endl;
  initialized = true;
  return true;
}

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

// Templated benchmark executor for Scenario A (Fragmented Batch)
template <typename T>
void RunBenchmarkScenarioA(TpuPjrtManager* manager,
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

  // We only copy ODD blocks -> exactly 50% of data.
  // D2H: Device Odd -> Host Odd
  std::vector<int64_t> d2h_src_offsets;
  std::vector<int64_t> d2h_dst_offsets;
  // H2D: Host Odd -> Device Even (different offset on device for verification)
  std::vector<int64_t> h2d_src_offsets;
  std::vector<int64_t> h2d_dst_offsets;

  std::vector<int64_t> copy_sizes;

  for (int b = 1; b < kNumBlocks; b += 2) {
    d2h_src_offsets.push_back(b);
    d2h_dst_offsets.push_back(b);

    h2d_src_offsets.push_back(b);
    h2d_dst_offsets.push_back(b - 1);  // Copy to even block

    copy_sizes.push_back(1);
  }
  size_t bytes_transferred = (bytes_per_layer / 2) * kNumLayers;

  std::cout << "[INFO] Scenario A: Fragmented Batch (Raw D2H/H2D) ["
            << type_label << "]. Total layers: " << kNumLayers
            << ", Layer size: " << bytes_per_layer / (1024.0 * 1024.0) << " MB"
            << ", Total size: " << total_bytes / (1024.0 * 1024.0) << " MB"
            << ", Transferred size: " << bytes_transferred / (1024.0 * 1024.0)
            << " MB" << std::endl;

  // 1. Allocate Host Memory to initialize TPU buffers (temporary)
  auto allocator_or = HostMemoryAllocator::Create(manager->client());
  ASSERT_OK(allocator_or.status());
  auto host_allocator = std::move(allocator_or).value();

  std::vector<HostBufferAllocation> host_init_buffers;
  std::vector<const uint8_t*> host_init_ptrs;
  for (int i = 0; i < kNumLayers; ++i) {
    auto alloc_or = host_allocator->AllocateDmaMappedForDevice(
        bytes_per_layer, devices[i % devices.size()]);
    ASSERT_OK(alloc_or.status());
    host_init_buffers.push_back(std::move(alloc_or).value());
    std::memset(host_init_buffers.back().ptr, 0, bytes_per_layer);

    T* src_ptr = reinterpret_cast<T*>(host_init_buffers.back().ptr);
    for (int64_t e = 0; e < elements_per_layer; ++e) {
      if (sizeof(T) == 1) {
        src_ptr[e] = static_cast<T>((i + e) % 64);
      } else {
        src_ptr[e] = static_cast<T>((i + e) % 65536);
      }
    }
    host_init_ptrs.push_back(host_init_buffers.back().ptr);
  }

  // 2. Allocate TPU Device Buffers and populate them
  std::vector<std::unique_ptr<xla::PjRtBuffer>> device_buffers;
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers(
      kNumLayers);
  for (int i = 0; i < kNumLayers; ++i) {
    xla::PjRtDevice* device = devices[i % devices.size()];
    auto buf_or = manager->BufferFromHost(host_init_ptrs[i], primitive_type,
                                          layer_shape, device);
    ASSERT_OK(buf_or.status());
    device_buffers.push_back(std::move(buf_or).value());
    auto handle_or =
        raiden::RaidenBufferHandle::Acquire(device_buffers.back().get());
    ASSERT_OK(handle_or.status());
    layer_buffers[i].push_back(std::move(handle_or).value());
  }

  for (auto& bufs : layer_buffers) {
    ASSERT_OK(bufs[0].buffer->GetReadyFuture().Await());
  }

  // Free initial host buffers early to reduce peak memory
  host_init_buffers.clear();
  host_init_ptrs.clear();

  // 3. Create KVCacheManagerBase
  HostMemoryAllocator* raw_allocator = host_allocator.get();
  HostBufferAllocator allocator_fn =
      [raw_allocator](size_t size, const xla::PjRtDevice* device) {
        return raw_allocator->AllocateDmaMappedForDevice(size, device);
      };

  const int64_t host_blocks_to_allocate = kNumBlocks;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<kv_cache::KVCacheManagerBase> engine,
      kv_cache::KVCacheManagerBase::Create(
          layer_buffers,
          /*local_port=*/std::nullopt,
          /*host_blocks_to_allocate=*/host_blocks_to_allocate,
          /*unsafe_skip_buffer_lock=*/true,
          /*parallelism=*/1,
          /*host_allocator=*/allocator_fn));

  // 4. Warmup Phase
  constexpr int kWarmupIterations = 3;
  for (int iter = 0; iter < kWarmupIterations; ++iter) {
    auto d2h_future_or =
        engine->D2h(d2h_src_offsets, d2h_dst_offsets, copy_sizes);
    ASSERT_OK(AwaitAll(d2h_future_or));

    auto h2d_future_or =
        engine->H2d(h2d_src_offsets, h2d_dst_offsets, copy_sizes);
    ASSERT_OK(AwaitAll(h2d_future_or));
  }

  // 5. Timed Benchmark Loop (D2H)
  constexpr int kBenchmarkIterations = 50;
  std::vector<double> d2h_bandwidths;
  d2h_bandwidths.reserve(kBenchmarkIterations);

  for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
    absl::Time start = absl::Now();
    auto d2h_future_or =
        engine->D2h(d2h_src_offsets, d2h_dst_offsets, copy_sizes);
    ASSERT_OK(AwaitAll(d2h_future_or));
    absl::Duration duration = absl::Now() - start;
    double seconds = absl::ToDoubleSeconds(duration);
    double bandwidth =
        (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / seconds;
    d2h_bandwidths.push_back(bandwidth);
  }

  PrintDistribution(
      absl::StrCat("Scenario A D2H Bandwidth (Odd Blocks) [", type_label, "]"),
      d2h_bandwidths);

  // 6. Timed Benchmark Loop (H2D)
  std::vector<double> h2d_bandwidths;
  h2d_bandwidths.reserve(kBenchmarkIterations);

  for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
    absl::Time start = absl::Now();
    auto h2d_future_or =
        engine->H2d(h2d_src_offsets, h2d_dst_offsets, copy_sizes);
    ASSERT_OK(AwaitAll(h2d_future_or));
    absl::Duration duration = absl::Now() - start;
    double seconds = absl::ToDoubleSeconds(duration);
    double bandwidth =
        (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / seconds;
    h2d_bandwidths.push_back(bandwidth);
  }

  PrintDistribution(
      absl::StrCat("Scenario A H2D Bandwidth (Odd Blocks) [", type_label, "]"),
      h2d_bandwidths);

  // 7. Verify Correctness
  // We verify that the data copied from ODD blocks to EVEN blocks matches.
  size_t elements_per_block = elements_per_layer / kNumBlocks;
  for (int i = 0; i < kNumLayers; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto literal,
                            device_buffers[i]->ToLiteral().Await());
    auto read_back = literal->data<T>();

    for (int64_t b = 1; b < kNumBlocks; b += 2) {
      // Even block (b - 1) should now match odd block (b)
      for (size_t e = 0; e < elements_per_block; ++e) {
        size_t odd_idx = b * elements_per_block + e;
        size_t even_idx = (b - 1) * elements_per_block + e;
        EXPECT_EQ(read_back[even_idx], read_back[odd_idx])
            << "Mismatch in layer " << i << " between even block " << (b - 1)
            << " and odd block " << b << " at element " << e;
      }
    }
  }
}

// Templated benchmark executor for Scenario B (Baked-in Tensor)
template <typename T>
void RunBenchmarkScenarioB(TpuPjrtManager* manager,
                           const std::vector<xla::PjRtDevice*>& devices,
                           xla::PrimitiveType primitive_type,
                           absl::string_view type_label, int num_layers,
                           int64_t num_blocks) {
  int num_devices = devices.size();
  const int kNumLayers = std::max(1, num_layers / num_devices);
  const int64_t kNumBlocks = num_blocks;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;

  std::vector<int64_t> baked_shape = {kNumBlocks, kNumLayers, kBlockSize,
                                      kNumHeads,  2,          kHeadDim};
  int64_t total_elements =
      kNumBlocks * kNumLayers * kBlockSize * kNumHeads * 2 * kHeadDim;
  size_t total_bytes = total_elements * sizeof(T);

  // We only copy ODD blocks -> exactly 50% of data.
  // D2H: Device Odd -> Host Odd
  std::vector<int64_t> d2h_src_offsets;
  std::vector<int64_t> d2h_dst_offsets;
  // H2D: Host Odd -> Device Even (different offset on device for verification)
  std::vector<int64_t> h2d_src_offsets;
  std::vector<int64_t> h2d_dst_offsets;

  std::vector<int64_t> copy_sizes;

  for (int b = 1; b < kNumBlocks; b += 2) {
    d2h_src_offsets.push_back(b);
    d2h_dst_offsets.push_back(b);

    h2d_src_offsets.push_back(b);
    h2d_dst_offsets.push_back(b - 1);  // Copy to even block

    copy_sizes.push_back(1);
  }
  size_t bytes_transferred = (total_bytes / 2) * num_devices;

  std::cout << "[INFO] Scenario B: Baked-in Tensor (Raw D2H/H2D) ["
            << type_label << "]. Shape: [" << kNumBlocks << ", " << kNumLayers
            << ", " << kBlockSize << ", " << kNumHeads << ", 2, " << kHeadDim
            << "]"
            << ", Total size: "
            << (total_bytes * num_devices) / (1024.0 * 1024.0) << " MB"
            << ", Transferred size: " << bytes_transferred / (1024.0 * 1024.0)
            << " MB" << ", Devices: " << num_devices << std::endl;

  // 1. Allocate Host Memory to initialize TPU buffers (temporary)
  auto allocator_or = HostMemoryAllocator::Create(manager->client());
  ASSERT_OK(allocator_or.status());
  auto host_allocator = std::move(allocator_or).value();

  auto alloc_or =
      host_allocator->AllocateDmaMappedForDevice(total_bytes, devices[0]);
  ASSERT_OK(alloc_or.status());
  auto host_init = std::move(alloc_or).value();
  std::memset(host_init.ptr, 0, total_bytes);

  T* src_ptr = reinterpret_cast<T*>(host_init.ptr);
  for (int64_t e = 0; e < total_elements; ++e) {
    if (sizeof(T) == 1) {
      src_ptr[e] = static_cast<T>(e % 64);
    } else {
      src_ptr[e] = static_cast<T>(e % 65536);
    }
  }

  // 2. Allocate TPU Device Buffers (1 layer, multi-device)
  std::vector<std::unique_ptr<xla::PjRtBuffer>> device_buffers;
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers(
      1);  // 1 layer
  for (int i = 0; i < num_devices; ++i) {
    auto buf_or = manager->BufferFromHost(host_init.ptr, primitive_type,
                                          baked_shape, devices[i]);
    ASSERT_OK(buf_or.status());
    device_buffers.push_back(std::move(buf_or).value());
    auto handle_or =
        raiden::RaidenBufferHandle::Acquire(device_buffers.back().get());
    ASSERT_OK(handle_or.status());
    layer_buffers[0].push_back(std::move(handle_or).value());
  }

  for (auto& handle : layer_buffers[0]) {
    ASSERT_OK(handle.buffer->GetReadyFuture().Await());
  }

  // Free initial host buffer early to reduce peak memory
  host_init = {};

  // 3. Create KVCacheManagerBase
  HostMemoryAllocator* raw_allocator = host_allocator.get();
  HostBufferAllocator allocator_fn =
      [raw_allocator](size_t size, const xla::PjRtDevice* device) {
        return raw_allocator->AllocateDmaMappedForDevice(size, device);
      };

  const int64_t host_blocks_to_allocate = kNumBlocks;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<kv_cache::KVCacheManagerBase> engine,
      kv_cache::KVCacheManagerBase::Create(
          layer_buffers,
          /*local_port=*/std::nullopt,
          /*host_blocks_to_allocate=*/host_blocks_to_allocate,
          /*unsafe_skip_buffer_lock=*/true,
          /*parallelism=*/1,
          /*host_allocator=*/allocator_fn));

  // 4. Warmup Phase
  constexpr int kWarmupIterations = 3;
  for (int iter = 0; iter < kWarmupIterations; ++iter) {
    auto d2h_future_or =
        engine->D2h(d2h_src_offsets, d2h_dst_offsets, copy_sizes);
    ASSERT_OK(AwaitAll(d2h_future_or));

    auto h2d_future_or =
        engine->H2d(h2d_src_offsets, h2d_dst_offsets, copy_sizes);
    ASSERT_OK(AwaitAll(h2d_future_or));
  }

  // 5. Timed Benchmark Loop (D2H)
  constexpr int kBenchmarkIterations = 50;
  std::vector<double> d2h_bandwidths;
  d2h_bandwidths.reserve(kBenchmarkIterations);

  for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
    absl::Time start = absl::Now();
    auto d2h_future_or =
        engine->D2h(d2h_src_offsets, d2h_dst_offsets, copy_sizes);
    ASSERT_OK(AwaitAll(d2h_future_or));
    absl::Duration duration = absl::Now() - start;
    double seconds = absl::ToDoubleSeconds(duration);
    double bandwidth =
        (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / seconds;
    d2h_bandwidths.push_back(bandwidth);
  }

  PrintDistribution(
      absl::StrCat("Scenario B D2H Bandwidth (Odd Blocks) [", type_label, "]"),
      d2h_bandwidths);

  // 6. Timed Benchmark Loop (H2D)
  std::vector<double> h2d_bandwidths;
  h2d_bandwidths.reserve(kBenchmarkIterations);

  for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
    absl::Time start = absl::Now();
    auto h2d_future_or =
        engine->H2d(h2d_src_offsets, h2d_dst_offsets, copy_sizes);
    ASSERT_OK(AwaitAll(h2d_future_or));
    absl::Duration duration = absl::Now() - start;
    double seconds = absl::ToDoubleSeconds(duration);
    double bandwidth =
        (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / seconds;
    h2d_bandwidths.push_back(bandwidth);
  }

  PrintDistribution(
      absl::StrCat("Scenario B H2D Bandwidth (Odd Blocks) [", type_label, "]"),
      h2d_bandwidths);

  // 7. Verify Correctness
  // We verify that the data copied from ODD blocks to EVEN blocks matches.
  size_t elements_per_block = total_elements / kNumBlocks;
  for (int i = 0; i < num_devices; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto literal,
                            device_buffers[i]->ToLiteral().Await());
    auto read_back = literal->data<T>();

    for (int64_t b = 1; b < kNumBlocks; b += 2) {
      // Even block (b - 1) should now match odd block (b)
      for (size_t e = 0; e < elements_per_block; ++e) {
        size_t odd_idx = b * elements_per_block + e;
        size_t even_idx = (b - 1) * elements_per_block + e;
        EXPECT_EQ(read_back[even_idx], read_back[odd_idx])
            << "Mismatch in device " << i << " between even block " << (b - 1)
            << " and odd block " << b << " at element " << e;
      }
    }
  }
}

}  // namespace
}  // namespace tpu_raiden

#ifndef ASSERT_OK
#define ASSERT_OK(status) ASSERT_TRUE((status).ok())
#endif

namespace tpu_raiden {
namespace {

class KVCacheManagerPerfTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { InitializeLibtpuOnce(); }

  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(manager_, tpu_raiden::TpuPjrtManager::GetDefault());

    int num_tpus = absl::GetFlag(FLAGS_num_tpus);
    auto all_devices = manager_->client()->addressable_devices();
    int available_tpus = all_devices.size();
    std::cout << "[INFO] Available TPUs: " << available_tpus << std::endl;
    tpu_raiden::PrintTpuHardwareTopology();

    if (num_tpus > available_tpus) {
      std::cout << "[WARNING] Requested " << num_tpus << " TPUs, but only "
                << available_tpus << " are available. Using " << available_tpus
                << " TPUs." << std::endl;
      num_tpus = available_tpus;
    }

    for (int i = 0; i < num_tpus; ++i) {
      devices_.push_back(all_devices[i]);
    }

    std::cout << "[INFO] Using " << devices_.size() << " TPUs:" << std::endl;
    for (auto* dev : devices_) {
      std::cout << "  Device " << dev->id() << ": " << dev->device_kind()
                << std::endl;
    }
  }

  void TearDown() override {
    // manager_ is a raw pointer to a singleton, do not delete or reset it.
  }

  tpu_raiden::TpuPjrtManager* manager_ = nullptr;
  std::vector<xla::PjRtDevice*> devices_;
};

struct BenchmarkParams {
  xla::PrimitiveType primitive_type;
  std::string type_label;
};

class ParameterizedKVCacheManagerPerfTest
    : public KVCacheManagerPerfTest,
      public ::testing::WithParamInterface<BenchmarkParams> {};

// Scenario A: Fragmented Batch (Independent Layer Buffers)
TEST_P(ParameterizedKVCacheManagerPerfTest, ScenarioA_FragmentedBatch) {
  const BenchmarkParams& params = GetParam();
  int num_layers = absl::GetFlag(FLAGS_num_layers);
  int64_t num_blocks = absl::GetFlag(FLAGS_num_blocks);
  switch (params.primitive_type) {
    case xla::BF16:
      RunBenchmarkScenarioA<xla::bfloat16>(
          manager_, devices_, params.primitive_type, params.type_label,
          num_layers, num_blocks);
      break;
    case xla::F32:
      RunBenchmarkScenarioA<float>(manager_, devices_, params.primitive_type,
                                   params.type_label, num_layers, num_blocks);
      break;
    case xla::S32:
      RunBenchmarkScenarioA<int32_t>(manager_, devices_, params.primitive_type,
                                     params.type_label, num_layers, num_blocks);
      break;
    case xla::F8E4M3FN:
      RunBenchmarkScenarioA<xla::float8_e4m3fn>(
          manager_, devices_, params.primitive_type, params.type_label,
          num_layers, num_blocks);
      break;
    default:
      FAIL() << "Unsupported primitive type: " << params.primitive_type;
  }
}

// Scenario B: Baked-in Layer Dimension (Single Massive Buffer)
TEST_P(ParameterizedKVCacheManagerPerfTest, ScenarioB_BakedInTensor) {
  const BenchmarkParams& params = GetParam();
  int num_layers = absl::GetFlag(FLAGS_num_layers);
  int64_t num_blocks = absl::GetFlag(FLAGS_num_blocks);
  switch (params.primitive_type) {
    case xla::BF16:
      RunBenchmarkScenarioB<xla::bfloat16>(
          manager_, devices_, params.primitive_type, params.type_label,
          num_layers, num_blocks);
      break;
    case xla::F32:
      RunBenchmarkScenarioB<float>(manager_, devices_, params.primitive_type,
                                   params.type_label, num_layers, num_blocks);
      break;
    case xla::S32:
      RunBenchmarkScenarioB<int32_t>(manager_, devices_, params.primitive_type,
                                     params.type_label, num_layers, num_blocks);
      break;
    case xla::F8E4M3FN:
      RunBenchmarkScenarioB<xla::float8_e4m3fn>(
          manager_, devices_, params.primitive_type, params.type_label,
          num_layers, num_blocks);
      break;
    default:
      FAIL() << "Unsupported primitive type: " << params.primitive_type;
  }
}

INSTANTIATE_TEST_SUITE_P(
    KVCacheManagerPerfTests, ParameterizedKVCacheManagerPerfTest,
    ::testing::Values(BenchmarkParams{xla::BF16, "BF16"},
                      BenchmarkParams{xla::F32, "F32"},
                      BenchmarkParams{xla::S32, "INT32"},
                      BenchmarkParams{xla::F8E4M3FN, "FP8"}),
    [](const ::testing::TestParamInfo<BenchmarkParams>& info) {
      return info.param.type_label;
    });

}  // namespace
}  // namespace tpu_raiden
