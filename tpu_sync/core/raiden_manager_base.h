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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_MANAGER_BASE_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_MANAGER_BASE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "peregrine/src/api/v1/rate_limiter.h"
#include "xla/future.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/transport/block_transport.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/buffer_push_task.h"

namespace tpu_raiden {

class RaidenManagerBase : public tpu_raiden::transport::BlockTransportDelegate {
 public:
  using ShardBufferInfoBase = ::tpu_raiden::ShardBufferInfoBase;
  using LayerInfoBase = ::tpu_raiden::LayerInfoBase;
  using StagingArena = ::tpu_raiden::StagingArena;

  // |shards| describes every local shard (one entry per shard, in shard order;
  // see `transport::ShardInfo`, `raiden::DetectShards()`), so `num_shards()`
  // equals `shards.size()`. Its NUMA nodes drive per-shard NIC selection
  // (`local_ips()` then spans the data NICs of every shard's NUMA node, see
  // `local_ips_for_shard()` / `shards_for_local_ip()`). |numa_nodes| lists the
  // distinct NUMA nodes of the incoming buffers (the first one becomes
  // `assigned_numa_node()`); when no shard in |shards| carries a NUMA node
  // (`numa_node >= 0`), every shard is attributed to `assigned_numa_node()`
  // and NIC selection uses the single-node behaviour.
  RaidenManagerBase(
      size_t num_layers, std::vector<transport::ShardInfo> shards,
      size_t slice_byte_size, std::optional<int> local_port = std::nullopt,
      int parallelism = 1, std::optional<std::string> bind_ip = std::nullopt,
      std::vector<int> numa_nodes = {},
      std::vector<HostNicAddress> host_nics = GetLocalHostNicAddresses());

  ~RaidenManagerBase() override;

  // Direct C++ H2H network write (Push)
  absl::StatusOr<std::vector<int>> H2hWriteDirect(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1);

  virtual void H2hWriteDirectAsync(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids, uint64_t uuid, int layer_idx,
      std::function<void(absl::StatusOr<std::vector<int>>)> on_complete);

  // Direct C++ H2H network read (Pull)
  absl::StatusOr<std::vector<int>> H2hReadDirect(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids);

  // Backward-compatible overloads
  absl::StatusOr<std::vector<int>> H2hWriteDirect(
      absl::string_view peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1) {
    return H2hWriteDirect(std::vector<std::string>{std::string(peer)},
                          src_block_ids, dst_block_ids, uuid, layer_idx);
  }

  absl::StatusOr<std::vector<int>> H2hReadDirect(
      absl::string_view peer, const std::vector<int>& src_block_ids) {
    return H2hReadDirect(std::vector<std::string>{std::string(peer)},
                         src_block_ids);
  }

  absl::Status PushWeightsChunk(absl::string_view peer, size_t dst_shard_idx,
                                size_t dst_offset_bytes,
                                const uint8_t* data_ptr, size_t size_bytes,
                                uint64_t uuid = 0, size_t layer_idx = 0);

  // Synchronously pulls a contiguous byte range from |peer|'s |src_shard_idx|
  // into this host's |dst_shard_idx| for |buffer_id| (layer index).
  absl::Status PullBuffer(absl::string_view peer, size_t buffer_id,
                          size_t src_shard_idx, size_t src_offset_bytes,
                          size_t dst_shard_idx, size_t dst_offset_bytes,
                          size_t size_bytes);

  absl::Status PushWeightsChunks(
      const std::vector<transport::BufferPushTask>& tasks, int parallelism,
      uint64_t uuid);

  virtual absl::Status RegisterExpectedChunks(uint64_t uuid,
                                              uint32_t expected_chunks);
  virtual absl::Status RegisterExpectedLayerChunks(
      uint64_t uuid,
      const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks);

  virtual void ForgetPushProgress(uint64_t uuid);

  void SetTestOnlyRateLimiters(
      std::shared_ptr<peregrine::v1::TestOnlyRateLimiter> egress,
      std::shared_ptr<peregrine::v1::TestOnlyRateLimiter> ingress);

  // Stops and joins the underlying raw transport server if active.
  void StopTransportServer();

  virtual std::optional<int> local_port() const;
  virtual std::string local_ip() const;
  virtual std::vector<std::string> local_ips() const;
  std::optional<int> assigned_numa_node() const { return assigned_numa_node_; }

  // Placement of every local shard (`local_index` order, one entry per
  // shard): the constructor's |shards|, or one entry per shard on
  // `assigned_numa_node()` with unknown global index when none was supplied.
  absl::Span<const transport::ShardInfo> shards() const override {
    return shards_;
  }
  // NUMA node of the device backing |shard_idx|, or nullopt when unknown.
  // Falls back to `assigned_numa_node()` when no per-shard map was supplied.
  std::optional<int> shard_numa_node(size_t shard_idx) const;
  // NUMA node of the NIC that owns local address |ip|, or nullopt when |ip| is
  // not one of the host NICs handed to the constructor.
  std::optional<int> numa_node_for_ip(absl::string_view ip) const;
  // Subset of `local_ips()` whose NIC sits on the NUMA node of |shard_idx|.
  // Returns all of `local_ips()` when the shard or the NICs have no NUMA
  // information, so callers can always use the result as a peer list.
  std::vector<std::string> local_ips_for_shard(size_t shard_idx) const;
  // Shards to advertise on local address |ip|: those whose device sits on the
  // NIC's NUMA node, plus every shard that no local IP is NUMA-local to (or
  // whose NUMA node is unknown), so that the union over `local_ips()` always
  // covers every shard. Returns every shard when |ip| is not a local address,
  // when |ip| or the shards carry no NUMA information, or when nothing else
  // would be selected.
  std::vector<int64_t> shards_for_local_ip(absl::string_view ip) const;

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) override;
  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override;

  virtual const uint8_t* GetHostPointer(size_t layer_idx,
                                        size_t shard_idx) const;
  virtual size_t GetHostSize(size_t layer_idx, size_t shard_idx) const;

  void SetExternalHostPointers(const std::vector<const uint8_t*>& host_ptrs,
                               const std::vector<size_t>& host_sizes);

  StagingArena& staging_arena() { return layers_; }
  const StagingArena& staging_arena() const { return layers_; }

  // Delegate overrides E2E
  size_t num_layers() const override { return num_layers_; }
  size_t num_shards() const override { return num_shards_; }
  size_t slice_byte_size() const override { return slice_byte_size_; }
  // Returns the layer-specific block size, falling back to slice_byte_size() if
  // the layer size is not initialized (0).
  size_t block_bytes(size_t layer_idx) const override;
  size_t shard_factor() const override { return shard_factor_; }

 protected:
  size_t num_layers_ = 0;
  size_t num_shards_ = 0;
  size_t slice_byte_size_ = 0;
  int parallelism_ = 1;
  size_t shard_factor_ = 1;
  int64_t major_dim_size_ = 0;
  std::optional<int> assigned_numa_node_ = std::nullopt;
  // Placement of every local shard; always `num_shards_` entries.
  std::vector<transport::ShardInfo> shards_;
  // Local IP address -> NUMA node of the owning NIC, for every NIC handed to
  // the constructor (not only the ones selected into `local_ips_`).
  absl::flat_hash_map<std::string, int> ip_numa_nodes_;
  int local_port_cfg_ = 0;
  std::optional<std::string> bind_ip_cfg_ = std::nullopt;
  std::vector<std::string> local_ips_;

  tpu_raiden::transport::BlockTransport* InitTransportServer(
      std::vector<HostNicAddress> host_nics = {});

  mutable absl::Mutex server_init_mu_;
  std::unique_ptr<tpu_raiden::transport::BlockTransport> server_;

  StagingArena layers_;

  // Delegate allocator overrides
  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  uint64_t uuid = 0) override {
    return absl::UnimplementedError("Block allocator is not available");
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

  absl::Status OnLayerDataReceived(size_t layer_idx,
                                   uint64_t uuid = 0) override {
    return absl::OkStatus();
  }

  absl::Status OnDataReceived(uint64_t uuid = 0) override {
    return absl::OkStatus();
  }

  // Runs |fn| once per shard, concurrently, each as a task on |pool| pinned to
  // that shard's NUMA node so host-side work on the shard's buffers touches
  // local memory. Returns as soon as the tasks are scheduled; the returned
  // future is the join of the futures returned by |fn|, indexed by shard, and
  // fails with the first error. Shard tasks never block on those futures. |fn|
  // outlives this call and is invoked concurrently, so it must own its state
  // and be const-callable. |pool| must outlive the scheduled tasks.
  template <typename T>
  xla::Future<std::vector<T>> ForEachShardNumaLocal(
      tpu_raiden::NumaThreadPool& pool,
      absl::AnyInvocable<xla::Future<T>(size_t shard_idx) const> fn);
  // Same, for |fn| without a result.
  xla::Future<> ForEachShardNumaLocal(
      tpu_raiden::NumaThreadPool& pool,
      absl::AnyInvocable<xla::Future<>(size_t shard_idx) const> fn);
};

template <typename T>
xla::Future<std::vector<T>> RaidenManagerBase::ForEachShardNumaLocal(
    tpu_raiden::NumaThreadPool& pool,
    absl::AnyInvocable<xla::Future<T>(size_t shard_idx) const> fn) {
  std::shared_ptr<const absl::AnyInvocable<xla::Future<T>(size_t) const>>
      shared_fn = std::make_shared<
          const absl::AnyInvocable<xla::Future<T>(size_t) const>>(
          std::move(fn));
  std::vector<xla::Future<T>> shard_futures;
  shard_futures.reserve(num_shards_);
  // One task per shard, pinned to that shard's NUMA node. The task forwards
  // the future returned by |fn| into the shard's promise without blocking on
  // it.
  for (size_t i = 0; i < num_shards_; ++i) {
    auto [promise, future] = xla::MakePromise<T>();
    shard_futures.push_back(std::move(future));
    pool.Schedule(
        shard_numa_node(i),
        [shared_fn, i, promise = std::move(promise).ToShared()]() {
          xla::Future<T> result = (*shared_fn)(i);
          if (!result.IsValid()) {
            promise->Set(absl::InternalError(
                "ForEachShardNumaLocal: shard task returned no future"));
            return;
          }
          std::move(result).OnReady(
              [promise](const auto& value) { promise->Set(value); });
        });
  }
  return xla::JoinFutures(absl::MakeSpan(shard_futures));
}

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_MANAGER_BASE_H_
