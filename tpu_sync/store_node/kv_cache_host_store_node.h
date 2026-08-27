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

#ifndef THIRD_PARTY_TPU_RAIDEN_STORE_NODE_KV_CACHE_HOST_STORE_NODE_H_
#define THIRD_PARTY_TPU_RAIDEN_STORE_NODE_KV_CACHE_HOST_STORE_NODE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/worker_service_server.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/store_node/kv_transfer_spec_source.h"
namespace tpu_raiden {
namespace store_node {

// A standalone raiden node that lends its host DRAM to the deployment as a
// KV cache pool: serving hosts evict prefix KV blocks into it and read them
// back later. It runs no engine; it is the store role of a serving host,
// deployed on its own (which is also why it can run on a machine with no
// accelerator at all).
//
// Not to be confused with kv_cache::KVCacheStoreServer, the gRPC wrapper
// hosting the peer-facing KVCacheStoreService: that is one component INSIDE
// this node, in the same way ControllerServer is one component inside
// RaidenController.
//
// Boot is two phases, split by the one input the node cannot know on its own:
//
//   Phase A (no spec needed): wait on the KVTransferSpecSource until the
//     deployment's KVTransferSpec is published, retrying with jittered
//     backoff. Everything the node is configured with (identity, addresses,
//     DRAM budget) is deployment config; the spec is the serving hosts'
//     runtime truth and must come from them.
//
//   Phase B (spec in hand): run the same assembly a serving host runs,
//     with CPU-backed managers:
//       1. One KVCacheManagerWithTransfer (CPU-only) per spec worker rank,
//          mirroring the serving hosts' transfer topology: each allocates
//          its own DRAM pool up front and opens its own raw-transfer data
//          endpoint.
//       2. KVCacheStore: builds the RaidenController, the LRU cache and the
//          peer-facing store server, and publishes this node to the global
//          registry (RegisterStore), so registration happens exactly when
//          the node can actually serve.
//       3. WorkerService + worker registration, one per manager: makes each
//          manager reachable from the controller under its rank's node_id,
//          completing the transfer path.
//
// After Create() returns, the node is discoverable and serves peers; there is
// no separate Start(). Destruction tears down in reverse: the store first
// (unregisters from the global registry, stops the store server and
// controller), then the worker server, then the manager (frees the pool).
class KVCacheHostStoreNode {
 public:
  struct Options {
    // Identity this node registers under in the global registry. The four
    // fields form one opaque composite key --
    // raiden never matches a field individually -- so all that matters is
    // that the tuple is unique within the deployment and stable across
    // restarts (re-registering the same id is how a restarted node replaces
    // its stale registration).
    kv_cache::RaidenId raiden_id;

    // The IP peers use to reach this node. Bind-and-advertise, same semantics
    // as KVCacheStore's store_server_ip: the store server, the controller and
    // the worker all bind it, and it is the host published to the global
    // registry. Required: an unreachable host store node is useless.
    std::string store_server_ip;

    // Controller port; 0 lets gRPC choose.
    int raiden_controller_port = 0;

    // Global registry to publish this node to. Per KVCacheStore's
    // construction rules the registry decides whether the peer-facing plane
    // exists at all: empty means no store server is stood up -- the manager,
    // controller and worker still boot (useful in tests), but peers can
    // neither discover nor dial this node.
    std::string global_registry_address;

    // KV pool group this node registers under (StoreInfo.kv_pool_group), the
    // same group whose KVTransferSpec it serves. Placement never crosses
    // groups, and a store registered with an empty group never serves as a
    // placement target.
    std::string kv_pool_group;

    // Placement tier this node registers under (StoreInfo.evict_tier).
    // Serving hosts sit on tier 0 and demote strictly upward, so a host
    // store node defaults to the tier right below them.
    int32_t evict_tier = 1;

    // Runs the StoreMonitor: its heartbeats put a TTL on this node's
    // registration (so a dead node ages out of placement) and refresh the
    // free capacity the placement RPC ranks by. Only takes effect with a
    // global_registry_address -- the monitor's whole job here is talking to
    // the registry. Off means a TTL-less registration that never reports
    // capacity.
    bool enable_store_monitor = true;

    // Host DRAM lent to the pool. Converted to whole blocks of the received
    // spec's block geometry; the remainder below one block is not allocated.
    size_t dram_budget_bytes = 0;

    // Transfer parallelism of the manager.
    int parallelism = 1;

    // KVTransferSpec wait: retry cadence while the source reports NotFound or
    // Unavailable. The interval starts at spec_poll_initial, doubles up
    // to spec_poll_max, and every sleep is jittered so a fleet of nodes
    // booting together spreads its polls. spec_wait_timeout bounds the wait
    // so a node that can never learn the spec (wrong registry address, dead
    // deployment) fails boot visibly instead of hanging forever.
    absl::Duration spec_poll_initial = absl::Seconds(1);
    absl::Duration spec_poll_max = absl::Minutes(1);
    absl::Duration spec_wait_timeout = absl::Minutes(120);
  };

  // Boots a host store node: blocks in WaitForSpec, then assembles the node.
  // On return the node is serving and (when a global registry is configured)
  // discoverable. `kv_transfer_spec_source` must outlive the call.
  static absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> Create(
      const Options& options, KVTransferSpecSource* kv_transfer_spec_source);

  KVCacheHostStoreNode(const KVCacheHostStoreNode&) = delete;
  KVCacheHostStoreNode& operator=(const KVCacheHostStoreNode&) = delete;

  // Phase A alone: polls `source` until it yields a valid spec, with
  // jittered exponential backoff per `options`. Returns the spec, or
  // DeadlineExceeded after options.spec_wait_timeout, or the source's
  // error when it is neither NotFound nor Unavailable.
  static absl::StatusOr<KVTransferSpec> WaitForSpec(
      KVTransferSpecSource& source, const Options& options);

  // Whole blocks of `spec`'s block geometry that fit in `dram_budget_bytes`.
  // Every worker's pool holds its own shards of every block, so one block
  // costs num_workers x num_kv_shards x (sum of block_array_bytes).
  // InvalidArgument if the budget does not cover even one block.
  static absl::StatusOr<size_t> NumBlocksForBudget(size_t dram_budget_bytes,
                                                   const KVTransferSpec& spec);

  const KVTransferSpec& spec() const { return spec_; }
  // Blocks in each worker's pool -- one flat block id space shared by all
  // workers, not a partition.
  size_t num_host_blocks() const { return num_host_blocks_; }

  kv_cache::KVCacheStore* store() const { return store_.get(); }
  const std::vector<std::unique_ptr<KVCacheManagerWithTransfer>>& managers()
      const {
    return managers_;
  }

  // Advertised "host:port" of the peer-facing store server / the controller.
  std::string store_server_address() const {
    return store_->store_server_address();
  }
  std::string raiden_controller_address() const {
    return store_->raiden_controller_address();
  }

 private:
  KVCacheHostStoreNode(
      const KVTransferSpec& spec, size_t num_host_blocks,
      std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> managers,
      std::vector<std::unique_ptr<controller::WorkerServiceServer>>
          worker_servers,
      std::unique_ptr<kv_cache::KVCacheStore> store)
      : spec_(spec),
        num_host_blocks_(num_host_blocks),
        managers_(std::move(managers)),
        worker_servers_(std::move(worker_servers)),
        store_(std::move(store)) {}

  KVTransferSpec spec_;
  size_t num_host_blocks_ = 0;

  // Declaration order is teardown order reversed: store_ goes down first.
  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> managers_;
  std::vector<std::unique_ptr<controller::WorkerServiceServer>> worker_servers_;
  std::unique_ptr<kv_cache::KVCacheStore> store_;
};

}  // namespace store_node
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_STORE_NODE_KV_CACHE_HOST_STORE_NODE_H_
