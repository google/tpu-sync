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

#include "tpu_sync/store_node/kv_cache_host_store_node.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/worker_service_server.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/core/status_macros.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/store_node/kv_transfer_spec_source.h"
namespace tpu_raiden {
namespace store_node {
namespace {

absl::Status ValidateOptions(const KVCacheHostStoreNode::Options& options) {
  if (options.raiden_id.empty()) {
    return absl::InvalidArgumentError("options.raiden_id must be set");
  }
  if (options.store_server_ip.empty()) {
    return absl::InvalidArgumentError(
        "options.store_server_ip must be set: a host store node exists to be "
        "reached by peers, so it always binds and advertises a concrete IP");
  }
  if (options.dram_budget_bytes == 0) {
    return absl::InvalidArgumentError("options.dram_budget_bytes must be set");
  }
  return absl::OkStatus();
}

std::string ComposeEndpoint(absl::string_view ip, int port) {
  return absl::StrContains(ip, ":") ? absl::StrCat("[", ip, "]:", port)
                                    : absl::StrCat(ip, ":", port);
}

}  // namespace

absl::StatusOr<KVTransferSpec> KVCacheHostStoreNode::WaitForSpec(
    KVTransferSpecSource& source, const Options& options) {
  absl::BitGen gen;
  absl::Duration interval = options.spec_poll_initial;
  const absl::Time deadline = absl::Now() + options.spec_wait_timeout;
  while (true) {
    absl::StatusOr<KVTransferSpec> spec = source.Get();
    if (spec.ok()) {
      RETURN_IF_ERROR(ValidateSpec(*spec));
      return spec;
    }
    if (!absl::IsNotFound(spec.status()) &&
        !absl::IsUnavailable(spec.status())) {
      return spec.status();
    }
    if (absl::Now() >= deadline) {
      return absl::DeadlineExceededError(
          absl::StrCat("KV transfer spec not published within ",
                       absl::FormatDuration(options.spec_wait_timeout),
                       "; last source status: ", spec.status().ToString()));
    }
    absl::SleepFor(interval * absl::Uniform(gen, 0.5, 1.5));
    interval = std::min(interval * 2, options.spec_poll_max);
  }
}

absl::StatusOr<size_t> KVCacheHostStoreNode::NumBlocksForBudget(
    size_t dram_budget_bytes, const KVTransferSpec& spec) {
  RETURN_IF_ERROR(ValidateSpec(spec));
  size_t bytes_per_shard = 0;
  for (uint64_t array_bytes : spec.block_array_bytes) {
    bytes_per_shard += array_bytes;
  }
  const size_t block_bytes =
      spec.num_workers * spec.num_kv_shards * bytes_per_shard;
  if (dram_budget_bytes < block_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "dram_budget_bytes=", dram_budget_bytes,
        " does not cover a single block of ", block_bytes,
        " bytes (num_workers=", spec.num_workers,
        " x num_kv_shards=", spec.num_kv_shards, " x ", bytes_per_shard,
        " bytes across ", spec.block_array_bytes.size(), " block arrays)"));
  }
  return dram_budget_bytes / block_bytes;
}

absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>>
KVCacheHostStoreNode::Create(const Options& options,
                             KVTransferSpecSource* kv_transfer_spec_source) {
  RETURN_IF_ERROR(ValidateOptions(options));
  if (kv_transfer_spec_source == nullptr) {
    return absl::InvalidArgumentError(
        "kv_transfer_spec_source must not be null");
  }

  // Phase A: the only input the node cannot know on its own.
  ASSIGN_OR_RETURN(const KVTransferSpec spec,
                   WaitForSpec(*kv_transfer_spec_source, options));
  ASSIGN_OR_RETURN(const size_t num_host_blocks,
                   NumBlocksForBudget(options.dram_budget_bytes, spec));

  // One stride per block array, in the spec's order.  Hybrid models register
  // arrays of differing block_bytes (a DeepSeek-V4 rank registers four).
  const size_t num_block_arrays = spec.block_array_bytes.size();
  const std::vector<size_t> block_bytes_per_array(
      spec.block_array_bytes.begin(), spec.block_array_bytes.end());

  // Phase B: the same assembly a serving host deployment runs, CPU-backed.
  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> managers;
  std::unique_ptr<kv_cache::KVCacheStore> store;
  try {
    // 1. The pools, one manager per spec worker rank, mirroring the serving
    // hosts' transfer topology: a peer pairs each of its workers with the
    // worker here sharing its node_id (unmatched node_ids are a hard
    // failure), so this node must field the same dense [0, num_workers)
    // roster. Block ids are one flat space -- every worker holds its own
    // shards of every block -- so each manager allocates and zeroes its
    // full num_host_blocks x num_block_arrays x num_kv_shards x
    // block_bytes_per_array pool here, at boot, never on a request path.
    // The constructor's first and third parameters are named num_layers and
    // slice_byte_size for historical reasons; they are the array count and
    // the per-array block stride (see KVTransferSpec).
    // local_control_port=-1: the slot protocol is engine-to-engine
    // coordination and has no role on a host store node.
    managers.reserve(spec.num_workers);
    for (size_t rank = 0; rank < spec.num_workers; ++rank) {
      managers.push_back(std::make_unique<KVCacheManagerWithTransfer>(
          num_block_arrays, spec.num_kv_shards, block_bytes_per_array,
          /*local_port=*/0,
          /*host_blocks_to_allocate=*/num_host_blocks, options.parallelism,
          /*node_id=*/static_cast<int64_t>(rank),
          /*local_control_port=*/-1));
    }

    // 2. Everything above the pool, assembled through the same
    // KVCacheStore::Create entry point a serving host uses: the
    // RaidenController coordination plane (num_blocks sized to the capacity,
    // so incoming writes can allocate block ids through AllocateBlockIds),
    // the HostOffloadBackend LRU cache, and the peer-facing store server.
    // Create publishes the node to the global registry (RegisterStore) once
    // the store server is bound, so registration happens exactly when the
    // advertised address is live. With no global registry configured
    // KVCacheStore stands up no store server at all (the registry decides
    // whether the peer plane exists). expected_worker_count stays at its
    // default 0: it gates construction on workers registering from outside,
    // and this node's workers only register below, after the store exists.
    kv_cache::BackendConfig backend_config;
    backend_config.type = "HostOffloadBackend";
    backend_config.kv_pool_group = options.kv_pool_group;
    backend_config.evict_tier = options.evict_tier;
    // Without a registry there is nothing to heartbeat, so the monitor only
    // runs when one is configured (Create rejects the combination).
    backend_config.monitor_config.enable =
        options.enable_store_monitor &&
        !options.global_registry_address.empty();
    ASSIGN_OR_RETURN(
        store, kv_cache::KVCacheStore::Create(
                   backend_config, /*capacity=*/num_host_blocks,
                   options.global_registry_address, options.raiden_id,
                   static_cast<int>(spec.num_kv_shards),
                   static_cast<int64_t>(block_bytes_per_array[0]),
                   options.store_server_ip, options.raiden_controller_port));
  } catch (const std::exception& e) {
    return absl::UnavailableError(
        absl::StrCat("host store node assembly failed: ", e.what()));
  }

  // 3. Make each manager reachable from the controller, one WorkerService
  // and one registration per rank. StartServer must precede RegisterWorker:
  // registration needs the worker's live control endpoint. Registration
  // goes through the controller's own RPC, the same path a serving host's
  // worker takes; node_id = rank is the pairing id peers match on.
  std::vector<std::unique_ptr<controller::WorkerServiceServer>> worker_servers;
  worker_servers.reserve(managers.size());
  core::controller::RaidenControllerClient controller_client(
      store->raiden_controller_address());
  for (size_t rank = 0; rank < managers.size(); ++rank) {
    auto worker_server = controller::WorkerServiceServer::Create();
    RETURN_IF_ERROR(worker_server->StartServer(
        /*host_allocator=*/nullptr, KVManagerHolder(managers[rank].get()),
        /*port=*/0));
    const std::string worker_endpoint = ComposeEndpoint(
        options.store_server_ip, worker_server->GetRaidenWorkerPort());
    absl::Status registered = controller_client.RegisterWorker(
        absl::StrCat("worker_", rank), worker_endpoint,
        managers[rank]->get_local_data_endpoints(),
        /*node_id=*/static_cast<int64_t>(rank));
    if (!registered.ok()) {
      // Without its full worker roster the node cannot pair with peers, so
      // fail the whole boot rather than come up half-alive.
      return absl::Status(
          registered.code(),
          absl::StrCat("host store node worker registration failed for rank ",
                       rank, ": ", registered.message()));
    }
    worker_servers.push_back(std::move(worker_server));
  }

  auto node = absl::WrapUnique(new KVCacheHostStoreNode(
      spec, num_host_blocks, std::move(managers), std::move(worker_servers),
      std::move(store)));
  LOG(INFO) << "KVCacheHostStoreNode up: " << node->num_host_blocks()
            << " host blocks x " << node->managers().size()
            << " workers (num_block_arrays=" << num_block_arrays
            << " num_kv_shards=" << spec.num_kv_shards
            << " block_bytes_per_array=["
            << absl::StrJoin(block_bytes_per_array, ",") << "]"
            << "), store server " << node->store_server_address()
            << ", controller " << node->raiden_controller_address();
  return node;
}

}  // namespace store_node
}  // namespace tpu_raiden
