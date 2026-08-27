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

// Standalone KV cache host store node.
//
// Lends this machine's DRAM to a raiden deployment as a KV cache pool.
// Start it alongside the serving jobs; it waits for the deployment's
// KVTransferSpec, assembles itself, publishes itself to the global registry,
// and serves until SIGTERM/SIGINT, on which it unregisters and tears down.

#include <csignal>
#include <cstddef>
#include <memory>
#include <string>

#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/store_node/grs_kv_transfer_spec_source.h"
#include "tpu_sync/store_node/kv_cache_host_store_node.h"

// Identity: the RaidenId this node registers under. Raiden treats the four
// fields as one opaque composite key -- no field is matched individually
// anywhere -- so the only hard requirements are that the tuple is unique
// within the deployment and stable across restarts (a restarted node replaces
// its own registration by re-registering the same id). The structure mirrors
// how serving jobs are addressed: job + replica locate the process, data
// name + replica index name the cache entity it hosts.
ABSL_FLAG(std::string, job_name, "",
          "Job this host store node is deployed as, e.g. "
          "'kv-host-store-node'.");
ABSL_FLAG(std::string, job_replica_id, "0",
          "Replica/task index of this node within --job_name.");
ABSL_FLAG(std::string, data_name, "kv_cache",
          "Name of the cache pool this node hosts. Any stable string; keep "
          "the default unless one deployment hosts several distinct pools.");
ABSL_FLAG(int, data_replica_idx, 0,
          "Replica index of the pool; 0 unless --data_name is replicated.");

// Addressing. The node listens on four ports: controller (gRPC), peer-facing
// store server (gRPC), worker control plane (WorkerService gRPC), and
// transfer data plane (BlockTransport raw TCP). Only the controller port is
// configurable, because KVCacheStore's API takes it alongside
// store_server_ip; the other three always bind OS-chosen ports. That is
// safe because every address is registered or published at boot -- worker
// to controller, store and controller to the registries -- so no peer ever
// discovers this node by computing a port.
ABSL_FLAG(std::string, store_server_ip, "",
          "IP peers reach this node on; bound and advertised. Required.");
ABSL_FLAG(int, raiden_controller_port, 0,
          "Controller port; 0 lets gRPC choose.");
ABSL_FLAG(std::string, global_registry_address, "",
          "GlobalRegistry host:port. Supplies the deployment's "
          "KVTransferSpec and is where this node publishes itself. "
          "Required.");
ABSL_FLAG(std::string, kv_pool_group, "",
          "KV pool group whose KVTransferSpec this node serves (the "
          "serving hosts publish their spec under this name; see "
          "global_registry.proto). Also the group this node registers "
          "under: placement never crosses groups. Required.");
ABSL_FLAG(int, evict_tier, 1,
          "Placement tier this node registers under (StoreInfo.evict_tier). "
          "Cold blocks demote strictly to greater tiers; serving hosts sit "
          "on tier 0, so the default puts this node right below them.");
ABSL_FLAG(bool, enable_store_monitor, true,
          "Heartbeat this node's status to the global registry. Heartbeats "
          "put a TTL on the registration (a dead node ages out of "
          "placement) and refresh the free capacity placement ranks by; "
          "without them this node is registered forever but never offers "
          "capacity.");

// Capacity.
ABSL_FLAG(size_t, dram_budget_bytes, 0,
          "Host DRAM lent to the pool, in bytes. Required.");

// Transfer plane.
ABSL_FLAG(int, parallelism, 1,
          "Parallel transport streams this node uses when it initiates "
          "transfers. Purely local performance tuning for this machine's "
          "NIC/CPU; peers need not match it.");


namespace {

using ::tpu_raiden::store_node::GrsKVTransferSpecSource;
using ::tpu_raiden::store_node::KVCacheHostStoreNode;

int Run() {
  // Block SIGTERM/SIGINT before the node spawns any thread: threads inherit
  // the mask, so delivery is confined to the sigwait below instead of
  // terminating the process with no teardown.
  sigset_t shutdown_signals;
  sigemptyset(&shutdown_signals);
  sigaddset(&shutdown_signals, SIGTERM);
  sigaddset(&shutdown_signals, SIGINT);
  pthread_sigmask(SIG_BLOCK, &shutdown_signals, nullptr);

  KVCacheHostStoreNode::Options options;
  options.raiden_id = tpu_raiden::kv_cache::RaidenId{
      absl::GetFlag(FLAGS_job_name), absl::GetFlag(FLAGS_job_replica_id),
      absl::GetFlag(FLAGS_data_name), absl::GetFlag(FLAGS_data_replica_idx)};
  options.store_server_ip = absl::GetFlag(FLAGS_store_server_ip);
  options.raiden_controller_port = absl::GetFlag(FLAGS_raiden_controller_port);
  options.global_registry_address =
      absl::GetFlag(FLAGS_global_registry_address);
  options.dram_budget_bytes = absl::GetFlag(FLAGS_dram_budget_bytes);
  options.parallelism = absl::GetFlag(FLAGS_parallelism);

  if (options.global_registry_address.empty()) {
    LOG(ERROR) << "--global_registry_address is required";
    return 1;
  }
  options.kv_pool_group = absl::GetFlag(FLAGS_kv_pool_group);
  if (options.kv_pool_group.empty()) {
    LOG(ERROR) << "--kv_pool_group is required";
    return 1;
  }
  options.evict_tier = absl::GetFlag(FLAGS_evict_tier);
  options.enable_store_monitor = absl::GetFlag(FLAGS_enable_store_monitor);
  GrsKVTransferSpecSource kv_transfer_spec_source(
      options.global_registry_address, options.kv_pool_group);

  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(options, &kv_transfer_spec_source);
  if (!node.ok()) {
    LOG(ERROR) << "host store node failed to boot: " << node.status();
    return 1;
  }

  // Park until asked to stop, then destroy the node so teardown runs in
  // order: unregister from the global registry, drain and stop the servers,
  // free the pool. A killed process skips all of that and leaves a stale
  // registry entry behind until the next boot re-registers the same id.
  int received_signal = 0;
  sigwait(&shutdown_signals, &received_signal);
  LOG(INFO) << "host store node shutting down on signal " << received_signal;
  node->reset();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  std::signal(SIGPIPE, SIG_IGN);
  return Run();
}
