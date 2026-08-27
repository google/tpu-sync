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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_CONTROLLER_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_CONTROLLER_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/buffer.h"
#include "tpu_sync/core/controller/controller_service.h"
#include "tpu_sync/core/controller/worker_registry.h"
#include "tpu_sync/core/controller/worker_service_client.h"
#include "tpu_sync/kv_cache/logical_block_manager.h"
#include "tpu_sync/proto/controller_service.grpc.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {
class RaidenControllerServiceImpl;
class ControllerServer;
}  // namespace controller
}  // namespace core

namespace controller {

// Per-read state for an in-flight ReadRemote (defined in the .cc).
struct RemoteReadState;

// Raiden Controller responsible for managing logical KV cache block allocations
// and synchronizing them with remote transfer workers via WorkerService gRPC.
class RaidenController {
 public:
  // Constructs a RaidenController for the given unit.
  // The RaidenController sets up its own ControllerService and local
  // LogicalBlockManager. Workers will dynamically register with it.
  //
  // This controller resolves no peers of its own: every address it dials is
  // handed to it (ReadRemote takes the source's controller address). Mapping a
  // peer's identity to that address belongs to whoever owns the directory --
  // for KVCacheStore, the global registry.
  //
  // `num_blocks` drives two independent things: the size of the logical-block
  // ledger (AllocateBlockIds), and how many physical buffers are pre-created
  // on every registering worker for the Legacy Physical/BufferProto mode
  // (Allocate/AllocateBuffers below). `preprovision_worker_buffers` = false
  // skips that pre-creation.
  //
  // `expected_worker_count` > 0 makes initialization block until that many
  // workers have registered, failing after RAIDEN_EXPECTED_WORKERS_TIMEOUT_S
  // seconds (default 120). Leave it 0 when workers register only after
  // Create() returns.
  //
  // TODO: Remove preprovision_worker_buffers and legacy
  // Physical/BufferProto mode.
  static absl::StatusOr<std::unique_ptr<RaidenController>> Create(
      const ::tpu_sync::rpc::RaidenIdProto& unit, int num_blocks,
      int num_shards, int64_t shard_size_bytes,
      absl::string_view raiden_controller_address = "",
      bool preprovision_worker_buffers = true, int expected_worker_count = 0);

  // Creates a fully initialized RaidenController for multiple worker addresses.
  // It will also start the ControllerServer to allow dynamic registrations.
  static absl::StatusOr<std::unique_ptr<RaidenController>> Create(
      const ::tpu_sync::rpc::RaidenIdProto& unit,
      absl::Span<const std::string> worker_addresses, int num_blocks,
      int num_shards, int64_t shard_size_bytes,
      absl::string_view raiden_controller_address = "",
      bool preprovision_worker_buffers = true, int expected_worker_count = 0);
  // Destructor automatically calls DeleteBuffers to clean up all pre-created
  // buffers on the registered workers.
  ~RaidenController();

  // Legacy API (Physical/BufferProto Mode):
  // Allocates `num_blocks` logical blocks and returns their associated
  // `BufferProto`s containing sharded physical buffer handles. Use these if you
  // need worker-level raw buffer handles. No RPC is performed.
  absl::StatusOr<std::vector<::tpu_sync::proto::BufferProto>> Allocate(
      int num_blocks);

  // Allocates num_blocks logical blocks from the pre-created buffer pool and
  // returns the corresponding Buffers. No gRPC call is made.
  absl::StatusOr<std::vector<Buffer>> AllocateBuffers(int num_blocks);

  // Deallocates the specified Buffers in the local logical block manager,
  // returning their blocks to the free pool. No gRPC call is made.
  absl::Status DeallocateBuffers(absl::Span<const Buffer> buffers);

  // Legacy API (Physical/BufferProto Mode):
  // Deallocates logical blocks associated with the provided `BufferProto`s.
  // No RPC is performed.
  absl::Status Deallocate(
      absl::Span<const ::tpu_sync::proto::BufferProto> sharded_buffers);

  // New API (Logical/Block ID Mode):
  // Allocates `num_blocks` logical blocks from the block manager and returns
  // their raw logical block IDs (0..N-1). No RPC is performed.
  absl::StatusOr<std::vector<int>> AllocateBlockIds(int num_blocks);

  // New API (Logical/Block ID Mode):
  // Deallocates the given logical block IDs in the block manager, returning
  // them to the free pool. No RPC is performed.
  absl::Status DeallocateBlockIds(absl::Span<const int> block_ids);

  // New API (Logical/Block ID Mode):
  // Allocates exactly the given logical block IDs in the block manager,
  // locked. The whole batch is validated first: fails without changing any
  // state if an ID is out of range, already allocated, or duplicated. No RPC
  // is performed.
  absl::Status AllocateTargetBlockIds(absl::Span<const int> block_ids);

  // Targeted worker transfer.
  // staging_host_buffers: host DRAM staging (bridge) block offsets, required
  // for 2-stage remote transfers (remote H2D read/write, remote D2H write);
  // always the middle hop of the data flow. Unused for local transfers.
  tsl::Future<> TransferBuffers(
      absl::string_view worker_id, absl::Span<const Buffer> src_buffers,
      absl::Span<const Buffer> dst_buffers,
      absl::Span<const Buffer> staging_host_buffers = {},
      absl::Span<const int64_t> copy_sizes = {});

  // Broadcast transfer to all registered workers (staging_host_buffers as
  // above).
  tsl::Future<> TransferBuffers(
      absl::Span<const Buffer> src_buffers,
      absl::Span<const Buffer> dst_buffers,
      absl::Span<const Buffer> staging_host_buffers = {},
      absl::Span<const int64_t> copy_sizes = {});

  // Reads blocks from a remote source, receiver-initiated: this controller's
  // own workers pull the bytes, so the write window belongs to the destination
  // and the H2D leg folds into the same transfer.
  //
  // Three phases against the source (see controller_service.proto):
  //   AcquireReadLease -> pull -> ReleaseReadLease.
  // The returned future succeeds ONLY if the transfer succeeded AND the source
  // reports verdict HELD, i.e. it kept the blocks pinned for the whole read.
  // Anything else fails the future and the caller must discard.
  //
  // dst_device_block_ids selects the mode:
  //   empty            -> ReadRemote-to-Host. Bytes land in dst_host_block_ids.
  //   size == hashes   -> ReadRemote-to-HBM. Bytes land in the caller's device
  //                       blocks, with dst_host_block_ids as a staging hop the
  //                       caller reclaims afterwards (KVCacheStore frees them
  //                       once the read settles and records nothing for them).
  //   any other size   -> InvalidArgument, before anything is pinned.
  //
  // src_host_block_ids is ADVISORY (registry-derived, possibly stale); the
  // source re-derives the authoritative ids while validating the hashes.
  //
  // src_controller_address is the source's ControllerService address, supplied
  // by the caller -- this controller holds no directory and resolves nothing.
  // An empty address is InvalidArgument, rejected before the source is asked to
  // pin anything.
  //
  // In HBM mode the caller's device blocks are written before the verdict is
  // known, so a discarded read leaves them holding undefined bytes. That is by
  // design: nothing in the LRU ever points at them (a remote read records
  // nothing in the store -- the caller alone tracks the device blocks), and
  // callers overwrite device blocks when they reuse them. Treat supplied
  // device blocks as scratch until the read reports success.
  tsl::Future<> ReadRemote(
      absl::string_view src_controller_address,
      const std::vector<int32_t>& src_host_block_ids,
      const std::vector<int32_t>& dst_host_block_ids,
      const std::vector<std::string>& block_hashes = {},
      const std::vector<int32_t>& dst_device_block_ids = {});

  // Weak handle to a live controller. An in-flight ReadRemote must be able to
  // call back into the controller to issue the pull, but the read outlives the
  // calling frame, so capturing `this` would make controller teardown
  // mid-read a use-after-free. The destructor clears `ctrl` under `mu`, and a
  // continuation holds `mu` for as long as it touches the controller -- so
  // teardown either happens before the callback (which then fails cleanly) or
  // waits for it.
  struct Lifetime {
    absl::Mutex mu;
    RaidenController* ctrl ABSL_GUARDED_BY(mu) = nullptr;
  };

  // Registers the source-side ReadRemote hooks, invoked when this controller
  // acts as the SOURCE of a remote read: AcquireReadLease calls the first to
  // validate the requested hashes and pin their host blocks, and lease
  // release/expiry calls the second to unpin them. KVCacheStore registers its
  // ValidateAndPinHostBlocks/UnpinHostBlocks here. Forwards to the hosted
  // ControllerService.
  void SetReadRemoteHooks(
      core::controller::RaidenControllerServiceImpl::ValidateAndPinCallback
          validate_and_pin,
      core::controller::RaidenControllerServiceImpl::UnpinCallback unpin);

  // Forwards an opaque transfer program to the named registered worker over
  // its persistent WorkerService channel. The controller does not interpret
  // the program or hold reshard state; an unknown worker id or missing client
  // resolves the future with an error status.
  tsl::Future<::tpu_sync::proto::TransferProgramResponse> SubmitTransferProgram(
      absl::string_view worker_id,
      const ::tpu_sync::proto::TransferProgramRequest& request);

  // Cached peer-controller channels. Test observability only: what it pins is
  // that the cache stays bounded as peers restart onto new addresses.
  size_t CachedStubCountForTest() const {
    absl::MutexLock lock(mutex_);
    return stubs_.size();
  }
  static size_t MaxCachedStubsForTest() { return kMaxCachedStubs; }

  // Accessors for state inspection and testing.
  const ::tpu_sync::rpc::RaidenIdProto& unit() const { return unit_; }
  kv_cache::LogicalBlockManager* block_manager() const {
    return block_manager_.get();
  }
  std::shared_ptr<core::controller::WorkerRegistry> worker_registry() const {
    return worker_registry_;
  }

  int num_shards() const { return num_shards_; }
  int64_t shard_size_bytes() const { return shard_size_bytes_; }
  std::string controller_address() const { return raiden_controller_address_; }
  const std::vector<::tpu_sync::proto::BufferProto>& all_sharded_buffers()
      const {
    return all_sharded_buffers_;
  }

 private:
  absl::StatusOr<::tpu_sync::proto::TransferBuffersRequest>
  BuildTransferBuffersRequest(
      absl::Span<const Buffer> src_buffers,
      absl::Span<const Buffer> dst_buffers,
      absl::Span<const Buffer> staging_host_buffers = {},
      absl::Span<const int64_t> copy_sizes = {});

 private:
  RaidenController(const ::tpu_sync::rpc::RaidenIdProto& unit, int num_blocks,
                   int num_shards, int64_t shard_size_bytes,
                   bool preprovision_worker_buffers);

  absl::Status Init(absl::Span<const std::string> worker_addresses,
                    absl::string_view raiden_controller_address,
                    int expected_worker_count);

  void Cleanup();

  absl::Status InitializeWorkerBuffers(
      core::controller::WorkerRegistration& reg);
  ::tpu_sync::rpc::RaidenIdProto unit_;

  int num_shards_;
  int64_t shard_size_bytes_;
  int num_total_blocks_;
  bool preprovision_worker_buffers_ = true;
  std::vector<::tpu_sync::proto::BufferProto> all_sharded_buffers_;
  std::shared_ptr<core::controller::WorkerRegistry> worker_registry_;
  mutable absl::Mutex mutex_;
  std::unique_ptr<kv_cache::LogicalBlockManager> block_manager_
      ABSL_GUARDED_BY(mutex_);
  std::string raiden_controller_address_;

  // Issues the pull for an in-flight read and chains the release. Runs on a
  // gRPC callback thread with lifetime_->mu held.
  void PullAndRelease(const std::vector<int32_t>& src_host_block_ids,
                      const std::vector<RaidenWorkerEndpoints>& src_groups,
                      const std::shared_ptr<RemoteReadState>& state,
                      bool to_hbm);

  std::shared_ptr<Lifetime> lifetime_ = std::make_shared<Lifetime>();

  std::unique_ptr<core::controller::ControllerServer>
      private_controller_server_;
  // The active ControllerServer hosting this controller's service (either
  // private_controller_server_ or the shared singleton). Used to register the
  // ReadRemote verify/pin and unpin hooks after construction. Not owned.
  core::controller::ControllerServer* active_server_ = nullptr;

  // Channel cache, keyed by address -- not a directory. A peer that restarts on
  // a new address is simply a new key, and its old entry is never looked up
  // again.
  //
  // Which is why this is BOUNDED. While the caller cached peer addresses
  // forever, a restarted peer's new address was never learned, so this map
  // could not grow past one entry per peer; resolving per read removed that
  // accidental bound and made every peer restart leave a dead channel behind.
  // Eviction is oldest-first: in steady state one address per peer stays
  // resident, so the entries that age out are the ones that stopped being
  // dialled -- the dead ones. An evicted stub that is still mid-read stays
  // alive through the shared_ptr its RemoteReadState holds.
  //
  // The precise version of this would drop a stub when a read against it fails
  // to connect (as the store does for its own peer clients). That needs the
  // failing address plumbed back through the acquire callback; the bound is
  // what makes the growth safe without it.
  static constexpr size_t kMaxCachedStubs = 256;
  absl::flat_hash_map<
      std::string,
      std::shared_ptr<::tpu_sync::proto::RaidenControllerService::Stub>>
      stubs_ ABSL_GUARDED_BY(mutex_);
  // Insertion order for the bound above.
  std::deque<std::string> stub_order_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace controller
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_CONTROLLER_H_
