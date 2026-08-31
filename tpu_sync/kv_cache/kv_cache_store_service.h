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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_SERVICE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/proto/kv_cache_store_service.grpc.pb.h"
#include "tpu_sync/proto/kv_cache_store_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

class WriteRemoteServerReactor;
struct WriteRemoteReactorGate;

class KVCacheStoreServiceImpl
    : public ::tpu_raiden::kv_cache::proto::KVCacheStoreService::CallbackService {
 public:
  // A stand-in for the pull. Tests only; production always calls
  // controller_->TransferBuffers. Lets a test hold a transfer open and
  // choose when it resolves.
  using TransferFn = std::function<tsl::Future<>(absl::Span<const Buffer>,
                                                 absl::Span<const Buffer>)>;

  KVCacheStoreServiceImpl(KVCacheStoreBackend* backend,
                          tpu_raiden::controller::RaidenController* controller);
  ~KVCacheStoreServiceImpl() override;

  ::grpc::ServerUnaryReactor* Fetch(
      ::grpc::CallbackServerContext* context,
      const ::tpu_raiden::kv_cache::proto::FetchRequest* request,
      ::tpu_raiden::kv_cache::proto::FetchResponse* response) override;

  // The destination side of a peer's save(dst). Applies the existence rule,
  // allocates landing blocks, issues the pull, and acks without waiting for
  // the bytes. When the operation goes terminal, writes the result on the
  // same call and closes it.
  ::grpc::ServerWriteReactor<::tpu_raiden::kv_cache::proto::WriteRemoteEvent>*
  WriteRemote(
      ::grpc::CallbackServerContext* context,
      const ::tpu_raiden::kv_cache::proto::WriteRemoteRequest* request) override;

  // Reports the state of an accepted operation; recovery for a source that
  // lost its WriteRemote call. `wait_ms > 0` holds the answer (capped by
  // MaxPollWait()) until the operation is terminal. UNKNOWN = no record.
  ::grpc::ServerUnaryReactor* PollWriteRemote(
      ::grpc::CallbackServerContext* context,
      const ::tpu_raiden::kv_cache::proto::PollWriteRemoteRequest* request,
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse* response)
      override;

  // Replaces the pull with `transfer_fn`. Tests only; set it before the
  // server serves, because the member is not mutex-guarded.
  void SetTransferFnForTesting(TransferFn transfer_fn);

  // Stops the deadline thread firing deadlines, without stopping the thread.
  // Tests only; lets a test show that the commit claim checks the wall clock
  // itself.
  void PauseDeadlineFiringForTesting();

  // Number of tracked operations in write_ops_. Tests only.
  size_t InFlightWriteOpsCountForTesting() const;

 private:
  // Withdraws `hash` from the global registry, unless this store still holds it
  // in host DRAM. Fire and forget; the result is not inspected. The residency
  // check costs one index probe and counts an eviction candidate as held,
  // since a candidate still has its host block and a local access promotes it
  // back.
  void WithdrawEntryIfUnbacked(const std::string& hash);

  // Keeps late callbacks away from a destroyed service. A callback holds `mu`
  // and checks `svc` while it runs; the destructor clears `svc` under `mu`,
  // so it either runs first or waits for the callback to finish.
  struct Lifetime {
    absl::Mutex mu;
    KVCacheStoreServiceImpl* svc ABSL_GUARDED_BY(mu) = nullptr;
  };

  // The destination's state machine for one operation. kCompleting has no
  // wire form: to the source it looks the same as PENDING.
  enum class OpState {
    // Accepted; the pull is running.
    kPending,
    // Claimed by a finished transfer; being committed right now.
    kCompleting,
    // Inserted and registered globally. Success.
    kCommitted,
    // Every hash was already here. Success.
    kAllExist,
    // Some hashes were already here. Failure.
    kPartialExist,
    // The transfer failed, or finished past the deadline.
    kFailed,
    // Inserted here, but not published to the registry. The source decides
    // what to do; see PollWriteRemoteResponse::STORED_UNREGISTERED.
    kStoredUnregistered,
  };

  // Maps an OpState to its wire state. kCompleting reports PENDING.
  static ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::State
  ToWireState(OpState state);

  // One accepted offer, tracked until its record expires.
  struct WriteOp {
    // Current state; terminal transitions go through MarkTerminal.
    OpState state = OpState::kPending;
    // The offered hashes.
    std::vector<std::string> block_hashes;
    // Host blocks allocated here for the pull to land in.
    std::vector<int32_t> landing_block_ids;
    // Populated only for PARTIAL_EXIST.
    std::vector<std::string> existing_hashes;
    // Populated only for STORED_UNREGISTERED.
    std::vector<std::string> unregistered_hashes;

    // When the offer was accepted.
    absl::Time accepted_at;
    // Last moment a finished transfer may commit. The commit claim checks
    // the wall clock against this directly.
    absl::Time deadline;
    // When the record may be collected: deadline + kRecordMargin.
    absl::Time expires_at;
    // The deadline granted to this operation.
    absl::Duration granted_deadline;

    // True once the landing blocks were freed, or the cache kept them. Stops
    // a second path freeing them again.
    bool blocks_released = false;
    // When to next warn that the landing blocks are still held.
    absl::Time next_leak_warning = absl::InfiniteFuture();

    // Path to the reactor that streams events back to the source. A sender
    // moves it out under write_mutex_, then locks the gate to reach the
    // reactor; see the gate's definition.
    std::shared_ptr<WriteRemoteReactorGate> reactor_gate;

    // True once the operation is terminal and its blocks are settled.
    // Teardown waits on this flag, not on transfer futures.
    bool settle_done = false;
  };

  // Runs when a transfer resolves. Static so it cannot capture `this`.
  static void OnTransferComplete(std::shared_ptr<Lifetime> lifetime,
                                 uint64_t op_id, absl::Status transfer_status);

  // A registry publish still in flight. `future` resolves when the registry
  // answers; `op` is the operation waiting on it.
  struct PendingPublish {
    tsl::Future<> future;
    std::shared_ptr<WriteOp> op;
  };

  // Handles a finished transfer: claims the operation, re-checks existence,
  // inserts, and starts the registry publish. Returns a PendingPublish when
  // the registry has not answered yet; the caller attaches the continuation,
  // because this runs with lifetime_->mu held and OnReady can run inline.
  std::optional<PendingPublish> CompleteWriteRemote(uint64_t op_id,
                                                    absl::Status
                                                        transfer_status);

  // Second half of CompleteWriteRemote, run once the registry has answered.
  void FinishPublish(const std::shared_ptr<WriteOp>& op, uint64_t op_id,
                     absl::Status registered);

  // The one place an operation goes terminal. Sets the state, fills the hash
  // lists, and marks blocks_released when the cache keeps the blocks.
  void MarkTerminal(const std::shared_ptr<WriteOp>& op, OpState state,
                    std::vector<std::string> existing)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(write_mutex_);

  // Marks the operation terminal, sends the result to the source, releases
  // the landing blocks if the cache did not keep them, and settles. Must not
  // be called with write_mutex_ held.
  void Finish(const std::shared_ptr<WriteOp>& op, OpState state,
              std::vector<std::string> existing);

  // One thread that sleeps until the next deadline, leak warning, or expired
  // record. Fires deadlines and collects settled records.
  void DeadlineLoop();

  // Returns the landing blocks to the pool, exactly once. Must not be called
  // with write_mutex_ held.
  void ReleaseLandingBlocks(const std::shared_ptr<WriteOp>& op);

  // Marks the operation settled, so teardown can stop waiting for it.
  void Settle(const std::shared_ptr<WriteOp>& op);

  // Looks up an operation by id; nullptr when the record is gone.
  std::shared_ptr<WriteOp> FindOp(uint64_t op_id)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(write_mutex_);

  // Builds the result message from the operation's terminal state.
  static proto::WriteRemoteEvent MakeResultEvent(
      const std::shared_ptr<WriteOp>& op);

  KVCacheStoreBackend* const backend_ = nullptr;
  tpu_raiden::controller::RaidenController* const controller_ = nullptr;

  // Test-only pull override; see SetTransferFnForTesting.
  TransferFn transfer_fn_override_;

  // Fence between this service and callbacks that outlive a call.
  std::shared_ptr<Lifetime> lifetime_;

  // Guards write_ops_ and everything reached through it.
  mutable absl::Mutex write_mutex_;
  // Live and recently settled operations, keyed by operation id.
  absl::flat_hash_map<uint64_t, std::shared_ptr<WriteOp>> write_ops_
      ABSL_GUARDED_BY(write_mutex_);
  // Generates operation ids.
  absl::BitGen op_id_rng_ ABSL_GUARDED_BY(write_mutex_);
  // Tells DeadlineLoop to exit.
  bool stop_deadline_thread_ ABSL_GUARDED_BY(write_mutex_) = false;
  // Tests only; see PauseDeadlineFiringForTesting.
  bool deadline_firing_paused_for_testing_ ABSL_GUARDED_BY(write_mutex_) =
      false;
  // Wakes DeadlineLoop early: a new operation, a terminal one, or shutdown.
  absl::CondVar deadline_cv_;
  // Runs DeadlineLoop.
  std::unique_ptr<std::thread> deadline_thread_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_SERVICE_H_
