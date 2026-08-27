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

class KVCacheStoreServiceImpl
    : public ::tpu_raiden::kv_cache::proto::KVCacheStoreService::Service {
 public:
  // A stand-in for the pull, used ONLY by tests. Production calls
  // controller_->TransferBuffers directly and never touches this.
  //
  // It exists because everything that makes a remote write correct happens
  // between "the pull was issued" and "the bytes arrived": the commit claim, a
  // deadline firing mid-transfer, the deferred free, teardown with an
  // operation outstanding. Testing any of those means holding a transfer
  // suspended and choosing the moment it resolves. The real call cannot do
  // that -- RaidenController is concrete with no virtual methods, and its
  // future resolves on its own schedule -- so a test supplies a future backed
  // by a promise it sets by hand.
  using TransferFn = std::function<tsl::Future<>(absl::Span<const Buffer>,
                                                 absl::Span<const Buffer>)>;

  KVCacheStoreServiceImpl(KVCacheStoreBackend* backend,
                          tpu_raiden::controller::RaidenController* controller);
  ~KVCacheStoreServiceImpl() override;

  ::grpc::Status Fetch(
      ::grpc::ServerContext* context,
      const ::tpu_raiden::kv_cache::proto::FetchRequest* request,
      ::tpu_raiden::kv_cache::proto::FetchResponse* response) override;

  // Accepts an offer of blocks from a peer -- the destination side of that
  // peer's save(dst): decides whether the write is worth doing, allocates
  // landing blocks, issues the pull, and answers with an operation id. Never
  // waits for the bytes.
  ::grpc::Status WriteRemote(
      ::grpc::ServerContext* context,
      const ::tpu_raiden::kv_cache::proto::WriteRemoteRequest* request,
      ::tpu_raiden::kv_cache::proto::WriteRemoteResponse* response) override;

  // Reports what became of an accepted operation. UNKNOWN once the record has
  // aged out, which the source cannot distinguish from "never happened" and
  // must treat as failure.
  ::grpc::Status PollWriteRemote(
      ::grpc::ServerContext* context,
      const ::tpu_raiden::kv_cache::proto::PollWriteRemoteRequest* request,
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse* response)
      override;

  // Diverts the pull through `transfer_fn` instead of the controller. TESTS
  // ONLY, and only before the server serves: the member is not mutex-guarded,
  // so setting it on a live service races with in-flight handlers. The
  // CHECK-fail on a non-empty operation map catches only the obvious half of
  // that.
  void SetTransferFnForTesting(TransferFn transfer_fn);

  // Stops the deadline thread firing deadlines, without stopping the thread.
  // Exists so a test can show that a late transfer is refused by the wall-clock
  // check in the claim itself, rather than only because the deadline thread
  // happened to run first. That distinction is the whole point of the
  // wall-clock term: a descheduled or wedged deadline thread must not be able
  // to let a commit through after the source has released its pin.
  void PauseDeadlineFiringForTesting();

 private:
  // Withdraws `hash` from the global registry, unless this store still holds it
  // in host DRAM. Fire and forget; the result is not inspected. The residency
  // check costs one index probe and counts an eviction candidate as held,
  // since a candidate still has its host block and a local access promotes it
  // back.
  void WithdrawEntryIfUnbacked(const std::string& hash);

  // A transfer-completion callback and the deadline thread both outlive the
  // call that started them, so neither may capture `this`. Both hold `mu` for
  // as long as they touch the service; the destructor clears `svc` under `mu`,
  // so teardown either happens first (and the callback then does nothing) or
  // waits for the callback to finish.
  struct Lifetime {
    absl::Mutex mu;
    KVCacheStoreServiceImpl* svc ABSL_GUARDED_BY(mu) = nullptr;
  };

  // The destination's own state machine. COMPLETING has no wire
  // representation on purpose: it means "claimed, and being finished right
  // now", which to the source is indistinguishable from PENDING. Keeping it
  // out of the proto stops it becoming a state a peer can reason about.
  enum class OpState {
    kPending,
    kCompleting,
    kCommitted,
    kAllExist,
    kPartialExist,
    kFailed,
    // Inserted here, but not published to the registry. See
    // PollWriteRemoteResponse::STORED_UNREGISTERED -- the source decides what
    // to do about it, because only the source knows whether it was going to
    // free its copy.
    kStoredUnregistered,
  };

  static ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::State
  ToWireState(OpState state);

  struct WriteOp {
    uint64_t id = 0;
    OpState state = OpState::kPending;
    std::vector<std::string> block_hashes;
    std::vector<int32_t> landing_block_ids;
    // Populated only for PARTIAL_EXIST.
    std::vector<std::string> existing_hashes;
    // Populated only for STORED_UNREGISTERED.
    std::vector<std::string> unregistered_hashes;

    absl::Time accepted_at;
    // Wall clock, not a timer reading: the commit check compares against this
    // directly so that commit safety does not depend on the deadline thread
    // having been scheduled.
    absl::Time deadline;
    // deadline + margin. Derived rather than a separate retention constant --
    // see the comment on kRecordMargin.
    absl::Time expires_at;
    absl::Duration granted_deadline;

    // The landing blocks return to the pool exactly once, from whichever path
    // reaches them first. COMMITTED and STORED_UNREGISTERED never release
    // them -- the cache owns the blocks then, and they are marked released so
    // no later path frees them out from under it.
    bool blocks_released = false;
    // When to next log a warning that a free is still deferred; only
    // meaningful while blocks are outstanding.
    absl::Time next_leak_warning = absl::InfiniteFuture();

    // Resolves once the operation is terminal AND its blocks are settled.
    // Teardown waits on these rather than on the raw transfer futures: Await()
    // on a future does not order with its own OnReady continuation.
    tsl::Future<> settled;
    tsl::Promise<> settled_promise;
    bool settle_done = false;
  };

  // Runs when a transfer resolves. Static, taking the lifetime handle, so it
  // cannot accidentally capture `this`.
  static void OnTransferComplete(std::shared_ptr<Lifetime> lifetime,
                                 uint64_t op_id, absl::Status transfer_status);

  // A publish that CompleteWriteRemote started and could not finish inline.
  // `future` resolves when the registry answers; `op` is the operation waiting
  // on it.
  struct PendingPublish {
    tsl::Future<> future;
    std::shared_ptr<WriteOp> op;
  };

  // Returns a PendingPublish when the operation reached the registry and the
  // answer has not arrived yet. The continuation is deliberately NOT attached
  // here: this runs with `lifetime_->mu` held, and Future::OnReady on an
  // already-resolved future runs inline on the calling thread, so attaching
  // here would re-enter that non-reentrant mutex. A backend with no registry
  // configured returns an already-resolved future, so that is the common case,
  // not a rare race.
  std::optional<PendingPublish> CompleteWriteRemote(uint64_t op_id,
                                                    absl::Status
                                                        transfer_status);

  // Second half of CompleteWriteRemote, run once the registry has answered.
  void FinishPublish(const std::shared_ptr<WriteOp>& op, uint64_t op_id,
                     absl::Status registered);

  // Records an operation's verdict, releases its landing blocks when the cache
  // did not keep them, and settles it. Must NOT be called with write_mutex_
  // held.
  void Finish(const std::shared_ptr<WriteOp>& op, OpState state,
              std::vector<std::string> existing);

  // Wakes at the earliest interesting time across all operations: a deadline
  // to fire, or a deferred free to complain about. One joined thread, never
  // one per operation.
  void DeadlineLoop();

  // Marks the blocks free exactly once and returns them to the pool. Must NOT
  // be called with write_mutex_ held.
  void ReleaseLandingBlocks(const std::shared_ptr<WriteOp>& op);

  void Settle(const std::shared_ptr<WriteOp>& op);

  std::shared_ptr<WriteOp> FindOp(uint64_t op_id)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(write_mutex_);

  KVCacheStoreBackend* const backend_ = nullptr;
  tpu_raiden::controller::RaidenController* const controller_ = nullptr;

  TransferFn transfer_fn_override_;

  std::shared_ptr<Lifetime> lifetime_;

  mutable absl::Mutex write_mutex_;
  absl::flat_hash_map<uint64_t, std::shared_ptr<WriteOp>> write_ops_
      ABSL_GUARDED_BY(write_mutex_);
  absl::BitGen op_id_rng_ ABSL_GUARDED_BY(write_mutex_);
  bool stop_deadline_thread_ ABSL_GUARDED_BY(write_mutex_) = false;
  bool deadline_firing_paused_for_testing_ ABSL_GUARDED_BY(write_mutex_) =
      false;
  absl::CondVar deadline_cv_;
  std::unique_ptr<std::thread> deadline_thread_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_SERVICE_H_
