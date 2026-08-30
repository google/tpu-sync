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

// Service implementation for KVCacheStoreService.

#include "tpu_sync/kv_cache/kv_cache_store_service.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/buffer.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/proto/kv_cache_store_service.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {

// The longest deadline this destination grants, whatever the source asked
// for. Kept below the source's default HOLD (30 s), so the source protects
// its blocks until this destination has given up.
constexpr absl::Duration kDefaultDeadlineCap = absl::Seconds(25);

// Grace period after an operation's deadline before its record may be
// collected. With default settings this is long enough for the source's one
// recovery ask to still find the record.
constexpr absl::Duration kRecordMargin = absl::Seconds(5);

// A free still deferred this far past the granted deadline is logged: nothing
// bounds a transfer, so those blocks may never come back.
constexpr int kLeakWarningDeadlineMultiple = 3;
// How often to repeat that warning.
constexpr absl::Duration kLeakWarningInterval = absl::Seconds(60);

// See ~KVCacheStoreServiceImpl for why teardown does not wait indefinitely.
constexpr absl::Duration kTeardownQuiesceTimeout = absl::Seconds(10);

// The deadline cap; RAIDEN_REMOTE_WRITE_DEADLINE_S overrides the default.
absl::Duration DeadlineCap() {
  const char* env = std::getenv("RAIDEN_REMOTE_WRITE_DEADLINE_S");
  if (env == nullptr) {
    return kDefaultDeadlineCap;
  }
  int seconds = 0;
  if (!absl::SimpleAtoi(env, &seconds) || seconds <= 0) {
    LOG(WARNING) << "Ignoring RAIDEN_REMOTE_WRITE_DEADLINE_S=\"" << env
                 << "\": expected a positive number of seconds.";
    return kDefaultDeadlineCap;
  }
  return absl::Seconds(seconds);
}

// The longest a waiting PollWriteRemote may hold its gRPC thread: the longest
// an operation can live, plus the margin its record survives after that. The
// handler clamps every requested wait to this.
absl::Duration MaxPollWait() { return DeadlineCap() + kRecordMargin; }

template <typename ProtoContainer>
std::vector<RaidenWorkerEndpoints> UnpackWorkerEndpointsProto(
    const ProtoContainer& proto_groups) {
  std::vector<RaidenWorkerEndpoints> result;
  result.reserve(proto_groups.size());
  for (const auto& group_proto : proto_groups) {
    std::vector<RaidenTransferEndpoint> eps;
    eps.reserve(group_proto.endpoints_size());
    for (const auto& ep_proto : group_proto.endpoints()) {
      eps.push_back({ep_proto.endpoint(),
                     std::vector<int64_t>(ep_proto.shards().begin(),
                                          ep_proto.shards().end())});
    }
    result.push_back(
        {group_proto.node_id(), group_proto.worker_id(), std::move(eps)});
  }
  return result;
}

}  // namespace

// Co-owned by a WriteOp and the reactor answering it. Senders reach the
// reactor only while holding `mu`; OnDone nulls `reactor` under `mu` before
// the reactor deletes itself. So a sender either finds a live reactor, or
// finds null and drops the answer. Never copy the raw pointer out.
struct WriteRemoteReactorGate {
  absl::Mutex mu;
  WriteRemoteServerReactor* reactor ABSL_GUARDED_BY(mu) = nullptr;
};

// Streams the ack and, later, the result of one WriteRemote call back to the
// source. Owns itself; deletes itself in OnDone.
class WriteRemoteServerReactor
    : public ::grpc::ServerWriteReactor<
          ::tpu_raiden::kv_cache::proto::WriteRemoteEvent> {
 public:
  // `gate` is null when the offer creates no operation (a refusal or an
  // existence answer): nothing needs to reach this reactor later.
  explicit WriteRemoteServerReactor(std::shared_ptr<WriteRemoteReactorGate> gate)
      : gate_(std::move(gate)) {}

  void StartAck(proto::WriteRemoteEvent ack_event) {
    {
      absl::MutexLock lock(mu_);
      // The destructor's sweep can finish this call before the ack goes out;
      // a write after Finish is undefined, so the ack is dropped.
      if (finished_) {
        return;
      }
      ack_event_ = std::move(ack_event);
    }
    StartWrite(&ack_event_);
  }

  // Sends the ack and closes the call; for offers that create no operation.
  void StartAckAndFinish(proto::WriteRemoteEvent ack_event) {
    {
      absl::MutexLock lock(mu_);
      ack_event_ = std::move(ack_event);
      finished_ = true;
    }
    StartWriteAndFinish(&ack_event_, grpc::WriteOptions(), ::grpc::Status::OK);
  }

  // Closes the call with `status`. Safe to call more than once.
  void FinishWithError(::grpc::Status status) {
    bool should_finish = false;
    {
      absl::MutexLock lock(mu_);
      if (!finished_) {
        finished_ = true;
        should_finish = true;
      }
    }
    if (should_finish) {
      Finish(status);
    }
  }

  // Sends the result and closes the call, once the ack write has finished.
  void SendResultAndFinish(proto::WriteRemoteEvent result_event) {
    bool should_write = false;
    {
      absl::MutexLock lock(mu_);
      result_event_ = std::move(result_event);
      has_result_ = true;
      if (ack_written_ && !finished_) {
        finished_ = true;
        should_write = true;
      }
    }
    if (should_write) {
      StartWriteAndFinish(&result_event_, grpc::WriteOptions(),
                          ::grpc::Status::OK);
    }
  }

  void OnWriteDone(bool ok) override {
    if (!ok) {
      FinishWithError(
          ::grpc::Status(::grpc::StatusCode::CANCELLED, "Write failed"));
      return;
    }
    bool should_write = false;
    {
      absl::MutexLock lock(mu_);
      if (!ack_written_) {
        ack_written_ = true;
        if (has_result_ && !finished_) {
          finished_ = true;
          should_write = true;
        }
      }
    }
    if (should_write) {
      StartWriteAndFinish(&result_event_, grpc::WriteOptions(),
                          ::grpc::Status::OK);
    }
  }

  void OnCancel() override {
    // A cancelled call must still be finished: gRPC withholds OnDone until
    // Finish, and an unfinished call stays a pending RPC forever. Only the
    // call ends here; the operation carries on.
    FinishWithError(::grpc::Status::CANCELLED);
  }

  void OnDone() override {
    if (gate_ != nullptr) {
      // After this, no sender can reach the object again; anyone already
      // inside a call through the gate holds gate_->mu, which this waits for.
      absl::MutexLock lock(gate_->mu);
      gate_->reactor = nullptr;
    }
    delete this;
  }

 private:
  // Shared with the WriteOp; OnDone nulls the reactor in it before deleting.
  const std::shared_ptr<WriteRemoteReactorGate> gate_;
  absl::Mutex mu_;
  // The two messages; they must outlive their gRPC writes.
  proto::WriteRemoteEvent ack_event_;
  proto::WriteRemoteEvent result_event_ ABSL_GUARDED_BY(mu_);
  // True once the ack write completed.
  bool ack_written_ ABSL_GUARDED_BY(mu_) = false;
  // True once a result is stored, waiting for the ack write to complete.
  bool has_result_ ABSL_GUARDED_BY(mu_) = false;
  // True once Finish was requested; no further writes are allowed.
  bool finished_ ABSL_GUARDED_BY(mu_) = false;
};

namespace {

// The one way a result reaches a reactor. A null gate or a null reactor
// means nobody is listening; the answer is dropped.
void SendResultThroughGate(const std::shared_ptr<WriteRemoteReactorGate>& gate,
                           proto::WriteRemoteEvent result_event) {
  if (gate == nullptr) {
    return;
  }
  absl::MutexLock lock(gate->mu);
  if (gate->reactor != nullptr) {
    gate->reactor->SendResultAndFinish(std::move(result_event));
  }
}

}  // namespace

KVCacheStoreServiceImpl::KVCacheStoreServiceImpl(
    KVCacheStoreBackend* backend,
    tpu_raiden::controller::RaidenController* controller)
    : backend_(backend),
      controller_(controller),
      lifetime_(std::make_shared<Lifetime>()) {
  {
    absl::MutexLock lock(lifetime_->mu);
    lifetime_->svc = this;
  }
  deadline_thread_ = std::make_unique<std::thread>(
      &KVCacheStoreServiceImpl::DeadlineLoop, this);
}

KVCacheStoreServiceImpl::~KVCacheStoreServiceImpl() {
  {
    absl::MutexLock lock(write_mutex_);
    stop_deadline_thread_ = true;
    deadline_cv_.Signal();
  }
  if (deadline_thread_ && deadline_thread_->joinable()) {
    deadline_thread_->join();
  }

  // Wait (bounded) for in-flight operations to settle, so their landing
  // blocks go back to the pool before the controller that owns it dies.
  // Bounded because this runs inside KVCacheStoreServer::Shutdown() with the
  // server's mutex held, and nothing bounds a transfer or a registry call.
  // The wait is on the settle_done flag: a resolved transfer future does not
  // mean its completion has run, and Await() takes no deadline.
  {
    absl::MutexLock lock(write_mutex_);
    const auto all_settled = [this]() ABSL_NO_THREAD_SAFETY_ANALYSIS {
      for (const auto& [id, op] : write_ops_) {
        if (!op->settle_done) return false;
      }
      return true;
    };
    if (!write_mutex_.AwaitWithDeadline(
            absl::Condition(&all_settled),
            absl::Now() + kTeardownQuiesceTimeout)) {
      LOG(ERROR) << "KVCacheStoreServiceImpl teardown gave up after "
                 << kTeardownQuiesceTimeout
                 << " waiting for in-flight remote writes to settle; their "
                     "landing blocks stay held. A source is wedged, or the "
                     "global registry is not answering.";
    }
  }

  // End the calls of operations that never answered, so their sources see
  // UNAVAILABLE instead of waiting out their deadlines.
  std::vector<std::shared_ptr<WriteRemoteReactorGate>> reactors_to_finish;
  {
    absl::MutexLock lock(write_mutex_);
    for (auto& [id, op] : write_ops_) {
      if (op->reactor_gate != nullptr) {
        reactors_to_finish.push_back(std::move(op->reactor_gate));
        op->reactor_gate = nullptr;
      }
    }
  }
  for (const auto& gate : reactors_to_finish) {
    absl::MutexLock lock(gate->mu);
    if (gate->reactor != nullptr) {
      gate->reactor->FinishWithError(::grpc::Status(
          ::grpc::StatusCode::UNAVAILABLE, "Server shutting down"));
    }
  }

  // Clearing `svc` under `mu` makes a late completion safe: it either finds a
  // null service and does nothing, or is already running and this waits for
  // it. Unbounded on purpose; the alternative is freeing the object while a
  // callback is still inside it.
  absl::MutexLock lock(lifetime_->mu);
  lifetime_->svc = nullptr;
}

void KVCacheStoreServiceImpl::PauseDeadlineFiringForTesting() {
  absl::MutexLock lock(write_mutex_);
  deadline_firing_paused_for_testing_ = true;
}

size_t KVCacheStoreServiceImpl::InFlightWriteOpsCountForTesting() const {
  absl::MutexLock lock(write_mutex_);
  return write_ops_.size();
}

void KVCacheStoreServiceImpl::SetTransferFnForTesting(TransferFn transfer_fn) {
  {
    absl::MutexLock lock(write_mutex_);
    CHECK(write_ops_.empty())
        << "SetTransferFnForTesting must be called before the server starts "
           "serving; the member is not mutex-guarded.";
  }
  transfer_fn_override_ = std::move(transfer_fn);
}

void KVCacheStoreServiceImpl::WithdrawEntryIfUnbacked(const std::string& hash) {
  if (backend_ == nullptr) {
    return;
  }
  // A lookup miss does not mean the block is gone: the lookup cannot see an
  // eviction candidate, and a candidate still holds its host block. So probe
  // with AlreadyPresentHostResident, which does see candidates, and leave the
  // entry alone if one is what the lookup missed.
  if (!backend_->AlreadyPresentHostResident({hash}).empty()) {
    return;
  }
  backend_->UnregisterBlocksAsync({hash}).OnReady([](absl::Status) {});
}

proto::WriteRemoteEvent KVCacheStoreServiceImpl::MakeResultEvent(
    const std::shared_ptr<WriteOp>& op) {
  proto::WriteRemoteEvent event;
  auto* result = event.mutable_result();
  result->set_state(ToWireState(op->state));
  for (const auto& hash : op->existing_hashes) {
    result->add_existing_hashes(hash);
  }
  for (const auto& hash : op->unregistered_hashes) {
    result->add_unregistered_hashes(hash);
  }
  return event;
}

::grpc::ServerUnaryReactor* KVCacheStoreServiceImpl::Fetch(
    ::grpc::CallbackServerContext* context,
    const ::tpu_raiden::kv_cache::proto::FetchRequest* request,
    ::tpu_raiden::kv_cache::proto::FetchResponse* response) {
  auto* reactor = context->DefaultReactor();
  if (backend_ == nullptr || controller_ == nullptr) {
    reactor->Finish(::grpc::Status(::grpc::StatusCode::FAILED_PRECONDITION,
                                  "Backend or RaidenController non-initialized"));
    return reactor;
  }

  const std::vector<std::string> block_hashes(request->block_hashes().begin(),
                                              request->block_hashes().end());
  const std::vector<int32_t> dst_host_block_ids(
      request->host_block_ids().begin(), request->host_block_ids().end());

  if (block_hashes.empty()) {
    reactor->Finish(::grpc::Status::OK);
    return reactor;
  }

  if (dst_host_block_ids.size() != block_hashes.size()) {
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        "Mismatched host_block_ids count vs block_hashes count."));
    return reactor;
  }

  // =========================================================================
  // STEP 1: Request validation
  // Answerable from the request alone, so it runs before anything is matched
  // or pinned -- a refusal here then has nothing to give back.
  // =========================================================================
  RaidenId client_id{
      request->client_raiden_id().job_name(),
      request->client_raiden_id().job_replica_id(),
      request->client_raiden_id().data_name(),
      request->client_raiden_id().data_replica_idx(),
  };
  const auto& server_unit = controller_->unit();
  bool is_cross_node =
      !client_id.empty() &&
      (client_id.job_name != server_unit.job_name() ||
       client_id.job_replica_id != server_unit.job_replica_id() ||
       client_id.data_name != server_unit.data_name() ||
       client_id.data_replica_idx != server_unit.data_replica_idx());
  if (is_cross_node && request->client_worker_endpoints().empty()) {
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        "Cross-node FetchRequest requires non-empty client_worker_endpoints."));
    return reactor;
  }

  // =========================================================================
  // STEP 2: Match the blocks and pin them, in one step
  // Validate block_hashes exist in local index and reside in host memory.
  // =========================================================================
  //
  // pin_found, so that the pin is taken under the same lock that matched the
  // entry. Matching first and pinning afterwards leaves a window in which the
  // block can be evicted, and the host block ids read out of the match then
  // name blocks that have gone back to the pool and been refilled with
  // unrelated bytes.
  LookupOptions options;
  options.enable_global = false;
  options.pin_found = true;
  auto lookup_or = backend_->Lookup(block_hashes, options);
  if (!lookup_or.ok()) {
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::NOT_FOUND,
        absl::StrCat("Validation failed: ", lookup_or.status().message())));
    return reactor;
  }
  const auto& lookup_slices = lookup_or.value();

  // The lookup pinned everything it returned. Release that on every exit,
  // including the refusals below.
  std::vector<std::string> pinned;
  pinned.reserve(lookup_slices.size());
  for (const auto& entry : lookup_slices) {
    pinned.push_back(entry.first);
  }
  auto unpin_cleanup =
      absl::MakeCleanup([this, &pinned]() { backend_->Release(pinned); });

  if (lookup_slices.size() < block_hashes.size()) {
    WithdrawEntryIfUnbacked(block_hashes[lookup_slices.size()]);
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::NOT_FOUND,
        absl::StrCat("Partial block match: found ", lookup_slices.size(),
                     " out of ", block_hashes.size())));
    return reactor;
  }

  std::vector<int32_t> src_host_block_ids;
  src_host_block_ids.reserve(lookup_slices.size());
  for (const auto& [hash, slice] : lookup_slices) {
    if (slice.status != BlockStatus::HOST &&
        slice.status != BlockStatus::HOST_AND_HBM) {
      reactor->Finish(::grpc::Status(
          ::grpc::StatusCode::FAILED_PRECONDITION,
          absl::StrCat("Block hash '", hash, "' is not resident in host DRAM")));
      return reactor;
    }
    src_host_block_ids.push_back(slice.host_block_id);
  }

  // =========================================================================
  // STEP 3: Transfer Execution
  // Transfer data from local source host DRAM directly to destination host DRAM
  // using RaidenController::TransferBuffers with Buffer structs.
  // =========================================================================
  std::vector<RaidenWorkerEndpoints> client_groups =
      UnpackWorkerEndpointsProto(request->client_worker_endpoints());

  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_host_block_ids.size());
  for (int id : src_host_block_ids) {
    src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  }

  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(dst_host_block_ids.size());
  for (int id : dst_host_block_ids) {
    Buffer dst_buf(id, std::vector<BufferShard>{}, std::nullopt,
                   ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
    if (!client_groups.empty()) {
      dst_buf.set_remote_worker_endpoints(client_groups);
    }
    dst_buffers.push_back(std::move(dst_buf));
  }

  tsl::Future<> transfer_future =
      controller_->TransferBuffers(src_buffers, dst_buffers);

  absl::Status transfer_status = transfer_future.Await();

  if (!transfer_status.ok()) {
    reactor->Finish(::grpc::Status(::grpc::StatusCode::INTERNAL,
                          absl::StrCat("Fetch TransferBuffers failed: ",
                                       transfer_status.message())));
    return reactor;
  }

  for (const auto& hash : block_hashes) {
    response->add_done_block_hashes(hash);
  }

  reactor->Finish(::grpc::Status::OK);
  return reactor;
}

::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::State
KVCacheStoreServiceImpl::ToWireState(OpState state) {
  switch (state) {
    case OpState::kPending:
    // COMPLETING has no wire representation: to the source it is still pending.
    case OpState::kCompleting:
      return ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::PENDING;
    case OpState::kCommitted:
      return ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::COMMITTED;
    case OpState::kAllExist:
      return ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::ALL_EXIST;
    case OpState::kPartialExist:
      return ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::
          PARTIAL_EXIST;
    case OpState::kFailed:
      return ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::FAILED;
    case OpState::kStoredUnregistered:
      return ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::
          STORED_UNREGISTERED;
  }
}

std::shared_ptr<KVCacheStoreServiceImpl::WriteOp>
KVCacheStoreServiceImpl::FindOp(uint64_t op_id) {
  auto it = write_ops_.find(op_id);
  return it != write_ops_.end() ? it->second : nullptr;
}

void KVCacheStoreServiceImpl::ReleaseLandingBlocks(
    const std::shared_ptr<WriteOp>& op) {
  std::vector<int32_t> to_free;
  {
    absl::MutexLock lock(write_mutex_);
    if (op->blocks_released || op->landing_block_ids.empty()) {
      return;
    }
    op->blocks_released = true;
    op->next_leak_warning = absl::InfiniteFuture();
    to_free = op->landing_block_ids;
    deadline_cv_.Signal();
  }
  // Deallocate outside write_mutex_: controller_ has its own lock.
  (void)controller_->DeallocateBlockIds(
      std::vector<int>(to_free.begin(), to_free.end()));
}

void KVCacheStoreServiceImpl::Settle(const std::shared_ptr<WriteOp>& op) {
  absl::MutexLock lock(write_mutex_);
  op->settle_done = true;
}

::grpc::ServerWriteReactor<::tpu_raiden::kv_cache::proto::WriteRemoteEvent>*
KVCacheStoreServiceImpl::WriteRemote(
    ::grpc::CallbackServerContext* context,
    const ::tpu_raiden::kv_cache::proto::WriteRemoteRequest* request) {
  if (backend_ == nullptr || controller_ == nullptr) {
    auto* reactor = new WriteRemoteServerReactor(/*gate=*/nullptr);
    reactor->Finish(::grpc::Status(::grpc::StatusCode::FAILED_PRECONDITION,
                                  "Backend or RaidenController non-initialized"));
    return reactor;
  }

  const std::vector<std::string> block_hashes(request->block_hashes().begin(),
                                              request->block_hashes().end());
  const std::vector<int32_t> src_host_block_ids(
      request->src_host_block_ids().begin(),
      request->src_host_block_ids().end());

  if (block_hashes.empty()) {
    auto* reactor = new WriteRemoteServerReactor(/*gate=*/nullptr);
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        "WriteRemoteRequest requires non-empty block_hashes."));
    return reactor;
  }
  if (src_host_block_ids.size() != block_hashes.size()) {
    auto* reactor = new WriteRemoteServerReactor(/*gate=*/nullptr);
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        "Mismatched src_host_block_ids count vs block_hashes count."));
    return reactor;
  }
  if (request->deadline_ms() <= 0) {
    auto* reactor = new WriteRemoteServerReactor(/*gate=*/nullptr);
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        absl::StrCat("WriteRemote requires a positive deadline_ms; got ",
                     request->deadline_ms(),
                     " (an unset field is indistinguishable from 0).")));
    return reactor;
  }

  const RaidenId src_id{
      request->src_raiden_id().job_name(),
      request->src_raiden_id().job_replica_id(),
      request->src_raiden_id().data_name(),
      request->src_raiden_id().data_replica_idx(),
  };
  const auto& unit = controller_->unit();
  if (!src_id.empty() && src_id.job_name == unit.job_name() &&
      src_id.job_replica_id == unit.job_replica_id() &&
      src_id.data_name == unit.data_name() &&
      src_id.data_replica_idx == unit.data_replica_idx()) {
    auto* reactor = new WriteRemoteServerReactor(/*gate=*/nullptr);
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        "WriteRemote destination is this store; there is nothing to transfer."));
    return reactor;
  }

  // Quick check: if we ALREADY hold every requested hash in host DRAM, answer
  // immediately without allocating landing blocks or pulling bytes.
  std::vector<std::string> present =
      backend_->AlreadyPresentHostResident(block_hashes);
  if (present.size() == block_hashes.size()) {
    auto* reactor = new WriteRemoteServerReactor(/*gate=*/nullptr);
    proto::WriteRemoteEvent ack_event;
    auto* ack = ack_event.mutable_ack();
    ack->set_exist_state(::tpu_raiden::kv_cache::proto::WRITE_ALL_EXIST);
    for (const auto& hash : present) {
      ack->add_existing_hashes(hash);
    }
    reactor->StartAckAndFinish(std::move(ack_event));
    return reactor;
  }
  if (!present.empty()) {
    auto* reactor = new WriteRemoteServerReactor(/*gate=*/nullptr);
    proto::WriteRemoteEvent ack_event;
    auto* ack = ack_event.mutable_ack();
    ack->set_exist_state(
        ::tpu_raiden::kv_cache::proto::WRITE_PARTIAL_EXIST);
    for (const auto& hash : present) {
      ack->add_existing_hashes(hash);
    }
    reactor->StartAckAndFinish(std::move(ack_event));
    return reactor;
  }

  // Only now do the source's worker endpoints matter.
  if (request->src_worker_endpoints().empty()) {
    auto* reactor = new WriteRemoteServerReactor(/*gate=*/nullptr);
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        "WriteRemote requires the source's worker endpoints: the destination "
        "pulls, so it needs somewhere to pull from."));
    return reactor;
  }

  // Cap the requested deadline.
  const absl::Duration granted_deadline =
      std::min(absl::Milliseconds(request->deadline_ms()), DeadlineCap());

  // Allocate landing blocks in destination host DRAM for the transfer.
  auto allocated_ids_or = controller_->AllocateBlockIds(block_hashes.size());
  if (!allocated_ids_or.ok()) {
    auto* reactor = new WriteRemoteServerReactor(/*gate=*/nullptr);
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::RESOURCE_EXHAUSTED,
        absl::StrCat("Failed to allocate destination landing blocks: ",
                     allocated_ids_or.status().message())));
    return reactor;
  }
  std::vector<int32_t> landing_block_ids(allocated_ids_or->begin(),
                                         allocated_ids_or->end());

  const absl::Time now = absl::Now();
  const absl::Time deadline = now + granted_deadline;
  const absl::Time expires_at = deadline + kRecordMargin;

  auto op = std::make_shared<WriteOp>();
  op->state = OpState::kPending;
  op->block_hashes = block_hashes;
  op->landing_block_ids = landing_block_ids;
  op->accepted_at = now;
  op->deadline = deadline;
  op->expires_at = expires_at;
  op->granted_deadline = granted_deadline;
  op->next_leak_warning =
      now + (granted_deadline * kLeakWarningDeadlineMultiple);

  // Auto-generate operation_id (avoiding 0, which is reserved).
  uint64_t op_id = 0;
  {
    absl::MutexLock lock(write_mutex_);
    do {
      op_id = absl::Uniform<uint64_t>(op_id_rng_);
    } while (op_id == 0 || write_ops_.contains(op_id));
    write_ops_[op_id] = op;
    deadline_cv_.Signal();
  }

  auto gate = std::make_shared<WriteRemoteReactorGate>();
  auto* reactor = new WriteRemoteServerReactor(gate);
  {
    absl::MutexLock lock(gate->mu);
    gate->reactor = reactor;
  }
  {
    absl::MutexLock lock(write_mutex_);
    op->reactor_gate = gate;
  }

  // Issue the DMA pull from source to local landing blocks.
  std::vector<RaidenWorkerEndpoints> client_groups =
      UnpackWorkerEndpointsProto(request->src_worker_endpoints());

  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_host_block_ids.size());
  for (int id : src_host_block_ids) {
    Buffer src_buf(id, std::vector<BufferShard>{}, std::nullopt,
                   ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
    if (!client_groups.empty()) {
      src_buf.set_remote_worker_endpoints(client_groups);
    }
    src_buffers.push_back(std::move(src_buf));
  }

  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(landing_block_ids.size());
  for (int id : landing_block_ids) {
    dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  }

  // Production goes straight to the controller; transfer_fn_override_ is empty
  // unless a test installed one.
  tsl::Future<> transfer_future =
      transfer_fn_override_
          ? transfer_fn_override_(src_buffers, dst_buffers)
          : controller_->TransferBuffers(src_buffers, dst_buffers);

  std::shared_ptr<Lifetime> lifetime = lifetime_;
  transfer_future.OnReady([lifetime, op_id](absl::Status status) {
    OnTransferComplete(lifetime, op_id, status);
  });

  proto::WriteRemoteEvent ack_event;
  auto* ack = ack_event.mutable_ack();
  ack->set_exist_state(
      ::tpu_raiden::kv_cache::proto::WRITE_EXIST_STATE_UNSPECIFIED);
  ack->set_operation_id(op_id);
  ack->set_granted_deadline_ms(
      absl::ToInt64Milliseconds(granted_deadline));
  // How long the record survives from this ack, so a source that lost the
  // stream can tell "asked too late" from "never happened".
  ack->set_record_ttl_ms(
      absl::ToInt64Milliseconds(granted_deadline + kRecordMargin));

  // Through the gate, not the raw pointer: the destructor's sweep can finish
  // (and, via OnDone, delete) this reactor between the operation being
  // recorded above and this line.
  {
    absl::MutexLock lock(gate->mu);
    if (gate->reactor != nullptr) {
      gate->reactor->StartAck(std::move(ack_event));
    }
  }
  return reactor;
}

void KVCacheStoreServiceImpl::OnTransferComplete(
    std::shared_ptr<Lifetime> lifetime, uint64_t op_id,
    absl::Status transfer_status) {
  std::optional<PendingPublish> pending;
  {
    absl::MutexLock lock(lifetime->mu);
    if (lifetime->svc != nullptr) {
      pending = lifetime->svc->CompleteWriteRemote(op_id, transfer_status);
    }
  }
  if (!pending.has_value()) {
    return;
  }
  // Attach with `lifetime->mu` released: OnReady on an already-resolved
  // future (the no-registry case) runs inline, and would deadlock if this
  // thread still held the mutex. Taking `mu` inside the callback keeps the
  // service alive through FinishPublish.
  std::move(pending->future)
      .OnReady([lifetime, op = std::move(pending->op),
                op_id](absl::Status registered) {
        absl::MutexLock lock(lifetime->mu);
        if (lifetime->svc != nullptr) {
          lifetime->svc->FinishPublish(op, op_id, registered);
        }
      });
}

std::optional<KVCacheStoreServiceImpl::PendingPublish>
KVCacheStoreServiceImpl::CompleteWriteRemote(uint64_t op_id,
                                             absl::Status transfer_status) {
  std::shared_ptr<WriteOp> op;
  bool claimed = false;
  {
    absl::MutexLock lock(write_mutex_);
    op = FindOp(op_id);
    if (op == nullptr) {
      return std::nullopt;  // Aged out.
    }
    const absl::Time now = absl::Now();
    if (transfer_status.ok() && op->state == OpState::kPending &&
        now < op->deadline) {
      // Claimed! No deadline thread can mark this op failed now, and no other
      // completion can touch it.
      op->state = OpState::kCompleting;
      claimed = true;
    } else if (op->state == OpState::kPending) {
      // Transfer was OK but past the deadline, or transfer failed before the
      // deadline thread noticed. Set the verdict here rather than leaving
      // the op PENDING forever.
      MarkTerminal(op, OpState::kFailed, {});
      deadline_cv_.Signal();
    }
  }

  if (!claimed) {
    // Deferred free: transfer failed, or succeeded too late. Release landing
    // blocks and settle.
    std::shared_ptr<WriteRemoteReactorGate> gate;
    proto::WriteRemoteEvent result_event;
    {
      absl::MutexLock lock(write_mutex_);
      gate = std::move(op->reactor_gate);
      if (gate != nullptr) {
        result_event = MakeResultEvent(op);
      }
    }
    SendResultThroughGate(gate, std::move(result_event));
    ReleaseLandingBlocks(op);
    Settle(op);
    return std::nullopt;
  }

  // Claimed path: write_mutex_ NOT held across backend and registry calls.

  // Re-check existence.
  std::vector<std::string> present =
      backend_->AlreadyPresentHostResident(op->block_hashes);
  if (present.size() == op->block_hashes.size()) {
    Finish(op, OpState::kAllExist, {});
    return std::nullopt;
  }
  if (!present.empty()) {
    Finish(op, OpState::kPartialExist, std::move(present));
    return std::nullopt;
  }

  std::vector<RaidenBlockId> slices;
  slices.reserve(op->landing_block_ids.size());
  const auto& unit = controller_->unit();
  const RaidenId local_id{unit.job_name(), unit.job_replica_id(),
                          unit.data_name(), unit.data_replica_idx()};
  for (int32_t block_id : op->landing_block_ids) {
    slices.push_back(RaidenBlockId(local_id, block_id, BlockStatus::HOST));
  }
  if (!backend_->InsertAllOrNothing(op->block_hashes, slices)) {
    Finish(op, OpState::kFailed, {});
    return std::nullopt;
  }

  // The blocks are in the cache; only the publish is left. The caller
  // attaches the continuation -- see the declaration for why not here.
  return PendingPublish{
      backend_->RegisterBlocksAsync(op->block_hashes, op->landing_block_ids),
      op};
}

void KVCacheStoreServiceImpl::FinishPublish(const std::shared_ptr<WriteOp>& op,
                                            uint64_t op_id,
                                            absl::Status registered) {
  if (!registered.ok()) {
    LOG(WARNING) << "Remote write " << op_id << " landed "
                 << op->block_hashes.size()
                 << " block(s) but could not publish them to the global "
                    "registry: "
                 << registered.message()
                 << ". Keeping them; no peer will find them until something "
                    "re-registers them.";
    Finish(op, OpState::kStoredUnregistered, {});
    return;
  }
  Finish(op, OpState::kCommitted, {});
}

void KVCacheStoreServiceImpl::MarkTerminal(const std::shared_ptr<WriteOp>& op,
                                           OpState state,
                                           std::vector<std::string> existing) {
  op->state = state;
  op->existing_hashes = std::move(existing);
  if (state == OpState::kStoredUnregistered) {
    op->unregistered_hashes = op->block_hashes;
  }
  const bool kept_by_cache =
      state == OpState::kCommitted || state == OpState::kStoredUnregistered;
  if (kept_by_cache) {
    op->blocks_released = true;
    op->next_leak_warning = absl::InfiniteFuture();
  }
}

void KVCacheStoreServiceImpl::Finish(const std::shared_ptr<WriteOp>& op,
                                     OpState state,
                                     std::vector<std::string> existing) {
  const bool kept_by_cache =
      state == OpState::kCommitted || state == OpState::kStoredUnregistered;
  std::shared_ptr<WriteRemoteReactorGate> gate;
  proto::WriteRemoteEvent result_event;
  {
    absl::MutexLock lock(write_mutex_);
    MarkTerminal(op, state, std::move(existing));
    deadline_cv_.Signal();
    gate = std::move(op->reactor_gate);
    if (gate != nullptr) {
      result_event = MakeResultEvent(op);
    }
  }
  SendResultThroughGate(gate, std::move(result_event));
  if (!kept_by_cache) {
    ReleaseLandingBlocks(op);
  }
  Settle(op);
}

::grpc::ServerUnaryReactor* KVCacheStoreServiceImpl::PollWriteRemote(
    ::grpc::CallbackServerContext* context,
    const ::tpu_raiden::kv_cache::proto::PollWriteRemoteRequest* request,
    ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse* response) {
  auto* reactor = context->DefaultReactor();
  if (request->operation_id() == 0) {
    reactor->Finish(::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        "operation_id 0 is reserved and never identifies an operation."));
    return reactor;
  }

  absl::MutexLock lock(write_mutex_);

  // Opportunistic collection of expired records; DeadlineLoop is what
  // guarantees it.
  const absl::Time now = absl::Now();
  absl::erase_if(write_ops_, [now](const auto& entry) {
    const auto& op = entry.second;
    return op->state != OpState::kCompleting &&
           op->state != OpState::kPending && op->blocks_released &&
           now >= op->expires_at;
  });

  auto it = write_ops_.find(request->operation_id());
  if (it != write_ops_.end() && request->wait_ms() > 0 &&
      (it->second->state == OpState::kPending ||
       it->second->state == OpState::kCompleting)) {
    struct WaitArg {
      KVCacheStoreServiceImpl* svc;
      uint64_t id;
    } arg{this, request->operation_id()};
    write_mutex_.AwaitWithTimeout(
        absl::Condition(
            +[](WaitArg* a) {
              a->svc->write_mutex_.AssertHeld();
              auto found = a->svc->write_ops_.find(a->id);
              if (found == a->svc->write_ops_.end()) return true;
              return found->second->state != OpState::kPending &&
                     found->second->state != OpState::kCompleting;
            },
            &arg),
        std::min(absl::Milliseconds(request->wait_ms()), MaxPollWait()));
    it = write_ops_.find(request->operation_id());
  }

  if (it == write_ops_.end()) {
    response->set_state(
        ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::UNKNOWN);
    reactor->Finish(::grpc::Status::OK);
    return reactor;
  }

  const auto& op = it->second;
  response->set_state(ToWireState(op->state));
  if (op->state == OpState::kCommitted) {
    for (const auto& hash : op->block_hashes) {
      response->add_committed_hashes(hash);
    }
  } else if (op->state == OpState::kFailed) {
    for (const auto& hash : op->block_hashes) {
      response->add_failed_hashes(hash);
    }
  }
  for (const auto& hash : op->unregistered_hashes) {
    response->add_unregistered_hashes(hash);
  }
  for (const auto& hash : op->existing_hashes) {
    response->add_existing_hashes(hash);
  }
  reactor->Finish(::grpc::Status::OK);
  return reactor;
}

void KVCacheStoreServiceImpl::DeadlineLoop() {
  while (true) {
    std::vector<std::pair<std::shared_ptr<WriteRemoteReactorGate>,
                          proto::WriteRemoteEvent>>
        reactors_to_notify;
    absl::Time wake_at = absl::InfiniteFuture();
    {
      absl::MutexLock lock(write_mutex_);
      if (stop_deadline_thread_) {
        break;
      }
      const absl::Time now = absl::Now();

      for (auto& [id, op] : write_ops_) {
        if (op->state == OpState::kPending &&
            !deadline_firing_paused_for_testing_) {
          if (now >= op->deadline) {
            MarkTerminal(op, OpState::kFailed, {});
            if (op->reactor_gate != nullptr) {
              reactors_to_notify.push_back(
                  {std::move(op->reactor_gate), MakeResultEvent(op)});
              op->reactor_gate = nullptr;
            }
          } else {
            wake_at = std::min(wake_at, op->deadline);
          }
        }
        if (!op->blocks_released && !op->landing_block_ids.empty()) {
          if (now >= op->next_leak_warning) {
            LOG(ERROR) << "Remote write " << id << " has held "
                       << op->landing_block_ids.size()
                       << " landing block(s) for " << (now - op->accepted_at)
                       << ", past " << kLeakWarningDeadlineMultiple
                       << "x its granted deadline of " << op->granted_deadline
                       << ". The transfer has not resolved, and nothing bounds "
                          "it: these blocks may not come back until this "
                          "process restarts.";
            op->next_leak_warning = now + kLeakWarningInterval;
          }
          wake_at = std::min(wake_at, op->next_leak_warning);
        }
        if (op->state != OpState::kCompleting &&
            op->state != OpState::kPending && op->blocks_released) {
          if (now < op->expires_at) {
            wake_at = std::min(wake_at, op->expires_at);
          }
        }
      }

      if (reactors_to_notify.empty()) {
        absl::erase_if(write_ops_, [now](const auto& entry) {
          const auto& op = entry.second;
          return op->state != OpState::kCompleting &&
                 op->state != OpState::kPending && op->blocks_released &&
                 now >= op->expires_at;
        });
        if (wake_at == absl::InfiniteFuture()) {
          deadline_cv_.Wait(&write_mutex_);
        } else {
          deadline_cv_.WaitWithDeadline(&write_mutex_, wake_at);
        }
      }
    }

    for (auto& [gate, event] : reactors_to_notify) {
      SendResultThroughGate(gate, std::move(event));
    }
  }
}

}  // namespace kv_cache
}  // namespace tpu_raiden
