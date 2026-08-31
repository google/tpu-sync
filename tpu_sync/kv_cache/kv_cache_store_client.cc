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

#include "tpu_sync/kv_cache/completion_executor.h"
#include "tpu_sync/kv_cache/kv_cache_store_client.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/client_context.h"
#include "grpcpp/impl/status.h"
#include "grpcpp/support/client_callback.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/proto/kv_cache_store_service.grpc.pb.h"
#include "tpu_sync/proto/kv_cache_store_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

// RPC deadline for WriteRemote and PollWriteRemote. Both handlers answer
// from local state without waiting on the transfer, so this only needs to
// cover bookkeeping plus network latency. Without it, a peer that accepts
// connections but never answers would hang the caller -- and the evict
// sweep shares its thread with the store's heartbeats.
constexpr std::chrono::seconds kRpcDeadline{10};

}  // namespace

KVCacheStoreClient::KVCacheStoreClient(
    std::shared_ptr<::grpc::ChannelInterface> channel)
    : stub_(::tpu_raiden::kv_cache::proto::KVCacheStoreService::NewStub(
          channel)) {}

KVCacheStoreClient::KVCacheStoreClient(
    std::unique_ptr<
        ::tpu_raiden::kv_cache::proto::KVCacheStoreService::StubInterface>
        stub)
    : stub_(std::move(stub)) {}

tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse>
KVCacheStoreClient::Fetch(
    absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> device_block_ids,
    absl::Span<const int32_t> host_block_ids,
    const ::tpu_sync::rpc::RaidenIdProto& client_raiden_id,
    absl::Span<const ::tpu_sync::proto::RaidenWorkerEndpointsProto>
        client_worker_endpoints) {
  if (block_hashes.empty()) {
    return tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse>(
        ::tpu_raiden::kv_cache::proto::FetchResponse{});
  }

  if (!device_block_ids.empty() &&
      device_block_ids.size() != block_hashes.size()) {
    return tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse>(
        absl::InvalidArgumentError(absl::StrCat(
            "Mismatched device_block_ids count (", device_block_ids.size(),
            ") vs block_hashes count (", block_hashes.size(), ").")));
  }

  if (!host_block_ids.empty() && host_block_ids.size() != block_hashes.size()) {
    return tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse>(
        absl::InvalidArgumentError(absl::StrCat(
            "Mismatched host_block_ids count (", host_block_ids.size(),
            ") vs block_hashes count (", block_hashes.size(), ").")));
  }

  ::tpu_raiden::kv_cache::proto::FetchRequest request;
  for (const auto& hash : block_hashes) {
    request.add_block_hashes(hash);
  }
  for (int32_t dev_id : device_block_ids) {
    request.add_device_block_ids(dev_id);
  }
  for (int32_t host_id : host_block_ids) {
    request.add_host_block_ids(host_id);
  }
  *request.mutable_client_raiden_id() = client_raiden_id;
  request.mutable_client_worker_endpoints()->Reserve(
      client_worker_endpoints.size());
  for (const auto& group : client_worker_endpoints) {
    *request.add_client_worker_endpoints() = group;
  }

  auto [promise, future] =
      tsl::MakePromise<::tpu_raiden::kv_cache::proto::FetchResponse>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response =
      std::make_shared<::tpu_raiden::kv_cache::proto::FetchResponse>();

  stub_->async()->Fetch(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) mutable {
        if (!status.ok()) {
          promise->Set(absl::Status(
              static_cast<absl::StatusCode>(status.error_code()),
              absl::StrCat("Fetch RPC failed: ", status.error_message())));
        } else {
          promise->Set(std::move(*response));
        }
      });
  return future;
}

class WriteRemoteClientReactor
    : public ::grpc::ClientReadReactor<
          ::tpu_raiden::kv_cache::proto::WriteRemoteEvent> {
 public:
  WriteRemoteClientReactor(
      ::tpu_raiden::kv_cache::proto::KVCacheStoreService::StubInterface* stub,
      ::tpu_raiden::kv_cache::proto::WriteRemoteRequest request,
      std::shared_ptr<::grpc::ClientContext> context,
      tsl::Promise<::tpu_raiden::kv_cache::proto::WriteRemoteAck> ack_promise,
      KVCacheStoreClient::WriteRemoteVerdictCallback on_verdict)
      : context_(std::move(context)),
        request_(std::move(request)),
        ack_promise_(std::move(ack_promise)),
        on_verdict_(std::move(on_verdict)) {
    stub->async()->WriteRemote(context_.get(), &request_, this);
    StartRead(&event_);
    StartCall();
  }

  void OnReadDone(bool ok) override {
    if (!ok) {
      return;
    }

    if (event_.has_ack()) {
      ack_received_ = true;
      operation_id_ = event_.ack().operation_id();
      ack_promise_.Set(event_.ack());
      if (event_.ack().exist_state() ==
          ::tpu_raiden::kv_cache::proto::WRITE_EXIST_STATE_UNSPECIFIED) {
        StartRead(&event_);
      } else {
        // Existence answers (ALL_EXIST / PARTIAL_EXIST) settle synchronously
        // inside SaveRemote via the ack promise. Clearing on_verdict_ prevents
        // OnDone from scheduling an unneeded verdict that races with SaveRemote.
        on_verdict_ = nullptr;
      }
    } else if (event_.has_result()) {
      has_result_ = true;
      result_ = event_.result();
      StartRead(&event_);
    }
  }

  void OnDone(const ::grpc::Status& status) override {
    bool initial_ack_failed = !ack_received_;
    if (!ack_received_) {
      ack_promise_.Set(absl::Status(
          static_cast<absl::StatusCode>(status.error_code()),
          status.error_message()));
      ack_received_ = true;
    }

    absl::Status rpc_status =
        status.ok()
            ? absl::OkStatus()
            : absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                           status.error_message());

    std::optional<proto::WriteRemoteResult> result;
    if (has_result_) {
      result = std::move(result_);
    }

    if (on_verdict_ && !initial_ack_failed) {
      CompletionExecutor::Schedule(
          [on_verdict = std::move(on_verdict_), rpc_status,
           result = std::move(result), op_id = operation_id_]() mutable {
            on_verdict(rpc_status, std::move(result), op_id);
          });
    }

    delete this;
  }

 private:
  // Shared, not owned outright, so a WriteRemoteCancel handle can reach it
  // without keeping the call's state alive past OnDone.
  std::shared_ptr<::grpc::ClientContext> context_;
  // Owned here: gRPC's callback API requires the request to stay valid for
  // the life of the call, and the reactor is the only thing that lives that
  // long.
  proto::WriteRemoteRequest request_;
  proto::WriteRemoteEvent event_;
  tsl::Promise<proto::WriteRemoteAck> ack_promise_;
  KVCacheStoreClient::WriteRemoteVerdictCallback on_verdict_;
  uint64_t operation_id_ = 0;
  bool ack_received_ = false;
  bool has_result_ = false;
  proto::WriteRemoteResult result_;
};

KVCacheStoreClient::WriteRemoteCall KVCacheStoreClient::WriteRemote(
    const ::tpu_sync::rpc::RaidenIdProto& src_raiden_id,
    absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> src_host_block_ids,
    absl::Span<const ::tpu_sync::proto::RaidenWorkerEndpointsProto>
        src_worker_endpoints,
    int64_t deadline_ms,
    absl::Duration hold_window,
    WriteRemoteVerdictCallback on_verdict) {
  if (block_hashes.empty()) {
    return WriteRemoteCall{
        tsl::Future<::tpu_raiden::kv_cache::proto::WriteRemoteAck>(
            absl::InvalidArgumentError(
                "WriteRemote requires at least one hash.")),
        nullptr};
  }
  if (src_host_block_ids.size() != block_hashes.size()) {
    return WriteRemoteCall{
        tsl::Future<::tpu_raiden::kv_cache::proto::WriteRemoteAck>(
            absl::InvalidArgumentError(absl::StrCat(
                "Mismatched src_host_block_ids count (",
                src_host_block_ids.size(), ") vs block_hashes count (",
                block_hashes.size(), ")."))),
        nullptr};
  }
  if (deadline_ms <= 0) {
    return WriteRemoteCall{
        tsl::Future<::tpu_raiden::kv_cache::proto::WriteRemoteAck>(
            absl::InvalidArgumentError(absl::StrCat(
                "WriteRemote requires a positive deadline_ms, got ",
                deadline_ms, "."))),
        nullptr};
  }

  ::tpu_raiden::kv_cache::proto::WriteRemoteRequest request;
  *request.mutable_src_raiden_id() = src_raiden_id;
  for (const auto& hash : block_hashes) {
    request.add_block_hashes(hash);
  }
  for (int32_t host_id : src_host_block_ids) {
    request.add_src_host_block_ids(host_id);
  }
  request.mutable_src_worker_endpoints()->Reserve(src_worker_endpoints.size());
  for (const auto& group : src_worker_endpoints) {
    *request.add_src_worker_endpoints() = group;
  }
  request.set_deadline_ms(deadline_ms);

  auto [promise, future] =
      tsl::MakePromise<::tpu_raiden::kv_cache::proto::WriteRemoteAck>();

  // THE call deadline is the HOLD window. Nothing on the source times a
  // remote write; this is what ends one that never gets an answer.
  auto context = std::make_shared<::grpc::ClientContext>();
  context->set_deadline(
      std::chrono::system_clock::now() +
      std::chrono::milliseconds(absl::ToInt64Milliseconds(hold_window)));

  // Made here rather than inside the reactor because the reactor owns itself
  // and there is no safe moment afterwards to reach into it. The handle holds
  // the context weakly; the reactor holds it strongly for the life of the
  // call.
  auto cancel = std::make_shared<WriteRemoteCancel>();
  {
    absl::MutexLock lock(&cancel->state_->mutex);
    cancel->state_->context = context;
  }

  // Owns itself until OnDone.
  new WriteRemoteClientReactor(stub_.get(), std::move(request),
                               std::move(context), std::move(promise),
                               std::move(on_verdict));
  return WriteRemoteCall{std::move(future), std::move(cancel)};
}

void WriteRemoteCancel::TryCancel() {
  std::shared_ptr<::grpc::ClientContext> context;
  {
    absl::MutexLock lock(&state_->mutex);
    context = state_->context.lock();
  }
  // Outside the lock: TryCancel does not block, but there is no reason to
  // hold anything while calling into grpc. A call that has already finished
  // leaves nothing to lock onto, and cancelling it is a no-op by omission.
  if (context != nullptr) {
    context->TryCancel();
  }
}

tsl::Future<::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse>
KVCacheStoreClient::PollWriteRemote(uint64_t operation_id, int64_t wait_ms) {
  if (operation_id == 0) {
    return tsl::Future<::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse>(
        absl::InvalidArgumentError(
            "operation_id 0 is reserved and never identifies an operation."));
  }

  ::tpu_raiden::kv_cache::proto::PollWriteRemoteRequest request;
  request.set_operation_id(operation_id);
  request.set_wait_ms(wait_ms);

  auto [promise, future] = tsl::MakePromise<
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse>();
  auto context = std::make_shared<grpc::ClientContext>();
  const auto deadline_duration =
      (wait_ms > 0) ? (std::chrono::milliseconds(wait_ms) + kRpcDeadline)
                    : kRpcDeadline;
  context->set_deadline(std::chrono::system_clock::now() + deadline_duration);
  auto response = std::make_shared<
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse>();

  stub_->async()->PollWriteRemote(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) mutable {
        if (!status.ok()) {
          promise->Set(
              absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                           absl::StrCat("PollWriteRemote RPC failed: ",
                                         status.error_message())));
        } else {
          promise->Set(std::move(*response));
        }
      });
  return future;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
