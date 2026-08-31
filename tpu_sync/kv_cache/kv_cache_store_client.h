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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_CLIENT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/proto/kv_cache_store_service.grpc.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

// Cancels an open WriteRemote call from outside it. ~KVCacheStore uses this
// to end abandoned offers, so the destination is not left holding a call
// nobody will answer. Holds the call's context weakly: the handle does not
// keep the call alive, and TryCancel does nothing after the call has ended.
class WriteRemoteCancel {
 public:
  void TryCancel();

 private:
  friend class KVCacheStoreClient;

  // Shared with the reactor, which owns itself. The call starts before the
  // handle is returned, so `context` is set before TryCancel can run.
  struct State {
    absl::Mutex mutex;
    std::weak_ptr<::grpc::ClientContext> context ABSL_GUARDED_BY(mutex);
  };

  std::shared_ptr<State> state_ = std::make_shared<State>();
};

class KVCacheStoreClient {
 public:
  explicit KVCacheStoreClient(
      std::shared_ptr<::grpc::ChannelInterface> channel);
  explicit KVCacheStoreClient(
      std::unique_ptr<
          ::tpu_raiden::kv_cache::proto::KVCacheStoreService::StubInterface>
          stub);

  // Asynchronous Non-Blocking Fetch RPC.
  // Returns a Future that resolves with FetchResponse upon RPC completion.
  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> Fetch(
      absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> device_block_ids = {},
      absl::Span<const int32_t> host_block_ids = {},
      const ::tpu_sync::rpc::RaidenIdProto& client_raiden_id = {},
      absl::Span<const ::tpu_sync::proto::RaidenWorkerEndpointsProto>
          client_worker_endpoints = {});

  // Reports how the offer's call ended, once an ack has arrived: `result` is
  // the streamed verdict if the peer sent one; otherwise `rpc_status` says
  // how the call ended.
  using WriteRemoteVerdictCallback = std::function<void(
      absl::Status rpc_status,
      std::optional<::tpu_raiden::kv_cache::proto::WriteRemoteResult> result,
      uint64_t operation_id)>;
  // Offers `block_hashes` to the connected peer, on one streaming call whose
  // deadline is `hold_window`. `ack` resolves when the peer has decided; it
  // does not wait for the bytes. `on_verdict` runs on the CompletionExecutor
  // when the call ends. If the call fails before any ack, the error goes to
  // `ack` and `on_verdict` never runs. `deadline_ms` must be > 0.
  struct WriteRemoteCall {
    // The peer's decision, or the error that ended the call before it.
    tsl::Future<::tpu_raiden::kv_cache::proto::WriteRemoteAck> ack;
    // Cancels the open call. Null only if this client refused to make it.
    std::shared_ptr<WriteRemoteCancel> cancel;
  };
  WriteRemoteCall WriteRemote(
      const ::tpu_sync::rpc::RaidenIdProto& src_raiden_id,
      absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> src_host_block_ids,
      absl::Span<const ::tpu_sync::proto::RaidenWorkerEndpointsProto>
          src_worker_endpoints,
      int64_t deadline_ms,
      absl::Duration hold_window,
      WriteRemoteVerdictCallback on_verdict = nullptr);

  // Asks the peer what became of an accepted operation. `wait_ms > 0` asks
  // it to hold the answer until the operation is terminal. UNKNOWN means the
  // peer's record is gone; callers must treat that as failure.
  tsl::Future<::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse>
  PollWriteRemote(uint64_t operation_id, int64_t wait_ms = 0);

 private:
  std::unique_ptr<
      ::tpu_raiden::kv_cache::proto::KVCacheStoreService::StubInterface>
      stub_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_CLIENT_H_
