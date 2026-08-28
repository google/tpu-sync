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

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/proto/kv_cache_store_service.grpc.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

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

  using WriteRemoteVerdictCallback = std::function<void(
      absl::Status rpc_status,
      std::optional<::tpu_raiden::kv_cache::proto::WriteRemoteResult> result)>;

  // Asynchronous non-blocking WriteRemote RPC: offer `block_hashes` to the
  // peer this client is connected to -- the source side of KVCacheStore's
  // save to a destination. The peer decides, allocates landing
  // blocks and starts pulling; it does NOT wait for the bytes, so the
  // returned future resolves as soon as the offer ack is received.
  // When the operation reaches a terminal state, on_verdict is invoked.
  //
  // `deadline_ms` must be > 0 -- see WriteRemoteRequest.deadline_ms.
  tsl::Future<::tpu_raiden::kv_cache::proto::WriteRemoteAck> WriteRemote(
      const ::tpu_sync::rpc::RaidenIdProto& src_raiden_id,
      absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> src_host_block_ids,
      absl::Span<const ::tpu_sync::proto::RaidenWorkerEndpointsProto>
          src_worker_endpoints,
      int64_t deadline_ms,
      absl::Duration hold_window,
      WriteRemoteVerdictCallback on_verdict = nullptr);

  // Asynchronous non-blocking PollWriteRemote RPC: ask the peer what became
  // of an operation it accepted. Returns UNKNOWN once the peer's record has
  // aged out, which is indistinguishable from "never happened".
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
