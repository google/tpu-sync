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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_GRPC_CONTROL_PLANE_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_GRPC_CONTROL_PLANE_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.grpc.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.pb.h"

namespace tpu_raiden {

// Extracts the client IP address from a gRPC ServerContext::peer() URI string
// (e.g., "ipv4:10.0.0.1:54321", "ipv6:[::1]:54321", "ipv6:%5B::1%5D:54321",
// or "ipv6:[::ffff:10.0.0.1]:54321").
std::string ExtractIpFromGrpcPeer(absl::string_view peer);

// Callback service: a PullStream waiting on the handler holds a reactor, not a
// server thread, so requests the producer cannot answer yet do not starve
// the ones it can.
class KVCacheControlPlaneServiceImpl final
    : public control_plane::proto::KVCacheControlPlaneService::CallbackService {
 public:
  explicit KVCacheControlPlaneServiceImpl(ControlPlaneHandler* handler)
      : handler_(handler) {}

  grpc::ServerUnaryReactor* PullStream(
      grpc::CallbackServerContext* context,
      const control_plane::proto::PullStreamRequest* request,
      control_plane::proto::PullStreamResponse* response) override;

  grpc::ServerUnaryReactor* Ack(
      grpc::CallbackServerContext* context,
      const control_plane::proto::AckRequest* request,
      control_plane::proto::AckResponse* response) override;

 private:
  ControlPlaneHandler* handler_ = nullptr;
};

class GrpcControlPlaneBackend : public ControlPlaneBackend {
 public:
  static constexpr size_t kDefaultMaxCachedStubs = 100000;

  explicit GrpcControlPlaneBackend(
      size_t max_cached_stubs = kDefaultMaxCachedStubs);
  ~GrpcControlPlaneBackend() override;

  absl::StatusOr<int> StartServer(int requested_port,
                                  ControlPlaneHandler* handler) override;
  void StopServer() override;

  absl::StatusOr<PullStreamResponseSpec> SendPullRequest(
      absl::string_view remote_endpoint, const PullStreamRequestSpec& req,
      absl::Duration timeout) override;

  absl::Status SendAck(absl::string_view remote_endpoint, uint64_t uuid,
                       absl::Duration timeout) override;

  absl::string_view Name() const override { return "grpc"; }

  // Returns the current number of cached gRPC stubs.
  size_t TEST_CachedStubCount() const;

  // Returns true if a stub for |endpoint| is currently in the LRU cache.
  bool TEST_HasCachedStub(absl::string_view endpoint) const;

 private:
  struct StubCacheEntry {
    std::shared_ptr<control_plane::proto::KVCacheControlPlaneService::Stub>
        stub;
    std::list<std::string>::iterator lru_it;
  };

  std::shared_ptr<control_plane::proto::KVCacheControlPlaneService::Stub>
  GetOrCreateStub(absl::string_view endpoint);

  std::unique_ptr<KVCacheControlPlaneServiceImpl> service_impl_;
  std::unique_ptr<grpc::Server> server_;

  size_t max_cached_stubs_ = kDefaultMaxCachedStubs;
  mutable absl::Mutex stub_mu_;
  std::list<std::string> lru_order_ ABSL_GUARDED_BY(stub_mu_);
  absl::flat_hash_map<std::string, StubCacheEntry> stubs_
      ABSL_GUARDED_BY(stub_mu_);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_GRPC_CONTROL_PLANE_BACKEND_H_
