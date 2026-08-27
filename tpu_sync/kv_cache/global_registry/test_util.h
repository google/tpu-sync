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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_TEST_UTIL_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_TEST_UTIL_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.grpc.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_server.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

// Wraps the real service and can hold Register RPCs open, so a test can
// observe a call that has reached the server and not yet been answered.
// Composes rather than derives because GlobalRegistryServiceImpl is final.
// Only Register stalls; every other RPC, RegisterStore included, is delegated
// straight through.
class StallingRegistryService final : public GlobalRegistryService::Service {
 public:
  explicit StallingRegistryService(
      std::unique_ptr<GlobalRegistryServiceImpl> impl)
      : impl_(std::move(impl)) {}

  grpc::Status Register(grpc::ServerContext* context,
                        const RegisterRequest* request,
                        RegisterResponse* response) override {
    if (stall_register_) {
      if (!in_stall_.HasBeenNotified()) {
        in_stall_.Notify();
      }
      stall_release_.WaitForNotification();
    }
    return impl_->Register(context, request, response);
  }

  grpc::Status Lookup(grpc::ServerContext* context,
                      const LookupRequest* request,
                      LookupResponse* response) override {
    return impl_->Lookup(context, request, response);
  }

  grpc::Status Unregister(grpc::ServerContext* context,
                          const UnregisterRequest* request,
                          UnregisterResponse* response) override {
    return impl_->Unregister(context, request, response);
  }

  grpc::Status PullOwned(
      grpc::ServerContext* context, const PullOwnedRequest* request,
      grpc::ServerWriter<PullOwnedResponse>* writer) override {
    return impl_->PullOwned(context, request, writer);
  }

  grpc::Status RegisterStore(grpc::ServerContext* context,
                             const RegisterStoreRequest* request,
                             RegisterStoreResponse* response) override {
    return impl_->RegisterStore(context, request, response);
  }

  grpc::Status ResolveStore(grpc::ServerContext* context,
                            const ResolveStoreRequest* request,
                            ResolveStoreResponse* response) override {
    return impl_->ResolveStore(context, request, response);
  }

  grpc::Status Heartbeat(grpc::ServerContext* context,
                         const HeartbeatRequest* request,
                         HeartbeatResponse* response) override {
    return impl_->Heartbeat(context, request, response);
  }

  grpc::Status GetPlacementTargets(
      grpc::ServerContext* context, const GetPlacementTargetsRequest* request,
      GetPlacementTargetsResponse* response) override {
    return impl_->GetPlacementTargets(context, request, response);
  }

  grpc::Status UnregisterStore(grpc::ServerContext* context,
                               const UnregisterStoreRequest* request,
                               UnregisterStoreResponse* response) override {
    return impl_->UnregisterStore(context, request, response);
  }

  grpc::Status RegisterKVTransferSpec(
      grpc::ServerContext* context,
      const RegisterKVTransferSpecRequest* request,
      RegisterKVTransferSpecResponse* response) override {
    return impl_->RegisterKVTransferSpec(context, request, response);
  }

  grpc::Status GetKVTransferSpec(grpc::ServerContext* context,
                                 const GetKVTransferSpecRequest* request,
                                 GetKVTransferSpecResponse* response) override {
    return impl_->GetKVTransferSpec(context, request, response);
  }

  void EnableStall() { stall_register_ = true; }
  void WaitForStall() { in_stall_.WaitForNotification(); }
  void ReleaseStall() {
    if (!stall_release_.HasBeenNotified()) {
      stall_release_.Notify();
    }
  }

 private:
  std::unique_ptr<GlobalRegistryServiceImpl> impl_;
  // Set from the test thread, read on a server handler thread.
  std::atomic<bool> stall_register_ = false;
  absl::Notification in_stall_;
  absl::Notification stall_release_;
};

// Holds an in-process GlobalRegistry service, gRPC server, channel, and client.
//
// Member order matters: `client` is destroyed before `channel` before `server`.
// Note that destroying the client no longer necessarily releases the channel --
// GlobalRegistryClient holds its stub by shared_ptr so an in-flight async
// callback can keep the transport alive past the client. A test that tears this
// struct down while a call is in flight should expect the call to complete,
// not to be severed.
struct TestGlobalRegistryServer {
  std::unique_ptr<GlobalRegistryServiceImpl> service;
  std::unique_ptr<GlobalRegistryService::Service> custom_service;
  std::unique_ptr<grpc::Server> server;
  std::string server_address;
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<GlobalRegistryClient> client;

  ~TestGlobalRegistryServer() {
    if (server) {
      server->Shutdown();
    }
  }
};

// Creates and starts an in-process gRPC TestGlobalRegistryServer hosting
// GlobalRegistryServiceImpl on an ephemeral port ("localhost:0").
inline std::unique_ptr<TestGlobalRegistryServer> CreateTestGlobalRegistryServer(
    absl::Duration default_ttl = absl::Seconds(60),
    absl::Duration cleanup_interval = absl::Seconds(10),
    int64_t pull_owned_batch_size =
        GlobalRegistryServiceImpl::kDefaultPullOwnedBatchSize) {
  auto test_server = std::make_unique<TestGlobalRegistryServer>();
  test_server->service = std::make_unique<GlobalRegistryServiceImpl>(
      default_ttl, cleanup_interval, pull_owned_batch_size);

  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(test_server->service.get());
  test_server->server = builder.BuildAndStart();

  test_server->server_address = absl::StrCat("localhost:", selected_port);
  test_server->channel = grpc::CreateChannel(
      test_server->server_address, grpc::InsecureChannelCredentials());
  test_server->client =
      std::make_unique<GlobalRegistryClient>(test_server->channel);
  return test_server;
}

// Creates and starts an in-process gRPC TestGlobalRegistryServer hosting
// a custom GlobalRegistryService::Service implementation on an ephemeral port.
inline std::unique_ptr<TestGlobalRegistryServer>
CreateTestGlobalRegistryServerWithService(
    std::unique_ptr<GlobalRegistryService::Service> custom_service) {
  auto test_server = std::make_unique<TestGlobalRegistryServer>();
  test_server->custom_service = std::move(custom_service);

  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(test_server->custom_service.get());
  test_server->server = builder.BuildAndStart();
  CHECK(test_server->server != nullptr)
      << "Failed to start test GlobalRegistry server";

  test_server->server_address = absl::StrCat("localhost:", selected_port);
  test_server->channel = grpc::CreateChannel(
      test_server->server_address, grpc::InsecureChannelCredentials());
  test_server->client =
      std::make_unique<GlobalRegistryClient>(test_server->channel);
  return test_server;
}

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_TEST_UTIL_H_
