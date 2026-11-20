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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_SERVER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_SERVER_H_

#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/kv_cache_offload/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache_offload/kv_cache_store_service.h"
namespace tpu_raiden {
namespace kv_cache {

// A gRPC server wrapper that hosts KVCacheStoreServiceImpl on an ephemeral or
// specified port.
class KVCacheStoreServer {
 public:
  KVCacheStoreServer() = default;

  KVCacheStoreServer(const KVCacheStoreServer&) = delete;
  KVCacheStoreServer& operator=(const KVCacheStoreServer&) = delete;

  // Creates an instance of KVCacheStoreServer.
  static std::unique_ptr<KVCacheStoreServer> Create();

  ~KVCacheStoreServer();

  // Starts the gRPC server hosting KVCacheStoreServiceImpl with backend and
  // controller pointers.
  absl::Status StartServer(KVCacheStoreBackend* backend,
                           tpu_raiden::controller::RaidenController* controller,
                           absl::string_view server_address = "");

  // Returns the port the gRPC server is listening on. Returns 0 if not running.
  int GetGrpcPort() const;

  // Returns the formatted server address (e.g. "localhost:12345" or
  // "[::]:12345"). Returns an empty string if the server is not running.
  std::string GetServerAddress() const;

  // Installs `fn` on the hosted service; see
  // KVCacheStoreServiceImpl::SetSpaceCheckFn. No-op if the server is not
  // running.
  void SetSpaceCheckFn(KVCacheStoreServiceImpl::SpaceCheckFn fn);

  // Gracefully shuts down the gRPC server and cleans up service state.
  void Shutdown();

 private:
  // De-duplicated helper method for building and starting the gRPC server.
  absl::Status StartServerInternal(absl::string_view server_address)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;
  std::unique_ptr<KVCacheStoreServiceImpl> service_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<grpc::Server> grpc_server_ ABSL_GUARDED_BY(mutex_);
  int grpc_port_ ABSL_GUARDED_BY(mutex_) = 0;
  std::string server_host_ ABSL_GUARDED_BY(mutex_) = "localhost";
  bool started_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_SERVER_H_
