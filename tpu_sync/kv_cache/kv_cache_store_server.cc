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

#include "tpu_sync/kv_cache/kv_cache_store_server.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_service.h"
namespace tpu_raiden {
namespace kv_cache {

std::unique_ptr<KVCacheStoreServer> KVCacheStoreServer::Create() {
  return std::make_unique<KVCacheStoreServer>();
}

KVCacheStoreServer::~KVCacheStoreServer() { Shutdown(); }

absl::Status KVCacheStoreServer::StartServer(
    KVCacheStoreBackend* backend,
    tpu_raiden::controller::RaidenController* controller,
    absl::string_view server_address) {
  absl::MutexLock lock(mutex_);
  if (started_) {
    return absl::OkStatus();
  }

  service_ = std::make_unique<KVCacheStoreServiceImpl>(backend, controller);
  return StartServerInternal(server_address);
}

absl::Status KVCacheStoreServer::StartServerInternal(
    absl::string_view server_address) {
  std::string target_address;
  std::string host = "localhost";
  if (server_address.empty()) {
    target_address = "[::]:0";
  } else {
    size_t last_colon = server_address.rfind(':');
    size_t last_bracket = server_address.rfind(']');
    bool has_port =
        (last_colon != absl::string_view::npos) &&
        (last_bracket == absl::string_view::npos || last_colon > last_bracket);
    if (!has_port) {
      target_address = absl::StrCat(server_address, ":0");
      host = std::string(server_address);
    } else {
      target_address = std::string(server_address);
      host = std::string(server_address.substr(0, last_colon));
    }
    if (host == "0.0.0.0" || host == "[::]") {
      host = "localhost";
    }
  }

  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort(target_address, grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(service_.get());
  grpc_server_ = builder.BuildAndStart();

  if (!grpc_server_ || selected_port == 0) {
    grpc_server_.reset();
    service_.reset();
    return absl::InternalError(
        absl::StrCat("Failed to start gRPC KVCacheStoreServer on address: ",
                     target_address));
  }

  grpc_port_ = selected_port;
  server_host_ = host;
  started_ = true;
  return absl::OkStatus();
}

int KVCacheStoreServer::GetGrpcPort() const {
  absl::MutexLock lock(mutex_);
  return grpc_port_;
}

std::string KVCacheStoreServer::GetServerAddress() const {
  absl::MutexLock lock(mutex_);
  if (grpc_port_ <= 0) {
    return "";
  }
  if (server_host_.empty() || server_host_ == "[::]" ||
      server_host_ == "0.0.0.0") {
    // Wildcard bind: no routable host to report, and no publishable one --
    // no in-tree caller uses a wildcard bind anymore.
    return "";
  }
  return absl::StrCat(server_host_, ":", grpc_port_);
}

void KVCacheStoreServer::Shutdown() {
  absl::MutexLock lock(mutex_);
  if (grpc_server_) {
    grpc_server_->Shutdown();
    grpc_server_.reset();
  }
  service_.reset();
  grpc_port_ = 0;
  server_host_ = "localhost";
  started_ = false;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
