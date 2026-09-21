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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_GRPC_CONTROL_PIPE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_GRPC_CONTROL_PIPE_H_

#include <cstddef>
#include <list>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/server.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.grpc.pb.h"
#include "tpu_sync/proto/control_pipe.pb.h"

namespace tpu_raiden {

class GrpcControlPipeServer : public ControlPipeServer {
 public:
  explicit GrpcControlPipeServer(const ControlPipeConfig& config);
  ~GrpcControlPipeServer() override;

  absl::StatusOr<int> Start(int requested_port) override;
  void Stop() override;
  void StopAccepting() override;

  int bound_port() const override { return bound_port_; }
  ControlDispatcher& dispatcher() override { return dispatcher_; }
  ControlPipeBackendType backend_type() const override {
    return ControlPipeBackendType::kGrpc;
  }

 private:
  class ControlPipeServiceImpl;
  class LegacyKVCacheServiceImpl;

  ControlPipeConfig config_;
  ControlDispatcher dispatcher_;
  int bound_port_ = 0;

  mutable absl::Mutex mu_;
  std::unique_ptr<grpc::Server> grpc_server_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<ControlPipeServiceImpl> pipe_service_;
  std::unique_ptr<LegacyKVCacheServiceImpl> kv_cache_service_;
};

class GrpcControlPipeClient : public ControlPipeClient {
 public:
  static constexpr size_t kDefaultMaxCachedStubs = 100000;

  explicit GrpcControlPipeClient(const ControlPipeConfig& config);
  ~GrpcControlPipeClient() override = default;

  absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> SendRaw(
      absl::string_view endpoint,
      const control_pipe::proto::ControlEnvelope& envelope,
      absl::Duration timeout) override;

  ControlPipeBackendType backend_type() const override {
    return ControlPipeBackendType::kGrpc;
  }

  // Returns the current number of cached gRPC stubs.
  size_t TEST_CachedStubCount() const;

  // Returns true if a stub for |endpoint| is currently in the LRU cache.
  bool TEST_HasCachedStub(absl::string_view endpoint) const;

 private:
  struct StubCacheEntry {
    std::shared_ptr<control_pipe::proto::ControlPipeService::Stub> stub;
    std::list<std::string>::iterator lru_it;
  };

  std::shared_ptr<control_pipe::proto::ControlPipeService::Stub>
  GetOrCreateStub(absl::string_view endpoint);

  ControlPipeConfig config_;
  mutable absl::Mutex stub_mu_;
  std::list<std::string> lru_order_ ABSL_GUARDED_BY(stub_mu_);
  absl::flat_hash_map<std::string, StubCacheEntry> stubs_
      ABSL_GUARDED_BY(stub_mu_);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_GRPC_CONTROL_PIPE_H_
