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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_TEST_UTIL_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_TEST_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/controller_service.h"
#include "tpu_sync/core/controller/worker_service_client.h"
#include "tpu_sync/core/controller/worker_service_impl.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/kv_cache/backends/backend.h"

namespace tpu_raiden {
namespace controller {

struct MockTransferManager {
  int d2h_calls = 0;
  int d2h_write_calls = 0;
  int d2h_read_calls = 0;
  int h2d_calls = 0;
  int h2d_write_calls = 0;
  int h2d_read_calls = 0;
  int h2h_calls = 0;
  int h2h_read_calls = 0;
  int h2h_write_calls = 0;
  std::string last_peer;
  std::vector<int64_t> last_src_offsets;
  std::vector<int64_t> last_dst_offsets;
  std::vector<int64_t> last_staging_offsets;
  std::vector<int64_t> last_copy_sizes;

  // Scripts every transfer to fail, so failure-path logic (pin retention,
  // failed-list reporting, block release) can be exercised on CPU where a
  // real transfer cannot run at all.
  bool fail_transfers = false;

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) {
    d2h_calls++;
    last_src_offsets = src_offsets;
    last_dst_offsets = dst_offsets;
    last_copy_sizes = copy_sizes;
    if (fail_transfers) return absl::InternalError("scripted transfer failure");
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hWrite(
      absl::string_view peer, const std::vector<int64_t>& src_device_offsets,
      const std::vector<int64_t>& src_host_offsets,
      const std::vector<int64_t>& dst_host_offsets,
      const std::vector<int64_t>& copy_sizes) {
    d2h_write_calls++;
    last_peer = std::string(peer);
    last_src_offsets = src_device_offsets;
    last_staging_offsets = src_host_offsets;
    last_dst_offsets = dst_host_offsets;
    last_copy_sizes = copy_sizes;
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hRead(
      absl::string_view peer, const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) {
    d2h_read_calls++;
    last_peer = std::string(peer);
    last_src_offsets = src_offsets;
    last_dst_offsets = dst_offsets;
    last_copy_sizes = copy_sizes;
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) {
    h2d_calls++;
    last_src_offsets = src_offsets;
    last_dst_offsets = dst_offsets;
    last_copy_sizes = copy_sizes;
    if (fail_transfers) return absl::InternalError("scripted transfer failure");
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2dWrite(
      absl::string_view peer, const std::vector<int64_t>& src_host_offsets,
      const std::vector<int64_t>& dst_host_offsets,
      const std::vector<int64_t>& dst_device_offsets,
      const std::vector<int64_t>& copy_sizes) {
    h2d_write_calls++;
    last_peer = std::string(peer);
    last_src_offsets = src_host_offsets;
    last_staging_offsets = dst_host_offsets;
    last_dst_offsets = dst_device_offsets;
    last_copy_sizes = copy_sizes;
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
      absl::string_view peer, const std::vector<int64_t>& src_host_offsets,
      const std::vector<int64_t>& dst_host_offsets,
      const std::vector<int64_t>& dst_device_offsets,
      const std::vector<int64_t>& copy_sizes) {
    h2d_read_calls++;
    last_peer = std::string(peer);
    last_src_offsets = src_host_offsets;
    last_staging_offsets = dst_host_offsets;
    last_dst_offsets = dst_device_offsets;
    last_copy_sizes = copy_sizes;
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1) {
    h2h_calls++;
    h2h_read_calls++;
    last_peer = peer;
    last_src_offsets.assign(src_block_ids.begin(), src_block_ids.end());
    last_dst_offsets.assign(dst_block_ids.begin(), dst_block_ids.end());
    return std::make_pair(std::vector<int>{}, raiden::PjRtCopyFuture());
  }
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1) {
    h2h_calls++;
    h2h_write_calls++;
    last_peer = peer;
    last_src_offsets.assign(src_block_ids.begin(), src_block_ids.end());
    last_dst_offsets.assign(dst_block_ids.begin(), dst_block_ids.end());
    return std::make_pair(std::vector<int>{}, raiden::PjRtCopyFuture());
  }

  absl::flat_hash_map<std::string,
                      std::shared_ptr<kv_cache::backends::KVBackend>>
      backends;

  void RegisterKVBackend(
      const std::string& backend_name,
      std::shared_ptr<kv_cache::backends::KVBackend> backend) {
    backends[backend_name] = std::move(backend);
  }

  std::shared_ptr<kv_cache::backends::KVBackend> GetKVBackend(
      absl::string_view backend_name) {
    auto it = backends.find(std::string(backend_name));
    if (it != backends.end()) return it->second;
    class MockKVBackend : public kv_cache::backends::KVBackend {
     public:
      explicit MockKVBackend(std::string name) : name_(std::move(name)) {}
      std::string name() const override { return name_; }
      void WriteAsync(
          const kv_cache::backends::BlockKey& key,
          absl::Span<const kv_cache::backends::BackendBufferDescriptor> slices,
          size_t total_bytes,
          std::function<void(absl::Status)> callback) override {
        if (callback) callback(absl::OkStatus());
      }
      void ReadAsync(
          const kv_cache::backends::BlockKey& key,
          absl::Span<const kv_cache::backends::BackendBufferDescriptor> slices,
          size_t total_bytes,
          std::function<void(absl::Status)> callback) override {
        if (callback) callback(absl::OkStatus());
      }
      void BatchExistsAsync(
          absl::Span<const kv_cache::backends::BlockKey> keys,
          std::function<void(std::vector<absl::StatusOr<bool>>)> callback)
          override {
        if (callback) {
          callback(std::vector<absl::StatusOr<bool>>(keys.size(), true));
        }
      }

     private:
      std::string name_;
    };

    class MockMapper : public kv_cache::backends::BlockKeyMapper {
     public:
      kv_cache::backends::BlockKey MapKey(
          const std::string& block_hash,
          const kv_cache::backends::KeyMappingOptions& options = {})
          const override {
        kv_cache::backends::BlockKey key;
        key.block_hash = block_hash;
        key.resolved_key = absl::StrCat(block_hash, ".bin");
        return key;
      }
      int tp_size() const override { return 1; }
    };
    auto backend = std::make_shared<MockKVBackend>(std::string(backend_name));
    backend->set_mapper(std::make_shared<MockMapper>());
    backends[std::string(backend_name)] = backend;
    return backend;
  }

  int h2d_read_from_backend_calls = 0;
  int d2h_write_to_backend_calls = 0;
  std::vector<kv_cache::backends::BlockKey> last_h2d_backend_keys;
  std::vector<kv_cache::backends::BlockKey> last_d2h_backend_keys;
  std::vector<int64_t> last_backend_src_block_ids;
  std::vector<int64_t> last_backend_dst_block_ids;

  absl::StatusOr<raiden::PjRtCopyFuture> H2dReadFromBackend(
      absl::Span<const std::shared_ptr<kv_cache::backends::KVBackend>> backends,
      const std::vector<kv_cache::backends::BlockKey>& block_keys,
      const std::vector<int64_t>& src_host_block_ids,
      const std::vector<int64_t>& dst_device_block_ids) {
    ++h2d_read_from_backend_calls;
    last_h2d_backend_keys = block_keys;
    last_backend_src_block_ids = src_host_block_ids;
    last_backend_dst_block_ids = dst_device_block_ids;
    return H2d(src_host_block_ids, dst_device_block_ids, {1});
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2dReadFromBackend(
      std::shared_ptr<kv_cache::backends::KVBackend> backend,
      const std::vector<kv_cache::backends::BlockKey>& block_keys,
      const std::vector<int64_t>& src_host_block_ids,
      const std::vector<int64_t>& dst_device_block_ids) {
    const std::shared_ptr<kv_cache::backends::KVBackend> b[] = {
        std::move(backend)};
    return H2dReadFromBackend(absl::MakeSpan(b), block_keys, src_host_block_ids,
                              dst_device_block_ids);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hWriteToBackend(
      absl::Span<const std::shared_ptr<kv_cache::backends::KVBackend>> backends,
      const std::vector<kv_cache::backends::BlockKey>& block_keys,
      const std::vector<int64_t>& src_device_block_ids,
      const std::vector<int64_t>& dst_host_block_ids) {
    ++d2h_write_to_backend_calls;
    last_d2h_backend_keys = block_keys;
    last_backend_src_block_ids = src_device_block_ids;
    last_backend_dst_block_ids = dst_host_block_ids;
    return D2h(src_device_block_ids, dst_host_block_ids, {1});
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hWriteToBackend(
      std::shared_ptr<kv_cache::backends::KVBackend> backend,
      const std::vector<kv_cache::backends::BlockKey>& block_keys,
      const std::vector<int64_t>& src_device_block_ids,
      const std::vector<int64_t>& dst_host_block_ids) {
    const std::shared_ptr<kv_cache::backends::KVBackend> b[] = {
        std::move(backend)};
    return D2hWriteToBackend(absl::MakeSpan(b), block_keys,
                             src_device_block_ids, dst_host_block_ids);
  }
};

// Like MockTransferManager but ALSO implements the vector
// (RaidenTransferEndpoint) H2hRead/H2hWrite overloads. Because it exposes these,
// KVManagerHolder's has_vector_h2h_*_v trait is true, so the holder dispatches
// remote reads/writes to the SHARD-MATCHING (vector) path instead of the
// single-endpoint string fallback. Records the full shard-tagged descriptor list
// the worker received, so a test can assert that path was actually triggered
// with the shards intact.
struct ShardAwareMockTransferManager : MockTransferManager {
  // Keep the base string overloads visible (the vector declarations below would
  // otherwise hide them, and KVManagerHolder still references the string form).
  using MockTransferManager::H2dRead;
  using MockTransferManager::H2hRead;
  using MockTransferManager::H2hWrite;

  int vector_h2h_read_calls = 0;
  int vector_h2h_write_calls = 0;
  int vector_h2d_read_calls = 0;
  std::vector<::tpu_raiden::RaidenTransferEndpoint> last_read_descriptors;
  std::vector<::tpu_raiden::RaidenTransferEndpoint> last_write_descriptors;
  std::vector<::tpu_raiden::RaidenTransferEndpoint> last_h2d_read_descriptors;

  absl::StatusOr<raiden::PjRtCopyFuture> H2hReadExplicit(
      const std::vector<::tpu_raiden::RaidenTransferEndpoint>&
          remote_descriptors,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids) {
    vector_h2h_read_calls++;
    last_read_descriptors = remote_descriptors;
    last_src_offsets.assign(src_block_ids.begin(), src_block_ids.end());
    last_dst_offsets.assign(dst_block_ids.begin(), dst_block_ids.end());
    if (fail_transfers) return absl::InternalError("scripted transfer failure");
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
      const std::vector<::tpu_raiden::RaidenTransferEndpoint>&
          remote_descriptors,
      const std::vector<int64_t>& src_host_offsets,
      const std::vector<int64_t>& dst_host_offsets,
      const std::vector<int64_t>& dst_device_offsets,
      const std::vector<int64_t>& copy_sizes) {
    vector_h2d_read_calls++;
    last_h2d_read_descriptors = remote_descriptors;
    last_src_offsets = src_host_offsets;
    last_staging_offsets = dst_host_offsets;
    last_dst_offsets = dst_device_offsets;
    last_copy_sizes = copy_sizes;
    if (fail_transfers) return absl::InternalError("scripted transfer failure");
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      const std::vector<::tpu_raiden::RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1) {
    vector_h2h_write_calls++;
    last_write_descriptors = remote_descriptors;
    last_src_offsets.assign(src_block_ids.begin(), src_block_ids.end());
    last_dst_offsets.assign(dst_block_ids.begin(), dst_block_ids.end());
    return std::make_pair(std::vector<int>{}, raiden::PjRtCopyFuture());
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      const std::vector<::tpu_raiden::RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int>& src_block_ids) {
    vector_h2h_read_calls++;
    last_read_descriptors = remote_descriptors;
    if (fail_transfers) return absl::InternalError("scripted transfer failure");
    last_src_offsets.assign(src_block_ids.begin(), src_block_ids.end());
    return std::make_pair(std::vector<int>{}, raiden::PjRtCopyFuture());
  }
};

// Helper struct wrapping an in-process gRPC test server and client for
// WorkerService.
struct TestWorkerServer {
  std::unique_ptr<WorkerServiceImpl> service;
  std::unique_ptr<grpc::Server> server;
  std::string server_address;
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<WorkerServiceClient> client;

  ~TestWorkerServer() {
    if (server) {
      server->Shutdown();
    }
  }
};

// Creates and starts an in-process gRPC TestWorkerServer hosting
// WorkerServiceImpl on an ephemeral port.
inline std::unique_ptr<TestWorkerServer> CreateTestWorkerServer() {
  auto test_server = std::make_unique<TestWorkerServer>();
  test_server->service = std::make_unique<WorkerServiceImpl>();

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
      std::make_unique<WorkerServiceClient>(test_server->channel);
  return test_server;
}

}  // namespace controller
}  // namespace tpu_raiden

namespace tpu_raiden {
namespace core {
namespace controller {

// Helper struct wrapping an in-process gRPC test server and client for
// RaidenControllerService.
struct TestControllerServer {
  std::unique_ptr<RaidenControllerServiceImpl> service;
  std::unique_ptr<grpc::Server> server;
  std::string server_address;
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<RaidenControllerClient> client;

  ~TestControllerServer() {
    if (server) {
      server->Shutdown();
    }
  }
};

// Creates and starts an in-process gRPC TestControllerServer hosting
// RaidenControllerServiceImpl on an ephemeral port.
inline std::unique_ptr<TestControllerServer> CreateTestControllerServer() {
  auto test_server = std::make_unique<TestControllerServer>();
  test_server->service =
      std::make_unique<RaidenControllerServiceImpl>();

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
      std::make_unique<RaidenControllerClient>(test_server->channel);
  return test_server;
}

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_TEST_UTIL_H_
