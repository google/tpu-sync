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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_IMPL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/proto/worker_service.grpc.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace controller {

using BufferHandle = uint64_t;

// Implementation of the WorkerService gRPC service running on transfer workers.
// Manages allocation and deallocation of sharded host memory buffers, and
// executes D2H and H2D transfers via KVManagerHolder.
class WorkerServiceImpl final
    : public ::tpu_sync::proto::WorkerService::Service {
 public:
  // Constructs a WorkerServiceImpl with the given host memory allocator and
  // transfer manager. If allocator is nullptr, defaults to
  // MallocHostMemoryAllocator.
  explicit WorkerServiceImpl(
      std::shared_ptr<HostMemoryAllocator> allocator = nullptr,
      KVManagerHolder transfer_manager = KVManagerHolder());

  void SetTransferManager(KVManagerHolder transfer_manager) {
    absl::MutexLock lock(mutex_);
    transfer_manager_ = std::move(transfer_manager);
  }

  grpc::Status CreateBuffers(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::CreateBuffersRequest* request,
      ::tpu_sync::proto::CreateBuffersResponse* response) override;

  grpc::Status DeleteBuffers(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::DeleteBuffersRequest* request,
      ::tpu_sync::proto::DeleteBuffersResponse* response) override;

  // Transfers (copies) disjoint memory regions across memory spaces on the
  // worker. The transfer specification applies uniformly across all buffers,
  // i.e., all shards and major dimensions (layers or blocks).
  grpc::Status TransferBuffers(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::TransferBuffersRequest* request,
      ::tpu_sync::proto::TransferBuffersResponse* response) override;

  // Dynamically configures storage backends (e.g. POSIX disk, K5 thick client)
  // on this worker node.
  grpc::Status RegisterBackends(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::RegisterBackendsRequest* request,
      ::tpu_sync::proto::RegisterBackendsResponse* response) override;

  // Normalizes a pool-reshard transfer program, lowers it to the
  // byte-identical StartTransferRequest the framed entry would deliver, and
  // drives the same pool executor operations. Unsupported completion
  // contracts (AwaitInline/Barrier) return UNIMPLEMENTED as the gRPC status;
  // admission verdicts for supported programs ride the response body with
  // the framed entry's success/message semantics.
  grpc::Status SubmitTransferProgram(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::TransferProgramRequest* request,
      ::tpu_sync::proto::TransferProgramResponse* response) override;

  // Transfer-program completion remains on the connector's existing polling
  // path. These lifecycle RPCs fail closed until they have implementations.
  grpc::Status PollTransfer(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::PollTransferRequest* request,
      ::tpu_sync::proto::PollTransferResponse* response) override;
  grpc::Status AbortTransfer(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::AbortTransferRequest* request,
      ::tpu_sync::proto::AbortTransferResponse* response) override;

  // Retrieves an allocated buffer shard for inspection or transfer operations.
  // Returns NotFoundError if the buffer handle is invalid.
  absl::StatusOr<HostBufferAllocation> GetBuffer(BufferHandle handle) const;

  // Overload accepting BufferHandleProto for convenience.
  absl::StatusOr<HostBufferAllocation> GetBuffer(
      const ::tpu_sync::proto::BufferHandleProto& handle_proto) const {
    return GetBuffer(BufferHandle(handle_proto.handle()));
  }

  // Returns the total number of currently allocated buffer shards across all
  // buffers.
  size_t GetBufferCount() const;

 private:
  std::shared_ptr<HostMemoryAllocator> allocator_;
  mutable absl::Mutex mutex_;
  uint64_t next_buffer_handle_ ABSL_GUARDED_BY(mutex_) = 1;
  absl::flat_hash_map<BufferHandle, HostBufferAllocation> buffers_
      ABSL_GUARDED_BY(mutex_);
  KVManagerHolder transfer_manager_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace controller
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_IMPL_H_
