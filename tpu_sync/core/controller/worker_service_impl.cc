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

#include "tpu_sync/core/controller/worker_service_impl.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/transfer_program_reshard.h"
#include "tpu_sync/kv_cache/storage/k5_backend.h"
#include "tpu_sync/kv_cache/storage/storage.h"
#include "tpu_sync/proto/transfer_program.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {

namespace {

bool HasStage(const google::protobuf::RepeatedPtrField<std::string>& pipeline,
              absl::string_view stage) {
  for (const auto& s : pipeline) {
    if (s == stage) return true;
  }
  return false;
}

bool IsD2H(::tpu_sync::rpc::MemoryType src_mem_type,
           ::tpu_sync::rpc::MemoryType dst_mem_type) {
  return src_mem_type == ::tpu_sync::rpc::MEMORY_TYPE_HBM &&
         (dst_mem_type == ::tpu_sync::rpc::MEMORY_TYPE_DRAM ||
          dst_mem_type == ::tpu_sync::rpc::MEMORY_TYPE_UNSPECIFIED);
}

bool IsH2D(::tpu_sync::rpc::MemoryType src_mem_type,
           ::tpu_sync::rpc::MemoryType dst_mem_type) {
  return (src_mem_type == ::tpu_sync::rpc::MEMORY_TYPE_DRAM ||
          src_mem_type == ::tpu_sync::rpc::MEMORY_TYPE_UNSPECIFIED) &&
         dst_mem_type == ::tpu_sync::rpc::MEMORY_TYPE_HBM;
}

bool IsH2H(::tpu_sync::rpc::MemoryType src_mem_type,
           ::tpu_sync::rpc::MemoryType dst_mem_type) {
  return src_mem_type == ::tpu_sync::rpc::MEMORY_TYPE_DRAM &&
         dst_mem_type == ::tpu_sync::rpc::MEMORY_TYPE_DRAM;
}

}  // namespace

WorkerServiceImpl::WorkerServiceImpl(
    std::shared_ptr<HostMemoryAllocator> allocator,
    KVManagerHolder transfer_manager)
    : allocator_(allocator ? std::move(allocator)
                           : std::make_shared<MallocHostMemoryAllocator>()),
      transfer_manager_(std::move(transfer_manager)) {}

grpc::Status WorkerServiceImpl::CreateBuffers(
    grpc::ServerContext* context,
    const ::tpu_sync::proto::CreateBuffersRequest* request,
    ::tpu_sync::proto::CreateBuffersResponse* response) {
  absl::MutexLock lock(mutex_);
  for (const auto& spec : request->buffers()) {
    if (spec.num_shards() <= 0 || spec.size_bytes() <= 0) {
      response->set_success(false);
      response->set_message("num_shards and size_bytes must be positive");
      return grpc::Status::OK;
    }
    ::tpu_sync::proto::BufferProto sharded_buf;
    for (int32_t s = 0; s < spec.num_shards(); ++s) {
      auto alloc_or = allocator_->Allocate(spec.size_bytes());
      if (!alloc_or.ok()) {
        response->set_success(false);
        response->set_message(absl::StrCat("Failed to allocate memory: ",
                                           alloc_or.status().message()));
        return grpc::Status::OK;
      }
      BufferHandle handle = next_buffer_handle_++;
      buffers_[handle] = std::move(*alloc_or);
      sharded_buf.add_buffer_handles()->set_handle(handle);
    }
    *response->add_buffers() = std::move(sharded_buf);
  }
  response->set_success(true);
  response->set_message("Buffers created successfully");
  return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::DeleteBuffers(
    grpc::ServerContext* context,
    const ::tpu_sync::proto::DeleteBuffersRequest* request,
    ::tpu_sync::proto::DeleteBuffersResponse* response) {
  absl::MutexLock lock(mutex_);
  for (const auto& sharded_buffer : request->sharded_buffers()) {
    if (sharded_buffer.buffer_handles().empty()) {
      response->set_success(false);
      response->set_message("BufferProto has no buffer_handles");
      return grpc::Status::OK;
    }
    for (const auto& handle_proto : sharded_buffer.buffer_handles()) {
      BufferHandle handle = handle_proto.handle();
      auto it = buffers_.find(handle);
      if (it == buffers_.end()) {
        response->set_success(false);
        response->set_message(
            absl::StrCat("Buffer not found for deletion: handle ", handle));
        return grpc::Status::OK;
      }
      buffers_.erase(it);
    }
  }
  response->set_success(true);
  response->set_message("Buffers deleted successfully");
  return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::TransferBuffers(
    grpc::ServerContext* context,
    const ::tpu_sync::proto::TransferBuffersRequest* request,
    ::tpu_sync::proto::TransferBuffersResponse* response) {
  absl::MutexLock lock(mutex_);
  const auto& transfer = request->transfer();

  if (transfer.has_storage_spec()) {
    if (!transfer_manager_) {
      response->set_success(false);
      response->set_message(
          "Transfer manager is not configured on WorkerService");
      return grpc::Status::OK;
    }
    const auto& storage_spec = transfer.storage_spec();
    std::string scheme = storage_spec.scheme();
    std::shared_ptr<kv_cache::storage::KVBackend> driver =
        transfer_manager_.GetBackend(scheme);
    if (!driver) {
      LOG(ERROR) << "[Worker] No storage backend registered for scheme: "
                 << scheme;
      response->set_success(false);
      response->set_message(
          absl::StrCat("No storage backend registered for scheme: ", scheme));
      return grpc::Status::OK;
    }

    // Unpack per-block descriptors from keys_by_buffer_index map
    std::vector<kv_cache::storage::BlockKey> block_keys;
    std::vector<int64_t> src_offsets;
    std::vector<int64_t> dst_offsets;
    std::vector<int64_t> copy_sizes;

    size_t num_buffers = storage_spec.keys_by_buffer_index().size();
    block_keys.reserve(num_buffers);
    src_offsets.reserve(num_buffers);
    dst_offsets.reserve(num_buffers);
    copy_sizes.reserve(num_buffers);

    for (size_t i = 0; i < num_buffers; ++i) {
      auto it =
          storage_spec.keys_by_buffer_index().find(static_cast<int32_t>(i));
      if (it == storage_spec.keys_by_buffer_index().end()) {
        response->set_success(false);
        response->set_message(
            absl::StrCat("Missing StorageKeyDescriptor for buffer_index: ", i));
        return grpc::Status::OK;
      }
      kv_cache::storage::BlockKey key;
      key.resolved_key = it->second.storage_key();
      block_keys.push_back(key);

      int64_t src_off = (i < transfer.src_offsets().size())
                            ? transfer.src_offsets(i)
                        : (i < transfer.src_buffers_size() &&
                           transfer.src_buffers(i).has_index())
                            ? transfer.src_buffers(i).index()
                            : 0;
      int64_t dst_off = (i < transfer.dst_offsets().size())
                            ? transfer.dst_offsets(i)
                        : (i < transfer.dst_buffers_size() &&
                           transfer.dst_buffers(i).has_index())
                            ? transfer.dst_buffers(i).index()
                            : 0;
      int64_t copy_sz = (i < transfer.copy_sizes().size())
                            ? transfer.copy_sizes(i)
                            : transfer_manager_.bytes_per_block();

      src_offsets.push_back(src_off);
      dst_offsets.push_back(dst_off);
      copy_sizes.push_back(copy_sz);
    }

    absl::Status status;
    switch (storage_spec.direction()) {
      case ::tpu_sync::proto::TRANSFER_DIR_OFFLOAD: {
        auto fut_or = transfer_manager_.D2hWriteToBackend(
            driver, block_keys, src_offsets, dst_offsets, copy_sizes);
        if (fut_or.ok()) {
          status = fut_or.value().Await();
        } else {
          status = fut_or.status();
        }
        break;
      }

      case ::tpu_sync::proto::TRANSFER_DIR_RECALL: {
        auto fut_or = transfer_manager_.H2dReadFromBackend(
            driver, block_keys, src_offsets, dst_offsets, copy_sizes);
        if (fut_or.ok()) {
          status = fut_or.value().Await();
        } else {
          status = fut_or.status();
        }
        break;
      }

      default:
        status = absl::InvalidArgumentError(
            "Unrecognized storage transfer direction");
        break;
    }

    response->set_success(status.ok());
    if (!status.ok()) {
      response->set_message(std::string(status.message()));
    }
    return grpc::Status::OK;
  }

  ::tpu_sync::rpc::MemoryType src_mem_type = transfer.src_mem_type();
  ::tpu_sync::rpc::MemoryType dst_mem_type = transfer.dst_mem_type();

  if (src_mem_type == ::tpu_sync::rpc::MEMORY_TYPE_UNSPECIFIED &&
      transfer.src_buffers_size() > 0) {
    src_mem_type = transfer.src_buffers(0).memory_type();
  }
  if (dst_mem_type == ::tpu_sync::rpc::MEMORY_TYPE_UNSPECIFIED &&
      transfer.dst_buffers_size() > 0) {
    dst_mem_type = transfer.dst_buffers(0).memory_type();
  }

  bool is_d2h = IsD2H(src_mem_type, dst_mem_type);
  bool is_h2d = IsH2D(src_mem_type, dst_mem_type);
  bool is_h2h = IsH2H(src_mem_type, dst_mem_type);

  if (!is_d2h && !is_h2d && !is_h2h) {
    response->set_success(false);
    response->set_message(
        "Only D2H, H2D, and H2H (DRAM to DRAM) transfers are "
        "currently supported");
    return grpc::Status::OK;
  }
  if (is_h2h && transfer.copy_sizes_size() > 0) {
    for (int64_t size : transfer.copy_sizes()) {
      if (size != 1) {
        response->set_success(false);
        response->set_message(
            "H2H transfers only support copy size of 1 per segment");
        return grpc::Status::OK;
      }
    }
  }
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;

  if (transfer.src_offsets_size() > 0) {
    src_offsets.assign(transfer.src_offsets().begin(),
                       transfer.src_offsets().end());
    dst_offsets.assign(transfer.dst_offsets().begin(),
                       transfer.dst_offsets().end());
  } else if (transfer.src_buffers_size() > 0 &&
             transfer.src_buffers_size() == transfer.dst_buffers_size()) {
    src_offsets.reserve(transfer.src_buffers_size());
    for (const auto& src_buf : transfer.src_buffers()) {
      if (!src_buf.has_index() || src_buf.index() < 0) {
        response->set_success(false);
        response->set_message("BufferProto missing valid index");
        return grpc::Status::OK;
      }
      src_offsets.push_back(src_buf.index());
    }
    dst_offsets.reserve(transfer.dst_buffers_size());
    for (const auto& dst_buf : transfer.dst_buffers()) {
      if (!dst_buf.has_index() || dst_buf.index() < 0) {
        response->set_success(false);
        response->set_message("BufferProto missing valid index");
        return grpc::Status::OK;
      }
      dst_offsets.push_back(dst_buf.index());
    }
  }

  if (src_offsets.empty() || src_offsets.size() != dst_offsets.size()) {
    response->set_success(false);
    response->set_message(
        "Source and destination offsets must have the same non-zero length");
    return grpc::Status::OK;
  }
  if (transfer.copy_sizes_size() > 0 &&
      transfer.copy_sizes_size() != src_offsets.size()) {
    response->set_success(false);
    response->set_message(
        "copy_sizes, if provided, must match the length of src_offsets");
    return grpc::Status::OK;
  }

  if (!transfer_manager_) {
    response->set_success(false);
    response->set_message(
        "Transfer manager is not configured on WorkerService");
    return grpc::Status::OK;
  }

  std::vector<int64_t> copy_sizes;
  if (transfer.copy_sizes_size() > 0) {
    copy_sizes.assign(transfer.copy_sizes().begin(),
                      transfer.copy_sizes().end());
  } else {
    copy_sizes.assign(src_offsets.size(), 1);
  }

  // Host staging (bridge) offsets for 2-stage remote transfers -- the middle
  // hop of the data flow. Required for remote H2D read/write and remote D2H
  // write; the transfer manager rejects those transfers when it is missing.
  std::vector<int64_t> staging_host_offsets;
  staging_host_offsets.reserve(transfer.staging_host_buffers_size());
  for (const auto& buf : transfer.staging_host_buffers()) {
    staging_host_offsets.push_back(buf.index());
  }

  std::vector<RaidenTransferEndpoint> dst_remote_descriptors;
  if (transfer.dst_buffers_size() > 0 &&
      transfer.dst_buffers(0).remote_descriptors_size() > 0) {
    dst_remote_descriptors.reserve(
        transfer.dst_buffers(0).remote_descriptors_size());
    for (const auto& ep_proto : transfer.dst_buffers(0).remote_descriptors()) {
      std::vector<int64_t> shards(ep_proto.shards().begin(),
                                  ep_proto.shards().end());
      dst_remote_descriptors.push_back(
          {ep_proto.endpoint(), std::move(shards)});
    }
  }

  std::vector<RaidenTransferEndpoint> src_remote_descriptors;
  if (transfer.src_buffers_size() > 0 &&
      transfer.src_buffers(0).remote_descriptors_size() > 0) {
    src_remote_descriptors.reserve(
        transfer.src_buffers(0).remote_descriptors_size());
    for (const auto& ep_proto : transfer.src_buffers(0).remote_descriptors()) {
      std::vector<int64_t> shards(ep_proto.shards().begin(),
                                  ep_proto.shards().end());
      src_remote_descriptors.push_back(
          {ep_proto.endpoint(), std::move(shards)});
    }
  }

  absl::StatusOr<raiden::PjRtCopyFuture> future_or;
  if (is_d2h) {
    if (transfer.src_buffers_size() > 0 &&
        !transfer.src_buffers(0).remote_address().empty()) {
      std::string src_peer = transfer.src_buffers(0).remote_address();
      future_or = transfer_manager_.D2hRead(src_peer, src_offsets, dst_offsets,
                                            copy_sizes);
    } else if (transfer.dst_buffers_size() > 0 &&
               !transfer.dst_buffers(0).remote_address().empty()) {
      std::string dst_peer = transfer.dst_buffers(0).remote_address();
      // Flow order: local device src -> local host staging -> remote host dst.
      future_or = transfer_manager_.D2hWrite(
          dst_peer, src_offsets, staging_host_offsets, dst_offsets, copy_sizes);
    } else {
      future_or = transfer_manager_.D2h(src_offsets, dst_offsets, copy_sizes);
    }
  } else if (is_h2d) {
    if (!src_remote_descriptors.empty()) {
      // Receiver-initiated pull: remote host src -> local host staging ->
      // local device dst, with per-shard endpoint routing.
      future_or = transfer_manager_.H2dRead(src_remote_descriptors, src_offsets,
                                            staging_host_offsets, dst_offsets,
                                            copy_sizes);
    } else if (transfer.src_buffers_size() > 0 &&
               !transfer.src_buffers(0).remote_address().empty()) {
      std::string src_peer = transfer.src_buffers(0).remote_address();
      // Flow order: remote host src -> local host staging -> local device dst.
      future_or = transfer_manager_.H2dRead(
          src_peer, src_offsets, staging_host_offsets, dst_offsets, copy_sizes);
    } else if (transfer.dst_buffers_size() > 0 &&
               !transfer.dst_buffers(0).remote_address().empty()) {
      std::string dst_peer = transfer.dst_buffers(0).remote_address();
      // Flow order: local host src -> remote host staging -> remote device dst.
      future_or = transfer_manager_.H2dWrite(
          dst_peer, src_offsets, staging_host_offsets, dst_offsets, copy_sizes);
    } else {
      future_or = transfer_manager_.H2d(src_offsets, dst_offsets, copy_sizes);
    }
  } else if (!src_remote_descriptors.empty()) {
    future_or = transfer_manager_.H2hRead(src_remote_descriptors, src_offsets,
                                          dst_offsets);
  } else if (!dst_remote_descriptors.empty()) {
    future_or = transfer_manager_.H2hWrite(dst_remote_descriptors, src_offsets,
                                           dst_offsets);
  } else if (transfer.src_buffers_size() > 0 &&
             !transfer.src_buffers(0).remote_address().empty()) {
    future_or = transfer_manager_.H2hRead(
        transfer.src_buffers(0).remote_address(), src_offsets, dst_offsets);
  } else if (transfer.dst_buffers_size() > 0 &&
             !transfer.dst_buffers(0).remote_address().empty()) {
    future_or = transfer_manager_.H2hWrite(
        transfer.dst_buffers(0).remote_address(), src_offsets, dst_offsets);
  } else {
    response->set_success(false);
    response->set_message("Peer address must be provided for H2H transfers");
    return grpc::Status::OK;
  }
  if (!future_or.ok()) {
    response->set_success(false);
    response->set_message(absl::StrCat(
        is_d2h ? "D2H" : (is_h2d ? "H2D" : "H2H"),
        " transfer dispatch failed: ", future_or.status().message()));
    return grpc::Status::OK;
  }
  absl::Status status = future_or->Await();
  if (!status.ok()) {
    response->set_success(false);
    response->set_message(
        absl::StrCat("Transfer execution failed: ", status.message()));
    return grpc::Status::OK;
  }
  response->set_success(true);
  response->set_message("Buffers transferred successfully");
  return grpc::Status::OK;
}

absl::StatusOr<HostBufferAllocation> WorkerServiceImpl::GetBuffer(
    BufferHandle handle) const {
  absl::MutexLock lock(mutex_);
  auto it = buffers_.find(handle);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Buffer not found: handle ", handle));
  }
  return it->second;
}

size_t WorkerServiceImpl::GetBufferCount() const {
  absl::MutexLock lock(mutex_);
  return buffers_.size();
}

grpc::Status WorkerServiceImpl::SubmitTransferProgram(
    grpc::ServerContext* context,
    const ::tpu_sync::proto::TransferProgramRequest* request,
    ::tpu_sync::proto::TransferProgramResponse* response) {
  // Class rejection is a dispatch failure, not an admission verdict: the
  // dormant contract classes and malformed programs fail loudly at the gRPC
  // layer so nothing mistakes them for an executed-and-failed transfer.
  absl::StatusOr<core::ReshardLoweringClass> lowering_class =
      core::NormalizeReshardProgram(*request);
  if (!lowering_class.ok()) {
    grpc::StatusCode code =
        lowering_class.status().code() == absl::StatusCode::kUnimplemented
            ? grpc::StatusCode::UNIMPLEMENTED
            : grpc::StatusCode::INVALID_ARGUMENT;
    return grpc::Status(code,
                        std::string(lowering_class.status().message()));
  }

  absl::StatusOr<::tpu_sync::rpc::StartTransferRequest> lowered =
      core::LowerToStartTransfer(*request);
  if (!lowered.ok()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        std::string(lowered.status().message()));
  }

  // From here down the semantics are the framed door's: run the same pool
  // executor op the listener would, and carry its verdict in-band.
  absl::MutexLock lock(mutex_);
  if (transfer_manager_ == nullptr) {
    response->set_success(false);
    response->set_message("Transfer manager is not initialized");
    return grpc::Status::OK;
  }

  absl::Status status;
  if (*lowering_class == core::ReshardLoweringClass::kPoolReshardSender) {
    const std::vector<int64_t> src_block_ids =
        core::DeriveSenderSourceBlockIds(*lowered);
    status = transfer_manager_.PoolReshardPush(*lowered, src_block_ids,
                                               lowered->parallelism());
  } else {
    const std::vector<int64_t> chip_block_ids =
        core::DeriveArmChipBlockIds(*lowered);
    status = transfer_manager_.PoolReshardRegisterRecv(*lowered,
                                                       chip_block_ids);
  }

  response->set_success(status.ok());
  response->set_message(status.ok() ? "SUCCESS"
                                    : std::string(status.message()));
  return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::PollTransfer(
    grpc::ServerContext* context,
    const ::tpu_sync::proto::PollTransferRequest* request,
    ::tpu_sync::proto::PollTransferResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                      "PollTransfer is not implemented; transfer completion "
                      "uses the existing connector polling path");
}

grpc::Status WorkerServiceImpl::AbortTransfer(
    grpc::ServerContext* context,
    const ::tpu_sync::proto::AbortTransferRequest* request,
    ::tpu_sync::proto::AbortTransferResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                      "AbortTransfer is not implemented");
}

grpc::Status WorkerServiceImpl::RegisterBackends(
    grpc::ServerContext* context,
    const ::tpu_sync::proto::RegisterBackendsRequest* request,
    ::tpu_sync::proto::RegisterBackendsResponse* response) {
  absl::MutexLock lock(mutex_);
  if (!transfer_manager_) {
    response->set_success(false);
    response->set_error_message(
        "Transfer manager is not configured on WorkerService");
    return grpc::Status::OK;
  }
  for (const auto& config : request->configs()) {
    std::string name = config.name();
    std::string scheme = config.scheme();
    if (scheme.empty()) {
      auto scheme_it = config.properties().find("scheme");
      if (scheme_it != config.properties().end()) {
        scheme = scheme_it->second;
      }
    }
    if (name == "PosixBackend") {
      if (scheme.empty()) {
        response->set_success(false);
        response->set_error_message("Missing 'scheme' in backend config");
        return grpc::Status::OK;
      }
      auto backend = std::make_shared<kv_cache::storage::PosixBackend>();
      transfer_manager_.RegisterBackend(scheme, backend);
      LOG(INFO)
          << "[Worker] Dynamically registered storage backend for scheme '"
          << scheme << "' (name: " << name << ")";
    } else if (name == "K5Backend" || name == "K5BackendMock") {
      auto uds_it = config.properties().find("uds_path");
      if (scheme.empty() || uds_it == config.properties().end()) {
        response->set_success(false);
        response->set_error_message(
            "Missing 'scheme' or 'uds_path' in K5Backend config");
        return grpc::Status::OK;
      }
      std::string uds_path = uds_it->second;
      auto backend =
          std::make_shared<kv_cache::storage::K5BackendMock>(uds_path);
      transfer_manager_.RegisterBackend(scheme, backend);
      LOG(INFO) << "[Worker] Dynamically registered K5 backend for scheme '"
                << scheme << "' and UDS: " << uds_path;
    } else {
      response->set_success(false);
      response->set_error_message("Unsupported backend name: " + name);
      return grpc::Status::OK;
    }
  }
  response->set_success(true);
  return grpc::Status::OK;
}

}  // namespace controller
}  // namespace tpu_raiden
