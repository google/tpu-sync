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

#include "tpu_sync/frameworks/torch/kv_cache_manager.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "c10/core/ScalarType.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/buffer_utils.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/worker_service_server.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/core/utils.h"
#include "tpu_sync/frameworks/torch/torch_utils.h"
#include "tpu_sync/kv_cache/kv_cache_listener.h"
#include "tpu_sync/transport/block_transport.h"

namespace tpu_raiden {
namespace torch {
namespace {

using TensorList = std::vector<at::Tensor>;

std::vector<std::vector<at::Tensor>> SingleShardLayers(
    const TensorList& kv_caches) {
  std::vector<std::vector<at::Tensor>> layers;
  layers.reserve(kv_caches.size());
  for (const auto& kv_cache : kv_caches) {
    layers.push_back({kv_cache});
  }
  return layers;
}

std::string FormatAddressWithPort(absl::string_view ip, int port) {
  if (absl::StrContains(ip, ':')) {
    return absl::StrCat("[", ip, "]:", port);
  }
  return absl::StrCat(ip, ":", port);
}

}  // namespace

TorchKVCacheManager::UnpackedLayers TorchKVCacheManager::UnpackLayers(
    const std::vector<std::vector<at::Tensor>>& device_tensors,
    bool unsafe_skip_buffer_lock) {
  // Retain the owning TensorBufferHandles: they pin the base storage buffers
  // backing the tensors, so they must outlive every D2h/H2d the manager
  // dispatches.
  UnpackedTensors u =
      UnpackTorchTensors(device_tensors, unsafe_skip_buffer_lock);
  UnpackedLayers unpacked;
  unpacked.buffers = std::move(u.buffers);
  unpacked.refs = std::move(u.refs);
  unpacked.logical_dimensions = std::move(u.logical_dimensions);
  unpacked.logical_slice_byte_size = u.logical_slice_byte_size;
  unpacked.logical_physical_size = u.logical_physical_size;
  unpacked.has_logical_metadata = u.has_logical_metadata;
  if (!unpacked.buffers.empty() && !unpacked.buffers[0].empty() &&
      unpacked.buffers[0][0].device) {
    unpacked.client = unpacked.buffers[0][0].device->client();
  }
  return unpacked;
}

TorchKVCacheManager::UnpackedLayers TorchKVCacheManager::UnpackLayers(
    const std::vector<std::vector<xla::PjRtBuffer*>>& device_buffers,
    bool unsafe_skip_buffer_lock) {
  UnpackedLayers unpacked;
  unpacked.buffers =
      ::tpu_raiden::UnpackLayers(device_buffers, unsafe_skip_buffer_lock);
  if (!unpacked.buffers.empty() && !unpacked.buffers[0].empty() &&
      unpacked.buffers[0][0].device) {
    unpacked.client = unpacked.buffers[0][0].device->client();
  }
  return unpacked;
}

TorchKVCacheManager::TorchKVCacheManager(
    const std::vector<std::vector<at::Tensor>>& device_tensors,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism, int64_t node_id,
    bool enable_shm)
    : TorchKVCacheManager(UnpackLayers(device_tensors, unsafe_skip_buffer_lock),
                          local_port, host_blocks_to_allocate,
                          unsafe_skip_buffer_lock, parallelism, node_id,
                          /*local_control_port=*/-1,
                          /*max_blocks=*/0, /*num_slots=*/0,
                          /*timeout_s=*/120.0,
                          /*kv_caches=*/{}, enable_shm) {}

TorchKVCacheManager::TorchKVCacheManager(
    const std::vector<std::vector<xla::PjRtBuffer*>>& device_buffers,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism, int64_t node_id,
    bool enable_shm)
    : TorchKVCacheManager(UnpackLayers(device_buffers, unsafe_skip_buffer_lock),
                          local_port, host_blocks_to_allocate,
                          unsafe_skip_buffer_lock, parallelism, node_id,
                          /*local_control_port=*/-1,
                          /*max_blocks=*/0, /*num_slots=*/0,
                          /*timeout_s=*/120.0,
                          /*kv_caches=*/{}, enable_shm) {}

TorchKVCacheManager::TorchKVCacheManager(
    UnpackedLayers unpacked, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
    int parallelism, int64_t node_id, int64_t local_control_port,
    int64_t max_blocks, int64_t num_slots, double timeout_s,
    std::vector<at::Tensor> kv_caches, bool enable_shm)
    : KVCacheManagerWithTransfer(
          unpacked.buffers,
          unpacked.has_logical_metadata ? unpacked.logical_slice_byte_size : 0,
          unpacked.has_logical_metadata ? unpacked.logical_dimensions
                                        : std::vector<int64_t>{},
          unpacked.has_logical_metadata ? unpacked.logical_physical_size : 0,
          local_port, host_blocks_to_allocate, unsafe_skip_buffer_lock,
          parallelism,
          // The same resolution the KVCacheManagerWithTransfer base applies,
          // so the segment identity records the pool size actually allocated.
          CreateHostMemoryAllocator(
              unpacked.client, enable_shm,
              host_blocks_to_allocate.has_value()
                  ? static_cast<int64_t>(*host_blocks_to_allocate)
                  : num_slots * max_blocks,
              (unpacked.buffers.empty() || unpacked.buffers[0].empty() ||
               unpacked.buffers[0][0].buffer == nullptr)
                  ? 0
                  : unpacked.buffers[0][0]
                        .buffer->GetOnDeviceSizeInBytes()
                        .value_or(0)),

          node_id, local_control_port, max_blocks, num_slots, timeout_s),
      kv_caches_(std::move(kv_caches)),
      // Move the keep-alive refs in AFTER the base ctor has acquired the
      // buffers; they pin the materialized device buffers for our lifetime.
      buffer_refs_(std::move(unpacked.refs)) {}

TorchKVCacheManager::TorchKVCacheManager(
    const std::vector<at::Tensor>& kv_caches, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, bool unsafe_skip_buffer_lock, int parallelism,
    std::optional<int> listener_port, bool enable_shm)
    : TorchKVCacheManager(UnpackLayers(SingleShardLayers(kv_caches)),
                          /*local_port=*/std::nullopt,
                          /*host_blocks_to_allocate=*/std::nullopt,
                          unsafe_skip_buffer_lock, parallelism, node_id,
                          local_control_port, max_blocks, num_slots, timeout_s,
                          kv_caches, enable_shm) {
  if (listener_port) {
    listener_ =
        std::make_unique<kv_cache::KVCacheListener>(this, *listener_port);
  }
}

TorchKVCacheManager::TorchKVCacheManager(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    int64_t node_id, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, int parallelism)
    : KVCacheManagerWithTransfer(
          num_layers, num_shards, slice_byte_size, local_port,
          host_blocks_to_allocate, parallelism, node_id,
          /*local_control_port=*/-1,
          /*max_blocks=*/host_blocks_to_allocate.value_or(0),
          /*num_slots=*/0, /*timeout_s=*/120.0),
      kv_caches_({}) {}

TorchKVCacheManager::~TorchKVCacheManager() {
  // Drain external copies here rather than leaving it to ~KVCacheManagerBase.
  // The base destructor runs after this subobject has already been destroyed,
  // so any completion that reaches back into the derived manager would touch
  // freed memory.
  if (is_shared_memory_mapped()) {
    const absl::Status status = UnmapSharedMemory();
    if (!status.ok()) {
      LOG(ERROR) << "TorchKVCacheManager shared memory unmap failed: "
                 << status;
    }
  }
}

std::optional<int> TorchKVCacheManager::listener_port() const {
  if (listener_) {
    return listener_->listener_port();
  }
  return std::nullopt;
}

bool TorchKVCacheManager::is_listener_active() const {
  if (listener_) {
    return listener_->is_active();
  }
  return false;
}

std::string TorchKVCacheManager::transfer_address() const {
  auto port = local_port();
  if (!port.has_value()) return "";
  return FormatAddressWithPort(local_ip(), *port);
}

std::string TorchKVCacheManager::listener_address() const {
  auto port = listener_port();
  if (!port.has_value()) return "";
  return FormatAddressWithPort(local_ip(), *port);
}

absl::Status TorchKVCacheManager::PushRegisteredPlan(
    uint64_t uuid, const std::string& peer,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, int layer_idx, int parallelism) {
  if (peer.empty()) {
    return absl::InvalidArgumentError("peer must not be empty");
  }
  if (src_block_ids.empty()) {
    return absl::InvalidArgumentError("src_block_ids must not be empty");
  }
  if (src_block_ids.size() != dst_block_ids.size()) {
    return absl::InvalidArgumentError(
        "src_block_ids and dst_block_ids must have the same length");
  }
  InitTransportServer();
  // Copy the transport pointer and release server_init_mu_ before the blocking
  // Push: holding the lock across it serializes concurrent pushes from the
  // same manager (one per destination peer).
  transport::BlockTransport* transport = nullptr;
  {
    absl::MutexLock lock(server_init_mu_);
    transport = server_.get();
  }
  if (!transport) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  auto status_or =
      transport->SyncPush({peer}, src_block_ids, dst_block_ids, parallelism,
                          transport::MajorOrder::kLayerMajor, uuid, layer_idx);
  if (!status_or.ok()) {
    return status_or.status();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> TorchKVCacheManager::ReadBlockBytes(
    size_t layer_idx, int block_id, size_t shard_idx) {
  if (block_id < 0) {
    return absl::InvalidArgumentError("block_id must be non-negative");
  }
  const size_t block_bytes = this->block_bytes(layer_idx);
  const size_t host_size = GetHostSize(layer_idx, shard_idx);
  const uint8_t* base = GetHostPointer(layer_idx, shard_idx);
  if (base == nullptr) {
    return absl::OutOfRangeError("layer or shard index out of range");
  }
  const size_t block = static_cast<size_t>(block_id);
  if (block_bytes == 0 || block > host_size / block_bytes ||
      block * block_bytes + block_bytes > host_size) {
    return absl::OutOfRangeError("block range exceeds host buffer");
  }
  const char* ptr = reinterpret_cast<const char*>(base + block * block_bytes);
  return std::string(ptr, block_bytes);
}

absl::Status TorchKVCacheManager::WriteBlockBytes(size_t layer_idx,
                                                  int block_id,
                                                  absl::string_view payload,
                                                  size_t shard_idx) {
  if (block_id < 0) {
    return absl::InvalidArgumentError("block_id must be non-negative");
  }
  const size_t block_bytes = this->block_bytes(layer_idx);
  if (payload.size() != block_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("payload size must equal block size: got ", payload.size(),
                     ", expected ", block_bytes));
  }
  const size_t host_size = GetHostSize(layer_idx, shard_idx);
  uint8_t* base = GetHostPointer(layer_idx, shard_idx);
  if (base == nullptr) {
    return absl::OutOfRangeError("layer or shard index out of range");
  }
  const size_t block = static_cast<size_t>(block_id);
  if (block_bytes == 0 || block > host_size / block_bytes ||
      block * block_bytes + block_bytes > host_size) {
    return absl::OutOfRangeError("block range exceeds host buffer");
  }
  std::memcpy(base + block * block_bytes, payload.data(), block_bytes);
  return absl::OkStatus();
}

KVCacheManager::KVCacheManager(
    const std::vector<std::vector<at::Tensor>>& device_tensors,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism, int raiden_worker_port,
    std::optional<std::string> raiden_controller_address,
    std::optional<std::string> worker_id, int64_t node_id, bool enable_shm)
    : torch_manager_(std::make_unique<TorchKVCacheManager>(
          device_tensors, local_port, host_blocks_to_allocate,
          unsafe_skip_buffer_lock, parallelism, node_id, enable_shm)) {
  StartGrpcServer(raiden_worker_port, raiden_controller_address, worker_id);
}

KVCacheManager::KVCacheManager(
    const std::vector<std::vector<xla::PjRtBuffer*>>& device_buffers,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism, int raiden_worker_port,
    std::optional<std::string> raiden_controller_address,
    std::optional<std::string> worker_id, int64_t node_id, bool enable_shm)
    : torch_manager_(std::make_unique<TorchKVCacheManager>(
          device_buffers, local_port, host_blocks_to_allocate,
          unsafe_skip_buffer_lock, parallelism, node_id, enable_shm)) {
  StartGrpcServer(raiden_worker_port, raiden_controller_address, worker_id);
}

KVCacheManager::KVCacheManager(
    const std::vector<at::Tensor>& kv_caches, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, bool unsafe_skip_buffer_lock, int parallelism,
    std::optional<int> listener_port, int raiden_worker_port,
    std::optional<std::string> raiden_controller_address,
    std::optional<std::string> worker_id, bool enable_shm)
    : torch_manager_(std::make_unique<TorchKVCacheManager>(
          kv_caches, node_id, local_control_port, max_blocks, num_slots,
          timeout_s, unsafe_skip_buffer_lock, parallelism, listener_port,
          enable_shm)) {
  StartGrpcServer(raiden_worker_port, raiden_controller_address, worker_id);
}

KVCacheManager::KVCacheManager(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    int64_t node_id, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, int parallelism,
    int raiden_worker_port,
    std::optional<std::string> raiden_controller_address,
    std::optional<std::string> worker_id)
    : torch_manager_(std::make_unique<TorchKVCacheManager>(
          num_layers, num_shards, slice_byte_size, node_id, local_port,
          host_blocks_to_allocate, parallelism)) {
  StartGrpcServer(raiden_worker_port, raiden_controller_address, worker_id);
}

KVCacheManager::~KVCacheManager() {
  if (private_grpc_server_) {
    private_grpc_server_->SetTransferManager(nullptr);
  }
  controller::WorkerServiceServer::GetInstance().SetTransferManager(nullptr);
}

void KVCacheManager::StartGrpcServer(
    int raiden_worker_port,
    std::optional<std::string> raiden_controller_address,
    std::optional<std::string> worker_id) {
  if (!raiden_controller_address.has_value() ||
      raiden_controller_address->empty()) {
    return;
  }

  bool use_private_server = false;
  const char* disable_singleton =
      std::getenv("RAIDEN_DISABLE_SINGLETON_WORKER");
  if (disable_singleton != nullptr &&
      (std::strcmp(disable_singleton, "true") == 0 ||
       std::strcmp(disable_singleton, "1") == 0)) {
    use_private_server = true;
  }

  absl::Status status;
  if (use_private_server) {
    private_grpc_server_ = controller::WorkerServiceServer::Create();
    status = private_grpc_server_->StartServer(
        /*host_allocator=*/nullptr, KVManagerHolder(torch_manager_.get()),
        raiden_worker_port);
  } else {
    status = controller::WorkerServiceServer::GetInstance().StartServer(
        /*host_allocator=*/nullptr, KVManagerHolder(torch_manager_.get()),
        raiden_worker_port);
  }

  if (!status.ok()) {
    throw std::runtime_error(absl::StrCat(
        "Failed to start gRPC server in KVCacheManager: ", status.message()));
  }

  if (raiden_controller_address.has_value() &&
      !raiden_controller_address->empty()) {
    int bound_port = GetRaidenWorkerPort();
    std::string w_id = worker_id.value_or("worker_0");

    std::string worker_ip = "127.0.0.1";
    auto ips = GetLocalHostIpAddresses();
    if (!ips.empty()) {
      worker_ip = ips[0];
    }
    std::string worker_endpoint;
    if (absl::StrContains(worker_ip, ":")) {
      worker_endpoint = absl::StrCat("[", worker_ip, "]:", bound_port);
    } else {
      worker_endpoint = absl::StrCat(worker_ip, ":", bound_port);
    }

    // Registry endpoints are what remote peers open DATA connections to (a
    // pull's H2hRead/H2dRead, a push's H2hWrite), so they must be data
    // endpoints even when this manager also runs a control server --
    // get_local_endpoints() would hand back the CONTROL port in that case, and
    // aiming a data-protocol connection at it hangs both sides with no error.
    std::string transfer_endpoint = "";
    auto local_eps = torch_manager_->get_local_data_endpoints();
    if (!local_eps.empty()) {
      transfer_endpoint = local_eps[0].endpoint;
    }

    std::vector<uint64_t> block_array_bytes;
    block_array_bytes.reserve(torch_manager_->num_block_arrays());
    for (size_t i = 0; i < torch_manager_->num_block_arrays(); ++i) {
      block_array_bytes.push_back(torch_manager_->block_bytes(i));
    }

    core::controller::RaidenControllerClient client(*raiden_controller_address);
    status = client.RegisterWorker(
        w_id, worker_endpoint, local_eps, torch_manager_->node_id(),
        block_array_bytes, static_cast<int32_t>(torch_manager_->num_shards()));
    if (!status.ok()) {
      LOG(ERROR) << "Failed to register worker with controller: "
                 << status.message();
    } else {
      LOG(INFO) << "Successfully registered worker " << w_id
                << " (worker_endpoint=" << worker_endpoint
                << ", transfer_endpoint=" << transfer_endpoint
                << ") with controller at " << *raiden_controller_address;
    }
  }
}

int KVCacheManager::GetRaidenWorkerPort() const {
  if (private_grpc_server_) {
    return private_grpc_server_->GetRaidenWorkerPort();
  }
  return controller::WorkerServiceServer::GetInstance().GetRaidenWorkerPort();
}

absl::StatusOr<raiden::PjRtCopyFuture> TorchKVCacheManager::H2d(
    const std::vector<int64_t>& block_ids,
    const std::vector<at::Tensor>& object_tensors, int64_t rank_id) {
  return CopyObjectBlocks(block_ids, object_tensors, rank_id, /*is_h2d=*/true);
}

absl::StatusOr<raiden::PjRtCopyFuture> TorchKVCacheManager::D2h(
    const std::vector<int64_t>& block_ids,
    const std::vector<at::Tensor>& object_tensors, int64_t rank_id) {
  return CopyObjectBlocks(block_ids, object_tensors, rank_id, /*is_h2d=*/false);
}

absl::StatusOr<raiden::PjRtCopyFuture> TorchKVCacheManager::CopyObjectBlocks(
    const std::vector<int64_t>& block_ids,
    const std::vector<at::Tensor>& object_tensors, int64_t rank_id,
    bool is_h2d) {
  if (buffer_holds_.empty()) {
    return absl::FailedPreconditionError(
        "KVCacheManager has no registered device KV cache");
  }
  if (block_ids.empty()) {
    return absl::InvalidArgumentError("block_ids must not be empty");
  }
  if (block_ids.size() != object_tensors.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "block_ids and object_tensors must have the same length; got ",
        block_ids.size(), " and ", object_tensors.size()));
  }
  if (rank_id < 0) {
    return absl::InvalidArgumentError("rank_id must be non-negative");
  }

  const size_t num_layers = num_layers_;
  // buffer_holds_ is only populated on the device-backed path, and nothing
  // guarantees it has one entry per layer; the submit loop indexes it with
  // layer_id, so check the length up front.
  if (buffer_holds_.size() < num_layers) {
    return absl::FailedPreconditionError(absl::StrCat(
        "KVCacheManager has ", buffer_holds_.size(),
        " registered layer buffers but num_layers=", num_layers));
  }
  if (object_tensors[0].dim() != 3 ||
      object_tensors[0].scalar_type() != at::kChar) {
    return absl::InvalidArgumentError(
        "object_tensors must be rank-3 CPU int8 tensors with shape "
        "[num_ranks, num_layers, page_nbytes]");
  }

  const int64_t page_nbytes_int = object_tensors[0].size(2);
  if (page_nbytes_int <= 0) {
    return absl::InvalidArgumentError("page_nbytes must be greater than zero");
  }
  const size_t page_nbytes = static_cast<size_t>(page_nbytes_int);
  if (slice_byte_size() > 0 && page_nbytes != slice_byte_size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("object_tensors page_nbytes=", page_nbytes,
                     " does not match manager slice_byte_size=",
                     slice_byte_size()));
  }

  const size_t dev_physical_size = buffer_holds_[0].physical_size;
  if (dev_physical_size == 0 || dev_physical_size % page_nbytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "device physical size ", dev_physical_size,
        " is not divisible by page_nbytes ", page_nbytes));
  }
  const size_t num_blocks = dev_physical_size / page_nbytes;

  // MapSharedMemory registered the pool against a single client, the one
  // GetPjRtClient() resolves to.  DmaMap registration is per client, so a
  // layer whose buffer belongs to a different client would hand the DMA
  // engine host memory that client never registered.  Devices may still
  // differ: the registration covers the client, not one device.
  xla::PjRtClient* const dma_client = GetPjRtClient();
  if (dma_client == nullptr) {
    return absl::FailedPreconditionError(
        "KVCacheManager has no active PJRT client for object tensor "
        "transfers");
  }

  // The transfer loop below addresses every layer with the same page size and
  // uses each layer's single shard.  Reject the geometries that would break
  // that assumption rather than silently reading the wrong bytes.
  for (size_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    const auto& layer_info = buffer_holds_[layer_id];
    if (layer_info.holds.size() != 1) {
      return absl::UnimplementedError(absl::StrCat(
          "object tensor transfers require exactly one shard per layer; layer ",
          layer_id, " has ", layer_info.holds.size(), " shards"));
    }
    if (layer_info.physical_size != dev_physical_size) {
      return absl::UnimplementedError(absl::StrCat(
          "object tensor transfers require a uniform per-layer device size; "
          "layer ",
          layer_id, " is ", layer_info.physical_size, " bytes but layer 0 is ",
          dev_physical_size, " bytes"));
    }
    // Resolve this layer's client the same way GetPjRtClient() does, so the
    // comparison is against the client the mapping was actually made on.
    const auto& hold = layer_info.holds[0];
    xla::PjRtClient* layer_client =
        hold.device != nullptr ? hold.device->client() : nullptr;
    if (layer_client == nullptr && hold.buffer != nullptr) {
      layer_client = hold.buffer->client();
    }
    if (layer_client == nullptr) {
      return absl::FailedPreconditionError(absl::StrCat(
          "layer ", layer_id, " has no resolvable PJRT client"));
    }
    if (layer_client != dma_client) {
      return absl::UnimplementedError(absl::StrCat(
          "object tensor transfers require every layer to share one PJRT "
          "client; layer ",
          layer_id, " belongs to a different client than the one the shared "
          "memory pool is DMA mapped on"));
    }
  }

  absl::flat_hash_set<int64_t> unique_blocks;
  unique_blocks.reserve(block_ids.size());
  std::vector<int64_t> device_offsets;
  device_offsets.reserve(block_ids.size());
  for (size_t i = 0; i < block_ids.size(); ++i) {
    const int64_t b = block_ids[i];
    if (b < 0 || static_cast<size_t>(b) >= num_blocks) {
      return absl::OutOfRangeError(
          absl::StrCat("block_ids[", i, "]=", b,
                       " is outside block range [0, ", num_blocks, ")"));
    }
    if (!unique_blocks.insert(b).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "block_ids contains duplicate block ", b,
          "; concurrent writes to one block have undefined ordering"));
    }
    device_offsets.push_back(b * static_cast<int64_t>(page_nbytes));
  }

  const size_t rank_bytes = num_layers * page_nbytes;
  size_t expected_num_ranks = 0;
  size_t expected_object_bytes = 0;
  std::vector<uint8_t*> host_bases;
  host_bases.reserve(object_tensors.size());

  for (size_t obj_id = 0; obj_id < object_tensors.size(); ++obj_id) {
    const at::Tensor& tensor = object_tensors[obj_id];
    if (!tensor.device().is_cpu()) {
      return absl::InvalidArgumentError(
          absl::StrCat("object_tensors[", obj_id, "] must be a CPU tensor"));
    }
    if (!tensor.is_contiguous()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "object_tensors[", obj_id,
          "] must be contiguous; a nonzero storage offset is allowed"));
    }
    if (tensor.dim() != 3 || tensor.scalar_type() != at::kChar) {
      return absl::InvalidArgumentError(absl::StrCat(
          "object_tensors[", obj_id,
          "] must be a rank-3 CPU int8 tensor with shape [num_ranks, ",
          num_layers, ", ", page_nbytes, "]"));
    }
    if (tensor.size(0) <= 0 ||
        tensor.size(1) != static_cast<int64_t>(num_layers) ||
        tensor.size(2) != static_cast<int64_t>(page_nbytes)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "object_tensors[", obj_id, "] must have shape [num_ranks, ",
          num_layers, ", ", page_nbytes, "] with num_ranks > 0"));
    }

    const size_t num_ranks = static_cast<size_t>(tensor.size(0));
    if (obj_id == 0) {
      expected_num_ranks = num_ranks;
      expected_object_bytes = num_ranks * rank_bytes;
    } else if (num_ranks != expected_num_ranks) {
      return absl::InvalidArgumentError(
          "all object_tensors must have the same num_ranks dimension");
    }
    if (static_cast<size_t>(rank_id) >= num_ranks) {
      return absl::OutOfRangeError(
          absl::StrCat("rank_id=", rank_id, " is outside object_tensors[",
                       obj_id, "] first dimension [0, ", num_ranks, ")"));
    }
    if (static_cast<size_t>(tensor.nbytes()) != expected_object_bytes) {
      return absl::InvalidArgumentError(
          absl::StrCat("object_tensors[", obj_id, "] has ", tensor.nbytes(),
                       " bytes, expected ", expected_object_bytes));
    }

    host_bases.push_back(static_cast<uint8_t*>(tensor.data_ptr()));
  }

  const size_t host_rank_offset = static_cast<size_t>(rank_id) * rank_bytes;
  std::vector<size_t> host_layer_offsets;
  host_layer_offsets.reserve(num_layers);
  for (size_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    host_layer_offsets.push_back(host_rank_offset + layer_id * page_nbytes);
  }

  // Pin the DMA registration for the rest of this call.  Checking
  // is_shared_memory_mapped() here instead would be racy: the pool could be
  // unmapped between the check and IssueH2dShard below.
  absl::StatusOr<ExternalCopyLease> lease = AcquireExternalCopyLease();
  if (!lease.ok()) {
    return lease.status();
  }

  // Every byte handed to the DMA engine must lie inside the pool that was
  // registered with DmaMap.  Each tensor contributes one contiguous run: the
  // `num_layers` pages belonging to `rank_id`.
  for (size_t obj_id = 0; obj_id < host_bases.size(); ++obj_id) {
    const absl::Status range_status =
        ValidateExternalRange(host_bases[obj_id] + host_rank_offset,
                              rank_bytes);
    if (!range_status.ok()) {
      return absl::Status(
          range_status.code(),
          absl::StrCat("object_tensors[", obj_id, "] is not DMA mappable: ",
                       range_status.message()));
    }
  }

  auto tensor_holds = std::make_shared<std::vector<at::Tensor>>(object_tensors);
  std::vector<raiden::PjRtCopyFuture> layer_futures;
  layer_futures.reserve(num_layers);

  const int64_t transfer_size = static_cast<int64_t>(page_nbytes);
  for (size_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    const size_t host_offset = host_layer_offsets[layer_id];
    const auto& shard_hold = buffer_holds_[layer_id].holds[0];

    absl::StatusOr<raiden::PjRtCopyFuture> future;
    if (is_h2d) {
      std::vector<raiden::H2dCopy> copies;
      copies.reserve(object_tensors.size());
      for (size_t obj_id = 0; obj_id < object_tensors.size(); ++obj_id) {
        copies.push_back(raiden::H2dCopy{
            .src = host_bases[obj_id] + host_offset,
            .dst_off = device_offsets[obj_id],
            .size = transfer_size,
        });
      }
      future = raiden::IssueH2dShard(shard_hold, copies);
    } else {
      std::vector<raiden::D2hCopy> copies;
      copies.reserve(object_tensors.size());
      for (size_t obj_id = 0; obj_id < object_tensors.size(); ++obj_id) {
        copies.push_back(raiden::D2hCopy{
            .dst = host_bases[obj_id] + host_offset,
            .src_off = device_offsets[obj_id],
            .size = transfer_size,
        });
      }
      future = raiden::IssueD2hShard(shard_hold, copies);
    }

    if (!future.ok()) {
      if (!layer_futures.empty()) {
        // Layers [0, layer_id) are already in flight and still reference the
        // caller's tensors; hand them to the lease so the mapping outlives
        // them even though the overall transfer failed.
        raiden::PjRtCopyFuture submitted =
            raiden::JoinPjRtCopyFutures(layer_futures);
        submitted.AddKeepAlive(tensor_holds);
        lease->Commit(std::move(submitted));
      }
      return future.status();
    }
    layer_futures.push_back(std::move(future.value()));
  }

  raiden::PjRtCopyFuture joined = raiden::JoinPjRtCopyFutures(layer_futures);
  joined.AddKeepAlive(std::move(tensor_holds));
  lease->Commit(joined);
  return joined;
}

}  // namespace torch
}  // namespace tpu_raiden
