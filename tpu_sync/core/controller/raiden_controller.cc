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

#include "tpu_sync/core/controller/raiden_controller.h"

// Dummy comment to trigger Copybara export and Kokoro presubmit.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/core/buffer.h"
#include "tpu_sync/core/controller/controller_server.h"
#include "tpu_sync/core/controller/worker_registry.h"
#include "tpu_sync/core/controller/worker_service_client.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/kv_cache/logical_block_manager.h"
#include "tpu_sync/proto/controller_service.grpc.pb.h"
#include "tpu_sync/proto/controller_service.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

bool CompareWorkerIds(absl::string_view a, absl::string_view b) {
  absl::string_view sv_a = a;
  absl::string_view sv_b = b;
  if (absl::ConsumePrefix(&sv_a, "worker_") &&
      absl::ConsumePrefix(&sv_b, "worker_")) {
    int id_a, id_b;
    if (absl::SimpleAtoi(sv_a, &id_a) && absl::SimpleAtoi(sv_b, &id_b)) {
      return id_a < id_b;
    }
  }
  return a < b;
}

absl::StatusOr<::tpu_sync::proto::CreateBuffersResponse> CreateBuffersForWorker(
    WorkerServiceClient& client,
    const ::tpu_sync::proto::CreateBuffersRequest& request,
    absl::string_view worker_id, int num_blocks) {
  auto resp_or = client.CreateBuffers(request).Await();
  if (!resp_or.ok()) {
    return absl::Status(
        resp_or.status().code(),
        absl::StrCat("Failed to create buffers on worker ", worker_id, ": ",
                     resp_or.status().message()));
  }
  if (!resp_or->success()) {
    return absl::FailedPreconditionError(
        absl::StrCat("WorkerService CreateBuffers failed on worker ", worker_id,
                     ": ", resp_or->message()));
  }
  if (resp_or->buffers_size() != num_blocks) {
    return absl::FailedPreconditionError(absl::StrCat(
        "WorkerService returned unexpected number of buffers on ", worker_id,
        ": ", resp_or->buffers_size(), " vs expected ", num_blocks));
  }
  return *resp_or;
}

// How long the destination waits for transfer + release before giving up and
// discarding. RAIDEN_REMOTE_READ_DEADLINE_S; defaults to the lease TTL plus a
// margin so that a pull outliving its lease still gets to release and LEARN
// that it was revoked. Must sit between the TTL and the source's revoked-
// lease retention (see controller_service.h).
absl::Duration RemoteReadDeadline() {
  static const absl::Duration kDeadline = [] {
    const char* raw = std::getenv("RAIDEN_REMOTE_READ_DEADLINE_S");
    // Derived from the TTL so overriding only RAIDEN_REMOTE_LOCK_TTL_S keeps a
    // coherent triple (retention > deadline > ttl); an explicit env var wins.
    absl::Duration fallback =
        core::controller::RaidenControllerServiceImpl::DefaultLeaseTtl() +
        absl::Seconds(30);
    if (raw == nullptr) return fallback;
    int64_t seconds = 0;
    if (!absl::SimpleAtoi(raw, &seconds) || seconds <= 0) {
      LOG(ERROR) << "RAIDEN_REMOTE_READ_DEADLINE_S=\"" << raw
                 << "\" is not a positive integer number of seconds; using "
                 << fallback;
      return fallback;
    }
    return absl::Seconds(seconds);
  }();
  return kDeadline;
}

// Per-phase tracing for a remote read. Off unless RAIDEN_LEASE_TRACE is set:
// a stalled read is otherwise invisible (the phases are all async callbacks,
// and a hang looks identical to a slow peer), and this is the first thing
// anyone debugging one will want.
bool LeaseTraceEnabled() {
  static const bool on = std::getenv("RAIDEN_LEASE_TRACE") != nullptr;
  return on;
}

#define RAIDEN_LEASE_TRACE(...)                                  \
  do {                                                           \
    if (LeaseTraceEnabled()) {                                   \
      LOG(INFO) << "RAIDEN_LEASE " << absl::StrCat(__VA_ARGS__); \
      std::fprintf(stderr, "RAIDEN_LEASE %s\n",                  \
                   absl::StrCat(__VA_ARGS__).c_str());           \
    }                                                            \
  } while (0)

// Validates the timing triple once per process. An incoherent set does not
// corrupt anything -- the commit gate is clock-free -- but it silently
// degrades every read (e.g. a deadline below the TTL means the TTL only ever
// fires for dead destinations, wasting the renewal path and hiding
// revocations), so say so loudly rather than let it pass.
void CheckTimingTripleOnce() {
  static const bool checked = [] {
    const absl::Duration ttl = ::tpu_raiden::core::controller::
        RaidenControllerServiceImpl::DefaultLeaseTtl();
    const absl::Duration deadline = RemoteReadDeadline();
    const absl::Duration retention = ::tpu_raiden::core::controller::
        RaidenControllerServiceImpl::RevokedLeaseRetention();
    if (!(retention > deadline && deadline > ttl)) {
      LOG(ERROR) << "Incoherent remote-read timing: expected retention > "
                    "deadline > ttl, got retention="
                 << retention << " deadline=" << deadline << " ttl=" << ttl
                 << ". Reads may report UNKNOWN instead of REVOKED, or be "
                    "abandoned while their lease is still valid.";
    }
    return true;
  }();
  (void)checked;
}

}  // namespace

absl::Status RaidenController::Init(
    absl::Span<const std::string> worker_addresses,
    absl::string_view raiden_controller_address, int expected_worker_count) {
  // On failure the caller (Create) drops the unique_ptr, so ~RaidenController
  // runs Cleanup() and detaches the registered callbacks and singleton
  // references.

  // Publish ourselves to in-flight reads (cleared in the destructor / Cleanup).
  {
    absl::MutexLock lock(&lifetime_->mu);
    lifetime_->ctrl = this;
  }
  // 1. Set callback on registry
  worker_registry_->SetOnRegisterCallback(
      [this](core::controller::WorkerRegistration& reg) {
        return this->InitializeWorkerBuffers(reg);
      });

  // 2. Start ControllerServer
  bool use_private_server = false;
  const char* disable_singleton =
      std::getenv("RAIDEN_DISABLE_SINGLETON_WORKER");
  if (disable_singleton != nullptr &&
      (std::strcmp(disable_singleton, "true") == 0 ||
       std::strcmp(disable_singleton, "1") == 0)) {
    use_private_server = true;
  }

  absl::Status server_status;
  core::controller::ControllerServer* server_ptr = nullptr;
  if (use_private_server) {
    private_controller_server_ = core::controller::ControllerServer::Create();
    server_status = private_controller_server_->StartServer(
        worker_registry_, raiden_controller_address);
    server_ptr = private_controller_server_.get();
  } else {
    auto& server = core::controller::ControllerServer::GetInstance();
    server_status =
        server.StartServer(worker_registry_, raiden_controller_address);
    server_ptr = &server;
  }

  if (!server_status.ok()) {
    LOG(WARNING) << "Failed to start ControllerServer in RaidenController: "
                 << server_status.message();
  }

  int actual_port = server_ptr->GetGrpcPort();
  if (raiden_controller_address.empty()) {
    raiden_controller_address_ = absl::StrCat("[::]:", actual_port);
  } else {
    absl::string_view host = raiden_controller_address;
    size_t closing_bracket = host.rfind(']');
    if (closing_bracket != absl::string_view::npos) {
      host = host.substr(0, closing_bracket + 1);
    } else {
      size_t colon_pos = host.rfind(':');
      if (colon_pos != absl::string_view::npos) {
        host = host.substr(0, colon_pos);
      }
    }
    raiden_controller_address_ = absl::StrCat(host, ":", actual_port);
  }

  active_server_ = server_ptr;

  for (size_t i = 0; i < worker_addresses.size(); ++i) {
    std::string worker_id = absl::StrCat("worker_", i);
    absl::Status status = worker_registry_->RegisterWorker(
        worker_id, worker_addresses[i], /*raiden_transfer_endpoints=*/{});
    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat("Failed to register static worker ", worker_addresses[i],
                       ": ", status.message()));
    }
  }

  // 4. Wait for expected workers to register (if requested)
  if (expected_worker_count > 0) {
    int timeout_s = 120;
    if (const char* env_timeout =
            std::getenv("RAIDEN_EXPECTED_WORKERS_TIMEOUT_S")) {
      int parsed = 0;
      if (absl::SimpleAtoi(env_timeout, &parsed) && parsed > 0) {
        timeout_s = parsed;
      }
    }
    if (!worker_registry_->AwaitWorkerCount(
            static_cast<size_t>(expected_worker_count),
            absl::Seconds(timeout_s))) {
      size_t actual = worker_registry_->GetRegisteredWorkers().size();
      return absl::DeadlineExceededError(absl::StrCat(
          "RaidenController timed out waiting for workers: expected ",
          expected_worker_count, " worker(s) within ", timeout_s, "s, got ",
          actual));
    }
  }

  return absl::OkStatus();
}

RaidenController::RaidenController(const ::tpu_sync::rpc::RaidenIdProto& unit,
                                   int num_blocks, int num_shards,
                                   int64_t shard_size_bytes,
                                   bool preprovision_worker_buffers)
    : unit_(unit),
      num_shards_(num_shards),
      shard_size_bytes_(shard_size_bytes),
      num_total_blocks_(num_blocks),
      preprovision_worker_buffers_(preprovision_worker_buffers),
      worker_registry_(std::make_shared<core::controller::WorkerRegistry>()),
      block_manager_(
          std::make_unique<kv_cache::LogicalBlockManager>(num_blocks)) {}

absl::StatusOr<std::unique_ptr<RaidenController>> RaidenController::Create(
    const ::tpu_sync::rpc::RaidenIdProto& unit, int num_blocks, int num_shards,
    int64_t shard_size_bytes, absl::string_view raiden_controller_address,
    bool preprovision_worker_buffers, int expected_worker_count) {
  return Create(unit, /*worker_addresses=*/{}, num_blocks, num_shards,
                shard_size_bytes, raiden_controller_address,
                preprovision_worker_buffers, expected_worker_count);
}

absl::StatusOr<std::unique_ptr<RaidenController>> RaidenController::Create(
    const ::tpu_sync::rpc::RaidenIdProto& unit,
    absl::Span<const std::string> worker_addresses, int num_blocks,
    int num_shards, int64_t shard_size_bytes,
    absl::string_view raiden_controller_address,
    bool preprovision_worker_buffers, int expected_worker_count) {
  auto controller = absl::WrapUnique(
      new RaidenController(unit, num_blocks, num_shards, shard_size_bytes,
                           preprovision_worker_buffers));
  absl::Status status = controller->Init(
      worker_addresses, raiden_controller_address, expected_worker_count);
  if (!status.ok()) {
    return status;
  }
  return controller;
}

void RaidenController::Cleanup() {
  // Detach in-flight reads FIRST. A ReadRemote continuation holds lifetime_->mu
  // for as long as it touches this controller, so after this block either the
  // continuation has finished or it will see a null ctrl and fail the read
  // cleanly -- never a use-after-free.
  {
    absl::MutexLock lock(&lifetime_->mu);
    lifetime_->ctrl = nullptr;
  }
  if (worker_registry_) {
    worker_registry_->SetOnRegisterCallback(nullptr);
  }
  if (private_controller_server_) {
    private_controller_server_->SetWorkerRegistry(nullptr);
    private_controller_server_.reset();
  } else {
    auto& server = core::controller::ControllerServer::GetInstance();
    server.SetWorkerRegistry(nullptr);
  }

  if (all_sharded_buffers_.empty() || !worker_registry_) return;

  ::tpu_sync::proto::DeleteBuffersRequest request;
  *request.mutable_unit() = unit_;
  for (const auto& sharded_buf : all_sharded_buffers_) {
    *request.add_sharded_buffers() = sharded_buf;
  }

  for (const auto& reg : worker_registry_->GetRegisteredWorkers()) {
    if (reg.worker_service_client) {
      auto resp_or = reg.worker_service_client->DeleteBuffers(request).Await();
      if (!resp_or.ok()) {
        LOG(ERROR) << "Failed to delete buffers on worker " << reg.worker_id
                   << ": " << resp_or.status().message();
      } else if (!resp_or->success()) {
        LOG(ERROR) << "WorkerService DeleteBuffers failed on worker "
                   << reg.worker_id << ": " << resp_or->message();
      }
    }
  }
}

RaidenController::~RaidenController() { Cleanup(); }

absl::Status RaidenController::InitializeWorkerBuffers(
    core::controller::WorkerRegistration& reg) {
  absl::MutexLock lock(mutex_);

  if (!reg.worker_service_client) {
    return absl::FailedPreconditionError(absl::StrCat(
        "WorkerServiceClient for ", reg.worker_id, " is not initialized"));
  }

  // With pre-provisioning disabled the request is empty: the RPC still runs,
  // so registration keeps probing that the worker's WorkerService is
  // reachable, but no physical buffers are created on the worker.
  const int provision_blocks =
      preprovision_worker_buffers_ ? num_total_blocks_ : 0;
  ::tpu_sync::proto::CreateBuffersRequest request;
  *request.mutable_unit() = unit_;
  for (int block_id = 0; block_id < provision_blocks; ++block_id) {
    auto* spec = request.add_buffers();
    spec->set_num_shards(num_shards_);
    spec->set_size_bytes(shard_size_bytes_);
  }

  auto resp_or = CreateBuffersForWorker(*reg.worker_service_client, request,
                                        reg.worker_id, provision_blocks);
  if (!resp_or.ok()) {
    return absl::FailedPreconditionError(
        absl::StrCat("CreateBuffers RPC failed: ", resp_or.status().message()));
  }
  auto resp = *resp_or;

  std::vector<::tpu_sync::proto::BufferProto> created_buffers;
  created_buffers.reserve(provision_blocks);
  for (int block_id = 0; block_id < provision_blocks; ++block_id) {
    ::tpu_sync::proto::BufferProto buf = resp.buffers(block_id);
    buf.set_index(block_id);
    created_buffers.push_back(std::move(buf));
  }

  reg.buffers = std::move(created_buffers);

  if (all_sharded_buffers_.empty()) {
    all_sharded_buffers_ = reg.buffers;
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<::tpu_sync::proto::BufferProto>>
RaidenController::Allocate(int num_blocks) {
  absl::MutexLock lock(mutex_);
  // Without the pre-provisioned buffers (no worker registered yet, or
  // preprovision_worker_buffers=false) indexing all_sharded_buffers_ below
  // would be out of bounds -- fail before locking any ids.
  if (all_sharded_buffers_.empty()) {
    return absl::FailedPreconditionError(
        "No physical buffers provisioned: register a worker first, and note "
        "that Physical/BufferProto mode is unavailable when the controller "
        "was built with preprovision_worker_buffers=false");
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<int> block_ids,
                        block_manager_->Allocate(num_blocks, /*lock=*/true));
  std::vector<::tpu_sync::proto::BufferProto> result;
  result.reserve(num_blocks);
  for (int block_id : block_ids) {
    result.push_back(all_sharded_buffers_[block_id]);
  }
  return result;
}

absl::StatusOr<std::vector<Buffer>> RaidenController::AllocateBuffers(
    int num_blocks) {
  ABSL_ASSIGN_OR_RETURN(std::vector<::tpu_sync::proto::BufferProto> protos,
                        Allocate(num_blocks));
  std::vector<Buffer> buffers;
  buffers.reserve(protos.size());
  for (const auto& proto : protos) {
    Buffer buf = Buffer::FromProto(proto);
    buf.set_memory_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
    buffers.push_back(std::move(buf));
  }
  return buffers;
}

absl::Status RaidenController::Deallocate(
    absl::Span<const ::tpu_sync::proto::BufferProto> sharded_buffers) {
  std::vector<int> block_ids;
  block_ids.reserve(sharded_buffers.size());
  for (const auto& sharded_buf : sharded_buffers) {
    if (!sharded_buf.has_index() || sharded_buf.index() < 0) {
      return absl::InvalidArgumentError(
          "BufferProto has invalid or missing index");
    }
    block_ids.push_back(sharded_buf.index());
  }
  absl::MutexLock lock(mutex_);
  return block_manager_->Deallocate(block_ids);
}

absl::StatusOr<std::vector<int>> RaidenController::AllocateBlockIds(
    int num_blocks) {
  absl::MutexLock lock(mutex_);
  return block_manager_->Allocate(num_blocks, /*lock=*/true);
}

absl::Status RaidenController::DeallocateBlockIds(
    absl::Span<const int> block_ids) {
  absl::MutexLock lock(mutex_);
  return block_manager_->Deallocate(block_ids);
}

absl::Status RaidenController::AllocateTargetBlockIds(
    absl::Span<const int> block_ids) {
  absl::MutexLock lock(mutex_);
  return block_manager_->AllocateTarget(block_ids);
}

absl::Status RaidenController::DeallocateBuffers(
    absl::Span<const Buffer> buffers) {
  std::vector<::tpu_sync::proto::BufferProto> protos;
  protos.reserve(buffers.size());
  for (const auto& buf : buffers) {
    protos.push_back(buf.ToProto());
  }
  return Deallocate(protos);
}

absl::StatusOr<::tpu_sync::proto::TransferBuffersRequest>
RaidenController::BuildTransferBuffersRequest(
    absl::Span<const Buffer> src_buffers, absl::Span<const Buffer> dst_buffers,
    absl::Span<const Buffer> staging_host_buffers,
    absl::Span<const int64_t> copy_sizes) {
  if (src_buffers.empty() || src_buffers.size() != dst_buffers.size()) {
    return absl::InvalidArgumentError(
        "Source and destination buffers must have the same non-zero length");
  }
  if (!copy_sizes.empty() && copy_sizes.size() != src_buffers.size()) {
    return absl::InvalidArgumentError(
        "copy_sizes, if provided, must match the length of src_buffers");
  }

  ::tpu_sync::rpc::MemoryType src_mem_type = src_buffers[0].memory_type();
  ::tpu_sync::rpc::MemoryType dst_mem_type = dst_buffers[0].memory_type();

  ::tpu_sync::proto::TransferBuffersRequest request;
  auto* transfer = request.mutable_transfer();
  transfer->set_src_mem_type(src_mem_type);
  transfer->set_dst_mem_type(dst_mem_type);

  for (const auto& buf : src_buffers) {
    if (buf.index() < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Source buffer has invalid negative index: ", buf.index()));
    }
    auto* added_buf = transfer->add_src_buffers();
    *added_buf = buf.ToProto();
    added_buf->set_index(buf.index());
  }
  for (const auto& buf : dst_buffers) {
    if (buf.index() < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Destination buffer has invalid negative index: ", buf.index()));
    }
    auto* added_buf = transfer->add_dst_buffers();
    *added_buf = buf.ToProto();
    added_buf->set_index(buf.index());
  }
  for (int64_t size : copy_sizes) {
    transfer->add_copy_sizes(size);
  }
  for (const auto& buf : staging_host_buffers) {
    if (buf.index() < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Staging host buffer has invalid negative index: ", buf.index()));
    }
    auto* added_buf = transfer->add_staging_host_buffers();
    *added_buf = buf.ToProto();
    added_buf->set_index(buf.index());
  }

  return request;
}

tsl::Future<> RaidenController::TransferBuffers(
    absl::string_view worker_id, absl::Span<const Buffer> src_buffers,
    absl::Span<const Buffer> dst_buffers,
    absl::Span<const Buffer> staging_host_buffers,
    absl::Span<const int64_t> copy_sizes) {
  auto request_or = BuildTransferBuffersRequest(
      src_buffers, dst_buffers, staging_host_buffers, copy_sizes);
  if (!request_or.ok()) {
    return tsl::Future<>(request_or.status());
  }

  auto worker_or = worker_registry_->GetWorker(worker_id);
  if (!worker_or.ok()) {
    return tsl::Future<>(absl::FailedPreconditionError(
        absl::StrCat("Worker ", worker_id, " is not registered. ",
                     "Did you wait for the worker to register?")));
  }
  auto worker_client = worker_or->worker_service_client;
  if (!worker_client) {
    return tsl::Future<>(absl::FailedPreconditionError(absl::StrCat(
        "WorkerServiceClient for ", worker_id, " is not initialized.")));
  }

  return worker_client->TransferBuffers(*request_or);
}

tsl::Future<> RaidenController::TransferBuffers(
    absl::Span<const Buffer> src_buffers, absl::Span<const Buffer> dst_buffers,
    absl::Span<const Buffer> staging_host_buffers,
    absl::Span<const int64_t> copy_sizes) {
  if (src_buffers.empty() || src_buffers.size() != dst_buffers.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(
        "Source and destination buffers must have the same non-zero length"));
  }
  if (!copy_sizes.empty() && copy_sizes.size() != src_buffers.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(
        "copy_sizes, if provided, must match the length of src_buffers"));
  }

  // 1. Identify if a transfer is crossing node boundaries (remote vs local).
  //    A buffer is considered remote if it has remote worker endpoints
  //    or a remote address populated. We check both source and destination
  //    buffers because a transfer could act as a push (remote destination)
  //    or pull (remote source).
  bool is_src_remote = false;
  bool is_dst_remote = false;
  for (const auto& buf : src_buffers) {
    if (!buf.remote_worker_endpoints().empty() ||
        buf.remote_address().has_value()) {
      is_src_remote = true;
      break;
    }
  }
  for (const auto& buf : dst_buffers) {
    if (!buf.remote_worker_endpoints().empty() ||
        buf.remote_address().has_value()) {
      is_dst_remote = true;
      break;
    }
  }
  bool is_remote = is_src_remote || is_dst_remote;

  std::optional<std::vector<int>> auto_allocated_staging_ids;
  std::vector<Buffer> local_staging_buffers;
  absl::Span<const Buffer> request_staging = staging_host_buffers;

  if (is_remote) {
    bool src_is_hbm =
        src_buffers.front().memory_type() == ::tpu_sync::rpc::MEMORY_TYPE_HBM;
    bool dst_is_hbm =
        dst_buffers.front().memory_type() == ::tpu_sync::rpc::MEMORY_TYPE_HBM;

    // 2. Determine if a staging host block is required.
    //    For cross-node transfers involving TPU HBM (High Bandwidth Memory),
    //    direct HBM-to-HBM transfers across the network are not supported.
    //    Thus, the node must stage data in its host DRAM before sending or
    //    after receiving.
    //    - If the source is local and resides in HBM, or the destination is
    //    local
    //      and resides in HBM, we need a local host staging buffer.
    //    - Conversely, if the source is remote and in HBM, or the destination
    //    is
    //      remote and in HBM, the remote node will need a host staging buffer.
    bool local_host_staging_required =
        (!is_src_remote && src_is_hbm) || (!is_dst_remote && dst_is_hbm);
    bool remote_host_staging_required =
        (is_src_remote && src_is_hbm) || (is_dst_remote && dst_is_hbm);

    if (local_host_staging_required && request_staging.empty()) {
      // 3a. Safeguard logic: Auto-allocating local staging blocks.
      //     If local staging blocks are required but none were explicitly
      //     provided, we can safely auto-allocate them from the controller's
      //     pool because this controller manages the local resources.
      auto host_blocks_or = this->AllocateBlockIds(src_buffers.size());
      if (!host_blocks_or.ok()) {
        return tsl::Future<>(host_blocks_or.status());
      }
      auto_allocated_staging_ids = *host_blocks_or;
      local_staging_buffers.reserve(auto_allocated_staging_ids->size());
      for (int id : *auto_allocated_staging_ids) {
        local_staging_buffers.emplace_back(id, std::vector<BufferShard>{},
                                           std::nullopt,
                                           ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
      }
      request_staging = local_staging_buffers;
    }

    if (remote_host_staging_required && request_staging.empty()) {
      // 3b. Safeguard logic: Returning error for missing remote blocks.
      //     If remote staging blocks are required but none were provided, we
      //     must return an error. We cannot auto-allocate them from here
      //     because this controller doesn't manage the remote peer's block IDs
      //     and memory pool. The caller must pre-allocate them on the remote
      //     worker and supply them.
      return tsl::Future<>(absl::InvalidArgumentError(
          "Remote transfer requires remote host staging blocks, but none were "
          "provided."));
    }
  }

  auto workers = worker_registry_->GetRegisteredWorkers();
  if (workers.empty()) {
    return tsl::Future<>(absl::FailedPreconditionError(
        "No registered workers available for TransferBuffers"));
  }

  std::sort(workers.begin(), workers.end(),
            [](const core::controller::WorkerRegistration& a,
               const core::controller::WorkerRegistration& b) {
              return CompareWorkerIds(a.worker_id, b.worker_id);
            });

  absl::flat_hash_map<int64_t, const RaidenWorkerEndpoints*>
      peer_node_id_to_endpoints;
  for (const auto& dst_buf : dst_buffers) {
    for (const auto& group : dst_buf.remote_worker_endpoints()) {
      peer_node_id_to_endpoints.emplace(group.node_id, &group);
    }
  }
  // Pull direction (remote src): the peer's worker endpoint groups arrive on
  // the src buffers instead.
  if (peer_node_id_to_endpoints.empty()) {
    for (const auto& src_buf : src_buffers) {
      for (const auto& group : src_buf.remote_worker_endpoints()) {
        peer_node_id_to_endpoints.emplace(group.node_id, &group);
      }
    }
  }

  std::vector<tsl::Future<>> worker_futures;
  worker_futures.reserve(workers.size());

  for (size_t w = 0; w < workers.size(); ++w) {
    if (!workers[w].worker_service_client) continue;

    std::vector<Buffer> worker_src;
    std::vector<Buffer> worker_dst;
    std::vector<int64_t> worker_copy_sizes;
    worker_src.reserve(src_buffers.size());
    worker_dst.reserve(dst_buffers.size());

    // Every buffer/block is transferred by every worker (each worker owns a
    // shard of every block), so there is no per-block partitioning here.
    for (size_t i = 0; i < src_buffers.size(); ++i) {
      Buffer src_buf = src_buffers[i];
      if (!src_buf.remote_worker_endpoints().empty()) {
        // Pull: pair this local worker with the remote source worker that
        // shares its node_id. Never broadcast — an unmatched node_id would
        // corrupt the shards.
        auto it = peer_node_id_to_endpoints.find(workers[w].node_id);
        if (it == peer_node_id_to_endpoints.end()) {
          return tsl::Future<>(absl::FailedPreconditionError(absl::StrCat(
              "TransferBuffers: no source worker group with node_id ",
              workers[w].node_id, " to match local worker '",
              workers[w].worker_id, "' (source provided ",
              src_buf.remote_worker_endpoints().size(), " group(s))")));
        }
        src_buf.set_remote_descriptors(it->second->endpoints);
        src_buf.set_remote_worker_endpoints({});
      }
      worker_src.push_back(std::move(src_buf));

      Buffer dst_buf = dst_buffers[i];
      if (!dst_buf.remote_worker_endpoints().empty()) {
        auto it = peer_node_id_to_endpoints.find(workers[w].node_id);
        if (it == peer_node_id_to_endpoints.end()) {
          return tsl::Future<>(absl::FailedPreconditionError(absl::StrCat(
              "TransferBuffers: no destination worker group with node_id ",
              workers[w].node_id, " to match source worker '",
              workers[w].worker_id, "' (destination provided ",
              dst_buf.remote_worker_endpoints().size(), " group(s))")));
        }
        dst_buf.set_remote_descriptors(it->second->endpoints);
        dst_buf.set_remote_worker_endpoints({});
      } else if (dst_buf.remote_descriptors().empty() &&
                 !dst_buf.remote_address().has_value() &&
                 !workers[w].raiden_transfer_endpoints.empty()) {
        // Default fallback for local (intra-node / single-host) buffer
        // transfers where remote_worker_endpoints was omitted. Inject local
        // worker w's registered sub-manager transfer endpoints into
        // remote_descriptors.
        dst_buf.set_remote_descriptors(workers[w].raiden_transfer_endpoints);
      }
      worker_dst.push_back(std::move(dst_buf));

      if (!copy_sizes.empty()) {
        worker_copy_sizes.push_back(copy_sizes[i]);
      }
    }

    if (worker_src.empty()) continue;

    // Every worker owns a shard of every block, so the (host) staging offsets
    // are identical across workers.
    auto req_or = BuildTransferBuffersRequest(
        worker_src, worker_dst, request_staging, worker_copy_sizes);
    if (!req_or.ok()) {
      return tsl::Future<>(req_or.status());
    }

    worker_futures.push_back(
        workers[w].worker_service_client->TransferBuffers(*req_or));
  }

  if (worker_futures.empty()) {
    return tsl::Future<>(absl::FailedPreconditionError(
        "No active WorkerServiceClient available for TransferBuffers"));
  }

  auto aggregate_future = tsl::JoinFutures(absl::MakeSpan(worker_futures));
  if (auto_allocated_staging_ids.has_value()) {
    auto [promise, future] = tsl::MakePromise<>();
    aggregate_future.OnReady(
        [this, promise = std::move(promise),
         ids = *auto_allocated_staging_ids](absl::Status status) mutable {
          (void)this->DeallocateBlockIds(ids);
          promise.Set(std::move(status));
        });
    return future;
  }

  return aggregate_future;
}

void RaidenController::SetReadRemoteHooks(
    core::controller::RaidenControllerServiceImpl::ValidateAndPinCallback
        validate_and_pin,
    core::controller::RaidenControllerServiceImpl::UnpinCallback unpin) {
  if (active_server_ != nullptr) {
    active_server_->SetReadRemoteHooks(std::move(validate_and_pin),
                                       std::move(unpin));
  }
}

// Everything a single in-flight read needs, owned by a shared_ptr so the
// acquire callback, the transfer future, and the release callback can chain
// without capturing `this` -- controller teardown mid-read must not be a
// use-after-free.
struct RemoteReadState {
  std::shared_ptr<::tpu_sync::proto::RaidenControllerService::Stub> stub;
  std::shared_ptr<tsl::Promise<>> promise;
  std::vector<int32_t> dst_host_block_ids;
  std::vector<int32_t> dst_device_block_ids;
  uint64_t lease_id = 0;
  absl::Mutex mu;
  bool settled ABSL_GUARDED_BY(mu) = false;

  // Idempotent: the deadline timer and the release callback race by design.
  void Settle(absl::Status status) {
    {
      absl::MutexLock lock(&mu);
      if (settled) return;
      settled = true;
    }
    promise->Set(std::move(status));
  }
};

tsl::Future<> RaidenController::ReadRemote(
    absl::string_view src_controller_address,
    const std::vector<int32_t>& src_host_block_ids,
    const std::vector<int32_t>& dst_host_block_ids,
    const std::vector<std::string>& block_hashes,
    const std::vector<int32_t>& dst_device_block_ids) {
  CheckTimingTripleOnce();

  if (src_host_block_ids.size() != dst_host_block_ids.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(absl::StrCat(
        "Mismatch in block IDs sizes: src size ", src_host_block_ids.size(),
        " vs dst size ", dst_host_block_ids.size())));
  }
  if (block_hashes.empty()) {
    // There is no unvalidated transfer path any more: without hashes the
    // source cannot verify or pin anything.
    return tsl::Future<>(absl::InvalidArgumentError(
        "ReadRemote requires block_hashes to validate at the source"));
  }
  if (block_hashes.size() != dst_host_block_ids.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(absl::StrCat(
        "block_hashes size ", block_hashes.size(),
        " does not match block id count ", dst_host_block_ids.size())));
  }
  // Reject a wrong-sized device list BEFORE acquiring, so a caller error never
  // pins anything at the source.
  const bool to_hbm = !dst_device_block_ids.empty();
  if (to_hbm && dst_device_block_ids.size() != dst_host_block_ids.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(
        absl::StrCat("dst_device_block_ids size ", dst_device_block_ids.size(),
                     " must be empty (read to host) or match the block count ",
                     dst_host_block_ids.size())));
  }

  if (worker_registry_->GetRegisteredWorkers().empty()) {
    return tsl::Future<>(absl::FailedPreconditionError(
        "No registered workers available for ReadRemote"));
  }

  // The caller owns peer resolution; this controller keeps no directory. Reject
  // an empty address here rather than letting gRPC fail on it, which reports a
  // parse error naming nothing.
  if (src_controller_address.empty()) {
    return tsl::Future<>(absl::InvalidArgumentError(
        "ReadRemote requires the source's controller address; the caller "
        "resolves it (KVCacheStore reads it from the global registry)"));
  }
  const std::string controller_address(src_controller_address);

  std::shared_ptr<::tpu_sync::proto::RaidenControllerService::Stub> stub;
  {
    absl::MutexLock lock(mutex_);
    auto it = stubs_.find(controller_address);
    if (it != stubs_.end()) stub = it->second;
  }
  if (stub == nullptr) {
    auto channel = grpc::CreateChannel(controller_address,
                                       grpc::InsecureChannelCredentials());
    stub = ::tpu_sync::proto::RaidenControllerService::NewStub(channel);
    {
      absl::MutexLock lock(mutex_);
      if (stubs_.emplace(controller_address, stub).second) {
        stub_order_.push_back(controller_address);
        // Bounded: every peer restart mints one address that is never dialled
        // again (see the header). Oldest-first, so what ages out is what
        // stopped being used.
        while (stub_order_.size() > kMaxCachedStubs) {
          stubs_.erase(stub_order_.front());
          stub_order_.pop_front();
        }
      } else {
        // Another thread got there first; use its stub so both share a channel.
        stub = stubs_[controller_address];
      }
    }
  }

  auto [promise, future] = tsl::MakePromise<>();
  auto state = std::make_shared<RemoteReadState>();
  state->stub = stub;
  state->promise = std::move(promise).ToShared();
  state->dst_host_block_ids = dst_host_block_ids;
  state->dst_device_block_ids = dst_device_block_ids;

  // ---- Phase 1: Prepare -- acquire the lease. -----------------------------
  auto acquire_ctx = std::make_shared<grpc::ClientContext>();
  auto acquire_req =
      std::make_shared<::tpu_sync::proto::AcquireReadLeaseRequest>();
  for (const auto& hash : block_hashes) acquire_req->add_block_hashes(hash);
  acquire_req->set_ttl_ms(absl::ToInt64Milliseconds(
      core::controller::RaidenControllerServiceImpl::DefaultLeaseTtl()));
  *acquire_req->mutable_requester_raiden_id() = unit_;
  auto acquire_resp =
      std::make_shared<::tpu_sync::proto::AcquireReadLeaseResponse>();

  // Capture only shared state and by-value copies -- never `this`.
  auto lifetime = lifetime_;

  stub->async()->AcquireReadLease(
      acquire_ctx.get(), acquire_req.get(), acquire_resp.get(),
      [state, acquire_ctx, acquire_req, acquire_resp, lifetime,
       to_hbm](grpc::Status status) {
        if (!status.ok()) {
          // grpc and absl codes share canonical integers, so the source's
          // verify-hook distinctions survive: NOT_FOUND (hash absent) vs
          // FAILED_PRECONDITION (present, not host-resident). No retry here --
          // neither is retryable against the same source, and a hidden retry
          // loop would turn a fast failure into unbounded TTFT.
          RAIDEN_LEASE_TRACE("acquire FAILED code=", status.error_code(), " ",
                             status.error_message());
          state->Settle(
              absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                           absl::StrCat("AcquireReadLease failed: ",
                                        status.error_message())));
          return;
        }
        state->lease_id = acquire_resp->lease_id();
        RAIDEN_LEASE_TRACE(
            "acquired lease=", state->lease_id,
            " src_ids=", acquire_resp->src_host_block_ids_size(),
            " groups=", acquire_resp->src_worker_endpoints_size(),
            " ttl_ms=", acquire_resp->granted_ttl_ms());

        // ---- Phase 2: Work -- our own workers pull the bytes. -------------
        // Use the source's AUTHORITATIVE ids, not the advisory ones we sent.
        std::vector<int32_t> src_ids(acquire_resp->src_host_block_ids().begin(),
                                     acquire_resp->src_host_block_ids().end());
        std::vector<RaidenWorkerEndpoints> src_groups;
        src_groups.reserve(acquire_resp->src_worker_endpoints_size());
        for (const auto& group : acquire_resp->src_worker_endpoints()) {
          std::vector<::tpu_raiden::RaidenTransferEndpoint> eps;
          eps.reserve(group.endpoints_size());
          for (const auto& ep : group.endpoints()) {
            eps.push_back(
                {ep.endpoint(),
                 std::vector<int64_t>(ep.shards().begin(), ep.shards().end())});
          }
          src_groups.push_back(
              {group.node_id(), group.worker_id(), std::move(eps)});
        }

        absl::MutexLock lock(&lifetime->mu);
        if (lifetime->ctrl == nullptr) {
          // The controller went away mid-read. Release best-effort so the
          // source does not hold pins for a full TTL, then fail the read.
          auto ctx = std::make_shared<grpc::ClientContext>();
          auto req =
              std::make_shared<::tpu_sync::proto::ReleaseReadLeaseRequest>();
          req->set_lease_id(state->lease_id);
          auto resp =
              std::make_shared<::tpu_sync::proto::ReleaseReadLeaseResponse>();
          state->stub->async()->ReleaseReadLease(
              ctx.get(), req.get(), resp.get(),
              [ctx, req, resp](grpc::Status) {});
          state->Settle(absl::CancelledError(
              "ReadRemote abandoned: controller shut down mid-read"));
          return;
        }
        lifetime->ctrl->PullAndRelease(src_ids, src_groups, state, to_hbm);
      });

  // ---- Deadline. ----------------------------------------------------------
  // Without this a hung worker leaves the future unresolved forever, which
  // leaves the hashes permanently inadmissible at the store (reading_hashes_
  // is never cleared) and contradicts the retry story. Expiry routes through
  // the normal discard path.
  std::thread([state, deadline = RemoteReadDeadline()] {
    absl::SleepFor(deadline);
    {
      absl::MutexLock lock(&state->mu);
      if (state->settled) return;
    }
    if (state->lease_id != 0) {
      // Best-effort, fire-and-forget: let the source reclaim early rather than
      // wait out the TTL.
      auto ctx = std::make_shared<grpc::ClientContext>();
      auto req = std::make_shared<::tpu_sync::proto::ReleaseReadLeaseRequest>();
      req->set_lease_id(state->lease_id);
      auto resp =
          std::make_shared<::tpu_sync::proto::ReleaseReadLeaseResponse>();
      state->stub->async()->ReleaseReadLease(ctx.get(), req.get(), resp.get(),
                                             [ctx, req, resp](grpc::Status) {});
    }
    RAIDEN_LEASE_TRACE("DEADLINE lease=", state->lease_id);
    state->Settle(absl::DeadlineExceededError(absl::StrCat(
        "ReadRemote exceeded the destination deadline (", deadline, ")")));
  }).detach();

  return future;
}

void RaidenController::PullAndRelease(
    const std::vector<int32_t>& src_host_block_ids,
    const std::vector<RaidenWorkerEndpoints>& src_groups,
    const std::shared_ptr<RemoteReadState>& state, bool to_hbm) {
  // Pull direction: the peer's endpoint groups ride on the SRC buffers, and
  // TransferBuffers pairs each local worker with the source group sharing its
  // node_id (hard failure if unmatched -- never a broadcast).
  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_host_block_ids.size());
  for (int32_t id : src_host_block_ids) {
    Buffer src_buf(id, std::vector<BufferShard>{}, std::nullopt,
                   ::tpu_sync::rpc::MemoryType::MEMORY_TYPE_DRAM);
    src_buf.set_remote_worker_endpoints(src_groups);
    src_buffers.push_back(std::move(src_buf));
  }

  std::vector<Buffer> dst_buffers;
  std::vector<Buffer> staging_buffers;
  if (to_hbm) {
    // remote host src -> local host staging -> local device dst. The host
    // blocks are a staging hop only, not a destination the caller keeps:
    // KVCacheStore returns them to the pool once the read settles, success or
    // failure, and records nothing for them -- the caller's device blocks are
    // the read's only product.
    dst_buffers.reserve(state->dst_device_block_ids.size());
    for (int32_t id : state->dst_device_block_ids) {
      dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                               ::tpu_sync::rpc::MemoryType::MEMORY_TYPE_HBM);
    }
    staging_buffers.reserve(state->dst_host_block_ids.size());
    for (int32_t id : state->dst_host_block_ids) {
      staging_buffers.emplace_back(
          id, std::vector<BufferShard>{}, std::nullopt,
          ::tpu_sync::rpc::MemoryType::MEMORY_TYPE_DRAM);
    }
  } else {
    dst_buffers.reserve(state->dst_host_block_ids.size());
    for (int32_t id : state->dst_host_block_ids) {
      dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                               ::tpu_sync::rpc::MemoryType::MEMORY_TYPE_DRAM);
    }
  }

  RAIDEN_LEASE_TRACE("pull issue lease=", state->lease_id,
                     " blocks=", src_buffers.size(),
                     " to_hbm=", to_hbm ? 1 : 0);
  tsl::Future<> transfer =
      TransferBuffers(src_buffers, dst_buffers, staging_buffers);
  RAIDEN_LEASE_TRACE("pull issued lease=", state->lease_id);

  // ---- Phase 3: Commit -- release and read the verdict. -------------------
  // This runs on EVERY path, including transfer failure: leaving a failed
  // read's lease to the sweeper would hold the source's blocks for a full TTL
  // for no reason.
  transfer.OnReady([state](absl::Status transfer_status) {
    RAIDEN_LEASE_TRACE("pull done lease=", state->lease_id,
                       " status=", transfer_status.ToString());
    auto ctx = std::make_shared<grpc::ClientContext>();
    auto req = std::make_shared<::tpu_sync::proto::ReleaseReadLeaseRequest>();
    req->set_lease_id(state->lease_id);
    auto resp = std::make_shared<::tpu_sync::proto::ReleaseReadLeaseResponse>();
    state->stub->async()->ReleaseReadLease(
        ctx.get(), req.get(), resp.get(),
        [state, ctx, req, resp, transfer_status](grpc::Status rpc_status) {
          if (!transfer_status.ok()) {
            // Report the transfer error; the release was still sent.
            state->Settle(transfer_status);
            return;
          }
          if (!rpc_status.ok()) {
            // We cannot tell whether the pin held, so we must assume it did
            // not.
            state->Settle(absl::UnavailableError(
                absl::StrCat("ReleaseReadLease failed, verdict unknown: ",
                             rpc_status.error_message())));
            return;
          }
          RAIDEN_LEASE_TRACE("release lease=", state->lease_id,
                             " verdict=", static_cast<int>(resp->verdict()));
          switch (resp->verdict()) {
            case ::tpu_sync::proto::LEASE_HELD:
              state->Settle(absl::OkStatus());
              return;
            case ::tpu_sync::proto::LEASE_REVOKED:
              state->Settle(absl::FailedPreconditionError(
                  "read discarded: the source lease was revoked mid-read (its "
                  "TTL expired), so the bytes may be from reused blocks"));
              return;
            default:
              state->Settle(absl::FailedPreconditionError(
                  "read discarded: the source has no record of this lease "
                  "(aged out or restarted), so the pin cannot be attested"));
              return;
          }
        });
  });
}

tsl::Future<::tpu_sync::proto::TransferProgramResponse>
RaidenController::SubmitTransferProgram(
    absl::string_view worker_id,
    const ::tpu_sync::proto::TransferProgramRequest& request) {
  auto fail = [](absl::Status status) {
    auto [promise, future] =
        tsl::MakePromise<::tpu_sync::proto::TransferProgramResponse>();
    std::move(promise).ToShared()->Set(std::move(status));
    return future;
  };
  absl::StatusOr<core::controller::WorkerRegistration> registration =
      worker_registry_->GetWorker(worker_id);
  if (!registration.ok()) {
    return fail(absl::UnavailableError(
        absl::StrCat("SubmitTransferProgram: worker '", worker_id,
                     "' is not registered with this controller: ",
                     registration.status().message())));
  }
  if (registration->worker_service_client == nullptr) {
    return fail(absl::UnavailableError(
        absl::StrCat("SubmitTransferProgram: worker '", worker_id,
                     "' has no WorkerService channel")));
  }
  return registration->worker_service_client->SubmitTransferProgram(request);
}

}  // namespace controller
}  // namespace tpu_raiden
