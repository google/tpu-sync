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

#include "tpu_sync/kv_cache/reshard/reshard_client.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/reshard/framed_rpc.h"
#include "tpu_sync/rpc/controller_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

namespace {

// Facade parity: connect_socket(address, timeout=300.0) per call.
constexpr absl::Duration kCallTimeout = absl::Seconds(300);

tpu_sync::rpc::RaidenIdProto RaidenIdProtoOf(const RaidenId& unit) {
  tpu_sync::rpc::RaidenIdProto proto;
  proto.set_job_name(unit.job_name);
  proto.set_job_replica_id(unit.job_replica_id);
  proto.set_data_name(unit.data_name);
  proto.set_data_replica_idx(unit.data_replica_idx);
  return proto;
}

}  // namespace

ReshardClient::ReshardClient(std::string address, FramedTransport* transport)
    : address_(std::move(address)) {
  if (transport == nullptr) {
    owned_transport_ = std::make_unique<SocketFramedTransport>();
    transport_ = owned_transport_.get();
  } else {
    transport_ = transport;
  }
}

tpu_sync::rpc::ControlRequest ReshardClient::BuildRegisterWorkUnit(
    const RegisterWorkUnitArgs& args) {
  tpu_sync::rpc::ControlRequest req;
  req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_REGISTER_WORK_UNIT);
  tpu_sync::rpc::RegisterWorkUnitRequest* reg =
      req.mutable_register_work_unit_request();
  *reg->mutable_unit() = RaidenIdProtoOf(args.unit);
  for (const std::string& shard : args.shards) reg->add_shards(shard);
  // Python assigns "" when None; proto3 implicit presence drops it either
  // way, so unconditional set_ matches the wire exactly.
  reg->set_control_plane_rpc_address(args.control_plane_rpc_address);
  // `if mesh_shape:` — present-but-empty is absent (same for layout and
  // global_shape); `if itemsize:` — zero is absent.
  for (int64_t v : args.mesh_shape) reg->add_mesh_shape(v);
  for (int32_t v : args.layout) reg->add_layout(v);
  for (int64_t v : args.global_shape) reg->add_global_shape(v);
  if (args.itemsize != 0) reg->set_itemsize(args.itemsize);
  if (args.has_pool_manifest) {
    for (const ClientPoolSpec& pool : args.pool_manifest) {
      tpu_sync::rpc::PoolSpecProto* pool_proto = reg->add_pools();
      pool_proto->set_tag(pool.tag);
      pool_proto->set_storage_index(pool.storage_index);
      pool_proto->set_base_offset_bytes(pool.base_offset_bytes);
      pool_proto->set_block_stride_bytes(pool.block_stride_bytes);
      pool_proto->set_num_blocks(pool.num_blocks);
      pool_proto->set_dtype_tag(pool.dtype_tag);
      for (const ClientPoolRegion& region : pool.regions) {
        tpu_sync::rpc::RegionSpecProto* region_proto =
            pool_proto->add_regions();
        region_proto->set_name(region.name);
        region_proto->set_offset_bytes(region.offset_bytes);
        region_proto->set_stride_bytes(region.stride_bytes);
        region_proto->set_unit_bytes(region.unit_bytes);
        region_proto->set_num_units(region.num_units);
        region_proto->set_units_per_stride(region.units_per_stride);
      }
    }
  }
  if (args.layout_fingerprint.has_value()) {
    reg->set_layout_fingerprint(*args.layout_fingerprint);
  }
  if (args.page_tokens.has_value()) reg->set_page_tokens(*args.page_tokens);
  if (args.transfer_parallelism.has_value()) {
    reg->set_transfer_parallelism(*args.transfer_parallelism);
  }
  if (args.transfer_rank.has_value()) {
    reg->set_transfer_rank(*args.transfer_rank);
  }
  if (args.has_variables) {
    for (const std::string& payload : args.variables) {
      // Byte-exact pass-through of the caller's VariableMetadataProto.
      reg->add_variables()->ParseFromString(payload);
    }
  }
  return req;
}

tpu_sync::rpc::ControllerRequest ReshardClient::BuildRegisterRequestBlocks(
    const std::string& req_id, int64_t uuid, const RaidenId& unit,
    const std::vector<int64_t>& block_ids,
    const std::vector<ClientPoolSpans>& pool_spans) {
  tpu_sync::rpc::ControllerRequest req;
  req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_REGISTER_REQUEST_BLOCKS);
  tpu_sync::rpc::RegisterRequestBlocksRequest* block_req =
      req.mutable_register_request_blocks_request();
  block_req->set_req_id(req_id);
  block_req->set_uuid(uuid);
  *block_req->mutable_unit() = RaidenIdProtoOf(unit);
  for (int64_t block_id : block_ids) block_req->add_block_ids(block_id);
  for (const ClientPoolSpans& entry : pool_spans) {
    auto* entry_proto = block_req->add_pool_spans();
    entry_proto->set_tag(entry.tag);
    for (int64_t block_id : entry.block_ids) {
      entry_proto->add_block_ids(block_id);
    }
    entry_proto->set_declared_bytes(entry.declared_bytes);
    entry_proto->set_dst_space_version(entry.dst_space_version);
    for (const ClientByteSpan& span : entry.spans) {
      auto* span_proto = entry_proto->add_spans();
      span_proto->set_src_block_ordinal(span.src_block_ordinal);
      span_proto->set_src_offset_bytes(span.src_offset_bytes);
      span_proto->set_dst_block_index(span.dst_block_index);
      span_proto->set_dst_offset_bytes(span.dst_offset_bytes);
      span_proto->set_size_bytes(span.size_bytes);
      span_proto->set_src_stride_bytes(span.src_stride_bytes);
      span_proto->set_dst_stride_bytes(span.dst_stride_bytes);
      span_proto->set_count(span.count);
      if (span.dst_unit_ordinal >= 0) {
        span_proto->set_dst_unit_ordinal(span.dst_unit_ordinal);
      }
    }
  }
  return req;
}

tpu_sync::rpc::ControllerRequest ReshardClient::BuildReleaseRequestBlocks(
    const std::string& req_id, int64_t uuid) {
  tpu_sync::rpc::ControllerRequest req;
  req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_RELEASE_REQUEST_BLOCKS);
  auto* release = req.mutable_release_request_blocks_request();
  release->set_req_id(req_id);
  release->set_uuid(uuid);
  release->set_force(true);  // facade: force=True always
  return req;
}

tpu_sync::rpc::ControllerRequest ReshardClient::BuildCompleteRequestBlocks(
    const std::string& req_id, int64_t uuid, const RaidenId& unit) {
  tpu_sync::rpc::ControllerRequest req;
  req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_COMPLETE_REQUEST_BLOCKS);
  auto* complete = req.mutable_complete_request_blocks_request();
  complete->set_req_id(req_id);
  complete->set_uuid(uuid);
  *complete->mutable_unit() = RaidenIdProtoOf(unit);
  return req;
}

tpu_sync::rpc::ControllerRequest
ReshardClient::BuildCancelRequestBlocksIfUnclaimed(const std::string& req_id,
                                                   int64_t uuid) {
  tpu_sync::rpc::ControllerRequest req;
  req.set_command(tpu_sync::rpc::ControllerRequest::
                      COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED);
  auto* cancel = req.mutable_cancel_request_blocks_if_unclaimed_request();
  cancel->set_req_id(req_id);
  cancel->set_uuid(uuid);
  return req;
}

tpu_sync::rpc::ControllerRequest ReshardClient::BuildStartTransfer(
    const StartTransferArgs& args) {
  tpu_sync::rpc::ControllerRequest req;
  req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_COORDINATE_TRANSFER);
  tpu_sync::rpc::CoordinateTransferRequest* coord =
      req.mutable_coordinate_transfer_request();
  for (const RaidenId& unit : args.src_units) {
    *coord->add_src_units() = RaidenIdProtoOf(unit);
  }
  for (const RaidenId& unit : args.dst_units) {
    *coord->add_dst_units() = RaidenIdProtoOf(unit);
  }
  coord->set_use_block_chunks(args.use_block_chunks);
  coord->set_is_sender(args.is_sender);
  coord->set_expected_block_count(args.expected_block_count);
  coord->set_uuid(args.uuid);
  coord->set_req_id(args.req_id);
  coord->set_dst_controller_address(args.dst_controller_address);
  coord->set_src_controller_address(args.src_controller_address);
  coord->set_dst_mem_type(
      static_cast<tpu_sync::rpc::MemoryType>(args.dst_mem_type));
  if (args.has_dst_device_block_ids) {
    for (int64_t block : args.dst_device_block_ids) {
      coord->add_dst_device_block_ids(block);
    }
  }
  // num_tokens: accepted by the shim for caller compatibility, retired
  // from the wire (facade parity).
  if (args.has_transfer_pool_tags) {
    for (const std::string& tag : args.transfer_pool_tags) {
      coord->add_transfer_pool_tags(tag);
    }
  }
  // `if dst_block_counts:` — truthy; empty is absent.
  for (int64_t count : args.dst_block_counts) {
    coord->add_dst_block_counts(static_cast<int32_t>(count));
  }
  // `if dst_skip_bytes:` — truthy; empty (all-zero clip) is absent, which
  // keeps skip-free requests byte-identical to pre-clip encodings.
  for (int64_t skip : args.dst_skip_bytes) {
    coord->add_dst_skip_bytes(skip);
  }
  return req;
}

tpu_sync::rpc::ControllerRequest ReshardClient::BuildGetTransferStatus(
    const std::string& req_id, int64_t uuid) {
  tpu_sync::rpc::ControllerRequest req;
  req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_GET_TRANSFER_STATUS);
  auto* status = req.mutable_get_transfer_status_request();
  status->set_req_id(req_id);
  status->set_uuid(uuid);
  return req;
}

tpu_sync::rpc::ControllerRequest ReshardClient::BuildGetRequestBlockStatus(
    const std::vector<std::pair<std::string, int64_t>>& keys) {
  tpu_sync::rpc::ControllerRequest req;
  req.set_command(
      tpu_sync::rpc::ControllerRequest::COMMAND_GET_REQUEST_BLOCK_STATUS);
  auto* status = req.mutable_get_request_block_status_request();
  for (const auto& [req_id, uuid] : keys) {
    auto* key = status->add_keys();
    key->set_req_id(req_id);
    key->set_uuid(uuid);
  }
  return req;
}

tpu_sync::rpc::ControlRequest ReshardClient::BuildGetMetadata() {
  tpu_sync::rpc::ControlRequest req;
  req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_GET_METADATA);
  return req;
}

tpu_sync::rpc::ControlRequest ReshardClient::BuildShutdown() {
  tpu_sync::rpc::ControlRequest req;
  req.set_command(tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN);
  return req;
}

absl::StatusOr<tpu_sync::rpc::ControllerResponse> ReshardClient::CallController(
    const tpu_sync::rpc::ControllerRequest& request) {
  absl::StatusOr<std::string> body =
      transport_->Call(address_, request.SerializeAsString(), kCallTimeout);
  if (!body.ok()) return body.status();
  tpu_sync::rpc::ControllerResponse response;
  if (!response.ParseFromString(*body)) {
    return absl::InternalError("Failed to parse ControllerResponse");
  }
  if (!response.success()) {
    // Verbatim facade text: the connector's bounded retry substring-matches
    // the embedded server message.
    return absl::InternalError(absl::StrCat(
        "Remote Controller Server execution failed: ", response.message()));
  }
  return response;
}

absl::StatusOr<tpu_sync::rpc::ControlResponse> ReshardClient::CallRaiden(
    const tpu_sync::rpc::ControlRequest& request) {
  absl::StatusOr<std::string> body =
      transport_->Call(address_, request.SerializeAsString(), kCallTimeout);
  if (!body.ok()) return body.status();
  tpu_sync::rpc::ControlResponse response;
  if (!response.ParseFromString(*body)) {
    return absl::InternalError("Failed to parse ControlResponse");
  }
  if (!response.success()) {
    return absl::InternalError(absl::StrCat(
        "Remote Controller Server execution failed: ", response.message()));
  }
  return response;
}

absl::Status ReshardClient::RegisterWorkUnit(
    const RegisterWorkUnitArgs& args) {
  return CallRaiden(BuildRegisterWorkUnit(args)).status();
}

absl::Status ReshardClient::RegisterRequestBlocks(
    const std::string& req_id, int64_t uuid, const RaidenId& unit,
    const std::vector<int64_t>& block_ids,
    const std::vector<ClientPoolSpans>& pool_spans) {
  return CallController(BuildRegisterRequestBlocks(req_id, uuid, unit,
                                                   block_ids, pool_spans))
      .status();
}

absl::Status ReshardClient::ReleaseRequestBlocks(const std::string& req_id,
                                                 int64_t uuid) {
  return CallController(BuildReleaseRequestBlocks(req_id, uuid)).status();
}

absl::Status ReshardClient::CompleteRequestBlocks(const std::string& req_id,
                                                  int64_t uuid,
                                                  const RaidenId& unit) {
  return CallController(BuildCompleteRequestBlocks(req_id, uuid, unit))
      .status();
}

absl::StatusOr<bool> ReshardClient::CancelRequestBlocksIfUnclaimed(
    const std::string& req_id, int64_t uuid) {
  absl::StatusOr<tpu_sync::rpc::ControllerResponse> response =
      CallController(BuildCancelRequestBlocksIfUnclaimed(req_id, uuid));
  if (!response.ok()) return response.status();
  if (response->response_data() == "true") return true;
  if (response->response_data() == "false") return false;
  return absl::InternalError(absl::StrCat(
      "Remote Controller Server returned an invalid cancellation result: '",
      response->response_data(), "'"));
}

absl::StatusOr<bool> ReshardClient::StartTransfer(
    const StartTransferArgs& args) {
  absl::Status status = CallController(BuildStartTransfer(args)).status();
  if (!status.ok()) return status;
  return true;
}

absl::StatusOr<int32_t> ReshardClient::GetTransferStatus(
    const std::string& req_id, int64_t uuid) {
  absl::StatusOr<tpu_sync::rpc::ControllerResponse> response =
      CallController(BuildGetTransferStatus(req_id, uuid));
  if (!response.ok()) return response.status();
  return static_cast<int32_t>(
      response->get_transfer_status_response().status());
}

absl::StatusOr<std::vector<int32_t>> ReshardClient::GetRequestBlockStatus(
    const std::vector<std::pair<std::string, int64_t>>& keys) {
  absl::StatusOr<tpu_sync::rpc::ControllerResponse> response =
      CallController(BuildGetRequestBlockStatus(keys));
  if (!response.ok()) return response.status();
  const auto& body = response->get_request_block_status_response();
  if (static_cast<size_t>(body.statuses_size()) != keys.size()) {
    return absl::InternalError(
        absl::StrCat("Remote Controller Server returned ", body.statuses_size(),
                     " request block statuses for ", keys.size(), " keys"));
  }
  std::vector<int32_t> statuses;
  statuses.reserve(keys.size());
  for (int status : body.statuses()) statuses.push_back(status);
  return statuses;
}

absl::StatusOr<std::vector<std::string>> ReshardClient::GetMetadata() {
  absl::StatusOr<tpu_sync::rpc::ControlResponse> response =
      CallRaiden(BuildGetMetadata());
  if (!response.ok()) return response.status();
  std::vector<std::string> serialized;
  serialized.reserve(response->get_metadata_response().metadata_size());
  for (const auto& metadata : response->get_metadata_response().metadata()) {
    serialized.push_back(metadata.SerializeAsString());
  }
  return serialized;
}

absl::StatusOr<bool> ReshardClient::Shutdown() {
  absl::Status status = CallRaiden(BuildShutdown()).status();
  if (!status.ok()) return status;
  return true;
}

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
