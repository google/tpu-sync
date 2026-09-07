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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_CLIENT_H_

// C++ implementation of the reshard client command surface previously served
// by RaidenControllerClientFacade in Python, ported byte-compatibly. Every
// encoding decision (field presence, defaults, truthiness quirks) lives here;
// the Python shim only normalizes object shapes. Error contract: a failed
// command surfaces the server message as
// "Remote Controller Server execution failed: <message>" verbatim because
// the vLLM connector's bounded retry logic substring-matches the message text.
//
// Request builders are exposed separately from the transport wrappers so
// tests can byte-compare encodings against recorded facade output without
// sockets.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/reshard/declaration_types.h"
#include "tpu_sync/kv_cache/reshard/framed_rpc.h"
#include "tpu_sync/rpc/controller_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

// Plain-value mirrors of the Python-side inputs. Defaults reproduce
// _coerce_pool_spec_proto's mapping defaults.
struct ClientPoolRegion {
  std::string name;
  int64_t offset_bytes = 0;
  int64_t stride_bytes = 0;
  int64_t unit_bytes = 0;
  int64_t num_units = 0;
  int64_t units_per_stride = 1;
};

struct ClientPoolSpec {
  std::string tag;
  int64_t storage_index = 0;
  int64_t base_offset_bytes = 0;
  int64_t block_stride_bytes = 0;
  int64_t num_blocks = 0;
  std::string dtype_tag;
  std::vector<ClientPoolRegion> regions;
};

struct ClientByteSpan {
  int64_t src_block_ordinal = 0;
  int64_t src_offset_bytes = 0;
  int64_t dst_block_index = 0;
  int64_t dst_offset_bytes = 0;
  int64_t size_bytes = 0;
  int64_t src_stride_bytes = 0;
  int64_t dst_stride_bytes = 0;
  int64_t count = 0;
  // -1 = every destination (absent on the wire); >= 0 routes the span to
  // that destination unit only.
  int32_t dst_unit_ordinal = -1;
};

struct ClientPoolSpans {
  std::string tag;
  std::vector<int64_t> block_ids;
  int64_t declared_bytes = 0;
  int32_t dst_space_version = 0;  // getattr(entry, "dst_space_version", 0)
  std::vector<ClientByteSpan> spans;
};

struct RegisterWorkUnitArgs {
  RaidenId unit;
  std::vector<std::string> shards;
  // Python: control_plane_rpc_address if control_plane_rpc_address else ""
  std::string control_plane_rpc_address;
  // Python presence quirks preserved: `if mesh_shape:` — present-but-empty
  // is ABSENT; same for layout/global_shape. `if itemsize:` — zero is
  // ABSENT. The *_is_none flags distinguish None from empty/zero where the
  // Python semantics differ (`is not None` fields).
  std::vector<int64_t> mesh_shape;
  std::vector<int32_t> layout;
  std::vector<int64_t> global_shape;
  int32_t itemsize = 0;
  bool has_pool_manifest = false;  // pool_manifest is not None
  std::vector<ClientPoolSpec> pool_manifest;
  std::optional<std::string> layout_fingerprint;  // is not None
  std::optional<int64_t> page_tokens;             // is not None
  std::optional<int32_t> transfer_parallelism;    // is not None
  std::optional<int32_t> transfer_rank;           // is not None
  bool has_variables = false;  // variables is not None
  // Serialized VariableMetadataProto payloads (the shim serializes the
  // caller's pb2 objects; parsing here keeps them byte-exact).
  std::vector<std::string> variables;
};

struct StartTransferArgs {
  std::vector<RaidenId> src_units;
  std::vector<RaidenId> dst_units;
  std::string req_id;  // Python: req_id if req_id else ""
  bool use_block_chunks = false;
  bool is_sender = true;
  int64_t expected_block_count = 0;
  int64_t uuid = 0;
  std::string dst_controller_address;  // "" when None
  std::string src_controller_address;  // "" when None
  int32_t dst_mem_type = 1;            // RaidenMemoryType.DRAM
  bool has_dst_device_block_ids = false;  // is not None
  std::vector<int64_t> dst_device_block_ids;
  bool has_transfer_pool_tags = false;  // is not None
  std::vector<std::string> transfer_pool_tags;
  // Python: `if dst_block_counts:` — truthy, so empty is absent.
  std::vector<int64_t> dst_block_counts;
  // Per-tag destination clip; `if dst_skip_bytes:` — truthy, so empty
  // (no clip) is absent and skip-free encodings stay byte-identical.
  std::vector<int64_t> dst_skip_bytes;
  // num_tokens is accepted by the shim for caller compatibility and
  // dropped, exactly like the facade (retired from the wire).
};

class ReshardClient {
 public:
  // `transport` is borrowed when non-null (tests); otherwise the client
  // owns a SocketFramedTransport. Timeout matches the facade's
  // connect_socket(timeout=300.0) per call.
  explicit ReshardClient(std::string address,
                         FramedTransport* transport = nullptr);

  // --- request builders (encoding ownership) ----------------------------
  static tpu_sync::rpc::ControlRequest BuildRegisterWorkUnit(
      const RegisterWorkUnitArgs& args);
  static tpu_sync::rpc::ControllerRequest BuildRegisterRequestBlocks(
      const std::string& req_id, int64_t uuid, const RaidenId& unit,
      const std::vector<int64_t>& block_ids,
      const std::vector<ClientPoolSpans>& pool_spans);
  static tpu_sync::rpc::ControllerRequest BuildReleaseRequestBlocks(
      const std::string& req_id, int64_t uuid);
  static tpu_sync::rpc::ControllerRequest BuildCompleteRequestBlocks(
      const std::string& req_id, int64_t uuid, const RaidenId& unit);
  static tpu_sync::rpc::ControllerRequest BuildCancelRequestBlocksIfUnclaimed(
      const std::string& req_id, int64_t uuid);
  static tpu_sync::rpc::ControllerRequest BuildStartTransfer(
      const StartTransferArgs& args);
  static tpu_sync::rpc::ControllerRequest BuildGetTransferStatus(
      const std::string& req_id, int64_t uuid);
  static tpu_sync::rpc::ControllerRequest BuildGetRequestBlockStatus(
      const std::vector<std::pair<std::string, int64_t>>& keys);
  static tpu_sync::rpc::ControlRequest BuildGetMetadata();
  static tpu_sync::rpc::ControlRequest BuildShutdown();

  // --- transport wrappers (facade-identical semantics) -------------------
  absl::Status RegisterWorkUnit(const RegisterWorkUnitArgs& args);
  absl::Status RegisterRequestBlocks(
      const std::string& req_id, int64_t uuid, const RaidenId& unit,
      const std::vector<int64_t>& block_ids,
      const std::vector<ClientPoolSpans>& pool_spans);
  absl::Status ReleaseRequestBlocks(const std::string& req_id, int64_t uuid);
  absl::Status CompleteRequestBlocks(const std::string& req_id, int64_t uuid,
                                     const RaidenId& unit);
  absl::StatusOr<bool> CancelRequestBlocksIfUnclaimed(
      const std::string& req_id, int64_t uuid);
  absl::StatusOr<bool> StartTransfer(const StartTransferArgs& args);
  absl::StatusOr<int32_t> GetTransferStatus(const std::string& req_id,
                                            int64_t uuid);
  // GetRequestBlockStatusResponse::Status values, parallel to keys.
  absl::StatusOr<std::vector<int32_t>> GetRequestBlockStatus(
      const std::vector<std::pair<std::string, int64_t>>& keys);
  // Serialized RegisterWorkUnitRequest payloads; the Python shim parses
  // them back into pb2 objects so callers see the facade's return shape.
  absl::StatusOr<std::vector<std::string>> GetMetadata();
  absl::StatusOr<bool> Shutdown();

  const std::string& address() const { return address_; }

 private:
  absl::StatusOr<tpu_sync::rpc::ControllerResponse> CallController(
      const tpu_sync::rpc::ControllerRequest& request);
  absl::StatusOr<tpu_sync::rpc::ControlResponse> CallRaiden(
      const tpu_sync::rpc::ControlRequest& request);

  std::string address_;
  std::unique_ptr<SocketFramedTransport> owned_transport_;
  FramedTransport* transport_;
};

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_CLIENT_H_
