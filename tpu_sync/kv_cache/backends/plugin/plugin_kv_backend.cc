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

#include "tpu_sync/kv_cache/backends/plugin/plugin_kv_backend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/plugin/raiden_storage_plugin_abi.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {

namespace {

constexpr size_t kMaxErrorBufferBytes = 1024;
constexpr size_t kDefaultResolvedKeyCapacity = 4096;

RaidenParallelismConfig ToCParallelism(const ParallelismConfig& parallelism) {
  RaidenParallelismConfig out;
  out.tp_size = parallelism.tp_size;
  out.tp_rank = parallelism.tp_rank;
  out.pcp_size = parallelism.pcp_size;
  out.pcp_rank = parallelism.pcp_rank;
  out.pp_size = parallelism.pp_size;
  out.pp_rank = parallelism.pp_rank;
  return out;
}

bool HasDeclaredParallelism(const ParallelismConfig& parallelism) {
  return parallelism.tp_size != kAxisUndeclared ||
         parallelism.pcp_size != kAxisUndeclared ||
         parallelism.pp_size != kAxisUndeclared;
}

struct IoCallbackContext {
  std::function<void(absl::Status)> callback;
};

void OnPluginIoComplete(void* user_data, RaidenStatusCode status,
                        const char* error_msg) {
  std::unique_ptr<IoCallbackContext> ctx(
      static_cast<IoCallbackContext*>(user_data));
  absl::string_view msg =
      error_msg != nullptr ? absl::string_view(error_msg) : absl::string_view();
  ctx->callback(ToAbslStatus(status, msg));
}

struct BatchExistsCallbackContext {
  size_t expected_num_keys = 0;
  std::function<void(std::vector<absl::StatusOr<bool>>)> callback;
};

void OnPluginBatchExistsComplete(void* user_data,
                                 const RaidenStatusCode* statuses,
                                 const uint8_t* exists, size_t num_keys,
                                 const char* error_msg) {
  std::unique_ptr<BatchExistsCallbackContext> ctx(
      static_cast<BatchExistsCallbackContext*>(user_data));
  absl::string_view msg =
      error_msg != nullptr ? absl::string_view(error_msg) : absl::string_view();
  std::vector<absl::StatusOr<bool>> results;
  results.reserve(ctx->expected_num_keys);
  if (statuses == nullptr || exists == nullptr ||
      num_keys != ctx->expected_num_keys) {
    absl::Status err = absl::InternalError(
        msg.empty() ? "Plugin batch_exists_async returned invalid array lengths"
                    : msg);
    results.assign(ctx->expected_num_keys, err);
    ctx->callback(std::move(results));
    return;
  }
  for (size_t i = 0; i < num_keys; ++i) {
    if (statuses[i] == RAIDEN_STATUS_OK) {
      results.push_back(exists[i] != 0);
    } else {
      results.push_back(ToAbslStatus(statuses[i], msg));
    }
  }
  ctx->callback(std::move(results));
}

}  // namespace

absl::Status ToAbslStatus(RaidenStatusCode code, absl::string_view message) {
  if (code == RAIDEN_STATUS_OK) {
    return absl::OkStatus();
  }
  const int raw_code = static_cast<int>(code);
  if (raw_code < static_cast<int>(absl::StatusCode::kCancelled) ||
      raw_code > static_cast<int>(absl::StatusCode::kUnauthenticated)) {
    return absl::Status(absl::StatusCode::kUnknown,
                        message.empty() ? "Unknown plugin error" : message);
  }
  return absl::Status(static_cast<absl::StatusCode>(raw_code), message);
}

PluginBlockKeyMapper::PluginBlockKeyMapper(
    const RaidenStorageOutofTreePluginV1* vtable, void* ctx,
    std::shared_ptr<void> ctx_lifetime, ParallelismConfig parallelism)
    : vtable_(vtable),
      ctx_(ctx),
      ctx_lifetime_(std::move(ctx_lifetime)),
      parallelism_(parallelism) {}

absl::StatusOr<BlockKey> PluginBlockKeyMapper::MapKey(
    const std::string& block_hash, const KeyMappingOptions& options) const {
  if (block_hash.empty()) {
    return absl::InvalidArgumentError("block_hash must not be empty");
  }
  if (vtable_->map_key == nullptr) {
    BlockKey key;
    key.block_hash = block_hash;
    key.resolved_key = block_hash;
    return key;
  }
  auto effective = [](int per_call, int configured) {
    return per_call == kAxisUndeclared ? configured : per_call;
  };
  const ParallelismConfig effective_parallelism = {
      .tp_size = effective(options.parallelism.tp_size, parallelism_.tp_size),
      .tp_rank = effective(options.parallelism.tp_rank, parallelism_.tp_rank),
      .pcp_size =
          effective(options.parallelism.pcp_size, parallelism_.pcp_size),
      .pcp_rank =
          effective(options.parallelism.pcp_rank, parallelism_.pcp_rank),
      .pp_size = effective(options.parallelism.pp_size, parallelism_.pp_size),
      .pp_rank = effective(options.parallelism.pp_rank, parallelism_.pp_rank),
  };
  const RaidenParallelismConfig c_parallelism =
      ToCParallelism(effective_parallelism);
  std::vector<char> key_buf(kDefaultResolvedKeyCapacity, '\0');
  size_t resolved_len = 0;
  int64_t offset = 0;
  int64_t size = 0;
  char err_buf[kMaxErrorBufferBytes] = {0};

  RaidenStatusCode code = vtable_->map_key(
      ctx_, reinterpret_cast<const uint8_t*>(block_hash.data()),
      block_hash.size(), &c_parallelism, key_buf.data(), key_buf.size(),
      &resolved_len, &offset, &size, err_buf, sizeof(err_buf));
  if (code != RAIDEN_STATUS_OK) {
    return ToAbslStatus(code, err_buf);
  }
  if (resolved_len > key_buf.size()) {
    return absl::InternalError(
        absl::StrCat("Plugin map_key returned length ", resolved_len,
                     " exceeding capacity ", key_buf.size()));
  }
  BlockKey key;
  key.block_hash = block_hash;
  key.resolved_key.assign(key_buf.data(), resolved_len);
  key.offset = offset;
  key.size = size;
  return key;
}

absl::StatusOr<std::shared_ptr<PluginKVBackend>> PluginKVBackend::Create(
    const RaidenStorageOutofTreePluginV1* vtable,
    absl::flat_hash_map<std::string, std::string> properties,
    const ParallelismConfig& parallelism) {
  if (vtable == nullptr) {
    return absl::InvalidArgumentError("Plugin vtable must not be null");
  }
  if (vtable->create == nullptr || vtable->destroy == nullptr ||
      vtable->write_async == nullptr || vtable->read_async == nullptr ||
      vtable->batch_exists_async == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Plugin '", vtable->type_name ? vtable->type_name : "<null>",
        "' is missing required vtable entry points"));
  }

  std::vector<RaidenPluginProperty> c_props;
  c_props.reserve(properties.size());
  for (const auto& [k, v] : properties) {
    c_props.push_back(RaidenPluginProperty{k.c_str(), v.c_str()});
  }
  RaidenPluginConfig c_config{c_props.data(), c_props.size()};
  RaidenParallelismConfig c_parallelism = ToCParallelism(parallelism);

  void* ctx = nullptr;
  char err_buf[kMaxErrorBufferBytes] = {0};
  RaidenStatusCode status =
      vtable->create(&c_config, &c_parallelism, &ctx, err_buf, sizeof(err_buf));
  if (status != RAIDEN_STATUS_OK) {
    return ToAbslStatus(status, err_buf);
  }

  std::shared_ptr<void> ctx_lifetime(ctx, [vtable](void* p) {
    if (p != nullptr && vtable->destroy != nullptr) {
      vtable->destroy(p);
    }
  });

  auto backend = std::shared_ptr<PluginKVBackend>(
      new PluginKVBackend(vtable, ctx, ctx_lifetime, std::move(properties)));
  backend->set_mapper(std::make_shared<PluginBlockKeyMapper>(
      vtable, ctx, std::move(ctx_lifetime), parallelism));
  return backend;
}

PluginKVBackend::PluginKVBackend(
    const RaidenStorageOutofTreePluginV1* vtable, void* ctx,
    std::shared_ptr<void> ctx_lifetime,
    absl::flat_hash_map<std::string, std::string> properties)
    : KVBackend(std::move(properties)),
      vtable_(vtable),
      ctx_(ctx),
      ctx_lifetime_(std::move(ctx_lifetime)) {}

absl::Status PluginKVBackend::OnHostMemoryInitialized(
    absl::Span<const HostBufferDescriptor> shard_regions) {
  if (vtable_->register_shm_arena == nullptr) {
    return absl::OkStatus();
  }
  // Deduplicate arenas by (fd, ptr, size, offset) so each distinct shared
  // memory segment is registered once.
  absl::flat_hash_set<std::tuple<int, const void*, size_t, int64_t>> seen;
  for (const HostBufferDescriptor& region : shard_regions) {
    if (region.ptr == nullptr || region.size == 0) {
      continue;
    }
    auto key =
        std::make_tuple(region.fd, region.ptr, region.size, region.offset);
    if (!seen.insert(key).second) {
      continue;
    }
    char err_buf[kMaxErrorBufferBytes] = {0};
    RaidenStatusCode status =
        vtable_->register_shm_arena(ctx_, region.fd, region.ptr, region.size,
                                    region.offset, err_buf, sizeof(err_buf));
    if (status != RAIDEN_STATUS_OK) {
      return ToAbslStatus(status, err_buf);
    }
  }
  return absl::OkStatus();
}

void PluginKVBackend::WriteAsync(const BlockKey& key,
                                 absl::Span<const HostBufferDescriptor> slices,
                                 size_t total_bytes,
                                 std::function<void(absl::Status)> callback) {
  std::vector<RaidenHostSlice> c_slices;
  c_slices.reserve(slices.size());
  for (const HostBufferDescriptor& s : slices) {
    c_slices.push_back(
        RaidenHostSlice{s.ptr, s.size, static_cast<int32_t>(s.fd), s.offset});
  }
  RaidenBlockIoDesc desc;
  desc.key.block_hash = reinterpret_cast<const uint8_t*>(key.block_hash.data());
  desc.key.block_hash_len = key.block_hash.size();
  desc.key.resolved_key = key.resolved_key.data();
  desc.key.resolved_key_len = key.resolved_key.size();
  desc.key.offset = key.offset;
  desc.key.size = key.size;
  desc.slices = c_slices.data();
  desc.num_slices = c_slices.size();
  desc.total_bytes = total_bytes;

  auto* cb_ctx = new IoCallbackContext{std::move(callback)};
  vtable_->write_async(ctx_, &desc, &OnPluginIoComplete, cb_ctx);
}

void PluginKVBackend::ReadAsync(const BlockKey& key,
                                absl::Span<const HostBufferDescriptor> slices,
                                size_t total_bytes,
                                std::function<void(absl::Status)> callback) {
  std::vector<RaidenHostSlice> c_slices;
  c_slices.reserve(slices.size());
  for (const HostBufferDescriptor& s : slices) {
    c_slices.push_back(
        RaidenHostSlice{s.ptr, s.size, static_cast<int32_t>(s.fd), s.offset});
  }
  RaidenBlockIoDesc desc;
  desc.key.block_hash = reinterpret_cast<const uint8_t*>(key.block_hash.data());
  desc.key.block_hash_len = key.block_hash.size();
  desc.key.resolved_key = key.resolved_key.data();
  desc.key.resolved_key_len = key.resolved_key.size();
  desc.key.offset = key.offset;
  desc.key.size = key.size;
  desc.slices = c_slices.data();
  desc.num_slices = c_slices.size();
  desc.total_bytes = total_bytes;

  auto* cb_ctx = new IoCallbackContext{std::move(callback)};
  vtable_->read_async(ctx_, &desc, &OnPluginIoComplete, cb_ctx);
}

void PluginKVBackend::BatchExistsAsync(
    absl::Span<const BlockKey> keys,
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback) {
  if (keys.empty()) {
    callback({});
    return;
  }
  std::vector<RaidenBlockKey> c_keys;
  c_keys.reserve(keys.size());
  for (const BlockKey& k : keys) {
    RaidenBlockKey ck;
    ck.block_hash = reinterpret_cast<const uint8_t*>(k.block_hash.data());
    ck.block_hash_len = k.block_hash.size();
    ck.resolved_key = k.resolved_key.data();
    ck.resolved_key_len = k.resolved_key.size();
    ck.offset = k.offset;
    ck.size = k.size;
    c_keys.push_back(ck);
  }
  auto* cb_ctx =
      new BatchExistsCallbackContext{keys.size(), std::move(callback)};
  vtable_->batch_exists_async(ctx_, c_keys.data(), c_keys.size(),
                              &OnPluginBatchExistsComplete, cb_ctx);
}

}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden
