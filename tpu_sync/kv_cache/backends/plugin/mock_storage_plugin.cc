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

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tpu_sync/kv_cache/backends/plugin/raiden_storage_plugin_abi.h"

namespace {

// The mock is built like an out-of-tree plugin: it uses only the C ABI header
// and the C++ standard library, so it locks with std::mutex, not absl::Mutex.
using Mutex = std::mutex;  // NOLINT(build/c++11)

struct SharedMockStore {
  Mutex mu;
  std::unordered_map<std::string, std::vector<uint8_t>> objects;
  size_t shm_arenas_registered = 0;
  int last_registered_fd = -1;
  size_t last_registered_size = 0;
};

SharedMockStore& GlobalMockStore() {
  static auto* store = new SharedMockStore();
  return *store;
}

struct MockPluginContext {
  std::string prefix = "mock";
  std::string root_dir;
  bool use_pread_on_write = false;
  RaidenParallelismConfig parallelism{};
};

std::string BytesToHex(const uint8_t* data, size_t len) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(len * 2);
  for (size_t i = 0; i < len; ++i) {
    out.push_back(kHex[(data[i] >> 4) & 0xF]);
    out.push_back(kHex[data[i] & 0xF]);
  }
  return out;
}

void WriteError(char* err_buf, size_t err_buf_len, const char* msg) {
  if (err_buf != nullptr && err_buf_len > 0) {
    std::snprintf(err_buf, err_buf_len, "%s", msg);
  }
}

RaidenStatusCode MockCreate(const RaidenPluginConfig* config,
                            const RaidenParallelismConfig* parallelism,
                            void** out_ctx, char* err_buf, size_t err_buf_len) {
  if (out_ctx == nullptr) {
    WriteError(err_buf, err_buf_len, "out_ctx is null");
    return RAIDEN_STATUS_INVALID_ARGUMENT;
  }
  auto* ctx = new MockPluginContext();
  if (parallelism != nullptr) {
    ctx->parallelism = *parallelism;
  }
  if (config != nullptr) {
    for (size_t i = 0; i < config->num_properties; ++i) {
      const char* k = config->properties[i].key;
      const char* v = config->properties[i].value;
      if (k == nullptr || v == nullptr) continue;
      if (std::strcmp(k, "fail_create") == 0 && std::strcmp(v, "true") == 0) {
        delete ctx;
        WriteError(err_buf, err_buf_len, "simulated create failure");
        return RAIDEN_STATUS_INVALID_ARGUMENT;
      }
      if (std::strcmp(k, "prefix") == 0) {
        ctx->prefix = v;
      } else if (std::strcmp(k, "root_dir") == 0) {
        ctx->root_dir = v;
      } else if (std::strcmp(k, "use_pread_on_write") == 0) {
        ctx->use_pread_on_write =
            (std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0);
      } else if (std::strcmp(k, "clear_store_on_create") == 0 &&
                 std::strcmp(v, "true") == 0) {
        SharedMockStore& store = GlobalMockStore();
        std::lock_guard<Mutex> lock(store.mu);
        store.objects.clear();
        store.shm_arenas_registered = 0;
        store.last_registered_fd = -1;
        store.last_registered_size = 0;
      }
    }
  }
  *out_ctx = ctx;
  return RAIDEN_STATUS_OK;
}

void MockDestroy(void* raw_ctx) {
  delete static_cast<MockPluginContext*>(raw_ctx);
}

RaidenStatusCode MockRegisterShmArena(void* /*raw_ctx*/, int fd, void* base_ptr,
                                      size_t size_bytes, int64_t /*fd_offset*/,
                                      char* err_buf, size_t err_buf_len) {
  if (base_ptr == nullptr || size_bytes == 0) {
    WriteError(err_buf, err_buf_len, "invalid arena base_ptr or size_bytes");
    return RAIDEN_STATUS_INVALID_ARGUMENT;
  }
  if (fd >= 0) {
    struct stat st{};
    if (fstat(fd, &st) != 0) {
      WriteError(err_buf, err_buf_len, "fstat failed on arena fd");
      return RAIDEN_STATUS_FAILED_PRECONDITION;
    }
  }
  SharedMockStore& store = GlobalMockStore();
  std::lock_guard<Mutex> lock(store.mu);
  store.shm_arenas_registered++;
  store.last_registered_fd = fd;
  store.last_registered_size = size_bytes;
  return RAIDEN_STATUS_OK;
}

RaidenStatusCode MockMapKey(
    void* raw_ctx, const uint8_t* block_hash, size_t block_hash_len,
    const RaidenParallelismConfig* parallelism, char* out_resolved_key,
    size_t out_resolved_key_capacity, size_t* out_resolved_key_len,
    int64_t* out_offset, int64_t* out_size, char* err_buf, size_t err_buf_len) {
  if (block_hash == nullptr || block_hash_len == 0) {
    WriteError(err_buf, err_buf_len, "empty block_hash");
    return RAIDEN_STATUS_INVALID_ARGUMENT;
  }
  auto* ctx = static_cast<MockPluginContext*>(raw_ctx);
  const RaidenParallelismConfig p =
      parallelism != nullptr ? *parallelism : ctx->parallelism;
  std::string key = ctx->prefix + "/" + BytesToHex(block_hash, block_hash_len) +
                    "_pp_" + std::to_string(p.pp_rank) + "_pcp_" +
                    std::to_string(p.pcp_rank) + "_tp_" +
                    std::to_string(p.tp_rank);
  if (key.size() > out_resolved_key_capacity) {
    WriteError(err_buf, err_buf_len, "resolved key exceeds capacity");
    return RAIDEN_STATUS_RESOURCE_EXHAUSTED;
  }
  std::memcpy(out_resolved_key, key.data(), key.size());
  *out_resolved_key_len = key.size();
  *out_offset = 0;
  *out_size = 0;
  return RAIDEN_STATUS_OK;
}

std::string ExtractKey(const RaidenBlockKey& key) {
  if (key.resolved_key != nullptr && key.resolved_key_len > 0) {
    return std::string(key.resolved_key, key.resolved_key_len);
  }
  if (key.block_hash != nullptr && key.block_hash_len > 0) {
    return std::string(reinterpret_cast<const char*>(key.block_hash),
                       key.block_hash_len);
  }
  return "";
}

void MockWriteAsync(void* raw_ctx, const RaidenBlockIoDesc* desc,
                    RaidenIoCompletionCallback callback, void* user_data) {
  auto* ctx = static_cast<MockPluginContext*>(raw_ctx);
  if (desc == nullptr) {
    callback(user_data, RAIDEN_STATUS_INVALID_ARGUMENT, "desc is null");
    return;
  }
  std::string key = ExtractKey(desc->key);
  if (key.empty()) {
    callback(user_data, RAIDEN_STATUS_INVALID_ARGUMENT, "empty key");
    return;
  }

  std::vector<uint8_t> payload;
  payload.reserve(desc->total_bytes);
  for (size_t i = 0; i < desc->num_slices; ++i) {
    const RaidenHostSlice& slice = desc->slices[i];
    if (slice.size == 0) continue;
    if (ctx->use_pread_on_write) {
      if (slice.fd < 0) {
        callback(user_data, RAIDEN_STATUS_FAILED_PRECONDITION,
                 "expected valid slice.fd for pread write");
        return;
      }
      std::vector<uint8_t> buf(slice.size);
      ssize_t n = pread(slice.fd, buf.data(), slice.size, slice.offset);
      if (n != static_cast<ssize_t>(slice.size)) {
        callback(user_data, RAIDEN_STATUS_INTERNAL,
                 "pread from slice.fd failed");
        return;
      }
      payload.insert(payload.end(), buf.begin(), buf.end());
    } else {
      if (slice.ptr == nullptr) {
        callback(user_data, RAIDEN_STATUS_INVALID_ARGUMENT,
                 "slice.ptr is null");
        return;
      }
      const auto* bytes = static_cast<const uint8_t*>(slice.ptr);
      payload.insert(payload.end(), bytes, bytes + slice.size);
    }
  }

  if (!ctx->root_dir.empty()) {
    std::filesystem::path file_path =
        std::filesystem::path(ctx->root_dir) / (key + ".bin");
    std::error_code ec;
    std::filesystem::create_directories(file_path.parent_path(), ec);
    if (ec) {
      callback(user_data, RAIDEN_STATUS_INTERNAL,
               "failed to create root_dir parent directory");
      return;
    }
    std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
    if (!ofs.write(reinterpret_cast<const char*>(payload.data()),
                   static_cast<std::streamsize>(payload.size()))) {
      callback(user_data, RAIDEN_STATUS_INTERNAL,
               "failed to write file under root_dir");
      return;
    }
    ofs.close();
  } else {
    SharedMockStore& store = GlobalMockStore();
    std::lock_guard<Mutex> lock(store.mu);
    store.objects[std::move(key)] = std::move(payload);
  }
  callback(user_data, RAIDEN_STATUS_OK, nullptr);
}

void MockReadAsync(void* raw_ctx, const RaidenBlockIoDesc* desc,
                   RaidenIoCompletionCallback callback, void* user_data) {
  auto* ctx = static_cast<MockPluginContext*>(raw_ctx);
  if (desc == nullptr) {
    callback(user_data, RAIDEN_STATUS_INVALID_ARGUMENT, "desc is null");
    return;
  }
  std::string key = ExtractKey(desc->key);
  std::vector<uint8_t> payload;
  if (!ctx->root_dir.empty()) {
    std::filesystem::path file_path =
        std::filesystem::path(ctx->root_dir) / (key + ".bin");
    std::ifstream ifs(file_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
      callback(user_data, RAIDEN_STATUS_NOT_FOUND, "object file not found");
      return;
    }
    const std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    payload.resize(static_cast<size_t>(size));
    if (size > 0 && !ifs.read(reinterpret_cast<char*>(payload.data()), size)) {
      callback(user_data, RAIDEN_STATUS_INTERNAL, "failed to read object file");
      return;
    }
  } else {
    SharedMockStore& store = GlobalMockStore();
    std::lock_guard<Mutex> lock(store.mu);
    auto it = store.objects.find(key);
    if (it == store.objects.end()) {
      callback(user_data, RAIDEN_STATUS_NOT_FOUND, "object not found");
      return;
    }
    payload = it->second;
  }

  size_t cursor = 0;
  for (size_t i = 0; i < desc->num_slices; ++i) {
    const RaidenHostSlice& slice = desc->slices[i];
    if (cursor + slice.size > payload.size()) {
      callback(user_data, RAIDEN_STATUS_DATA_LOSS,
               "stored payload shorter than requested slices");
      return;
    }
    std::memcpy(slice.ptr, payload.data() + cursor, slice.size);
    cursor += slice.size;
  }
  callback(user_data, RAIDEN_STATUS_OK, nullptr);
}

void MockBatchExistsAsync(void* raw_ctx, const RaidenBlockKey* keys,
                          size_t num_keys, RaidenBatchExistsCallback callback,
                          void* user_data) {
  auto* ctx = static_cast<MockPluginContext*>(raw_ctx);
  std::vector<RaidenStatusCode> statuses(num_keys, RAIDEN_STATUS_OK);
  std::vector<uint8_t> exists(num_keys, 0);
  if (!ctx->root_dir.empty()) {
    for (size_t i = 0; i < num_keys; ++i) {
      std::string key = ExtractKey(keys[i]);
      std::filesystem::path file_path =
          std::filesystem::path(ctx->root_dir) / (key + ".bin");
      std::error_code ec;
      exists[i] = std::filesystem::exists(file_path, ec) ? 1 : 0;
    }
  } else {
    SharedMockStore& store = GlobalMockStore();
    std::lock_guard<Mutex> lock(store.mu);
    for (size_t i = 0; i < num_keys; ++i) {
      std::string key = ExtractKey(keys[i]);
      exists[i] = store.objects.find(key) != store.objects.end() ? 1 : 0;
    }
  }
  callback(user_data, statuses.data(), exists.data(), num_keys, nullptr);
}

const RaidenStorageOutofTreePluginV1 kMockPluginVtable = {
    RAIDEN_STORAGE_PLUGIN_ABI_VERSION,
    sizeof(RaidenStorageOutofTreePluginV1),
    "mock_plugin",
    &MockCreate,
    &MockDestroy,
    &MockRegisterShmArena,
    &MockMapKey,
    &MockWriteAsync,
    &MockReadAsync,
    &MockBatchExistsAsync,
};

}  // namespace

extern "C" __attribute__((visibility("default")))
const RaidenStorageOutofTreePluginV1*
raiden_storage_plugin_get_v1(void) {
  return &kMockPluginVtable;
}
