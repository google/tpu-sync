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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_RAIDEN_STORAGE_PLUGIN_ABI_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_RAIDEN_STORAGE_PLUGIN_ABI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAIDEN_STORAGE_PLUGIN_ABI_VERSION 1
#define RAIDEN_STORAGE_PLUGIN_ENTRY_SYMBOL "raiden_storage_plugin_get_v1"

// Status codes matching canonical absl::StatusCode values.
typedef enum RaidenStatusCode {
  RAIDEN_STATUS_OK = 0,
  RAIDEN_STATUS_CANCELLED = 1,
  RAIDEN_STATUS_UNKNOWN = 2,
  RAIDEN_STATUS_INVALID_ARGUMENT = 3,
  RAIDEN_STATUS_DEADLINE_EXCEEDED = 4,
  RAIDEN_STATUS_NOT_FOUND = 5,
  RAIDEN_STATUS_ALREADY_EXISTS = 6,
  RAIDEN_STATUS_PERMISSION_DENIED = 7,
  RAIDEN_STATUS_RESOURCE_EXHAUSTED = 8,
  RAIDEN_STATUS_FAILED_PRECONDITION = 9,
  RAIDEN_STATUS_ABORTED = 10,
  RAIDEN_STATUS_OUT_OF_RANGE = 11,
  RAIDEN_STATUS_UNIMPLEMENTED = 12,
  RAIDEN_STATUS_INTERNAL = 13,
  RAIDEN_STATUS_UNAVAILABLE = 14,
  RAIDEN_STATUS_DATA_LOSS = 15,
  RAIDEN_STATUS_UNAUTHENTICATED = 16,
} RaidenStatusCode;

typedef struct RaidenPluginProperty {
  const char* key;
  const char* value;
} RaidenPluginProperty;

typedef struct RaidenPluginConfig {
  const RaidenPluginProperty* properties;
  size_t num_properties;
} RaidenPluginConfig;

// The host adds this property to RaidenPluginConfig on every create() call.
// Its value is the role of the new instance. The coordinator instance only maps
// keys and probes existence, and a worker instance reads and writes blocks, so
// a plugin can size the resources of an instance by its role.
#define RAIDEN_PLUGIN_PROPERTY_ROLE "raiden_role"
#define RAIDEN_PLUGIN_ROLE_COORDINATOR "coordinator"
#define RAIDEN_PLUGIN_ROLE_WORKER "worker"

// Serving parallelism coordinates (-1 means undeclared).
typedef struct RaidenParallelismConfig {
  int32_t tp_size;
  int32_t tp_rank;
  int32_t pcp_size;
  int32_t pcp_rank;
  int32_t pp_size;
  int32_t pp_rank;
} RaidenParallelismConfig;

// Resolved backend locator for a single block or rank shard.
typedef struct RaidenBlockKey {
  const uint8_t* block_hash;
  size_t block_hash_len;
  const char* resolved_key;
  size_t resolved_key_len;
  int64_t offset;
  int64_t size;
} RaidenBlockKey;

// Host DRAM slice for a single (layer, shard) sub-buffer of a block.
typedef struct RaidenHostSlice {
  void* ptr;
  size_t size;
  int32_t fd;
  int64_t offset;
} RaidenHostSlice;

typedef struct RaidenBlockIoDesc {
  RaidenBlockKey key;
  const RaidenHostSlice* slices;
  size_t num_slices;
  size_t total_bytes;
} RaidenBlockIoDesc;

typedef void (*RaidenIoCompletionCallback)(void* user_data,
                                           RaidenStatusCode status,
                                           const char* error_msg);

typedef void (*RaidenBatchExistsCallback)(void* user_data,
                                          const RaidenStatusCode* statuses,
                                          const uint8_t* exists,
                                          size_t num_keys,
                                          const char* error_msg);

// Versioned C function table exported by an out-of-tree storage plugin.
typedef struct RaidenStorageOutofTreePluginV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  const char* type_name;

  RaidenStatusCode (*create)(const RaidenPluginConfig* config,
                             const RaidenParallelismConfig* parallelism,
                             void** out_ctx, char* err_buf, size_t err_buf_len);

  void (*destroy)(void* ctx);

  // Optional hook called once per shared-memory arena during initialization.
  RaidenStatusCode (*register_shm_arena)(void* ctx, int fd, void* base_ptr,
                                         size_t size_bytes, int64_t fd_offset,
                                         char* err_buf, size_t err_buf_len);

  // Optional key mapper. Leave NULL when the coordinator probes block existence
  // directly through batch_exists_async with raw block hashes.
  RaidenStatusCode (*map_key)(void* ctx, const uint8_t* block_hash,
                              size_t block_hash_len,
                              const RaidenParallelismConfig* parallelism,
                              char* out_resolved_key,
                              size_t out_resolved_key_capacity,
                              size_t* out_resolved_key_len, int64_t* out_offset,
                              int64_t* out_size, char* err_buf,
                              size_t err_buf_len);

  void (*write_async)(void* ctx, const RaidenBlockIoDesc* desc,
                      RaidenIoCompletionCallback callback, void* user_data);

  void (*read_async)(void* ctx, const RaidenBlockIoDesc* desc,
                     RaidenIoCompletionCallback callback, void* user_data);

  void (*batch_exists_async)(void* ctx, const RaidenBlockKey* keys,
                             size_t num_keys,
                             RaidenBatchExistsCallback callback,
                             void* user_data);
} RaidenStorageOutofTreePluginV1;

// Minimum `struct_size` required for a valid v1.0 plugin vtable (up through
// `batch_exists_async`). Additive v1.x extensions may append optional function
// pointers after `batch_exists_async`; hosts must check
// `RAIDEN_PLUGIN_V1_HAS_FIELD` before reading fields beyond this minimum size.
#define RAIDEN_STORAGE_PLUGIN_V1_MIN_SIZE                         \
  (offsetof(RaidenStorageOutofTreePluginV1, batch_exists_async) + \
   sizeof(((const RaidenStorageOutofTreePluginV1*)0)->batch_exists_async))

#define RAIDEN_PLUGIN_V1_HAS_FIELD(vtable, field)         \
  ((vtable) != NULL &&                                    \
   (vtable)->struct_size >=                               \
       (offsetof(RaidenStorageOutofTreePluginV1, field) + \
        sizeof(((const RaidenStorageOutofTreePluginV1*)0)->field)))

typedef const RaidenStorageOutofTreePluginV1* (*RaidenStoragePluginGetV1Fn)(
    void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_RAIDEN_STORAGE_PLUGIN_ABI_H_
