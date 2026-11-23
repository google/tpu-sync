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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_KV_CACHE_MANAGER_FFI_BINDINGS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_KV_CACHE_MANAGER_FFI_BINDINGS_H_

#include <nanobind/nanobind.h>

namespace tpu_raiden::kv_cache {

// Registers the KVCacheManager FFI lifecycle helpers on `m`.
void BindKvCacheManagerFfi(nanobind::module_& m);

}  // namespace tpu_raiden::kv_cache

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_KV_CACHE_MANAGER_FFI_BINDINGS_H_
