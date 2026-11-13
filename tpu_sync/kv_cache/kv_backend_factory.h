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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_BACKEND_FACTORY_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_BACKEND_FACTORY_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {

// Factory registry for dynamic creation of worker-side KVBackend instances.
// Mirrors KVCacheStoreBackendFactory so built-in backends and dynamically
// loaded plugins share a single worker registration path.
class KVBackendFactory {
 public:
  using BackendCreator =
      std::function<absl::StatusOr<std::shared_ptr<backends::KVBackend>>(
          const BackendConfig& config)>;

  static KVBackendFactory& Instance();

  // Registers a creator function for `type_name`. Returns AlreadyExistsError if
  // `type_name` is already registered.
  absl::Status RegisterBackend(absl::string_view type_name,
                               BackendCreator creator);

  // Constructs a worker KVBackend instance from `config`.
  absl::StatusOr<std::shared_ptr<backends::KVBackend>> CreateBackend(
      const BackendConfig& config) const;

  bool IsRegistered(absl::string_view type_name) const;

  std::vector<std::string> GetRegisteredTypes() const;

  static absl::Status Register(absl::string_view type_name,
                               BackendCreator creator) {
    return Instance().RegisterBackend(type_name, std::move(creator));
  }

  static absl::StatusOr<std::shared_ptr<backends::KVBackend>> Create(
      const BackendConfig& config) {
    return Instance().CreateBackend(config);
  }

 private:
  friend class absl::NoDestructor<KVBackendFactory>;

  KVBackendFactory() = default;
  ~KVBackendFactory() = default;

  KVBackendFactory(const KVBackendFactory&) = delete;
  KVBackendFactory& operator=(const KVBackendFactory&) = delete;

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, BackendCreator> creators_
      ABSL_GUARDED_BY(mutex_);
};

namespace internal {
struct KVBackendRegistrar {
  KVBackendRegistrar(absl::string_view type_name,
                     KVBackendFactory::BackendCreator creator) {
    (void)KVBackendFactory::Instance().RegisterBackend(type_name,
                                                       std::move(creator));
  }
};
}  // namespace internal

}  // namespace kv_cache
}  // namespace tpu_raiden

#define REGISTER_KV_BACKEND_CONCAT_INNER(a, b) a##b
#define REGISTER_KV_BACKEND_CONCAT(a, b) REGISTER_KV_BACKEND_CONCAT_INNER(a, b)

#define REGISTER_KV_BACKEND(type_name, creator_func)                    \
  namespace {                                                           \
  static ::tpu_raiden::kv_cache::internal::KVBackendRegistrar           \
      REGISTER_KV_BACKEND_CONCAT(_raiden_kv_backend_reg_,               \
                                 __COUNTER__)(type_name, creator_func); \
  }  // namespace

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_KV_BACKEND_FACTORY_H_
