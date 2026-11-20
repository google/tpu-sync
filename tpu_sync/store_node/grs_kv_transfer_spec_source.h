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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_STORE_NODE_GRS_KV_TRANSFER_SPEC_SOURCE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_STORE_NODE_GRS_KV_TRANSFER_SPEC_SOURCE_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/kv_cache_offload/global_registry/global_registry_client.h"
#include "tpu_sync/store_node/kv_transfer_spec_source.h"

namespace tpu_raiden {
namespace store_node {

// KVTransferSpec from the global registry (GetKVTransferSpec RPC).
// `kv_pool_group` selects whose spec to read: specs are registered per
// group of geometry-identical raiden units (see global_registry.proto), so
// the group this node is deployed to serve must be named explicitly.
class GrsKVTransferSpecSource : public KVTransferSpecSource {
 public:
  GrsKVTransferSpecSource(absl::string_view global_registry_address,
                          absl::string_view kv_pool_group);

  // NotFound until a serving host has published the group's spec;
  // Unavailable while the registry is unreachable. Both are retried by
  // WaitForSpec.
  absl::StatusOr<KVTransferSpec> Get() override;

 private:
  kv_cache::global_registry::GlobalRegistryClient client_;
  const std::string kv_pool_group_;
};

}  // namespace store_node
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_STORE_NODE_GRS_KV_TRANSFER_SPEC_SOURCE_H_
