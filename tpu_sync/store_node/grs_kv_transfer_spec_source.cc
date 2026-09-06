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

#include "tpu_sync/store_node/grs_kv_transfer_spec_source.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_sync/store_node/kv_transfer_spec_source.h"
namespace tpu_raiden {
namespace store_node {

GrsKVTransferSpecSource::GrsKVTransferSpecSource(
    absl::string_view global_registry_address, absl::string_view kv_pool_group)
    : client_(grpc::CreateChannel(std::string(global_registry_address),
                                  grpc::InsecureChannelCredentials())),
      kv_pool_group_(kv_pool_group) {}

absl::StatusOr<KVTransferSpec> GrsKVTransferSpecSource::Get() {
  ABSL_ASSIGN_OR_RETURN(
      const ::tpu_raiden::kv_cache::global_registry::KVTransferSpec proto,
      client_.GetKVTransferSpec(kv_pool_group_));
  KVTransferSpec spec;
  spec.block_array_bytes.reserve(proto.block_arrays_size());
  for (const auto& array : proto.block_arrays()) {
    spec.block_array_bytes.push_back(
        static_cast<uint64_t>(array.block_bytes()));
  }
  spec.num_kv_shards = static_cast<size_t>(proto.num_kv_shards());
  spec.num_workers = static_cast<size_t>(proto.num_workers());
  return spec;
}

}  // namespace store_node
}  // namespace tpu_raiden
