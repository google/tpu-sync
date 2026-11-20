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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tpu_sync/kv_cache_offload/global_registry/global_registry.pb.h"
#include "tpu_sync/platform/grpc_credential_factory.h"
#include "tpu_sync/store_node/kv_transfer_spec_source.h"
namespace tpu_raiden {
namespace store_node {
namespace {

using KVTransferSpecProto =
    ::tpu_raiden::kv_cache::global_registry::KVTransferSpec;

// The registry rejects non-positive block_bytes when a spec is registered;
// check again so a bad value never wraps into a huge uint64_t.
absl::StatusOr<uint64_t> PositiveBlockBytes(
    const KVTransferSpecProto::BlockArray& array) {
  if (array.block_bytes() <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("registry returned a KVTransferSpec with block_bytes ",
                     array.block_bytes()));
  }
  return static_cast<uint64_t>(array.block_bytes());
}

std::shared_ptr<grpc::ChannelCredentials> CreateClientCredentialsOrDie() {
  absl::StatusOr<std::shared_ptr<grpc::ChannelCredentials>> creds =
      CreateGrpcCredentialFactory()->CreateClientCredentials();
  CHECK_OK(creds.status());
  return *std::move(creds);
}

}  // namespace

GrsKVTransferSpecSource::GrsKVTransferSpecSource(
    absl::string_view global_registry_address, absl::string_view kv_pool_group)
    : client_(grpc::CreateChannel(std::string(global_registry_address),
                                  CreateClientCredentialsOrDie())),
      kv_pool_group_(kv_pool_group) {}

absl::StatusOr<KVTransferSpec> GrsKVTransferSpecSource::Get() {
  ABSL_ASSIGN_OR_RETURN(const KVTransferSpecProto proto,
                        client_.GetKVTransferSpec(kv_pool_group_));
  KVTransferSpec spec;
  spec.block_array_bytes.reserve(proto.block_arrays_size());
  for (const auto& array : proto.block_arrays()) {
    ABSL_ASSIGN_OR_RETURN(const uint64_t bytes, PositiveBlockBytes(array));
    spec.block_array_bytes.push_back(bytes);
  }
  for (int w = 0; w < proto.workers_size(); ++w) {
    const auto& worker = proto.workers(w);
    if (worker.node_id() != w) {
      return absl::InvalidArgumentError(
          absl::StrCat("registry returned a KVTransferSpec whose workers[", w,
                       "] has node_id ", worker.node_id()));
    }
    std::vector<uint64_t>& arrays =
        spec.worker_block_array_bytes.emplace_back();
    arrays.reserve(worker.block_arrays_size());
    for (const auto& array : worker.block_arrays()) {
      ABSL_ASSIGN_OR_RETURN(const uint64_t bytes, PositiveBlockBytes(array));
      arrays.push_back(bytes);
    }
  }
  spec.num_kv_shards = static_cast<size_t>(proto.num_kv_shards());
  spec.num_workers = static_cast<size_t>(proto.num_workers());
  return spec;
}

}  // namespace store_node
}  // namespace tpu_raiden
