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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/kv_cache_offload/global_registry/global_registry.grpc.pb.h"
#include "tpu_sync/kv_cache_offload/global_registry/global_registry.pb.h"
#include "tpu_sync/kv_cache_offload/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache_offload/global_registry/global_registry_server.h"
#include "tpu_sync/kv_cache_offload/global_registry/test_util.h"
#include "tpu_sync/store_node/kv_transfer_spec_source.h"

namespace tpu_raiden {
namespace store_node {
namespace {

class GrsKVTransferSpecSourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    service_ = std::make_unique<
        ::tpu_raiden::kv_cache::global_registry::GlobalRegistryServiceImpl>();
    grpc::ServerBuilder builder;
    builder.AddListeningPort(
        "localhost:0",
        ::tpu_raiden::kv_cache::global_registry::CreateTestServerCredentials(),
        &port_);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
    address_ = absl::StrCat("localhost:", port_);
  }

  void TearDown() override { server_->Shutdown(); }

  // Publishes a spec under `kv_pool_group` the way a serving host will.
  void Publish(
      const ::tpu_raiden::kv_cache::global_registry::KVTransferSpec& spec,
      absl::string_view kv_pool_group = "prefill_pool") {
    ::tpu_raiden::kv_cache::global_registry::GlobalRegistryClient client(
        grpc::CreateChannel(address_, ::tpu_raiden::kv_cache::global_registry::
                                          CreateTestClientCredentials()));
    ASSERT_TRUE(client.RegisterKVTransferSpec(spec, kv_pool_group).ok());
  }

  std::unique_ptr<
      ::tpu_raiden::kv_cache::global_registry::GlobalRegistryServiceImpl>
      service_;
  std::unique_ptr<grpc::Server> server_;
  int port_ = 0;
  std::string address_;
};

TEST_F(GrsKVTransferSpecSourceTest, NotFoundBeforePublish) {
  GrsKVTransferSpecSource source(address_, "prefill_pool");
  EXPECT_TRUE(absl::IsNotFound(source.Get().status()));
}

TEST_F(GrsKVTransferSpecSourceTest, RoundTripsPublishedSpec) {
  ::tpu_raiden::kv_cache::global_registry::KVTransferSpec proto;
  proto.add_block_arrays()->set_block_bytes(4096);
  proto.add_block_arrays()->set_block_bytes(512);
  proto.set_num_kv_shards(2);
  proto.set_num_workers(1);
  Publish(proto);

  GrsKVTransferSpecSource source(address_, "prefill_pool");
  absl::StatusOr<KVTransferSpec> spec = source.Get();
  ASSERT_TRUE(spec.ok()) << spec.status();
  EXPECT_EQ(spec->block_array_bytes, (std::vector<uint64_t>{4096, 512}));
  EXPECT_EQ(spec->num_kv_shards, 2u);
  EXPECT_EQ(spec->num_workers, 1u);
}

TEST_F(GrsKVTransferSpecSourceTest, RoundTripsPerWorkerSpec) {
  ::tpu_raiden::kv_cache::global_registry::KVTransferSpec proto;
  for (int w = 0; w < 2; ++w) {
    auto* worker = proto.add_workers();
    worker->set_node_id(w);
    worker->add_block_arrays()->set_block_bytes(4096);
    if (w == 1) worker->add_block_arrays()->set_block_bytes(512);
  }
  proto.set_num_kv_shards(2);
  proto.set_num_workers(2);
  Publish(proto);

  GrsKVTransferSpecSource source(address_, "prefill_pool");
  absl::StatusOr<KVTransferSpec> spec = source.Get();
  ABSL_ASSERT_OK(spec);
  EXPECT_TRUE(spec->block_array_bytes.empty());
  EXPECT_EQ(spec->worker_block_array_bytes,
            (std::vector<std::vector<uint64_t>>{{4096}, {4096, 512}}));
  EXPECT_EQ(spec->num_workers, 2u);
}

TEST_F(GrsKVTransferSpecSourceTest, OtherGroupsSpecIsNotFound) {
  ::tpu_raiden::kv_cache::global_registry::KVTransferSpec proto;
  proto.add_block_arrays()->set_block_bytes(4096);
  proto.set_num_kv_shards(2);
  proto.set_num_workers(1);
  Publish(proto, "decode_pool");

  // The node follows its configured group only; another group's spec is
  // invisible, so the source keeps reporting NotFound (and WaitForSpec keeps
  // waiting).
  GrsKVTransferSpecSource source(address_, "prefill_pool");
  EXPECT_TRUE(absl::IsNotFound(source.Get().status()));
}

TEST_F(GrsKVTransferSpecSourceTest, UnavailableWhenRegistryUnreachable) {
  server_->Shutdown();
  GrsKVTransferSpecSource source(address_, "prefill_pool");
  EXPECT_TRUE(absl::IsUnavailable(source.Get().status()));
}

// Serves one fixed spec without validating it, as a faulty registry might.
class FixedSpecRegistry final : public ::tpu_raiden::kv_cache::global_registry::
                                    GlobalRegistryService::Service {
 public:
  explicit FixedSpecRegistry(
      ::tpu_raiden::kv_cache::global_registry::KVTransferSpec spec)
      : spec_(std::move(spec)) {}

  grpc::Status GetKVTransferSpec(
      grpc::ServerContext* context,
      const ::tpu_raiden::kv_cache::global_registry::GetKVTransferSpecRequest*
          request,
      ::tpu_raiden::kv_cache::global_registry::GetKVTransferSpecResponse*
          response) override {
    response->set_found(true);
    *response->mutable_spec() = spec_;
    return grpc::Status::OK;
  }

 private:
  const ::tpu_raiden::kv_cache::global_registry::KVTransferSpec spec_;
};

absl::Status GetFromFixedRegistry(
    ::tpu_raiden::kv_cache::global_registry::KVTransferSpec spec) {
  auto registry = ::tpu_raiden::kv_cache::global_registry::
      CreateTestGlobalRegistryServerWithService(
          std::make_unique<FixedSpecRegistry>(std::move(spec)));
  return GrsKVTransferSpecSource(registry->server_address, "prefill_pool")
      .Get()
      .status();
}

TEST(GrsKVTransferSpecSourceUnvalidatedTest, RejectsNegativeBlockBytes) {
  // A negative size would otherwise wrap into a huge uint64_t.
  ::tpu_raiden::kv_cache::global_registry::KVTransferSpec uniform;
  uniform.add_block_arrays()->set_block_bytes(-4096);
  uniform.set_num_kv_shards(1);
  uniform.set_num_workers(1);
  EXPECT_TRUE(absl::IsInvalidArgument(GetFromFixedRegistry(uniform)));

  ::tpu_raiden::kv_cache::global_registry::KVTransferSpec per_worker;
  per_worker.add_workers()->add_block_arrays()->set_block_bytes(-4096);
  per_worker.set_num_kv_shards(1);
  per_worker.set_num_workers(1);
  EXPECT_TRUE(absl::IsInvalidArgument(GetFromFixedRegistry(per_worker)));
}

TEST(GrsKVTransferSpecSourceUnvalidatedTest, RejectsMisnumberedWorkers) {
  ::tpu_raiden::kv_cache::global_registry::KVTransferSpec spec;
  for (int node_id : {1, 0}) {
    auto* worker = spec.add_workers();
    worker->set_node_id(node_id);
    worker->add_block_arrays()->set_block_bytes(4096 >> node_id);
  }
  spec.set_num_kv_shards(1);
  spec.set_num_workers(2);
  EXPECT_TRUE(absl::IsInvalidArgument(GetFromFixedRegistry(spec)));
}

}  // namespace
}  // namespace store_node
}  // namespace tpu_raiden
