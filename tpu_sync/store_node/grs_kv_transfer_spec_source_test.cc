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
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_server.h"
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
    builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
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
        grpc::CreateChannel(address_, grpc::InsecureChannelCredentials()));
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
  ASSERT_TRUE(spec.ok()) << spec.status();
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

}  // namespace
}  // namespace store_node
}  // namespace tpu_raiden
