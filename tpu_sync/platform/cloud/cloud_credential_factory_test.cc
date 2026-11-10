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

#include "tpu_sync/platform/cloud/cloud_credential_factory.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "tpu_sync/platform/grpc_credential_factory.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::NotNull;
using ::testing::WhenDynamicCastTo;

TEST(CloudCredentialFactoryTest, CreateServerCredentialsReturnsNonNull) {
  CloudCredentialFactory factory;
  EXPECT_THAT(factory.CreateServerCredentials(), IsOkAndHolds(NotNull()));
}

TEST(CloudCredentialFactoryTest, CreateClientCredentialsReturnsNonNull) {
  CloudCredentialFactory factory;
  EXPECT_THAT(factory.CreateClientCredentials(), IsOkAndHolds(NotNull()));
}

TEST(CloudCredentialFactoryTest,
     CreateGrpcCredentialFactoryReturnsWorkingFactory) {
  std::shared_ptr<GrpcCredentialFactory> factory =
      CreateGrpcCredentialFactory();
  ASSERT_THAT(factory, NotNull());
  EXPECT_THAT(factory.get(),
              WhenDynamicCastTo<CloudCredentialFactory*>(NotNull()));
  EXPECT_THAT(factory->CreateServerCredentials(), IsOkAndHolds(NotNull()));
  EXPECT_THAT(factory->CreateClientCredentials(), IsOkAndHolds(NotNull()));
}

}  // namespace
}  // namespace tpu_raiden
