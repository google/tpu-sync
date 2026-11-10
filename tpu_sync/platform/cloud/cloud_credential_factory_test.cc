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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tpu_sync/platform/grpc_credential_factory.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::WhenDynamicCastTo;

// Observed on Forge (non-GCP Linux workers where grpc_alts_is_running_on_gcp()
// returns false):
// - AltsServerCredentials() still constructs a non-null SecureServerCredentials
//   wrapper around a null core credential pointer, so CreateServerCredentials()
//   returns OK with a non-null pointer.
// - AltsCredentials() returns nullptr, so CreateClientCredentials() returns
//   FAILED_PRECONDITION ("grpc::experimental::AltsCredentials() returned
//   nullptr (ALTS credentials require running on GCP).").
// On GCP machines, both return OK with non-null pointers. Tests accept either
// non-null OK credentials (GCP) or ALTS kFailedPrecondition (non-GCP) so they
// remain deterministic across Forge and GCE environments.
auto IsNonNullOrAltsPrecondition() {
  return AnyOf(
      IsOkAndHolds(NotNull()),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("ALTS")));
}

TEST(CloudCredentialFactoryTest,
     CreateServerCredentialsReturnsNonNullOrAltsPrecondition) {
  CloudCredentialFactory factory;
  EXPECT_THAT(factory.CreateServerCredentials(), IsNonNullOrAltsPrecondition());
}

TEST(CloudCredentialFactoryTest,
     CreateClientCredentialsReturnsNonNullOrAltsPrecondition) {
  CloudCredentialFactory factory;
  EXPECT_THAT(factory.CreateClientCredentials(), IsNonNullOrAltsPrecondition());
}

TEST(CloudCredentialFactoryTest,
     CreateGrpcCredentialFactoryReturnsWorkingFactory) {
  std::shared_ptr<GrpcCredentialFactory> factory =
      CreateGrpcCredentialFactory();
  ASSERT_THAT(factory, NotNull());
  EXPECT_THAT(factory.get(),
              WhenDynamicCastTo<CloudCredentialFactory*>(NotNull()));
  EXPECT_THAT(factory->CreateServerCredentials(),
              IsNonNullOrAltsPrecondition());
  EXPECT_THAT(factory->CreateClientCredentials(),
              IsNonNullOrAltsPrecondition());
}

}  // namespace
}  // namespace tpu_raiden
