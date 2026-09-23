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

#include "tpu_sync/common/grpc_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "grpcpp/support/status.h"

namespace tpu_raiden {
namespace {

using ::testing::status::StatusIs;

TEST(GrpcUtilTest, FromGrpcStatusOk) {
  EXPECT_OK(FromGrpcStatus(::grpc::Status::OK));
}

TEST(GrpcUtilTest, FromGrpcStatusPreservesCodeAndMessage) {
  ::grpc::Status grpc_status(::grpc::StatusCode::UNIMPLEMENTED,
                             "Method not found");
  absl::Status status = FromGrpcStatus(grpc_status);
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kUnimplemented, "Method not found"));
}

TEST(GrpcUtilTest, FromGrpcStatusStreamRemovedMapsToUnavailable) {
  ::grpc::Status grpc_status(::grpc::StatusCode::UNKNOWN, "Stream removed");
  absl::Status status = FromGrpcStatus(grpc_status);
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kUnavailable, "Stream removed"));
}

TEST(GrpcUtilTest, ToGrpcStatusOk) {
  ::grpc::Status grpc_status = ToGrpcStatus(absl::OkStatus());
  EXPECT_TRUE(grpc_status.ok());
}

TEST(GrpcUtilTest, ToGrpcStatusPreservesCodeAndMessage) {
  ::grpc::Status grpc_status =
      ToGrpcStatus(absl::UnavailableError("peer down"));
  EXPECT_EQ(grpc_status.error_code(), ::grpc::StatusCode::UNAVAILABLE);
  EXPECT_EQ(grpc_status.error_message(), "peer down");
}

}  // namespace
}  // namespace tpu_raiden
