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

#include "tpu_sync/core/control_plane_backend.h"

#include <stdlib.h>

#include <gtest/gtest.h>

namespace tpu_raiden {
namespace {

TEST(ResolveControlPlaneBackendTypeTest, EnvVarAndOverridePrecedence) {
  unsetenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND");
  unsetenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE");

  // Default is kTcp
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kTcp);

  // Explicit override takes precedence
  EXPECT_EQ(ResolveControlPlaneBackendType(ControlPlaneBackendType::kGrpc),
            ControlPlaneBackendType::kGrpc);

  // TPU_RAIDEN_CONTROL_PLANE_BACKEND=grpc
  setenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "grpc", 1);
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kGrpc);
  EXPECT_EQ(ResolveControlPlaneBackendType(ControlPlaneBackendType::kTcp),
            ControlPlaneBackendType::kTcp);

  // TPU_RAIDEN_CONTROL_PLANE_BACKEND=tcp
  setenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "tcp", 1);
  setenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "1", 1);
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kTcp);

  // Fallback to TPU_RAIDEN_USE_GRPC_CONTROL_PLANE=1 when primary env unset
  unsetenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND");
  setenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "1", 1);
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kGrpc);

  setenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "true", 1);
  EXPECT_EQ(ResolveControlPlaneBackendType(), ControlPlaneBackendType::kGrpc);

  unsetenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE");
}

}  // namespace
}  // namespace tpu_raiden
