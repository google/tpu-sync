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

#include "tpu_sync/common/control_pipe/control_pipe_types.h"

#include <cstdlib>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "tpu_sync/proto/control_pipe.pb.h"

namespace tpu_raiden {
namespace {

class EnvVarScoper {
 public:
  EnvVarScoper(const char* name, const char* val) : name_(name) {
    if (const char* old = std::getenv(name)) {
      old_val_ = old;
    }
    if (val != nullptr) {
      setenv(name, val, 1);
    } else {
      unsetenv(name);
    }
  }

  ~EnvVarScoper() {
    if (old_val_.has_value()) {
      setenv(name_.c_str(), old_val_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> old_val_;
};

TEST(ControlPipeTypesTest, DefaultBackendTypeIsTcp) {
  EnvVarScoper backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", nullptr);
  EnvVarScoper flag_env("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", nullptr);
  EXPECT_EQ(ResolveControlPipeBackendType(), ControlPipeBackendType::kTcp);
  EXPECT_EQ(ControlPipeBackendTypeName(ControlPipeBackendType::kTcp), "tcp");
  EXPECT_EQ(ControlPipeBackendTypeName(ControlPipeBackendType::kGrpc), "grpc");
  EXPECT_EQ(ControlPipeBackendTypeName(ControlPipeBackendType::kZmq), "zmq");
}

TEST(ControlPipeTypesTest, ParseBackendType) {
  EXPECT_EQ(ParseControlPipeBackendType("tcp"), ControlPipeBackendType::kTcp);
  EXPECT_EQ(ParseControlPipeBackendType("TCP"), ControlPipeBackendType::kTcp);
  EXPECT_EQ(ParseControlPipeBackendType("grpc"), ControlPipeBackendType::kGrpc);
  EXPECT_EQ(ParseControlPipeBackendType("gRPC"), ControlPipeBackendType::kGrpc);
  EXPECT_EQ(ParseControlPipeBackendType("zmq"), ControlPipeBackendType::kZmq);
  EXPECT_EQ(ParseControlPipeBackendType("ZMQ"), ControlPipeBackendType::kZmq);
  EXPECT_EQ(ParseControlPipeBackendType("unknown"), std::nullopt);
}

TEST(ControlPipeTypesTest, OverrideTakesPrecedence) {
  EnvVarScoper backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "tcp");
  EXPECT_EQ(ResolveControlPipeBackendType(ControlPipeBackendType::kGrpc),
            ControlPipeBackendType::kGrpc);
  EXPECT_EQ(ResolveControlPipeBackendType(ControlPipeBackendType::kZmq),
            ControlPipeBackendType::kZmq);

  EnvVarScoper backend_env_grpc("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "grpc");
  EXPECT_EQ(ResolveControlPipeBackendType(ControlPipeBackendType::kTcp),
            ControlPipeBackendType::kTcp);
}

TEST(ControlPipeTypesTest, ResolvesFromEnvVar) {
  EnvVarScoper flag_env("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", nullptr);
  {
    EnvVarScoper backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "GRPC");
    EXPECT_EQ(ResolveControlPipeBackendType(), ControlPipeBackendType::kGrpc);
  }
  {
    EnvVarScoper backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "ZMQ");
    EXPECT_EQ(ResolveControlPipeBackendType(), ControlPipeBackendType::kZmq);
  }
  {
    EnvVarScoper backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "TCP");
    EXPECT_EQ(ResolveControlPipeBackendType(), ControlPipeBackendType::kTcp);
  }
  {
    EnvVarScoper backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "unknown");
    EXPECT_EQ(ResolveControlPipeBackendType(), ControlPipeBackendType::kTcp);
  }
}

TEST(ControlPipeTypesTest, ResolvesFromLegacyUseGrpcFlag) {
  EnvVarScoper backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", nullptr);
  {
    EnvVarScoper flag_env("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "1");
    EXPECT_EQ(ResolveControlPipeBackendType(), ControlPipeBackendType::kGrpc);
  }
  {
    EnvVarScoper flag_env("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "true");
    EXPECT_EQ(ResolveControlPipeBackendType(), ControlPipeBackendType::kGrpc);
  }
  {
    EnvVarScoper flag_env("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "0");
    EXPECT_EQ(ResolveControlPipeBackendType(), ControlPipeBackendType::kTcp);
  }
}

TEST(ControlPipeTypesTest, ControlEnvelopeProtoRoundtrip) {
  control_pipe::proto::ControlEnvelope env;
  env.set_message_type("tpu_sync.rpc.ControlRequest");
  env.set_request_id(42);
  env.set_payload("test_payload");
  (*env.mutable_metadata())["key"] = "val";

  std::string serialized;
  ASSERT_TRUE(env.SerializeToString(&serialized));

  control_pipe::proto::ControlEnvelope decoded;
  ASSERT_TRUE(decoded.ParseFromString(serialized));
  EXPECT_EQ(decoded.message_type(), "tpu_sync.rpc.ControlRequest");
  EXPECT_EQ(decoded.request_id(), 42);
  EXPECT_EQ(decoded.payload(), "test_payload");
  EXPECT_EQ(decoded.metadata().at("key"), "val");
}

}  // namespace
}  // namespace tpu_raiden
