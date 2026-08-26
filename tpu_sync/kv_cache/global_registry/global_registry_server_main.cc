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

#include <grpcpp/grpcpp.h>

#include <csignal>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_server.h"

ABSL_FLAG(int32_t, port, 50051, "Port to listen on");
ABSL_FLAG(absl::Duration, default_ttl, absl::InfiniteDuration(),
          "Default TTL for registrations");
ABSL_FLAG(absl::Duration, cleanup_interval, absl::Seconds(300),
          "Interval for cleanup thread");
ABSL_FLAG(
    int64_t, pull_owned_batch_size,
    tpu_raiden::kv_cache::global_registry::GlobalRegistryServiceImpl::
        kDefaultPullOwnedBatchSize,
    "Maximum number of entries per streamed PullOwned response message");

absl::Status RunServer() {
  std::string server_address =
      "[::]:" + std::to_string(absl::GetFlag(FLAGS_port));

  absl::Duration default_ttl = absl::GetFlag(FLAGS_default_ttl);
  absl::Duration cleanup_interval = absl::GetFlag(FLAGS_cleanup_interval);

  tpu_raiden::kv_cache::global_registry::GlobalRegistryServiceImpl service(
      default_ttl, cleanup_interval,
      absl::GetFlag(FLAGS_pull_owned_batch_size));

  grpc::ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (server == nullptr) {
    return absl::InternalError(
        absl::StrCat("Failed to listen on ", server_address,
                     ": the port is already in use or not permitted for this "
                     "process"));
  }

  std::cout << "Server listening on " << server_address << std::endl;
  LOG(INFO) << "Server listening on " << server_address;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
#ifndef _WIN32
  std::signal(SIGPIPE, SIG_IGN);
#endif
  absl::Status status = RunServer();
  if (!status.ok()) {
    // LOG(ERROR) reaches stderr on its own; only the announcement in
    // RunServer needs std::cout to be visible without a log sink.
    LOG(ERROR) << status;
    return 1;
  }
  return 0;
}
