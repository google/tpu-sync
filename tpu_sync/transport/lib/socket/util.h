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

#ifndef TPU_SYNC_TRANSPORT_LIB_SOCKET_UTIL_H_
#define TPU_SYNC_TRANSPORT_LIB_SOCKET_UTIL_H_

#include <atomic>
#include <chrono>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "grpcpp/channel.h"

namespace tpu_raiden::transport::lib {

// Connects to remote TCP peer with optional local IP binding and optional
// gRPC channel for TCP-over-PSP out-of-band key exchange.
absl::StatusOr<int> ConnectToPeer(
    absl::string_view peer, absl::string_view local_ip = "",
    bool require_psp = false, std::shared_ptr<grpc::Channel> channel = nullptr,
    const std::atomic<bool>* cancelled = nullptr,
    std::chrono::milliseconds timeout = std::chrono::seconds(30));

}  // namespace tpu_raiden::transport::lib

#endif  // TPU_SYNC_TRANSPORT_LIB_SOCKET_UTIL_H_
