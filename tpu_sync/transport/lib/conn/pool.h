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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_LIB_CONN_POOL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_LIB_CONN_POOL_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/channel.h"

namespace tpu_raiden::transport::lib {

// This class manages a pool of TCP connections between multiple pairs of
// local and remote peers.
// It is thread-safe.
class ConnPool {
 public:
  // Constructor.
  ConnPool() : stop_(false) {}

  // Destructor.
  ~ConnPool() { DCHECK(stop_ && pool_.empty()); }

  // Closes all connections.
  void Close();

  // Borrows a connection from the pool. If no connection is available, creates
  // a new one. Returns the socket descriptor of the connection if successful.
  // Otherwise returns an error status.
  absl::StatusOr<int> Borrow(
      absl::string_view peer, absl::string_view local_ip = "",
      bool require_psp = false,
      std::shared_ptr<grpc::Channel> channel = nullptr);

  // Returns a connection to the pool if ok is true. Otherwise, closes the
  // connection.
  void Return(bool ok, int fd, absl::string_view peer,
              absl::string_view local_ip = "");

 private:
  using Key = std::string;
  using Fds = std::vector<int>;

  // Generates a pool key from the local/peer ip address pair.
  static Key GenPoolKey(absl::string_view peer, absl::string_view local_ip) {
    return absl::StrCat(local_ip, "->", peer);
  }

 private:
  absl::Mutex mu_;
  bool stop_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<Key, Fds> pool_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_LIB_CONN_POOL_H_
