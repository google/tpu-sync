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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_CONTROL_PIPE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_CONTROL_PIPE_H_

#include <algorithm>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

// Creates a ControlPipeClient resolved from `backend_type` or
// `TPU_RAIDEN_CONTROL_PLANE_BACKEND`.
inline std::unique_ptr<ControlPipeClient> CreateReshardControlPipeClient(
    std::optional<ControlPipeBackendType> backend_type = std::nullopt) {
  ControlPipeConfig cfg;
  cfg.backend_type = ResolveControlPipeBackendType(backend_type);
  return CreateControlPipeClient(cfg);
}

// Invokes `ControlPipeClient::Call<ReqProto, RespProto>` against `address`,
// retrying on `UNAVAILABLE` until `timeout` so peers still initializing their
// listener during startup are awaited rather than immediately failed.
template <typename ReqProto, typename RespProto>
absl::StatusOr<RespProto> CallReshardControlPipe(ControlPipeClient* client,
                                                 absl::string_view address,
                                                 const ReqProto& request,
                                                 absl::Duration timeout) {
  const absl::Time deadline = absl::Now() + timeout;
  constexpr absl::Duration kInitialBackoff = absl::Milliseconds(50);
  constexpr absl::Duration kMaxBackoff = absl::Seconds(2);
  absl::Duration backoff = kInitialBackoff;
  absl::StatusOr<RespProto> resp;
  do {
    const absl::Duration remaining =
        std::max(deadline - absl::Now(), absl::Milliseconds(1));
    resp = client->Call<ReqProto, RespProto>(address, request, remaining);
    if (resp.ok() || !absl::IsUnavailable(resp.status())) {
      break;
    }
    const absl::Duration sleep_for =
        std::min(backoff, deadline - absl::Now());
    if (sleep_for <= absl::ZeroDuration()) {
      break;
    }
    absl::SleepFor(sleep_for);
    backoff = std::min(backoff * 2, kMaxBackoff);
  } while (absl::Now() < deadline);

  if (!resp.ok() && absl::IsUnavailable(resp.status())) {
    return absl::DeadlineExceededError(
        absl::StrCat("Timeout (", absl::ToInt64Seconds(timeout),
                     "s) failed to connect to robust endpoint ", address, ": ",
                     resp.status().message()));
  }
  return resp;
}

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_RESHARD_CONTROL_PIPE_H_
