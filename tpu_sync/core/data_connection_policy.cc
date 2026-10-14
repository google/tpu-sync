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

#include "tpu_sync/core/data_connection_policy.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/block_transport.h"

namespace tpu_raiden {

std::optional<DataConnectionReuse> ParseDataConnectionReuse(
    absl::string_view value) {
  const std::string lowered = absl::AsciiStrToLower(value);
  if (lowered == "pooled") return DataConnectionReuse::kPooled;
  if (lowered == "per_transfer") return DataConnectionReuse::kPerTransfer;
  return std::nullopt;
}

absl::StatusOr<DataConnectionReuse> ResolveDataConnectionReuse() {
  const char* raw =
      std::getenv(std::string(kDataConnectionReuseEnvVar).c_str());
  if (raw == nullptr || raw[0] == '\0') return DataConnectionReuse::kPooled;
  std::optional<DataConnectionReuse> parsed = ParseDataConnectionReuse(raw);
  if (!parsed.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        kDataConnectionReuseEnvVar, "=\"", raw,
        "\" is not one of pooled|per_transfer (case-insensitive)"));
  }
  return *parsed;
}

size_t ReleaseDataConnectionsAfterSend(transport::BlockTransport* transport,
                                       absl::Span<const std::string> peers,
                                       uint64_t uuid) {
  if (transport == nullptr) return 0;
  size_t closed = 0;
  for (const std::string& peer : peers) {
    const size_t n = transport->CloseIdleConnections(peer);
    if (n == 0) continue;
    closed += n;
    LOG(INFO) << "Released " << n << " idle data connection(s) to " << peer
              << " after send uuid=" << uuid
              << " (TPU_RAIDEN_DATA_CONN_REUSE=per_transfer)";
  }
  return closed;
}

}  // namespace tpu_raiden
