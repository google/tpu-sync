// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#ifndef TPU_SYNC_CORE_NAN_LOG_LEVEL_H_
#define TPU_SYNC_CORE_NAN_LOG_LEVEL_H_

#include <cstdlib>
#include <optional>
#include <string_view>

namespace tpu_raiden {
// Cumulative diagnostic scope shared with vllm-torchtpu.
enum class NanLogLevel {
  kOff = 0,
  kSummary = 1,
  kDetail = 2,
  kDump = 3,
  kAll = 4
};

// An invalid setting is rejected by the session creation boundary.
inline std::optional<NanLogLevel> GetNanLogLevel() {
  const char* value = std::getenv("TPU_NAN_LOG_LEVEL");
  if (value == nullptr) return NanLogLevel::kOff;
  const std::string_view text(value);
  if (text.size() != 1 || text[0] < '0' || text[0] > '4') {
    return std::nullopt;
  }
  return static_cast<NanLogLevel>(text[0] - '0');
}
}  // namespace tpu_raiden

#endif  // TPU_SYNC_CORE_NAN_LOG_LEVEL_H_
