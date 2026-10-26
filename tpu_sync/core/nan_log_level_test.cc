// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include "tpu_sync/core/nan_log_level.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

namespace tpu_raiden {
namespace {
class NanLogLevelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (const char* value = std::getenv("TPU_NAN_LOG_LEVEL")) saved_ = value;
  }
  void TearDown() override {
    if (saved_) {
      setenv("TPU_NAN_LOG_LEVEL", saved_->c_str(), 1);
    } else {
      unsetenv("TPU_NAN_LOG_LEVEL");
    }
  }
  std::optional<std::string> saved_;
};

TEST_F(NanLogLevelTest, DefaultsToOff) {
  unsetenv("TPU_NAN_LOG_LEVEL");
  EXPECT_EQ(GetNanLogLevel(), NanLogLevel::kOff);
}

TEST_F(NanLogLevelTest, CumulativeLevelsEnableReceiveCheckAtDump) {
  const NanLogLevel levels[] = {NanLogLevel::kOff, NanLogLevel::kSummary,
                                NanLogLevel::kDetail, NanLogLevel::kDump,
                                NanLogLevel::kAll};
  for (int i = 0; i < 5; ++i) {
    setenv("TPU_NAN_LOG_LEVEL", std::to_string(i).c_str(), 1);
    const auto level = GetNanLogLevel();
    ASSERT_TRUE(level.has_value());
    EXPECT_EQ(*level, levels[i]);
    EXPECT_EQ(*level >= NanLogLevel::kDump, i >= 3);
  }
}

TEST_F(NanLogLevelTest, RejectsInvalidConfig) {
  for (const char* value :
       {"", "-1", "5", "01", " 1", "1 ", "1.0", "ALL", "１"}) {
    setenv("TPU_NAN_LOG_LEVEL", value, 1);
    EXPECT_FALSE(GetNanLogLevel().has_value()) << value;
  }
}
}  // namespace
}  // namespace tpu_raiden
