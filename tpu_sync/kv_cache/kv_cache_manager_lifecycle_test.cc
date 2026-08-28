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

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>

#include "tpu_sync/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden::kv_cache {
namespace {

class LifecycleTestManager : public KVCacheManagerBase {
 public:
  explicit LifecycleTestManager(std::promise<void>* destruction_started)
      : KVCacheManagerBase(/*num_layers=*/0, /*num_shards=*/0,
                           /*slice_byte_size=*/1),
        destruction_started_(destruction_started) {}

  ~LifecycleTestManager() override { destruction_started_->set_value(); }

  std::shared_ptr<void> HoldManagerCallback() { return TrackManagerCallback(); }

 private:
  std::promise<void>* destruction_started_;
};

TEST(KVCacheManagerLifecycleTest, DestructorWaitsForTrackedCallback) {
  std::promise<void> destruction_started;
  std::future<void> destruction_started_future =
      destruction_started.get_future();
  auto manager = std::make_unique<LifecycleTestManager>(&destruction_started);
  std::shared_ptr<void> callback = manager->HoldManagerCallback();

  auto destroy =
      std::async(std::launch::async, [&manager]() { manager.reset(); });
  const std::future_status started =
      destruction_started_future.wait_for(std::chrono::seconds(1));
  EXPECT_EQ(started, std::future_status::ready);
  if (started != std::future_status::ready) {
    callback.reset();
    destroy.wait();
    return;
  }
  EXPECT_EQ(destroy.wait_for(std::chrono::milliseconds(40)),
            std::future_status::timeout);

  callback.reset();
  EXPECT_EQ(destroy.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  destroy.get();
}

}  // namespace
}  // namespace tpu_raiden::kv_cache
