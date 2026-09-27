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

#include "tpu_sync/core/raw_dma_pacer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"

namespace raiden {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::Field;
using ::testing::HasSubstr;

constexpr int64_t kKiB = 1024;
constexpr int64_t kMiB = 1024 * kKiB;

// Records every piece the pacer issues and lets the test complete them.
class FakeDma {
 public:
  struct Issued {
    int copy_id;
    int64_t offset;
    int64_t size;
  };

  RawDmaIssueFn Issuer(int copy_id) {
    return [this, copy_id](int64_t offset, int64_t size) {
      auto [promise, future] = xla::MakePromise();
      absl::MutexLock lock(mu_);
      issued_.push_back(Issued{copy_id, offset, size});
      promises_.emplace_back(std::move(promise));
      outstanding_ += size;
      max_outstanding_ = std::max(max_outstanding_, outstanding_);
      return std::move(future);
    };
  }

  // Blocks until at least `n` pieces have been issued.
  void WaitIssued(size_t n) {
    absl::MutexLock lock(mu_);
    auto enough = [this, n]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return issued_.size() >= n;
    };
    mu_.Await(absl::Condition(&enough));
  }

  // Completes the oldest issued, not yet completed piece.
  void CompleteNext(absl::Status status = absl::OkStatus()) {
    std::optional<xla::Promise<>> promise;
    {
      absl::MutexLock lock(mu_);
      ASSERT_LT(completed_, promises_.size());
      outstanding_ -= issued_[completed_].size;
      promise = std::move(promises_[completed_]);
      promises_[completed_].reset();
      ++completed_;
    }
    if (status.ok()) {
      promise->Set();
    } else {
      promise->Set(status);
    }
  }

  std::vector<Issued> issued() {
    absl::MutexLock lock(mu_);
    return issued_;
  }
  int64_t max_outstanding() {
    absl::MutexLock lock(mu_);
    return max_outstanding_;
  }
  size_t completed() {
    absl::MutexLock lock(mu_);
    return completed_;
  }

 private:
  absl::Mutex mu_;
  std::vector<Issued> issued_ ABSL_GUARDED_BY(mu_);
  std::vector<std::optional<xla::Promise<>>> promises_ ABSL_GUARDED_BY(mu_);
  size_t completed_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t outstanding_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t max_outstanding_ ABSL_GUARDED_BY(mu_) = 0;
};

// Drives every piece to completion, one at a time, in issue order.
void DrainInOrder(FakeDma& dma, size_t total_pieces) {
  for (size_t i = 0; i < total_pieces; ++i) {
    dma.WaitIssued(i + 1);
    dma.CompleteNext();
  }
}

TEST(RawDmaPacerTest, SplitsCopiesAndBoundsBytesInFlight) {
  RawDmaPacer pacer(
      {.max_inflight_bytes = 64 * kKiB, .chunk_bytes = 16 * kKiB});
  FakeDma dma;
  const int64_t size = 1024 * kKiB + 4 * kKiB;  // Odd tail piece.
  xla::Future<> done = pacer.Submit({{dma.Issuer(0), size}});

  // Only the first window goes out before anything completes.
  dma.WaitIssued(4);
  EXPECT_EQ(dma.issued().size(), 4);
  EXPECT_EQ(pacer.stats().inflight_bytes, 64 * kKiB);
  EXPECT_FALSE(done.IsReady());

  const size_t total = (size + 16 * kKiB - 1) / (16 * kKiB);
  DrainInOrder(dma, total);
  ABSL_EXPECT_OK(done.Await());

  // Pieces tile the copy exactly, in order, 4 KiB aligned.
  int64_t next = 0;
  for (const FakeDma::Issued& piece : dma.issued()) {
    EXPECT_EQ(piece.offset, next);
    EXPECT_EQ(piece.offset % (4 * kKiB), 0);
    EXPECT_LE(piece.size, 16 * kKiB);
    next += piece.size;
  }
  EXPECT_EQ(next, size);
  EXPECT_LE(dma.max_outstanding(), 64 * kKiB);
  EXPECT_EQ(pacer.stats().peak_inflight_bytes, 64 * kKiB);
  EXPECT_EQ(pacer.stats().inflight_bytes, 0);
}

TEST(RawDmaPacerTest, EarlierSubmissionsGoFirst) {
  RawDmaPacer pacer(
      {.max_inflight_bytes = 32 * kKiB, .chunk_bytes = 16 * kKiB});
  FakeDma dma;
  xla::Future<> first = pacer.Submit({{dma.Issuer(1), 64 * kKiB}});
  xla::Future<> second = pacer.Submit({{dma.Issuer(2), 64 * kKiB}});
  DrainInOrder(dma, 8);
  ABSL_EXPECT_OK(first.Await());
  ABSL_EXPECT_OK(second.Await());
  std::vector<int> order;
  for (const FakeDma::Issued& piece : dma.issued()) {
    order.push_back(piece.copy_id);
  }
  EXPECT_THAT(order, ::testing::ElementsAre(1, 1, 1, 1, 2, 2, 2, 2));
}

TEST(RawDmaPacerTest, ErrorResolvesOnlyAfterEveryPieceFinished) {
  RawDmaPacer pacer(
      {.max_inflight_bytes = 32 * kKiB, .chunk_bytes = 16 * kKiB});
  FakeDma dma;
  xla::Future<> done = pacer.Submit({{dma.Issuer(0), 48 * kKiB}});
  dma.WaitIssued(2);
  dma.CompleteNext(absl::InternalError("dma failed"));
  dma.WaitIssued(3);
  dma.CompleteNext();
  EXPECT_FALSE(done.IsReady());  // One piece is still on the wire.
  dma.CompleteNext();
  EXPECT_EQ(done.Await().code(), absl::StatusCode::kInternal);
}

TEST(RawDmaPacerTest, InvalidFutureFailsThePieceAndFreesItsWindow) {
  RawDmaPacer pacer(
      {.max_inflight_bytes = 32 * kKiB, .chunk_bytes = 16 * kKiB});
  // Three pieces: Submit issues two and the pacer thread the third, once
  // window space has been freed.
  RawDmaIssueFn invalid = [](int64_t /*offset*/, int64_t /*size*/) {
    return xla::Future<>();
  };
  EXPECT_THAT(pacer.Submit({{invalid, 48 * kKiB}}).Await(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("RawDmaPacer: issue function returned an "
                                 "invalid future for bytes [0, 16384)")));
  EXPECT_EQ(pacer.stats().inflight_bytes, 0);

  // Nothing leaked: a later copy through the same window still completes.
  FakeDma dma;
  xla::Future<> done = pacer.Submit({{dma.Issuer(0), 32 * kKiB}});
  DrainInOrder(dma, 2);
  ABSL_EXPECT_OK(done.Await());
  EXPECT_EQ(pacer.stats().inflight_bytes, 0);
}

TEST(RawDmaPacerTest, KeepAliveOutlivesDroppedFuture) {
  std::optional<RawDmaPacer> pacer;
  pacer.emplace(RawDmaPacerOptions{.max_inflight_bytes = 16 * kKiB,
                                   .chunk_bytes = 16 * kKiB});
  FakeDma dma;
  auto owned = std::make_shared<int>(7);
  std::weak_ptr<int> watch = owned;
  {
    xla::Future<> dropped =
        pacer->Submit({{dma.Issuer(0), 48 * kKiB}}, std::move(owned));
  }
  dma.WaitIssued(1);
  dma.CompleteNext();
  dma.WaitIssued(2);
  dma.CompleteNext();
  EXPECT_FALSE(watch.expired());
  dma.WaitIssued(3);
  dma.CompleteNext();
  // The last completion callback can run on the pacer thread (when the promise
  // is set before the pacer registers OnReady); destroying the pacer joins
  // that thread.
  pacer.reset();
  EXPECT_TRUE(watch.expired());
}

TEST(RawDmaPacerTest, DisabledIssuesWholeCopiesAtOnce) {
  RawDmaPacer pacer({.max_inflight_bytes = 0, .chunk_bytes = 16 * kKiB});
  FakeDma dma;
  xla::Future<> done =
      pacer.Submit({{dma.Issuer(0), 1024 * kKiB}, {dma.Issuer(1), 512 * kKiB}});
  // Issued synchronously on the caller, one request per copy.
  std::vector<FakeDma::Issued> issued = dma.issued();
  ASSERT_EQ(issued.size(), 2);
  EXPECT_EQ(issued[0].size, 1024 * kKiB);
  EXPECT_EQ(issued[1].size, 512 * kKiB);
  dma.CompleteNext();
  dma.CompleteNext();
  ABSL_EXPECT_OK(done.Await());
}

TEST(RawDmaPacerTest, PieceLargerThanWindowStillProgresses) {
  RawDmaPacer pacer({.max_inflight_bytes = 8 * kKiB, .chunk_bytes = 16 * kKiB});
  FakeDma dma;
  xla::Future<> done = pacer.Submit({{dma.Issuer(0), 32 * kKiB}});
  dma.WaitIssued(1);
  EXPECT_EQ(dma.issued().size(), 1);
  DrainInOrder(dma, 2);
  ABSL_EXPECT_OK(done.Await());
  EXPECT_EQ(dma.max_outstanding(), 16 * kKiB);
}

TEST(RawDmaPacerTest, EmptyCopiesCompleteImmediately) {
  RawDmaPacer pacer;
  FakeDma dma;
  ABSL_EXPECT_OK(pacer.Submit({}).Await());
  ABSL_EXPECT_OK(pacer.Submit({{dma.Issuer(0), 0}}).Await());
  EXPECT_TRUE(dma.issued().empty());
}

TEST(RawDmaPacerTest, NormalizesChunkToFourKiBMultiples) {
  EXPECT_EQ(RawDmaPacer::Normalize({.chunk_bytes = 5000}).chunk_bytes, 4096);
  EXPECT_EQ(RawDmaPacer::Normalize({.chunk_bytes = 100}).chunk_bytes, 4096);
  EXPECT_EQ(RawDmaPacer::Normalize({.chunk_bytes = 0}).chunk_bytes,
            RawDmaPacerOptions{}.chunk_bytes);
}

constexpr char kMaxInflightBytesEnv[] = "TPU_RAIDEN_RAW_DMA_MAX_INFLIGHT_BYTES";
constexpr char kChunkBytesEnv[] = "TPU_RAIDEN_RAW_DMA_CHUNK_BYTES";

// Sets environment variable `name` to `value` (unsets it for nullptr) for the
// lifetime of the object, then restores its previous value.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      old_value_ = old;
    }
    if (value != nullptr) {
      setenv(name, value, /*overwrite=*/1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (old_value_.has_value()) {
      setenv(name_.c_str(), old_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::optional<std::string> old_value_;
};

// Matches RawDmaPacerOptions holding exactly these values.
::testing::Matcher<RawDmaPacerOptions> OptionsAre(int64_t max_inflight_bytes,
                                                  int64_t chunk_bytes) {
  return AllOf(
      Field("max_inflight_bytes", &RawDmaPacerOptions::max_inflight_bytes,
            max_inflight_bytes),
      Field("chunk_bytes", &RawDmaPacerOptions::chunk_bytes, chunk_bytes));
}

// The OptionsFromEnv tests never call Global(): it is a process singleton that
// reads the environment only once.
TEST(RawDmaPacerTest, OptionsFromEnvKeepsDefaultsWhenUnsetOrEmpty) {
  {
    ScopedEnv max_inflight(kMaxInflightBytesEnv, nullptr);
    ScopedEnv chunk(kChunkBytesEnv, nullptr);
    EXPECT_THAT(RawDmaPacer::OptionsFromEnv(),
                IsOkAndHolds(OptionsAre(64 * kMiB, 16 * kMiB)));
  }
  {
    ScopedEnv max_inflight(kMaxInflightBytesEnv, "");
    ScopedEnv chunk(kChunkBytesEnv, "");
    EXPECT_THAT(RawDmaPacer::OptionsFromEnv(),
                IsOkAndHolds(OptionsAre(64 * kMiB, 16 * kMiB)));
  }
}

TEST(RawDmaPacerTest, OptionsFromEnvParsesIntegers) {
  {
    ScopedEnv max_inflight(kMaxInflightBytesEnv, "0");
    ScopedEnv chunk(kChunkBytesEnv, "8388608");
    EXPECT_THAT(RawDmaPacer::OptionsFromEnv(),
                IsOkAndHolds(OptionsAre(0, 8 * kMiB)));
  }
  {
    ScopedEnv max_inflight(kMaxInflightBytesEnv, "-1");
    ScopedEnv chunk(kChunkBytesEnv, nullptr);
    EXPECT_THAT(RawDmaPacer::OptionsFromEnv(),
                IsOkAndHolds(OptionsAre(-1, 16 * kMiB)));
  }
}

TEST(RawDmaPacerTest, OptionsFromEnvRejectsMalformedValues) {
  for (const char* name : {kMaxInflightBytesEnv, kChunkBytesEnv}) {
    for (const char* value : {"16M", "abc", "1.5", "99999999999999999999"}) {
      ScopedEnv max_inflight(kMaxInflightBytesEnv, nullptr);
      ScopedEnv chunk(kChunkBytesEnv, nullptr);
      ScopedEnv malformed(name, value);
      EXPECT_THAT(RawDmaPacer::OptionsFromEnv(),
                  StatusIs(absl::StatusCode::kInvalidArgument,
                           HasSubstr(absl::StrCat(name, "=\"", value, "\""))));
    }
  }
}

}  // namespace
}  // namespace raiden
