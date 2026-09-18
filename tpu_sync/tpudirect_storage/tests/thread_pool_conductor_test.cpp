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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <vector>

#include "../src/def_internal.h"
#include "../src/syscall_internal.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tdsul/def.h"
#include "util/IoConductor.h"
#include "util/ThreadPoolConductor.h"

namespace tdsul {
extern std::unique_ptr<IoConductor> g_io_conductor;
}

namespace {

class MockSyscall : public tdsul::Syscall {
 public:
  MOCK_METHOD(ssize_t, pread, (int fd, void* buf, size_t count, off_t offset),
              (override));
  MOCK_METHOD(ssize_t, pwrite,
              (int fd, const void* buf, size_t count, off_t offset),
              (override));
};

class ThreadPoolConductorTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {}

  void SetUp() override {
    original_syscall_ = tdsul::g_syscall;
    tdsul::g_syscall = &mock_syscall_;
  }

  void TearDown() override { tdsul::g_syscall = original_syscall_; }

  MockSyscall mock_syscall_;
  tdsul::Syscall* original_syscall_;
};

TEST_F(ThreadPoolConductorTest, TestWorkerThreadDispatch) {
  // Set up mock expectations. Since the worker thread doesn't yet call
  // pread/pwrite, we just allow it to be called any number of times.
  // When the logic is added to ThreadPoolConductor::Run, these tests can be
  // tightened.
  using ::testing::_;
  EXPECT_CALL(mock_syscall_, pread(_, _, _, _)).Times(testing::AnyNumber());
  EXPECT_CALL(mock_syscall_, pwrite(_, _, _, _)).Times(testing::AnyNumber());

  tdsul::ThreadPoolConductor pool(2);  // 2 threads
  pool.Start();

  // Each request carries a future so the test can prove the worker threads
  // actually executed the work. Asserting only "no crash" previously allowed a
  // pool whose threads failed to start to pass while silently dropping (and
  // leaking) every dispatched request.
  const int kNumTasks = 10;
  std::vector<tds_io_future> futures(kNumTasks);
  for (auto& future : futures) {
    future.status.status = TDS_PENDING;
  }

  for (int i = 0; i < kNumTasks; ++i) {
    tds_storage_io_t storage_io{};
    tds_buffer_io_t buffer_io{};
    tdsul::WorkRequest* req = new tdsul::SingleWorkRequest(
        storage_io, buffer_io, TDS_OP_READ, nullptr, &futures[i]);

    pool.Dispatch(req);
  }

  // Stop() waits for all tasks in the queues to be processed (and deleted).
  pool.Stop();

  // Every request must have reached a worker and completed. The heap checker
  // independently verifies that each `SingleWorkRequest` was freed.
  for (int i = 0; i < kNumTasks; ++i) {
    EXPECT_EQ(futures[i].status.status, TDS_SUCCESS)
        << "Request " << i << " was never executed by a worker thread";
  }
}

// A pool whose worker threads were never started must not silently retain (and
// leak) dispatched work. `Stop()` drains any queued requests and marks their
// futures failed, so callers observe an error instead of waiting forever.
TEST_F(ThreadPoolConductorTest, StopReleasesRequestsQueuedBeforeStart) {
  tdsul::ThreadPoolConductor pool(2);
  // Deliberately no Start(): the worker threads are constructed but idle,
  // mirroring the state left behind when pthread_create fails outright.

  tds_io_future future{};
  future.status.status = TDS_PENDING;

  tds_storage_io_t storage_io{};
  tds_buffer_io_t buffer_io{};
  pool.Dispatch(new tdsul::SingleWorkRequest(storage_io, buffer_io, TDS_OP_READ,
                                             nullptr, &future));

  pool.Stop();

  EXPECT_NE(future.status.status, TDS_PENDING)
      << "Abandoned request left its future pending forever";
}

}  // namespace
