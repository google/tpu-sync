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

#include "../src/def_internal.h"
#include "../src/util/IoQueue.h"
#include "tdsul/def.h"

namespace {

TEST(IoQueueTest, Push) {
  tdsul::IoQueue queue;

  tds_storage_io_t storage_io{};
  tds_buffer_io_t buffer_io{};
  tdsul::SingleWorkRequest req1(storage_io, buffer_io, TDS_OP_READ, nullptr,
                                nullptr);
  tdsul::SingleWorkRequest req2(storage_io, buffer_io, TDS_OP_WRITE, nullptr,
                                nullptr);

  // When pushed, request state is set to PENDING
  queue.Push(&req1);
  EXPECT_EQ(req1.state_, tdsul::RequestState::PENDING);
  EXPECT_EQ(queue.Size(), 1);

  queue.Push(&req2);
  EXPECT_EQ(req2.state_, tdsul::RequestState::PENDING);
  EXPECT_EQ(queue.Size(), 2);

  // Verify FIFO order at the front
  tdsul::WorkRequest* front_req = nullptr;
  EXPECT_TRUE(queue.Front(front_req));
  EXPECT_EQ(front_req, &req1);
}

TEST(IoQueueTest, Pop) {
  tdsul::IoQueue queue;

  tdsul::WorkRequest* popped_req = nullptr;
  // Pop on an empty queue should return false
  EXPECT_FALSE(queue.Pop(popped_req));

  tds_storage_io_t storage_io{};
  tds_buffer_io_t buffer_io{};
  tdsul::SingleWorkRequest req1(storage_io, buffer_io, TDS_OP_READ, nullptr,
                                nullptr);
  tdsul::SingleWorkRequest req2(storage_io, buffer_io, TDS_OP_WRITE, nullptr,
                                nullptr);

  queue.Push(&req1);
  queue.Push(&req2);

  // Pop requires the front request to be in FINISHED state.
  // Initially req1 is PENDING, so Pop should return false.
  EXPECT_FALSE(queue.Pop(popped_req));
  EXPECT_EQ(queue.Size(), 2);

  // Once req1 is marked FINISHED, Pop should succeed and remove req1.
  req1.state_ = tdsul::RequestState::FINISHED;
  EXPECT_TRUE(queue.Pop(popped_req));
  EXPECT_EQ(popped_req, &req1);
  EXPECT_EQ(queue.Size(), 1);

  // req2 is still PENDING, so another Pop should fail.
  EXPECT_FALSE(queue.Pop(popped_req));

  // Mark req2 as FINISHED and pop it.
  req2.state_ = tdsul::RequestState::FINISHED;
  EXPECT_TRUE(queue.Pop(popped_req));
  EXPECT_EQ(popped_req, &req2);
  EXPECT_EQ(queue.Size(), 0);

  // Now the queue is empty, Pop should fail.
  EXPECT_FALSE(queue.Pop(popped_req));
}

TEST(IoQueueTest, Size) {
  tdsul::IoQueue queue;

  EXPECT_EQ(queue.Size(), 0);

  tds_storage_io_t storage_io{};
  tds_buffer_io_t buffer_io{};
  tdsul::SingleWorkRequest req1(storage_io, buffer_io, TDS_OP_READ, nullptr,
                                nullptr);
  tdsul::SingleWorkRequest req2(storage_io, buffer_io, TDS_OP_WRITE, nullptr,
                                nullptr);
  tdsul::SingleWorkRequest req3(storage_io, buffer_io, TDS_OP_READ, nullptr,
                                nullptr);

  queue.Push(&req1);
  EXPECT_EQ(queue.Size(), 1);

  queue.Push(&req2);
  EXPECT_EQ(queue.Size(), 2);

  queue.Push(&req3);
  EXPECT_EQ(queue.Size(), 3);

  // Pop an element and check size decreases
  req1.state_ = tdsul::RequestState::FINISHED;
  tdsul::WorkRequest* popped = nullptr;
  EXPECT_TRUE(queue.Pop(popped));
  EXPECT_EQ(queue.Size(), 2);

  req2.state_ = tdsul::RequestState::FINISHED;
  EXPECT_TRUE(queue.Pop(popped));
  EXPECT_EQ(queue.Size(), 1);

  req3.state_ = tdsul::RequestState::FINISHED;
  EXPECT_TRUE(queue.Pop(popped));
  EXPECT_EQ(queue.Size(), 0);
}

TEST(IoQueueTest, Empty) {
  tdsul::IoQueue queue;

  EXPECT_TRUE(queue.Empty());

  tds_storage_io_t storage_io{};
  tds_buffer_io_t buffer_io{};
  tdsul::SingleWorkRequest req(storage_io, buffer_io, TDS_OP_READ, nullptr,
                               nullptr);

  queue.Push(&req);
  EXPECT_FALSE(queue.Empty());

  req.state_ = tdsul::RequestState::FINISHED;
  tdsul::WorkRequest* popped = nullptr;
  EXPECT_TRUE(queue.Pop(popped));
  EXPECT_TRUE(queue.Empty());
}

TEST(IoQueueTest, Front) {
  tdsul::IoQueue queue;

  tdsul::WorkRequest* front_req = nullptr;
  EXPECT_FALSE(queue.Front(front_req));

  tds_storage_io_t storage_io{};
  tds_buffer_io_t buffer_io{};
  tdsul::SingleWorkRequest req1(storage_io, buffer_io, TDS_OP_READ, nullptr,
                                nullptr);
  tdsul::SingleWorkRequest req2(storage_io, buffer_io, TDS_OP_WRITE, nullptr,
                                nullptr);

  queue.Push(&req1);
  EXPECT_TRUE(queue.Front(front_req));
  EXPECT_EQ(front_req, &req1);
  EXPECT_EQ(queue.Size(), 1);

  queue.Push(&req2);
  EXPECT_TRUE(queue.Front(front_req));
  EXPECT_EQ(front_req, &req1);
  EXPECT_EQ(queue.Size(), 2);
}

TEST(IoQueueTest, TestAndSetAssigned) {
  tdsul::IoQueue queue;

  EXPECT_TRUE(
      queue.TestAndSetAssigned());  // Succeeded in setting, returns true
  EXPECT_FALSE(queue.TestAndSetAssigned());  // Already true, returns false

  EXPECT_EQ(queue.CheckAndGetNextSchedulableRequest(), nullptr);

  // CheckAndGetNextSchedulableRequest sets is_assigned_ to false when returning
  // nullptr
  EXPECT_TRUE(queue.TestAndSetAssigned());
}

TEST(IoQueueTest, SetWorker) {
  tdsul::IoQueue queue;
  queue.SetWorker(nullptr);
}

TEST(IoQueueTest, GetTotalTransferSize) {
  tdsul::IoQueue queue;

  tds_storage_io_t storage_io{};
  tds_buffer_io_t buffer_io{};
  buffer_io.handle = nullptr;
  buffer_io.raw_ptr.size = 100;
  tdsul::SingleWorkRequest req1(storage_io, buffer_io, TDS_OP_READ, nullptr,
                                nullptr);

  queue.Push(&req1);
  EXPECT_EQ(queue.GetTotalTransferSize(), 100);

  buffer_io.raw_ptr.size = 200;
  tdsul::SingleWorkRequest req2(storage_io, buffer_io, TDS_OP_READ, nullptr,
                                nullptr);

  queue.Push(&req2);
  EXPECT_EQ(queue.GetTotalTransferSize(), 300);

  req1.state_ = tdsul::RequestState::FINISHED;
  tdsul::WorkRequest* popped = nullptr;
  EXPECT_TRUE(queue.Pop(popped));
  EXPECT_EQ(queue.GetTotalTransferSize(), 200);
}

TEST(IoQueueTest, CheckAndGetNextSchedulableRequest) {
  tdsul::IoQueue queue;

  tds_storage_io_t storage_io{};
  tds_buffer_io_t buffer_io{};
  tdsul::SingleWorkRequest req1(storage_io, buffer_io, TDS_OP_READ, nullptr,
                                nullptr);
  tdsul::SingleWorkRequest req2(storage_io, buffer_io, TDS_OP_READ, nullptr,
                                nullptr);

  queue.Push(&req1);
  queue.Push(&req2);

  auto req = queue.CheckAndGetNextSchedulableRequest();
  EXPECT_EQ(req, &req1);
  EXPECT_EQ(req->state_, tdsul::RequestState::SCHEDULED);

  EXPECT_EQ(queue.CheckAndGetNextSchedulableRequest(), nullptr);

  // After the first one is popped, the next one can be scheduled
  req1.state_ = tdsul::RequestState::FINISHED;
  tdsul::WorkRequest* popped = nullptr;
  EXPECT_TRUE(queue.Pop(popped));

  auto req_next = queue.CheckAndGetNextSchedulableRequest();
  EXPECT_EQ(req_next, &req2);
  EXPECT_EQ(req_next->state_, tdsul::RequestState::SCHEDULED);
}

}  // namespace
