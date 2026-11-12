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

#include "tpu_sync/transport/lib/adapter/peregrine_transport_adapter.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

constexpr absl::Duration kTransferTimeout = absl::Seconds(10);

int PickUnusedLoopbackPort() {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return 0;
  int reuse = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return 0;
  }
  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    close(fd);
    return 0;
  }
  const uint16_t port = ntohs(addr.sin_port);
  close(fd);
  return port;
}

// Adapter bound to 127.0.0.1 on a kernel-assigned port.
absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> CreateAdapter(
    int num_conns_per_peer = 1,
    PeregrineTransportAdapter::RecvHooks hooks = {}) {
  return PeregrineTransportAdapter::Create(
      "127.0.0.1", /*port=*/0, num_conns_per_peer, std::move(hooks));
}

// Explicit-push request of `len` bytes from `laddr` into receiver block
// `dst_block_id`, with the Post identity every request of one Post shares.
Request ExplicitPushRequest(uint8_t* laddr, size_t len, int32_t dst_block_id,
                            uint32_t shard = 0) {
  Request req = {};
  req.op = TransferOp::kBlockPushExplicit;
  req.laddr = laddr;
  req.raddr = nullptr;
  req.len = len;
  req.buffer_id = (uint64_t{3} << 32) | shard;  // layer 3 (= local_id)
  req.remote_id = 7;
  req.local_id = 3;
  req.uuid = 0xABCDEF0123456789;
  req.shard_mask = 0b11;
  req.dst_block_id = dst_block_id;
  req.src_block_id = dst_block_id + 10;
  req.chunk_idx = 0;
  return req;
}

std::tuple<uint32_t, int32_t, int32_t, uint16_t, uint32_t> FieldsOf(
    const RequestMetadata& req) {
  return {req.shard, req.block_id, req.src_block_id, req.chunk_idx, req.len};
}

std::tuple<uint64_t, uint32_t, uint32_t, uint16_t, TransferOp, uint8_t>
FieldsOf(const PostMetadata& post) {
  return {post.uuid,       post.remote_id, post.local_id,
          post.shard_mask, post.op,        post.version};
}

// Receiver-side recorder: `resolve_addr` hands out `blocks[block_id]` and
// `on_post_complete` records every completion.
class RecvRecorder {
 public:
  struct Resolve {
    PostMetadata post;
    RequestMetadata request;
  };
  struct Completion {
    PostMetadata post;
    absl::Status status;
    // Number of `resolve_addr` calls seen when this completion arrived.
    size_t resolves_before = 0;
  };

  explicit RecvRecorder(std::vector<std::vector<uint8_t>*> blocks)
      : blocks_(std::move(blocks)) {}

  PeregrineTransportAdapter::RecvHooks hooks() {
    return {
        .on_post_complete =
            [this](const PostMetadata& post, const absl::Status& status) {
              absl::MutexLock lock(mu_);
              completions_.push_back({post, status, resolves_.size()});
              return absl::OkStatus();
            },
        .resolve_addr = [this](const PostMetadata& post,
                               const RequestMetadata& req) -> uint8_t* {
          absl::MutexLock lock(mu_);
          resolves_.push_back({post, req});
          if (req.block_id < 0 ||
              static_cast<size_t>(req.block_id) >= blocks_.size()) {
            return nullptr;
          }
          return blocks_[req.block_id]->data();
        },
    };
  }

  bool WaitForCompletions(size_t count, absl::Duration timeout) {
    absl::MutexLock lock(mu_);
    const auto reached = [this, count]() {
      mu_.AssertReaderHeld();
      return completions_.size() >= count;
    };
    return mu_.AwaitWithTimeout(absl::Condition(&reached), timeout);
  }
  std::vector<Resolve> resolves() const {
    absl::MutexLock lock(mu_);
    return resolves_;
  }
  std::vector<Completion> completions() const {
    absl::MutexLock lock(mu_);
    return completions_;
  }

 private:
  const std::vector<std::vector<uint8_t>*> blocks_;
  mutable absl::Mutex mu_;
  std::vector<Resolve> resolves_ ABSL_GUARDED_BY(mu_);
  std::vector<Completion> completions_ ABSL_GUARDED_BY(mu_);
};

TEST(PeregrineTransportAdapterTest, CreateBindsKernelAssignedPort) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      CreateAdapter(/*num_conns_per_peer=*/4);
  ABSL_ASSERT_OK(adapter);
  EXPECT_GT((*adapter)->local_port(), 0);
}

TEST(PeregrineTransportAdapterTest, CreateBindsRequestedPort) {
  const int port = PickUnusedLoopbackPort();
  ASSERT_NE(port, 0);

  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      PeregrineTransportAdapter::Create("127.0.0.1", port,
                                        /*num_conns_per_peer=*/1);
  ABSL_ASSERT_OK(adapter);
  EXPECT_EQ((*adapter)->local_port(), port);
}

TEST(PeregrineTransportAdapterTest, CreateBindsIpv6Loopback) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      PeregrineTransportAdapter::Create("::1", /*port=*/0,
                                        /*num_conns_per_peer=*/1);
  if (!adapter.ok()) {
    GTEST_SKIP() << "IPv6 loopback unavailable: " << adapter.status();
  }
  EXPECT_GT((*adapter)->local_port(), 0);
}

TEST(PeregrineTransportAdapterTest, ReceiverAllocatedPushIsUnimplemented) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      CreateAdapter();
  ABSL_ASSERT_OK(adapter);

  uint8_t buf[4] = {};
  Request req = {};
  req.op = TransferOp::kBlockPush;
  req.laddr = buf;
  req.raddr = buf;
  req.len = sizeof(buf);

  absl::Status callback_status = absl::OkStatus();
  absl::StatusOr<Handle> handle = (*adapter)->Post(
      /*peers=*/{"127.0.0.1:1234"}, absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/{}, /*dst_block_ids=*/{},
      [&](absl::StatusOr<std::vector<int>> result) {
        callback_status = result.status();
      });
  EXPECT_THAT(handle, StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(callback_status, StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(PeregrineTransportAdapterTest, PostWithEmptyPeersFails) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      CreateAdapter();
  ABSL_ASSERT_OK(adapter);

  uint8_t buf[4] = {1, 2, 3, 4};
  Request req = {};
  req.op = TransferOp::kBlockPushExplicit;
  req.laddr = buf;
  req.raddr = buf;
  req.len = sizeof(buf);

  absl::Status callback_status = absl::OkStatus();
  absl::StatusOr<Handle> handle = (*adapter)->Post(
      /*peers=*/{}, /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/{}, /*dst_block_ids=*/{},
      [&](absl::StatusOr<std::vector<int>> result) {
        callback_status = result.status();
      });

  EXPECT_THAT(handle, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(callback_status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PeregrineTransportAdapterTest, PostWithEmptyRequestsFails) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      CreateAdapter();
  ABSL_ASSERT_OK(adapter);

  absl::Status callback_status = absl::OkStatus();
  absl::StatusOr<Handle> handle = (*adapter)->Post(
      /*peers=*/{"127.0.0.1:1234"}, /*requests=*/{},
      /*src_block_ids=*/{}, /*dst_block_ids=*/{},
      [&](absl::StatusOr<std::vector<int>> result) {
        callback_status = result.status();
      });

  EXPECT_THAT(handle, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(callback_status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PeregrineTransportAdapterTest, PostSynchronousFailureInvokesCallback) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      CreateAdapter();
  ABSL_ASSERT_OK(adapter);

  uint8_t buf[4] = {1, 2, 3, 4};
  Request req = {};
  req.op = TransferOp::kBlockPushExplicit;
  req.laddr = buf;
  req.raddr = buf;
  req.len = sizeof(buf);
  req.dst_block_id = 0;

  // Peregrine rejects a malformed peer endpoint synchronously.
  absl::Status callback_status = absl::OkStatus();
  absl::StatusOr<Handle> handle = (*adapter)->Post(
      /*peers=*/{"invalid"}, /*requests=*/absl::MakeConstSpan(&req, 1),
      /*src_block_ids=*/{}, /*dst_block_ids=*/{},
      [&](absl::StatusOr<std::vector<int>> result) {
        callback_status = result.status();
      });

  EXPECT_THAT(handle, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(callback_status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PeregrineTransportAdapterTest, PollUnknownHandleFails) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      CreateAdapter();
  ABSL_ASSERT_OK(adapter);

  EXPECT_THAT((*adapter)->Poll(12345), StatusIs(absl::StatusCode::kNotFound));
}

TEST(PeregrineTransportAdapterTest, EndToEndTcpWriteOverLoopback) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> local =
      CreateAdapter(/*num_conns_per_peer=*/2);
  ABSL_ASSERT_OK(local);
  std::vector<uint8_t> src_buf = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<uint8_t> remote_buf(src_buf.size(), 0);
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> remote =
      CreateAdapter(
          /*num_conns_per_peer=*/2,
          {.resolve_addr = [&](const PostMetadata&, const RequestMetadata&) {
            return remote_buf.data();
          }});
  ABSL_ASSERT_OK(remote);
  const std::string remote_endpoint =
      absl::StrCat("127.0.0.1:", (*remote)->local_port());

  Request write_req = {};
  write_req.op = TransferOp::kBlockPushExplicit;
  write_req.laddr = src_buf.data();
  write_req.len = src_buf.size();
  write_req.dst_block_id = 0;

  absl::Notification write_done;
  absl::StatusOr<std::vector<int>> write_result;
  absl::StatusOr<Handle> write_handle = (*local)->Post(
      /*peers=*/{remote_endpoint},
      /*requests=*/absl::MakeConstSpan(&write_req, 1),
      /*src_block_ids=*/{}, /*dst_block_ids=*/{},
      [&](absl::StatusOr<std::vector<int>> res) {
        write_result = std::move(res);
        write_done.Notify();
      });

  ABSL_ASSERT_OK(write_handle);
  ASSERT_TRUE(write_done.WaitForNotificationWithTimeout(absl::Seconds(10)));
  EXPECT_THAT(write_result, IsOkAndHolds(IsEmpty()));
  EXPECT_EQ(remote_buf, src_buf);
}

// Explicit-push validation runs before the peer is contacted: the peer port
// below has no listener, so a Post that reached Peregrine would not fail
// synchronously with InvalidArgument.
TEST(PeregrineTransportAdapterTest, InvalidExplicitPushFailsBeforePosting) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      CreateAdapter();
  ABSL_ASSERT_OK(adapter);
  uint8_t buf[4] = {1, 2, 3, 4};

  // Each case mutates one field of an otherwise valid request.
  struct Case {
    void (*mutate)(Request& req);
    const char* message;
  };
  for (const Case& c : {
           Case{[](Request& req) { req.dst_block_id = -1; },
                "needs a destination block"},
           Case{[](Request& req) { req.len = 0; },
                "len must be in [1, 2^32-1]"},
           Case{[](Request& req) { req.len = size_t{1} << 32; },
                "len must be in [1, 2^32-1]"},
       }) {
    SCOPED_TRACE(c.message);
    Request req = ExplicitPushRequest(buf, sizeof(buf), /*dst_block_id=*/0);
    c.mutate(req);
    absl::Status callback_status = absl::OkStatus();
    absl::StatusOr<Handle> handle = (*adapter)->Post(
        /*peers=*/{"127.0.0.1:1"}, absl::MakeConstSpan(&req, 1),
        /*src_block_ids=*/{}, /*dst_block_ids=*/{},
        [&](absl::StatusOr<std::vector<int>> result) {
          callback_status = result.status();
        });
    EXPECT_THAT(handle, StatusIs(absl::StatusCode::kInvalidArgument,
                                 HasSubstr(c.message)));
    EXPECT_THAT(callback_status, StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr(c.message)));
  }
}

TEST(PeregrineTransportAdapterTest,
     ExplicitPushIsReceiverResolvedAndRoundTripsMetadata) {
  std::vector<uint8_t> block0(8, 0);
  std::vector<uint8_t> block1(8, 0);
  RecvRecorder recorder({&block0, &block1});
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> local =
      CreateAdapter(/*num_conns_per_peer=*/2);
  ABSL_ASSERT_OK(local);
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> remote =
      CreateAdapter(/*num_conns_per_peer=*/2, recorder.hooks());
  ABSL_ASSERT_OK(remote);
  const std::string remote_endpoint =
      absl::StrCat("127.0.0.1:", (*remote)->local_port());

  std::vector<uint8_t> src0 = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<uint8_t> src1 = {11, 21, 31, 41, 51, 61, 71, 81};
  // Two requests, one per shard, each into its own receiver block.
  std::vector<Request> reqs = {
      ExplicitPushRequest(src0.data(), src0.size(), /*dst_block_id=*/0,
                          /*shard=*/0),
      ExplicitPushRequest(src1.data(), src1.size(), /*dst_block_id=*/1,
                          /*shard=*/1),
  };
  reqs[1].chunk_idx = 5;

  absl::Notification done;
  absl::StatusOr<std::vector<int>> result;
  absl::StatusOr<Handle> handle = (*local)->Post(
      /*peers=*/{remote_endpoint}, reqs, /*src_block_ids=*/{},
      /*dst_block_ids=*/{}, [&](absl::StatusOr<std::vector<int>> res) {
        result = std::move(res);
        done.Notify();
      });
  ABSL_ASSERT_OK(handle);
  ASSERT_TRUE(done.WaitForNotificationWithTimeout(kTransferTimeout));
  EXPECT_THAT(result, IsOkAndHolds(IsEmpty()));
  ASSERT_TRUE(recorder.WaitForCompletions(1, kTransferTimeout));

  // Bytes landed where the receiver resolved them.
  EXPECT_EQ(block0, src0);
  EXPECT_EQ(block1, src1);

  // Post metadata round-trips bit-identically, request metadata per request,
  // and every request was resolved before the Post completed.
  const PostMetadata expected_post{.uuid = 0xABCDEF0123456789,
                                   .remote_id = 7,
                                   .local_id = 3,
                                   .shard_mask = 0b11,
                                   .op = TransferOp::kBlockPushExplicit,
                                   .version = kMetadataVersionV1};
  const auto expected_req0 = std::make_tuple(
      uint32_t{0}, int32_t{0}, int32_t{10}, uint16_t{0}, uint32_t{8});
  const auto expected_req1 = std::make_tuple(
      uint32_t{1}, int32_t{1}, int32_t{11}, uint16_t{5}, uint32_t{8});

  const std::vector<RecvRecorder::Resolve> resolves = recorder.resolves();
  ASSERT_EQ(resolves.size(), 2);
  std::vector<std::tuple<uint32_t, int32_t, int32_t, uint16_t, uint32_t>>
      resolved_requests;
  for (const RecvRecorder::Resolve& resolve : resolves) {
    EXPECT_EQ(FieldsOf(resolve.post), FieldsOf(expected_post));
    resolved_requests.push_back(FieldsOf(resolve.request));
  }
  EXPECT_THAT(resolved_requests,
              UnorderedElementsAre(expected_req0, expected_req1));

  const std::vector<RecvRecorder::Completion> completions =
      recorder.completions();
  ASSERT_EQ(completions.size(), 1);
  ABSL_EXPECT_OK(completions[0].status);
  EXPECT_EQ(FieldsOf(completions[0].post), FieldsOf(expected_post));
  EXPECT_EQ(completions[0].resolves_before, 2);
}

TEST(PeregrineTransportAdapterTest,
     ExplicitPushSpanningLayersIsOnePostPerLayer) {
  std::vector<uint8_t> block0(8, 0);
  std::vector<uint8_t> block1(8, 0);
  RecvRecorder recorder({&block0, &block1});
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> local =
      CreateAdapter();
  ABSL_ASSERT_OK(local);
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> remote =
      CreateAdapter(/*num_conns_per_peer=*/1, recorder.hooks());
  ABSL_ASSERT_OK(remote);

  std::vector<uint8_t> src0 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint8_t> src1 = {9, 10, 11, 12, 13, 14, 15, 16};
  // Layer 0 -> block 0, layer 1 -> block 1 (`buffer_id` high word = layer).
  std::vector<Request> reqs = {
      ExplicitPushRequest(src0.data(), src0.size(), /*dst_block_id=*/0),
      ExplicitPushRequest(src1.data(), src1.size(), /*dst_block_id=*/1),
  };
  reqs[0].buffer_id = 0;
  reqs[1].buffer_id = uint64_t{1} << 32;

  absl::Notification done;
  int completions = 0;
  absl::StatusOr<std::vector<int>> result;
  ABSL_ASSERT_OK((*local)->Post(
      /*peers=*/{absl::StrCat("127.0.0.1:", (*remote)->local_port())}, reqs,
      /*src_block_ids=*/{}, /*dst_block_ids=*/{},
      [&](absl::StatusOr<std::vector<int>> res) {
        ++completions;
        result = std::move(res);
        done.Notify();
      }));
  ASSERT_TRUE(done.WaitForNotificationWithTimeout(kTransferTimeout));
  EXPECT_THAT(result, IsOkAndHolds(IsEmpty()));
  EXPECT_EQ(completions, 1);
  ASSERT_TRUE(recorder.WaitForCompletions(2, kTransferTimeout));
  EXPECT_EQ(block0, src0);
  EXPECT_EQ(block1, src1);

  // One Post per layer on the receiver, each request resolved under its own
  // layer's Post.
  std::vector<uint32_t> completed_layers;
  for (const RecvRecorder::Completion& c : recorder.completions()) {
    ABSL_EXPECT_OK(c.status);
    completed_layers.push_back(c.post.local_id);
  }
  EXPECT_THAT(completed_layers, UnorderedElementsAre(0, 1));
  const std::vector<RecvRecorder::Resolve> resolves = recorder.resolves();
  ASSERT_EQ(resolves.size(), 2);
  for (const RecvRecorder::Resolve& resolve : resolves) {
    EXPECT_EQ(resolve.post.local_id,
              static_cast<uint32_t>(resolve.request.block_id));
  }
}

TEST(PeregrineTransportAdapterTest,
     MultiLayerPushReportsSynchronousIssueFailureOnce) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      CreateAdapter();
  ABSL_ASSERT_OK(adapter);
  uint8_t buf[8] = {};
  std::vector<Request> reqs = {
      ExplicitPushRequest(buf, 4, /*dst_block_id=*/0),
      ExplicitPushRequest(buf + 4, 4, /*dst_block_id=*/1),
  };
  reqs[0].buffer_id = 0;
  reqs[1].buffer_id = uint64_t{1} << 32;

  // Peregrine rejects a malformed peer synchronously for every per-layer
  // Post; the aggregated callback must still fire exactly once.
  int calls = 0;
  absl::Status callback_status = absl::OkStatus();
  EXPECT_THAT((*adapter)->Post(/*peers=*/{"not-an-endpoint"}, reqs,
                               /*src_block_ids=*/{}, /*dst_block_ids=*/{},
                               [&](absl::StatusOr<std::vector<int>> result) {
                                 ++calls;
                                 callback_status = result.status();
                               }),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(calls, 1);
  EXPECT_THAT(callback_status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PeregrineTransportAdapterTest, MultiLayerPushWithoutCallbackIsRejected) {
  absl::StatusOr<std::unique_ptr<PeregrineTransportAdapter>> adapter =
      CreateAdapter();
  ABSL_ASSERT_OK(adapter);
  uint8_t buf[8] = {};
  std::vector<Request> reqs = {
      ExplicitPushRequest(buf, 4, /*dst_block_id=*/0),
      ExplicitPushRequest(buf + 4, 4, /*dst_block_id=*/1),
  };
  reqs[0].buffer_id = 0;
  reqs[1].buffer_id = uint64_t{1} << 32;

  EXPECT_THAT((*adapter)->Post(/*peers=*/{"127.0.0.1:1"}, reqs,
                               /*src_block_ids=*/{}, /*dst_block_ids=*/{},
                               /*on_complete=*/nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("requires a completion callback")));
}

}  // namespace
}  // namespace tpu_raiden::transport::lib
