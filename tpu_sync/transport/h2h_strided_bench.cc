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

// Two-host H2H strided-vs-contiguous KV resharding bandwidth benchmark
// driven by the real RawBufferTransport socket IO helpers
// (WriteExact / WriteVExact / ReadExact), i.e. the writev(2) path used by
// BlockTransport pushes.
//
// Scenario: Qwen3.5-397B-A17B, fp8 KV cache, 64k prompt, hybrid KV pool,
// prefill PCP8 -> decode TP2, one (prefill rank -> decode rank) flow,
// 15 full-attention layers.
//
//   Variant 0 "strided": per FA block, the destination TP rank's head-pair
//     slice of every token: 1024 iovecs x 512 B, source stride 1024 B.
//   Variant 1 "contig": pretend KV had been re-majored so one head's K+V for
//     the whole block is contiguous: 1 iovec of 524,288 B (the first 512
//     tokens). Same payload bytes, 1024x fewer iovec entries.
//
// --streams=N emulates KVCacheManager(parallelism=N) the way
// BlockTransport::Push does it: N TCP connections to the peer, the
// (layer, block) message list contiguously partitioned across N sender
// threads (one thread per stream, mirroring the socket worker pool), and the
// per-iteration completion defined by the slowest stream's ack.
// --scale=S replays the 64k-context plan S times per iteration so that
// multi-stream iterations remain long enough to time accurately.
//
// Usage:
//   VM2:  h2h_strided_bench --role=destination --port=18515
//   VM1:  h2h_strided_bench --role=source --peer=<VM2_IP>:18515 \
//             --streams=8 --scale=8 --iters=20

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>  // NOLINT
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "peregrine/src/api/socket_util.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_serializer.h"

namespace {

using ::peregrine::ReadExact;
using ::peregrine::WriteExact;
using ::peregrine::WriteVExact;
using ::tpu_raiden::transport::lib::ChunkHeader;
using ::tpu_raiden::transport::lib::DeserializeChunkHeader;
using ::tpu_raiden::transport::lib::kChunkHeaderSize;
using ::tpu_raiden::transport::lib::kOpBufferPush;
using ::tpu_raiden::transport::lib::SerializeChunkHeader;

// ---------------------------------------------------------------------------
// Qwen3.5-397B-A17B geometry (fp8 KV, hybrid unified block pool).
// See h2h_strided_bench.py for the full derivation.
// ---------------------------------------------------------------------------
constexpr int kNumFaLayers = 15;
constexpr int kNumKvHeads = 2;  // global GQA KV heads
constexpr int kHeadDim = 256;
constexpr int kItemSize = 1;                                             // fp8
constexpr size_t kHeadPairBytes = 2 * kHeadDim * kItemSize;              // 512
constexpr size_t kTokenStride = 2 * kNumKvHeads * kHeadDim * kItemSize;  // 1024
constexpr int kPcpSize = 8;

constexpr uint64_t kMagicUuid = 0x51E135;
constexpr size_t kHelloSize = 32;  // 8 x uint32 LE
const char* const kVariantNames[2] = {"strided", "contig"};

struct Geometry {
  int block_tokens;
  size_t slot_bytes;
  int blocks_per_rank;
  size_t payload_per_block;
  int msgs_per_plan;
  size_t payload_per_plan;
  int contig_tokens;

  static Geometry Make(int block_tokens, int context_tokens) {
    Geometry g;
    g.block_tokens = block_tokens;
    // 1024: GDN sharded TP4 over PCP8 ranks (default); 512: GDN TP8.
    g.slot_bytes = (block_tokens == 1024) ? 1067008 : 533504;
    const int blocks_total = context_tokens / block_tokens;
    g.blocks_per_rank = blocks_total / kPcpSize;
    g.payload_per_block = static_cast<size_t>(block_tokens) * kHeadPairBytes;
    g.msgs_per_plan = kNumFaLayers * g.blocks_per_rank;
    g.payload_per_plan = g.msgs_per_plan * g.payload_per_block;
    g.contig_tokens = static_cast<int>(g.payload_per_block / kTokenStride);
    return g;
  }
};

uint8_t FillPattern(int layer, int head) {
  return static_cast<uint8_t>((0x40 + head * 100 + layer) % 251);
}

absl::InlinedVector<char, kChunkHeaderSize> PackHeader(uint8_t variant,
                                                       uint16_t layer,
                                                       uint16_t block,
                                                       uint32_t payload) {
  const ChunkHeader header = {
      .version = 1,
      .op = kOpBufferPush,
      .flags = variant,
      .buffer_id = layer,
      .reserved = block,
      .remote_id = 0,
      .local_id = 0,
      .count_or_size = payload,
      .uuid = kMagicUuid,
  };
  return SerializeChunkHeader(header);
}

void CheckOk(const absl::Status& s, const char* what) {
  if (!s.ok()) {
    std::fprintf(stderr, "FATAL %s: %s\n", what, s.ToString().c_str());
    std::exit(1);
  }
}

int Die(const char* msg) {
  std::perror(msg);
  std::exit(1);
}

void SetNodelay(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

double CpuSeconds() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec +
         ru.ru_stime.tv_usec * 1e-6;
}

// One staging buffer per FA layer: [blocks_per_rank x slot_bytes], filled so
// every head-pair chunk carries FillPattern(layer, head).
std::vector<std::vector<uint8_t>> BuildSourceBuffers(const Geometry& g) {
  std::vector<std::vector<uint8_t>> buffers(kNumFaLayers);
  for (int layer = 0; layer < kNumFaLayers; ++layer) {
    auto& buf = buffers[layer];
    buf.assign(static_cast<size_t>(g.blocks_per_rank) * g.slot_bytes, 0);
    for (int block = 0; block < g.blocks_per_rank; ++block) {
      const size_t base = static_cast<size_t>(block) * g.slot_bytes;
      for (int tok = 0; tok < g.block_tokens; ++tok) {
        for (int head = 0; head < kNumKvHeads; ++head) {
          std::memset(&buf[base + tok * kTokenStride + head * kHeadPairBytes],
                      FillPattern(layer, head), kHeadPairBytes);
        }
      }
    }
  }
  return buffers;
}

struct BlockPlan {
  uint16_t layer;
  uint16_t block;
  std::vector<struct iovec> iov;
};

std::vector<BlockPlan> BuildPlans(const Geometry& g,
                                  std::vector<std::vector<uint8_t>>& buffers,
                                  int dest_rank, int variant) {
  std::vector<BlockPlan> plans;
  plans.reserve(g.msgs_per_plan);
  const size_t head_off = dest_rank * kHeadPairBytes;
  for (int layer = 0; layer < kNumFaLayers; ++layer) {
    uint8_t* data = buffers[layer].data();
    for (int block = 0; block < g.blocks_per_rank; ++block) {
      const size_t base = static_cast<size_t>(block) * g.slot_bytes;
      BlockPlan plan;
      plan.layer = static_cast<uint16_t>(layer);
      plan.block = static_cast<uint16_t>(block);
      if (variant == 0) {  // strided
        plan.iov.reserve(g.block_tokens);
        for (int t = 0; t < g.block_tokens; ++t) {
          plan.iov.push_back(
              {data + base + t * kTokenStride + head_off, kHeadPairBytes});
        }
      } else {  // contig
        plan.iov.push_back({data + base, g.payload_per_block});
      }
      plans.push_back(std::move(plan));
    }
  }
  return plans;
}

// Contiguous partition of [0, total) across streams, BlockTransport style
// (base count + remainder spread over the leading streams).
struct Partition {
  size_t offset;
  size_t count;
};

std::vector<Partition> PartitionMsgs(size_t total, int streams) {
  std::vector<Partition> parts(streams);
  const size_t base = total / streams;
  const size_t rem = total % streams;
  size_t off = 0;
  for (int i = 0; i < streams; ++i) {
    parts[i].offset = off;
    parts[i].count = base + (static_cast<size_t>(i) < rem ? 1 : 0);
    off += parts[i].count;
  }
  return parts;
}

int ConnectTo(const std::string& host, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) Die("socket");
  struct sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) Die("inet_pton");
  if (connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0)
    Die("connect");
  SetNodelay(fd);
  return fd;
}

int RunSource(const std::string& peer, const Geometry& g, int iters, int warmup,
              int dest_rank, int streams, int scale) {
  // BlockTransport::Push clamps parallelism to the block count; mirror it at
  // message granularity so short contexts do not open idle connections.
  if (streams > g.msgs_per_plan) {
    std::printf("[source] clamping streams %d -> %d (msgs per plan)\n", streams,
                g.msgs_per_plan);
    streams = g.msgs_per_plan;
  }
  const size_t colon = peer.rfind(':');
  CHECK_NE(colon, std::string::npos);
  const std::string host = peer.substr(0, colon);
  const int port = std::atoi(peer.c_str() + colon + 1);

  std::printf("[source] building staging buffers (pattern fill)...\n");
  auto buffers = BuildSourceBuffers(g);
  std::vector<BlockPlan> plans[2] = {
      BuildPlans(g, buffers, dest_rank, 0),
      BuildPlans(g, buffers, dest_rank, 1),
  };
  const auto parts = PartitionMsgs(g.msgs_per_plan, streams);
  const size_t payload_per_iter = g.payload_per_plan * scale;

  // One connection per stream, each announced with a per-stream hello:
  // {block_tokens, iters, warmup, dest_rank, msgs_this_stream,
  //  payload_per_block, num_streams, scale}.
  std::vector<int> fds(streams);
  for (int i = 0; i < streams; ++i) {
    fds[i] = ConnectTo(host, port);
    uint32_t hello[8] = {static_cast<uint32_t>(g.block_tokens),
                         static_cast<uint32_t>(iters),
                         static_cast<uint32_t>(warmup),
                         static_cast<uint32_t>(dest_rank),
                         static_cast<uint32_t>(parts[i].count),
                         static_cast<uint32_t>(g.payload_per_block),
                         static_cast<uint32_t>(streams),
                         static_cast<uint32_t>(scale)};
    static_assert(sizeof(hello) == kHelloSize);
    CheckOk(WriteExact(fds[i], hello, sizeof(hello)), "hello");
  }
  std::printf(
      "[source] connected to %s with %d stream(s), scale=%d, "
      "payload/iter=%.1f MiB\n",
      peer.c_str(), streams, scale, payload_per_iter / 1048576.0);

  std::vector<double> secs[2];
  double cpu[2] = {0.0, 0.0};
  std::atomic<bool> failed{false};

  for (int it = 0; it < warmup + iters; ++it) {
    for (int v = 0; v < 2; ++v) {
      const double cpu0 = CpuSeconds();
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<std::thread> threads;
      threads.reserve(streams);
      for (int i = 0; i < streams; ++i) {
        threads.emplace_back([&, i, v]() {
          const BlockPlan* base = plans[v].data() + parts[i].offset;
          for (int s = 0; s < scale; ++s) {
            for (size_t m = 0; m < parts[i].count; ++m) {
              const BlockPlan& plan = base[m];
              const auto header_bytes =
                  PackHeader(static_cast<uint8_t>(v), plan.layer, plan.block,
                             static_cast<uint32_t>(g.payload_per_block));
              absl::Status st =
                  WriteExact(fds[i], header_bytes.data(), header_bytes.size());
              if (st.ok()) {
                st = WriteVExact(fds[i], absl::MakeConstSpan(plan.iov));
              }
              if (!st.ok()) {
                std::fprintf(stderr, "FATAL stream %d send: %s\n", i,
                             st.ToString().c_str());
                failed = true;
                return;
              }
            }
          }
          uint8_t ack;
          absl::Status st = ReadExact(fds[i], &ack, 1);
          if (!st.ok() || ack != 1) {
            std::fprintf(stderr, "FATAL stream %d ack: %s\n", i,
                         st.ToString().c_str());
            failed = true;
          }
        });
      }
      for (auto& t : threads) t.join();
      if (failed) return 1;
      const std::chrono::duration<double> dt =
          std::chrono::steady_clock::now() - t0;
      const double cpu_dt = CpuSeconds() - cpu0;
      if (it >= warmup) {
        secs[v].push_back(dt.count());
        cpu[v] += cpu_dt;
      }
      std::printf(
          "[source] iter %2d %-8s %8.2f ms  %6.2f GiB/s (%6.1f Gbit/s)\n", it,
          kVariantNames[v], dt.count() * 1e3,
          payload_per_iter / dt.count() / (1 << 30),
          payload_per_iter * 8 / dt.count() / 1e9);
    }
  }

  std::printf(
      "\n=== H2H achievable bandwidth (payload bytes only, "
      "RawBufferTransport::WriteVExact, %d streams) ===\n",
      streams);
  for (int v = 0; v < 2; ++v) {
    std::sort(secs[v].begin(), secs[v].end());
    const auto gib = [&](double t) { return payload_per_iter / t / (1 << 30); };
    const auto gbit = [&](double t) { return payload_per_iter * 8 / t / 1e9; };
    double wall = 0;
    for (double t : secs[v]) wall += t;
    const double med = secs[v][secs[v].size() / 2];
    std::printf(
        "%-8s: best %6.2f GiB/s (%6.1f Gbit/s) | median %6.2f GiB/s "
        "(%6.1f Gbit/s) | worst %6.2f GiB/s | sender busy-cores %4.2f | "
        "iovecs/iter %zu\n",
        kVariantNames[v], gib(secs[v].front()), gbit(secs[v].front()), gib(med),
        gbit(med), gib(secs[v].back()), cpu[v] / wall,
        static_cast<size_t>(g.msgs_per_plan) * scale *
            (v == 0 ? g.block_tokens : 1));
  }
  for (int fd : fds) close(fd);
  return 0;
}

void Verify(const Geometry& g, int variant, int dest_rank, int layer,
            const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> expected(g.payload_per_block);
  if (variant == 0) {
    std::memset(expected.data(), FillPattern(layer, dest_rank),
                expected.size());
  } else {
    for (int t = 0; t < g.contig_tokens; ++t) {
      for (int head = 0; head < kNumKvHeads; ++head) {
        std::memset(&expected[t * kTokenStride + head * kHeadPairBytes],
                    FillPattern(layer, head), kHeadPairBytes);
      }
    }
  }
  if (std::memcmp(payload.data(), expected.data(), expected.size()) != 0) {
    std::fprintf(stderr, "FATAL verification failed: variant=%s layer=%d\n",
                 kVariantNames[variant], layer);
    std::exit(1);
  }
}

// Per-connection receive loop: mirrors RawBufferTransport::ConnectionWorker
// (one thread per accepted connection).
int ReceiveStream(int fd, const Geometry& g, int iters, int warmup,
                  int dest_rank, int scale, uint32_t msgs_this_stream,
                  std::atomic<bool>* failed) {
  std::vector<char> header_buf(kChunkHeaderSize);
  std::vector<uint8_t> payload(g.payload_per_block);
  const uint8_t ack = 1;
  for (int it = 0; it < warmup + iters && !*failed; ++it) {
    for (int v = 0; v < 2; ++v) {
      for (uint32_t m = 0; m < msgs_this_stream * scale; ++m) {
        absl::Status st = ReadExact(fd, header_buf.data(), kChunkHeaderSize);
        uint32_t size = 0;
        uint16_t layer = 0;
        if (st.ok()) {
          absl::StatusOr<ChunkHeader> parsed_header =
              DeserializeChunkHeader(header_buf);
          if (!parsed_header.ok()) {
            st = parsed_header.status();
          } else {
            const ChunkHeader& h = *parsed_header;
            size = h.count_or_size;
            layer = h.buffer_id;
            if (h.op != kOpBufferPush || h.flags != v || h.uuid != kMagicUuid ||
                size != g.payload_per_block) {
              st = absl::InternalError("bad header");
            }
          }
        }
        if (st.ok()) {
          st = ReadExact(fd, payload.data(), size);
        }
        if (!st.ok()) {
          std::fprintf(stderr, "FATAL recv (it=%d v=%d m=%u): %s\n", it, v, m,
                       st.ToString().c_str());
          *failed = true;
          return 1;
        }
        if (it == 0) {
          Verify(g, v, dest_rank, layer, payload);
        }
      }
      absl::Status st = WriteExact(fd, &ack, 1);
      if (!st.ok()) {
        *failed = true;
        return 1;
      }
    }
  }
  return 0;
}

int RunDestination(int port) {
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) Die("socket");
  int one = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  struct sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);
  if (bind(srv, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0)
    Die("bind");
  if (listen(srv, 32) < 0) Die("listen");
  std::printf("[dest] listening on :%d\n", port);

  // Accept the first stream to learn the stream count, then the rest.
  struct StreamConn {
    int fd;
    uint32_t msgs;
  };
  std::vector<StreamConn> conns;
  Geometry g = Geometry::Make(1024, 65536);
  int iters = 0, warmup = 0, dest_rank = 0, scale = 1;
  uint32_t num_streams = 1;
  do {
    int fd = accept(srv, nullptr, nullptr);
    if (fd < 0) Die("accept");
    SetNodelay(fd);
    uint32_t hello[8];
    CheckOk(ReadExact(fd, hello, sizeof(hello)), "hello");
    if (conns.empty()) {
      // Context length is a source-side concern (it only sets how many
      // messages exist); the receive loop is driven by per-stream hello
      // message counts, so any context value works here.
      g = Geometry::Make(static_cast<int>(hello[0]), 65536);
      iters = hello[1];
      warmup = hello[2];
      dest_rank = hello[3];
      num_streams = hello[6];
      scale = hello[7];
    }
    if (hello[5] != g.payload_per_block || hello[6] != num_streams) {
      std::fprintf(stderr, "FATAL inconsistent hello across streams\n");
      return 1;
    }
    conns.push_back({fd, hello[4]});
  } while (conns.size() < num_streams);
  std::printf(
      "[dest] %u stream(s) connected: block_tokens=%d msgs/plan=%d scale=%d "
      "payload/iter=%zu B\n",
      num_streams, g.block_tokens, g.msgs_per_plan, scale,
      g.payload_per_plan * scale);

  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  threads.reserve(conns.size());
  for (const StreamConn& c : conns) {
    threads.emplace_back(ReceiveStream, c.fd, g, iters, warmup, dest_rank,
                         scale, c.msgs, &failed);
  }
  for (auto& t : threads) t.join();
  if (failed) return 1;
  std::printf("[dest] done; all iterations verified/acked.\n");
  for (const StreamConn& c : conns) close(c.fd);
  close(srv);
  return 0;
}

std::string GetFlag(int argc, char** argv, absl::string_view name,
                    absl::string_view def) {
  const std::string prefix = absl::StrCat("--", name, "=");
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]).rfind(prefix, 0) == 0) {
      return std::string(argv[i]).substr(prefix.size());
    }
  }
  return std::string(def);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string role = GetFlag(argc, argv, "role", "");
  const std::string peer = GetFlag(argc, argv, "peer", "");
  const int port = std::atoi(GetFlag(argc, argv, "port", "18515").c_str());
  const int iters = std::atoi(GetFlag(argc, argv, "iters", "20").c_str());
  const int warmup = std::atoi(GetFlag(argc, argv, "warmup", "3").c_str());
  const int dest_rank =
      std::atoi(GetFlag(argc, argv, "dest_rank", "0").c_str());
  const int block_tokens =
      std::atoi(GetFlag(argc, argv, "block_tokens", "1024").c_str());
  const int streams = std::atoi(GetFlag(argc, argv, "streams", "1").c_str());
  const int scale = std::atoi(GetFlag(argc, argv, "scale", "1").c_str());
  const int context_tokens =
      std::atoi(GetFlag(argc, argv, "context_tokens", "65536").c_str());

  if (block_tokens != 1024 && block_tokens != 512) {
    std::fprintf(stderr, "--block_tokens must be 1024 or 512\n");
    return 1;
  }
  if (streams < 1 || streams > 64 || scale < 1 || scale > 64) {
    std::fprintf(stderr, "--streams and --scale must be in [1, 64]\n");
    return 1;
  }
  if (context_tokens <= 0 || context_tokens % (block_tokens * kPcpSize) != 0) {
    std::fprintf(stderr,
                 "--context_tokens must be a positive multiple of "
                 "block_tokens * PCP size (%d)\n",
                 block_tokens * kPcpSize);
    return 1;
  }
  const Geometry g = Geometry::Make(block_tokens, context_tokens);

  if (role == "source") {
    if (peer.empty()) {
      std::fprintf(stderr, "--peer=ip:port is required for --role=source\n");
      return 1;
    }
    return RunSource(peer, g, iters, warmup, dest_rank, streams, scale);
  }
  if (role == "destination") {
    return RunDestination(port);
  }
  std::fprintf(stderr,
               "usage: %s --role=source|destination [--peer=ip:port] "
               "[--port=N] [--iters=N] [--warmup=N] [--dest_rank=0|1] "
               "[--block_tokens=1024|512] [--streams=N] [--scale=N] "
               "[--context_tokens=N]\n",
               argv[0]);
  return 1;
}
