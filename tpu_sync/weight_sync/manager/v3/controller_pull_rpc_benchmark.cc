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

// End-to-end benchmark of the pull RPCs: thousands of synthetic sampler hosts
// run the pull loop over TCP against a `RaidenControllerV3` pull server (the
// test-only encoding), so every decision costs a real RPC: framing, epoll,
// the scheduler and the reply.
//
// Modes:
//   sync:  a whole scheduled pull phase. Every grant takes --pull_ms (0: it
//          completes at once); hosts keep up to k_b pulls in flight and report
//          each completion right away (superseding a parked request). With
//          --shared_host_bandwidth a grant takes n * --pull_ms when the host
//          has n pulls in flight, so a host moves one pull's worth of bytes per
//          --pull_ms (the real demand); otherwise the demand is k_b times that.
//   floor: every host sends report-only requests (max_grants = 0, answered
//          right away) in a closed loop for --duration_s: the RPC cost floor.
//
// Example:
//   blaze run -c opt :controller_pull_rpc_benchmark -- --replicas=1024 \
//     --hosts=4 --bundles=512 --pull_ms=40

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/manager/v3/async_control_server.h"
#include "tpu_sync/weight_sync/manager/v3/controller_v3.h"
#include "tpu_sync/weight_sync/manager/v3/dynamic_pull_engine.h"
#include "tpu_sync/weight_sync/manager/v3/entity_registry.h"
#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"
#include "tpu_sync/weight_sync/manager/v3/test_pull_wire_codec.h"

ABSL_FLAG(std::string, mode, "sync", "sync or floor");
ABSL_FLAG(int32_t, replicas, 1024, "Destination replicas (D)");
ABSL_FLAG(int32_t, hosts, 4, "Hosts per replica (H)");
ABSL_FLAG(int32_t, bundles, 512, "Bundles (B, one layer each)");
ABSL_FLAG(int32_t, grant_batch, 8, "Leases per host (k_b)");
ABSL_FLAG(int32_t, uploads, 8, "Uploads per source host (c)");
ABSL_FLAG(double, pull_ms, 0, "Duration of one pull (sync mode)");
ABSL_FLAG(bool, shared_host_bandwidth, false,
          "Pulls of a host share its bandwidth: a pull granted while the host "
          "has n pulls in flight takes n * --pull_ms (demand model). Otherwise "
          "every pull takes --pull_ms (k_b times the demand)");
ABSL_FLAG(double, duration_s, 10, "Duration (floor mode)");
ABSL_FLAG(int32_t, long_poll_ms, 2000, "Long poll of every request");
ABSL_FLAG(int32_t, client_threads, 2, "Client event loops");
ABSL_FLAG(int32_t, server_threads, 2, "Pull server event loops");

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

using control_pipe::proto::ControlEnvelope;
using control_pipe::proto::ControlResponseEnvelope;

int64_t ThreadCpuNs() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

int64_t Percentile(std::vector<int64_t>& v, double q) {
  if (v.empty()) return 0;
  const size_t k = std::min(v.size() - 1, static_cast<size_t>(q * v.size()));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

struct Host {
  int fd = -1;
  RaidenId unit;
  int32_t host_idx = 0;
  bool seed = false;
  bool seeded_sent = false;
  bool done = false;
  int32_t in_flight = 0;
  std::vector<uint64_t> completed;
  // Outstanding requests: (request id, send time ns).
  std::vector<std::pair<uint64_t, int64_t>> outstanding;
  uint64_t next_request_id = 1;
  std::string in;
  std::string out;
  size_t out_offset = 0;
};

struct ClientResult {
  int64_t requests = 0;
  int64_t grants = 0;
  int64_t done_hosts = 0;
  int64_t errors = 0;
  int64_t cpu_ns = 0;
  std::vector<int64_t> rtt_ns;
};

struct Config {
  std::string mode;
  std::string req_id;
  uint64_t uuid = 0;
  int32_t grant_batch = 8;
  int64_t pull_ns = 0;
  bool shared_host_bandwidth = false;
  int32_t long_poll_ms = 2000;
  absl::Time floor_until;
};

// One client event loop driving a subset of the hosts.
class ClientLoop {
 public:
  ClientLoop(const Config& config, std::vector<Host> hosts)
      : config_(config), hosts_(std::move(hosts)) {}

  ClientResult Run() {
    const int64_t cpu0 = ThreadCpuNs();
    epoll_ = epoll_create1(0);
    CHECK_GE(epoll_, 0);
    for (size_t i = 0; i < hosts_.size(); ++i) {
      epoll_event ev{};
      ev.events = EPOLLIN;
      ev.data.u64 = i;
      CHECK_EQ(epoll_ctl(epoll_, EPOLL_CTL_ADD, hosts_[i].fd, &ev), 0);
    }
    for (size_t i = 0; i < hosts_.size(); ++i) MaybeSend(i);
    std::vector<epoll_event> events(512);
    while (active_ > 0 || HasOutstanding()) {
      if (config_.mode == "floor" && absl::Now() >= config_.floor_until &&
          !HasOutstanding()) {
        break;
      }
      int timeout_ms = 100;
      if (!timers_.empty()) {
        const int64_t wait = timers_.top().first - absl::GetCurrentTimeNanos();
        timeout_ms = static_cast<int>(
            std::clamp<int64_t>((wait + 999999) / 1000000, 0, 100));
      }
      const int n = epoll_wait(epoll_, events.data(),
                               static_cast<int>(events.size()), timeout_ms);
      for (int e = 0; e < n; ++e) {
        const size_t i = events[e].data.u64;
        if (events[e].events & EPOLLOUT) Flush(i);
        if (events[e].events & (EPOLLIN | EPOLLHUP | EPOLLERR)) OnReadable(i);
      }
      FireTimers();
    }
    close(epoll_);
    result_.cpu_ns = ThreadCpuNs() - cpu0;
    return std::move(result_);
  }

 private:
  bool HasOutstanding() const { return outstanding_ > 0; }

  void FireTimers() {
    const int64_t now = absl::GetCurrentTimeNanos();
    std::vector<size_t> touched;
    while (!timers_.empty() && timers_.top().first <= now) {
      auto [t, entry] = timers_.top();
      timers_.pop();
      Host& host = hosts_[entry.first];
      host.completed.push_back(entry.second);
      --host.in_flight;
      touched.push_back(entry.first);
    }
    for (size_t i : touched) MaybeSend(i);
  }

  void MaybeSend(size_t i) {
    Host& host = hosts_[i];
    if (host.done) return;
    if (config_.mode == "floor") {
      if (!host.outstanding.empty()) return;
      if (absl::Now() >= config_.floor_until) {
        host.done = true;
        --active_;
        return;
      }
      Send(i, PullServiceRequest{.req_id = config_.req_id,
                                 .uuid = config_.uuid,
                                 .unit = host.unit,
                                 .host_idx = host.host_idx,
                                 .max_grants = 0,
                                 .long_poll = absl::ZeroDuration()});
      return;
    }
    // A host keeps one request parked; a completion supersedes it.
    const bool has_reports = !host.completed.empty();
    if (host.outstanding.size() >= 2) return;
    if (!host.outstanding.empty() && !has_reports) return;
    // With all k_b leases in flight and nothing to report, a request would be
    // answered right away with nothing: wait for a completion instead.
    const int32_t max_grants =
        std::max(0, config_.grant_batch - host.in_flight);
    const bool seeded = host.seed && !host.seeded_sent;
    if (max_grants == 0 && !has_reports && !seeded) return;
    PullServiceRequest req{
        .req_id = config_.req_id,
        .uuid = config_.uuid,
        .unit = host.unit,
        .host_idx = host.host_idx,
        .max_grants = max_grants,
        .long_poll = absl::Milliseconds(config_.long_poll_ms),
        .seeded = seeded,
        .completed = std::move(host.completed)};
    host.completed.clear();
    host.seeded_sent = true;
    Send(i, std::move(req));
  }

  void Send(size_t i, PullServiceRequest req) {
    Host& host = hosts_[i];
    ControlEnvelope env;
    env.set_message_type(std::string(TestPullWireCodec::kMessageType));
    const uint64_t id = host.next_request_id++;
    env.set_request_id(id);
    env.set_payload(TestPullWireCodec::EncodeRequest(req));
    const std::string bytes = env.SerializeAsString();
    const uint32_t net_len = htonl(static_cast<uint32_t>(bytes.size()));
    host.out.append("CPIP", 4);
    host.out.append(reinterpret_cast<const char*>(&net_len), sizeof(net_len));
    host.out.append(bytes);
    host.outstanding.emplace_back(id, absl::GetCurrentTimeNanos());
    ++outstanding_;
    ++result_.requests;
    Flush(i);
  }

  void Flush(size_t i) {
    Host& host = hosts_[i];
    while (host.out_offset < host.out.size()) {
      const ssize_t n = send(host.fd, host.out.data() + host.out_offset,
                             host.out.size() - host.out_offset, MSG_NOSIGNAL);
      if (n > 0) {
        host.out_offset += n;
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      break;
    }
    const bool pending = host.out_offset < host.out.size();
    if (!pending) {
      host.out.clear();
      host.out_offset = 0;
    }
    epoll_event ev{};
    ev.events = EPOLLIN | (pending ? EPOLLOUT : 0);
    ev.data.u64 = i;
    epoll_ctl(epoll_, EPOLL_CTL_MOD, host.fd, &ev);
  }

  void OnReadable(size_t i) {
    Host& host = hosts_[i];
    char buf[16384];
    while (true) {
      const ssize_t n = recv(host.fd, buf, sizeof(buf), 0);
      if (n > 0) {
        host.in.append(buf, n);
        if (static_cast<size_t>(n) < sizeof(buf)) break;
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      if (n == 0) {
        ++result_.errors;
        if (!host.done) {
          host.done = true;
          --active_;
        }
      }
      break;
    }
    size_t offset = 0;
    ControlResponseEnvelope resp;
    while (host.in.size() - offset >= 8) {
      uint32_t net_len = 0;
      std::memcpy(&net_len, host.in.data() + offset + 4, sizeof(net_len));
      const size_t len = ntohl(net_len);
      if (host.in.size() - offset < 8 + len) break;
      CHECK(resp.ParseFromString(
          absl::string_view(host.in.data() + offset + 8, len)));
      offset += 8 + len;
      OnReply(i, resp);
    }
    host.in.erase(0, offset);
  }

  void OnReply(size_t i, const ControlResponseEnvelope& resp) {
    Host& host = hosts_[i];
    const int64_t now = absl::GetCurrentTimeNanos();
    for (auto it = host.outstanding.begin(); it != host.outstanding.end();
         ++it) {
      if (it->first == resp.request_id()) {
        result_.rtt_ns.push_back(now - it->second);
        host.outstanding.erase(it);
        --outstanding_;
        break;
      }
    }
    if (resp.status_code() != 0) {
      ++result_.errors;
      return;
    }
    absl::StatusOr<PullServiceReply> reply =
        TestPullWireCodec::DecodeReply(resp.payload());
    CHECK_OK(reply.status());
    if (reply->kind == PullReplyKind::kDone) {
      if (!host.done) {
        host.done = true;
        --active_;
        ++result_.done_hosts;
      }
      return;
    }
    if (reply->kind == PullReplyKind::kAborted) {
      if (config_.mode != "floor") ++result_.errors;
      if (!host.done) {
        host.done = true;
        --active_;
      }
      return;
    }
    result_.grants += reply->grants.size();
    for (const PhysicalPullGrant& grant : reply->grants) {
      ++host.in_flight;
      if (config_.pull_ns <= 0) {
        host.completed.push_back(grant.lease_id);
        --host.in_flight;
      } else {
        const int64_t pull_ns = config_.shared_host_bandwidth
                                    ? config_.pull_ns * host.in_flight
                                    : config_.pull_ns;
        timers_.push({now + pull_ns, {i, grant.lease_id}});
      }
    }
    MaybeSend(i);
  }

  const Config config_;
  std::vector<Host> hosts_;
  int epoll_ = -1;
  int64_t active_ = 0;
  int64_t outstanding_ = 0;
  using Timer = std::pair<int64_t, std::pair<size_t, uint64_t>>;
  std::priority_queue<Timer, std::vector<Timer>, std::greater<Timer>> timers_;
  ClientResult result_;

 public:
  void set_active(int64_t n) { active_ = n; }
};

tpu_sync::rpc::RegisterWorkUnitRequest MakeReg(const RaidenId& unit,
                                               int32_t num_hosts,
                                               int32_t num_layers, int idx) {
  tpu_sync::rpc::RegisterWorkUnitRequest req;
  *req.mutable_unit() = RaidenIdToProto(unit);
  std::string ctrl;
  for (int32_t h = 0; h < num_hosts; ++h) {
    req.add_shards(
        absl::StrCat("10.", idx / 250, ".", idx % 250, ".", h, ":8000"));
    absl::StrAppend(&ctrl, h == 0 ? "" : ",", "10.", idx / 250, ".", idx % 250,
                    ".", h, ":9000");
  }
  req.set_control_plane_rpc_address(ctrl);
  for (int32_t l = 0; l < num_layers; ++l) {
    auto* v = req.add_variables();
    v->set_name(absl::StrCat("w", l));
    // The same global shape on every unit; host `h` owns rows
    // [h * 64 / num_hosts, (h + 1) * 64 / num_hosts).
    v->add_shape(64);
    v->add_shape(64);
    v->add_mesh_shape(num_hosts);
    v->add_mesh_shape(1);
    v->add_layout(1);
    v->add_layout(0);
    v->set_item_size(2);
    v->set_layer_idx(l);
    for (int32_t h = 0; h < num_hosts; ++h) v->add_global_shard_indices(h);
  }
  return req;
}

int Connect(int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  CHECK_GE(fd, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  CHECK_EQ(connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0)
      << std::strerror(errno);
  const int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  return fd;
}

int Main() {
  const std::string mode = absl::GetFlag(FLAGS_mode);
  const int32_t num_replicas = absl::GetFlag(FLAGS_replicas);
  const int32_t num_hosts = absl::GetFlag(FLAGS_hosts);
  const int32_t num_bundles = absl::GetFlag(FLAGS_bundles);
  const int32_t grant_batch = absl::GetFlag(FLAGS_grant_batch);
  const int32_t uploads = absl::GetFlag(FLAGS_uploads);
  const double pull_ms = absl::GetFlag(FLAGS_pull_ms);

  RaidenControllerV3::Options opts;
  opts.num_bundle_groups = num_bundles;
  opts.grant_batch_size = grant_batch;
  opts.max_concurrent_uploads_per_source = uploads;
  opts.lease_timeout_ms = 600000;
  opts.long_poll_timeout_ms = absl::GetFlag(FLAGS_long_poll_ms);
  opts.transfer_timeout_ms = 3600000;
  opts.pull_phase = RaidenControllerV3::PullPhaseKind::kScheduled;
  opts.pull_server_port = 0;
  opts.pull_server_threads = absl::GetFlag(FLAGS_server_threads);
  opts.pull_wire_codec = std::make_shared<TestPullWireCodec>();
  opts.custom_rpc_sender = [](absl::string_view,
                              const tpu_sync::rpc::ControlRequest&) {
    tpu_sync::rpc::ControlResponse resp;
    resp.set_success(true);
    return absl::StatusOr<tpu_sync::rpc::ControlResponse>(resp);
  };
  RaidenControllerV3 controller(opts);
  CHECK_OK(controller.StartServer().status());

  const RaidenId trainer{"trainer", "0", "weights", 0};
  CHECK_OK(controller.RegisterWorkUnit(MakeReg(trainer, 1, num_bundles, 0)));
  std::vector<RaidenId> samplers;
  samplers.reserve(num_replicas);
  for (int32_t r = 0; r < num_replicas; ++r) {
    samplers.push_back(RaidenId{"sampler", absl::StrCat(r), "weights", r});
    CHECK_OK(controller.RegisterWorkUnit(
        MakeReg(samplers.back(), num_hosts, num_bundles, r + 1)));
  }
  const std::string req_id = "bench";
  absl::StatusOr<MaterializedTransferPlan> plan =
      controller.BuildMaterializedPlan(req_id, /*uuid=*/1, {trainer}, samplers);
  CHECK_OK(plan.status());
  CHECK_EQ(static_cast<int32_t>(plan->variable_bundles.size()), num_bundles);
  absl::flat_hash_set<int32_t> seeds;
  std::vector<int32_t> seeded_count(num_replicas, 0);
  for (size_t g = 0; g < plan->seed_layout.stripe_seeds.size(); ++g) {
    for (int32_t r : plan->seed_layout.stripe_seeds[g]) {
      seeds.insert(r);
      seeded_count[r] += plan->seed_layout.stripe_bundles[g].size();
    }
  }
  CHECK_OK(controller.StartTransferAsync(req_id, 1, {trainer}, samplers));
  while (controller.GetPullServiceForTest(req_id) == nullptr) {
    absl::SleepFor(absl::Milliseconds(1));
  }

  const int port = controller.pull_server_port();
  const int32_t num_threads = absl::GetFlag(FLAGS_client_threads);
  std::vector<std::vector<Host>> per_thread(num_threads);
  int32_t next = 0;
  for (int32_t r = 0; r < num_replicas; ++r) {
    for (int32_t h = 0; h < num_hosts; ++h) {
      Host host;
      host.fd = Connect(port);
      host.unit = samplers[r];
      host.host_idx = h;
      host.seed = seeds.contains(r);
      per_thread[next++ % num_threads].push_back(std::move(host));
    }
  }
  for (auto& hosts : per_thread) {
    for (Host& host : hosts) {
      const int flags = fcntl(host.fd, F_GETFL, 0);
      CHECK_GE(flags, 0);
      CHECK_EQ(fcntl(host.fd, F_SETFL, flags | O_NONBLOCK), 0);
    }
  }

  Config config;
  config.mode = mode;
  config.req_id = req_id;
  config.uuid = 1;
  config.grant_batch = std::min(grant_batch, uploads);
  config.pull_ns = static_cast<int64_t>(pull_ms * 1e6);
  config.shared_host_bandwidth = absl::GetFlag(FLAGS_shared_host_bandwidth);
  config.long_poll_ms = absl::GetFlag(FLAGS_long_poll_ms);
  const absl::Time start = absl::Now();
  config.floor_until = start + absl::Seconds(absl::GetFlag(FLAGS_duration_s));
  const AsyncControlServer::Stats server0 = *controller.GetPullServerStats();
  std::vector<ClientResult> results(num_threads);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int32_t t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      ClientLoop loop(config, std::move(per_thread[t]));
      loop.set_active(static_cast<int64_t>(
          (num_replicas * num_hosts) / num_threads +
          (t < (num_replicas * num_hosts) % num_threads ? 1 : 0)));
      results[t] = loop.Run();
    });
  }
  for (std::thread& t : threads) t.join();
  const absl::Duration elapsed = absl::Now() - start;
  const AsyncControlServer::Stats server1 = *controller.GetPullServerStats();

  std::shared_ptr<TransferPullService> service =
      controller.GetPullServiceForTest(req_id);
  if (mode == "floor") {
    service->scheduler().Abort(absl::CancelledError("benchmark done"));
  }
  const absl::Status transfer = controller.WaitForTransfer(req_id);
  ClientResult total;
  for (ClientResult& r : results) {
    total.requests += r.requests;
    total.grants += r.grants;
    total.done_hosts += r.done_hosts;
    total.errors += r.errors;
    total.cpu_ns += r.cpu_ns;
    total.rtt_ns.insert(total.rtt_ns.end(), r.rtt_ns.begin(), r.rtt_ns.end());
  }
  int64_t sched_cpu = 0;
  int64_t sched_requests = 0;
  int64_t sched_p99 = 0;
  for (const PullShardStats& s : service->scheduler().GetStats()) {
    sched_cpu += s.thread_cpu_ns;
    sched_requests += s.requests;
    sched_p99 = std::max(sched_p99, s.request_p99_ns);
  }
  // Lower bound of a sync: the host with the most pulls, k_b at a time (or one
  // at a time when the pulls of a host share its bandwidth).
  double lower_bound_s = 0;
  if (pull_ms > 0) {
    const int32_t parallel =
        config.shared_host_bandwidth ? 1 : config.grant_batch;
    for (int32_t r = 0; r < num_replicas; ++r) {
      const int32_t pulls = num_bundles - seeded_count[r];
      lower_bound_s = std::max(
          lower_bound_s,
          std::ceil(static_cast<double>(pulls) / parallel) * pull_ms / 1e3);
    }
  }
  const double secs = absl::ToDoubleSeconds(elapsed);
  const int64_t rpcs = server1.requests - server0.requests;
  std::printf(
      "mode=%s D=%d H=%d B=%d k_b=%d c=%d pull_ms=%.1f server_threads=%d "
      "client_threads=%d\n",
      mode.c_str(), num_replicas, num_hosts, num_bundles, config.grant_batch,
      uploads, pull_ms, absl::GetFlag(FLAGS_server_threads), num_threads);
  std::printf(
      "transfer=%s elapsed=%.3fs lower_bound=%.3fs ratio=%.3f done_hosts=%lld "
      "errors=%lld\n",
      transfer.ok() ? "OK" : std::string(transfer.message()).c_str(), secs,
      lower_bound_s, lower_bound_s > 0 ? secs / lower_bound_s : 0.0,
      static_cast<long long>(total.done_hosts),
      static_cast<long long>(total.errors));
  std::printf(
      "rpcs=%lld rpc/s=%.0f grants=%lld rtt_p50=%.3fms rtt_p99=%.3fms "
      "rtt_max=%.3fms\n",
      static_cast<long long>(rpcs), rpcs / secs,
      static_cast<long long>(total.grants), Percentile(total.rtt_ns, 0.5) / 1e6,
      Percentile(total.rtt_ns, 0.99) / 1e6,
      Percentile(total.rtt_ns, 1.0) / 1e6);
  std::printf(
      "server_loop_cpu=%.2fus/rpc (%.0f%% of a core) scheduler_cpu=%.2fus/req "
      "(%.0f%% of a core, all shards) scheduler_request_p99=%.3fms "
      "client_cpu=%.2fus/rpc\n",
      rpcs > 0 ? (server1.loop_cpu_ns - server0.loop_cpu_ns) / 1e3 / rpcs : 0,
      100.0 * (server1.loop_cpu_ns - server0.loop_cpu_ns) / 1e9 / secs,
      sched_requests > 0 ? sched_cpu / 1e3 / sched_requests : 0,
      100.0 * sched_cpu / 1e9 / secs, sched_p99 / 1e6,
      rpcs > 0 ? total.cpu_ns / 1e3 / rpcs : 0);
  controller.StopServer();
  return 0;
}

}  // namespace
}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  return tpu_raiden::weight_sync::v3::Main();
}
