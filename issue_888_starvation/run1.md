# Issue #888 — Cross-peer handshake starvation on the consumer control path

**Run:** 1
**Date:** 2026-09-22
**Branch:** `investigate-888-peer-starvation` (branched from `origin/main`)
**Base commit:** `868ae48` ("Standardize MetricFamilyBuffer and BufferedMetricsExporter …")
**Issue:** https://github.com/google/tpu-sync/issues/888
**Issue's stated base:** `083aec4` (verified: an ancestor of `868ae48`, so line numbers in the issue have drifted — see §3)

> **Scope of this run:** investigation + failing reproduction tests + fix proposals.
> **No fix has been implemented.** Nothing in `tpu_sync/` is modified except the
> addition of four tests to an existing test file (§5).

---

## 0. How to use this document (for run 2, run 3, …)

This file is intended to be self-sufficient. A future session should be able to
read *only this file* (plus the repo) and continue.

- §1 is the verdict.
- §2 is the mechanism, with verified code pointers at `868ae48`.
- §3 lists where the issue text is **wrong or stale**, so we don't re-derive it.
- §4 is the **assumptions register** — each assumption is numbered (`A1`…`A12`),
  has a confidence level and a stated way to validate it. Future runs should
  amend this table rather than rewrite it.
- §5 is the reproduction tests that exist on this branch and their current status.
- §6 is the fix menu with code sketches and trade-offs.
- §7 is the recommendation.
- §8 is the open-questions list for the reviewer.
- §9 is the changelog / handoff for run 2.

Convention for future runs: `run2.md` should reference `run1.md` for everything
it does not change, and carry forward §4 (assumptions) and §9 (changelog) in
full with amendments marked.

---

## 1. Verdict

**This is a real issue, not a non-issue.** It reproduces deterministically.

The core claim in #888 — that a single unresponsive producer blocks control-plane
handshakes to *every other* producer on a consumer — is correct and is a genuine
fault-isolation defect. There is no per-peer admission control, no per-peer
queueing, and no peer health tracking anywhere on this path.

Three findings **beyond** what the issue describes, all of which make it worse:

1. **The handshake deadline is per-syscall, not per-handshake** (§2.4). `timeout_s`
   is applied as `SO_RCVTIMEO`/`SO_SNDTIMEO`, and `ReadExact` loops. A peer that
   answers one byte just inside each timeout holds a worker for
   `sizeof(ControlResponseHeader) × timeout_s` = **24 × 120s = 48 minutes**, not
   120s. The starvation window is not bounded by `timeout_s`.
2. **The handshake pools are shared with the data path, not just across peers**
   (§2.5). `push_pool_` carries both the consumer's outbound handshake and the
   producer's layer pushes; `pull_pool_` carries both the producer's inbound
   control handlers and the consumer's H2H data-pull chunks. So a stuck handshake
   also steals data-plane capacity, and vice versa.
3. **Staging slots starve before threads do, and the failure mode is worse**
   (§2.6). A receive session allocates its staging slot in `StartRead`, *before*
   the handshake is scheduled, and holds it for the whole handshake. Once a sick
   peer's in-flight sessions have taken every slot, `StartRead` for a healthy peer
   fails allocation and is **dropped outright** — reported in `failed_recving`
   within milliseconds, never attempted. This is a harder failure than the delay
   the issue describes: reads to healthy producers are not slow, they are
   rejected. Measured at 48 ms in `SickPeerStarvesStagingSlotsForHealthyPeer`.

---

## 2. Mechanism, verified at `868ae48`

### 2.1 The pools are four threads, fixed, and not peer-aware

`tpu_sync/kv_cache/kv_cache_manager_base.cc:396-399` (and an identical block at
`:496-499`, a second constructor):

```cpp
constexpr size_t kPoolSize = 4;
dma_pool_  = std::make_unique<NumaThreadPool>(kPoolSize);
push_pool_ = std::make_shared<NumaThreadPool>(kPoolSize);
pull_pool_ = std::make_unique<NumaThreadPool>(kPoolSize);
```

`kPoolSize` is a `constexpr` local — **not** configurable, not derived from
hardware concurrency, not overridable by the caller. Both constructors must be
changed together for any sizing fix.

`NumaThreadPool` is a single-queue FIFO pool
(`tpu_sync/core/numa_thread_pool.h:102-106`):

```cpp
std::vector<std::thread> workers;
std::queue<std::function<void()>> tasks;   // one queue, FIFO, no peer key
std::mutex queue_mutex;
std::condition_variable condition;
bool stop;
```

The `numa_node` argument to `Schedule` only pins the worker thread
(`numa_thread_pool.h:58-60`); it does **not** partition the queue. So there is
exactly one FIFO for all peers and all work types.

### 2.2 The consumer schedules the entire blocking handshake onto `push_pool_`

The handshake has moved since the issue was filed. It now lives in
`TransferReceiveSession::ExecutePullRequest`,
`tpu_sync/core/transfer_receive_session.cc:454-506`:

```cpp
void TransferReceiveSession::ExecutePullRequest(
    KVCacheManagerWithTransfer& manager, const std::string& remote_endpoint) {
  ...
  base_->push_pool()->Schedule(                                    // line 465
      target_node, [self = shared_from_this(), &manager, remote_endpoint,
                    session_req_id, load_plan = std::move(load_plan)]() {
        ...
        absl::StatusOr<PullStreamResponseSpec> response =
            manager.control_backend_->SendPullRequest(                // line 484
                remote_endpoint, req_spec, absl::Seconds(manager.timeout_s_));
        ...
      });
}
```

The closure is *entirely blocking*: connect, three writes, one-or-two reads. It
occupies its worker for the whole handshake. `StartRead`
(`tpu_sync/core/kv_cache_manager_with_transfer.cc:748-835`) calls it at line 834.

### 2.3 `timeout_s_` defaults to 120 seconds

`tpu_sync/core/kv_cache_manager_with_transfer.h:353`:

```cpp
double timeout_s_ = 120.0;
```

with the same default repeated on five public constructors/factories
(`:186`, `:196`, `:209`, `:217`, `:301`). This is the **bulk transfer** timeout.
The control handshake is a 168-byte request header + 16 bytes of block ids and a
24-byte response header, and it inherits this value verbatim. There is no
separate control-plane timeout anywhere in the codebase.

### 2.4 The timeout is per-syscall, so the handshake is effectively unbounded

`TcpControlPlaneBackend::SendPullRequest`
(`tpu_sync/core/tcp_control_plane_backend.cc:577-644`) applies the timeout once,
to `ConnectTcp`:

```cpp
double timeout_s = absl::ToDoubleSeconds(timeout);
absl::StatusOr<int> fd = ConnectTcp(remote_endpoint, timeout_s);   // line 581
```

`ConnectTcp` turns it into socket options (`:143`):

```cpp
if (absl::Status status = SetSocketTimeouts(fd, timeout_s); !status.ok()) { ... }
```

and `SetSocketTimeouts`
(`tpu_sync/common/control_pipe/tcp_control_pipe.cc:65-76`) sets:

```cpp
setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
```

`SO_RCVTIMEO` bounds **one `recv()`**. `ReadExact`
(`tpu_sync/core/tcp_control_plane_backend.cc:188-209`) loops:

```cpp
while (remaining > 0) {
  ssize_t bytes_read = read(fd, ptr, remaining);
  ...
  ptr += bytes_read;
  remaining -= bytes_read;     // a successful partial read restarts the clock
}
```

Every byte that arrives resets the budget. There is no absolute deadline
threaded through `SendPullRequest`. A peer that dribbles the 24-byte response
header one byte per `timeout_s − ε` holds the worker for `24 × timeout_s`.
At the 120s default that is **48 minutes on one of four workers**; four such
peers wedge the consumer's entire handshake path for the same duration.

Note this does not require malice — a peer under severe memory pressure or with
a saturated NIC produces the same pattern.

### 2.5 Both pools are shared between control and data

| Pool | Threads | Used for | Code pointer |
|---|---|---|---|
| `push_pool_` | 4 | Consumer's **outbound** control handshake | `transfer_receive_session.cc:465` |
| `push_pool_` | 4 | Producer's layer **data** push dispatch | `transfer_send_session.cc:396` |
| `push_pool_` | 4 | Data-path chunk fan-out | `kv_cache_manager_base.cc:885`, `:1099`, `:2604` |
| `pull_pool_` | 4 | Producer's **inbound** control handlers | `kv_cache_manager_with_transfer.cc:1303-1304` |
| `pull_pool_` | 4 | Consumer's H2H **data** pull chunks | `kv_cache_manager_base.cc:968` |
| `RawBufferTransport::push_pool_` | 16 | Parallel batched data push only | `raw_buffer_transport.cc:79`, `:837` |

The producer-side executor is wired in
`KVCacheManagerWithTransfer::InitializeControlPlane`
(`tpu_sync/core/kv_cache_manager_with_transfer.cc:1301-1309`):

```cpp
auto executor = [this](std::function<void()> task) {
  base_->pull_pool()->Schedule(base_->assigned_numa_node(), std::move(task));
};
control_backend_ = CreateControlPlaneBackend(
    ResolveControlPlaneBackendType(), std::move(executor), absl::Seconds(timeout_s_));
```

and consumed by the accept loop at `tcp_control_plane_backend.cc:411-423`, which
hands each accepted socket to `HandleControlConnection` — whose first act is a
blocking `ReadExact` of the 168-byte request header (`:455`). A consumer that
connects and then goes silent therefore holds a `pull_pool_` worker for
`timeout_s`, which is the mirror bug already covered by
`HandlersOutliveConsumersThatNeverSpeak`.

**Consequence not in the issue:** on the producer side, control handlers and
data-path pull chunks contend for the same 4 threads. A stuck handshake degrades
data throughput and a slow data pull delays handshake handling. On the consumer
side the same is true of `push_pool_`.

### 2.6 Staging slots are held across the handshake and starve first

`StartRead` creates the receive session — which allocates staging — *before*
scheduling the handshake (`kv_cache_manager_with_transfer.cc:800-819`, then
`:834`). `TransferReceiveSession::Create`
(`tpu_sync/core/transfer_receive_session.cc:82-105`):

```cpp
if (!session->AllocateStagingForLoad(req_id, local_block_ids,
                                     local_host_block_ids, &host_block_ids)) {
  return absl::ResourceExhaustedError(
      absl::StrCat("Failed to allocate staging for load req_id=", req_id,
                   ", uuid=", uuid));
}
```

and `StartRead`'s handling of that failure (`:805-808`):

```cpp
if (!created.ok()) {
  failed_recving_.insert(req_id);
  return;                      // never reaches ExecutePullRequest
}
```

So the slot is pinned for the entire handshake, including the whole timeout. The
staging pool is sized `num_slots`, independent of `kPoolSize`. Once a sick peer
holds all of them:

- reads to **healthy** peers fail `AllocateStagingForLoad`,
- they are inserted into `failed_recving_` and **return immediately**,
- the handshake is never scheduled, so per-peer *thread* fairness alone would
  not save them.

This is confirmed empirically: `SickPeerStarvesStagingSlotsForHealthyPeer`
(§5.3) shows the healthy read landing in `failed_recving` after **48 ms**.

The practical consequence for a fix: **admission control has to cover staging
slots, not only pool workers.** A per-peer thread cap (Fix A) that leaves slot
allocation unchanged still lets one sick peer deny service to every other peer,
just at a different resource. See §6 Fix A's "slot reservation" note.

### 2.7 There is no peer health tracking at all

```
$ grep -rn "circuit\|CircuitBreaker\|blacklist\|denylist\|unhealthy\|
           health_check\|peer_health\|quarantine" --include=*.cc --include=*.h \
    tpu_sync/core/ tpu_sync/kv_cache/ | grep -v _test
(no matches)
```

No circuit breaker, no ejection, no per-peer failure counters, no backoff. Every
request to a known-dead peer pays full freight, forever. This is why the issue's
"for as long as requests to the dead peer keep arriving" clause matters: nothing
in the system ever stops those requests from being issued.

### 2.8 Why the existing tests do not catch it

`tpu_sync/core/kv_cache_manager_with_transfer_control_test.cc:62-64` states the
single-peer version of the property as intended behaviour:

```cpp
// Both ends of a control handshake share one pool of four workers, so four
// stuck handshakes are enough to starve either side.
constexpr int kPoolSize = 4;
constexpr double kTimeoutS = 0.5;
```

`ConsumerGivesUpOnProducerThatNeverAnswers` (`:652`) dispatches `kPoolSize + 1`
reads at **one** `SilentProducer` and asserts the last one eventually connects.
With one peer this is benign — the delayed request was aimed at the dead peer
anyway. The gap is that no test uses **two** peers, so the cross-peer coupling
is untested and undocumented. §5 closes that gap.

---

## 3. Corrections to the issue text

| # | Issue says | Reality at `868ae48` |
|---|---|---|
| C1 | `push_pool_` at `kv_cache_manager_base.cc:391-394` and `491-494` | Now `:396-399` and `:496-499`. Content unchanged. |
| C2 | Handshake scheduled in `kv_cache_manager_with_transfer.cc:1982-2038`, `ConnectTcp` at 1988, `ReadControlResponseHeader` at 2016 | **Refactored.** The handshake now lives in `TransferReceiveSession::ExecutePullRequest`, `transfer_receive_session.cc:454-506`, scheduling at `:465` and calling `SendPullRequest` at `:484`. `kv_cache_manager_with_transfer.cc` is only ~1500 lines of relevant surface now; those line numbers do not exist as described. |
| C3 | `timeout_s_` at `kv_cache_manager_with_transfer.h:576` | Now `:353`. Value still `120.0`. |
| C4 | "for up to `timeout_s_` (default 120s)" | **Understated.** The bound is per-syscall, so the true worst case is `24 × timeout_s` for the response header alone (§2.4). |
| C5 | Implies one shared pool | There are **two** 4-thread pools, one per direction (`push_pool_` outbound, `pull_pool_` inbound), and each is *also* shared with the data path (§2.5). |

### Validation of the "Current Knowledge" in the task brief

| Claim | Verdict |
|---|---|
| "Handshake — only 4 threads" | **True**, with the refinement that there are two such pools (outbound `push_pool_`, inbound `pull_pool_`), 4 threads each, and neither is exclusively for handshakes. |
| "Data Transfer — 16 threads" | **Partly true.** The only 16-thread pool is `RawBufferTransport::push_pool_` (`kMaxPushThreads = 16`, `raw_buffer_transport.cc:79`), lazily created at `:837`, and used **only** for the producer's parallel batched push when `parallelism > 1 && batches.size() > 1`. The consumer's data receive/H2H-pull path runs on the 4-thread `pull_pool_` (`kv_cache_manager_base.cc:968`), not on 16 threads. |
| "Sick producer slow at handshake starves the decoder from responding to the healthy prefill" | **True and now proven** — see `SickPeerDoesNotDelayHandshakeToHealthyPeer` (§5.1). |
| "Sick producer slow at data transfer starves the data transfer threads" | **True, and worse than stated.** In pull mode the consumer's chunk reads run on `pull_pool_`, which is the *same* 4 threads that serve inbound control handlers. So slow data transfer from one peer degrades control-plane responsiveness to all peers. Not yet covered by a test — see §9 for run 2. |

---

## 4. Assumptions register

Each assumption is numbered so future runs can amend individual rows.

| # | Assumption | Confidence | Basis / how to invalidate |
|---|---|---|---|
| A1 | `kPoolSize = 4` is what production runs with; nothing overrides it at runtime. | **High** | It is a `constexpr` function-local in both constructors (`kv_cache_manager_base.cc:396`, `:496`). Invalidate by finding a build flag or subclass that swaps the pools. |
| A2 | Production uses the default `timeout_s = 120.0`. | **Medium** | It is the default on all five entry points, but callers may pass a value. **Needs confirmation from whoever owns the deployment config** — the blast radius scales linearly with it. |
| A3 | The TCP control-plane backend (not gRPC) is the one in use. | **Medium** | `ResolveControlPlaneBackendType()` (`kv_cache_manager_with_transfer.cc:1307`) picks at runtime; `grpc_control_plane_backend.cc:277` also implements `SendPullRequest`. **The gRPC path has not been analysed in this run** — see §8 Q1. The pool-level starvation is backend-independent (it is a property of `push_pool_`), but the per-syscall timeout finding (§2.4) is TCP-specific. |
| A4 | A "sick" producer accepts TCP connections but does not answer at the application layer. | **High** | This is exactly the failure the issue describes (no TCP RST). It is also the realistic one: the listener socket is in the kernel, so the backlog keeps completing handshakes even when the process is wedged in a GC pause, a page-fault storm, or a deadlock. A hard crash producing RST fails fast and is *not* the interesting case. |
| A5 | `SilentProducer.accepted()` incrementing is a faithful proxy for "the consumer's worker began this handshake". | **High** | The consumer's worker calls `connect()` as its first blocking act; the producer's accept thread is parked in `accept()` and returns immediately. Measuring accept therefore isolates *pool scheduling latency* from any handshake semantics. |
| A6 | Reads to distinct peers are independent work with no shared lock outside the pool. | **Medium-High** | `StartRead` takes `mu_` only to register the session (`kv_cache_manager_with_transfer.cc:780-819`) and releases it before `ExecutePullRequest`. If a future change holds `mu_` across the handshake the bug becomes worse, not better. |
| A7 | Fixing fairness in the pool is sufficient; the callers do not themselves serialise per-peer. | **Medium** | Not exhaustively verified. Before implementing a fix, confirm the scheduler above `StartRead` is not already rate-limiting per peer. See §8 Q2. |
| A8 | 2P1D is the motivating topology, but the defect is O(1 sick peer) regardless of fan-in. | **High** | One sick peer suffices to consume 4 workers given ≥4 in-flight reads to it. Larger fan-in makes it *more* likely, not less. |
| A9 | The `peer_fair_dispatcher.{h,cc,_test.cc}` files present as **untracked** in the working tree at the start of this session are a prior, independent fix attempt. | **High** | They are untracked, not on `origin/main`, and their header comment describes exactly this bug. **They are deliberately not used, not built, and not evaluated in this run** (the brief says propose, do not fix). Run 2 should decide whether to adopt, adapt, or discard them. See §8 Q3. |
| A10 | The reproduction tests' timing thresholds (500 ms prompt-contact, `kStarvationTimeoutS = 4.0`) are wide enough not to flake on a loaded CI machine. | **Medium-High** (measured) | Chosen so the two outcomes differ by ~8×. Observed delays were 4.054 s and 4.040 s against a 0.5 s bound — an 8× margin, and the values track `kStarvationTimeoutS` to within 55 ms. If CI is noisy, widen the gap by raising `kStarvationTimeoutS` rather than lowering the 500 ms bound. |
| A13 | Each 1-block read consumes exactly one staging slot, so `num_slots = 2 * kPoolSize = 8` bounds concurrent reads at 8 in `TestManager`. | **High** (measured) | `SickPeerStarvesStagingSlotsForHealthyPeer` asserts `free_slots() == 0` after exactly 8 reads and that assertion passes. This is what makes §5.2's sizing constraint real. |
| A14 | A fix that bounds per-peer *threads* without bounding per-peer *staging slots* is incomplete. | **High** | Follows from §2.6: allocation happens in `StartRead` before scheduling, so a thread-level dispatcher never sees the rejected reads. Invalidate by moving staging allocation after admission. |
| A11 | A dribbling peer is a legitimate fault mode to defend against, not only an attack. | **Medium-High** | TCP will deliver partial data whenever the sender's socket buffer drains slowly. A wedged-but-alive producer flushing a response header across a long stall reproduces it without malice. |
| A12 | The consumer's `push_pool_` and the producer's `pull_pool_` are separate instances even when one process acts as both P and D. | **High** | They are distinct members of the same `KVCacheManagerBase` (`kv_cache_manager_base.h:710-711`). But note that in a co-located P+D process, that process's *own* producer duties and consumer duties do contend (§2.5). |

---

## 5. Reproduction tests

Added to the **existing** test file so the `SilentProducer` / `TestManager`
harness is reused rather than duplicated:

**File:** `tpu_sync/core/kv_cache_manager_with_transfer_control_test.cc`
**Target:** `//tpu_sync/core:kv_cache_manager_with_transfer_control_test`
**Location:** appended after `ExpiredReceiveKeepsStagingUntilHandshakeEnds`, under
a banner comment marking them as the issue-#888 fault-isolation group.

These tests are written to **assert the desired behaviour**, so they fail on
`origin/main` and will pass once a fix lands. They are the acceptance criteria.

### 5.1 `SickPeerDoesNotDelayHandshakeToHealthyPeer` — the core repro

Directly implements the reproduction sketched in the issue.

```cpp
constexpr double kStarvationTimeoutS = 4.0;

TEST(ControlHandshakeTest, SickPeerDoesNotDelayHandshakeToHealthyPeer) {
  SilentProducer sick;
  SilentProducer healthy;
  TestManager consumer(/*timeout_s=*/kStarvationTimeoutS);

  // Saturate every handshake worker with reads aimed at the sick peer.
  for (int i = 0; i < kPoolSize; ++i) {
    consumer.StartRead(absl::StrCat("sick", i), /*uuid=*/300 + i,
                       sick.endpoint(), {0}, {0});
  }
  ASSERT_TRUE(sick.WaitUntilAccepted(kPoolSize, std::chrono::seconds(10)));

  const absl::Time start = absl::Now();
  consumer.StartRead("healthy0", /*uuid=*/400, healthy.endpoint(), {0}, {0});

  const bool connected_promptly =
      healthy.WaitUntilAccepted(1, std::chrono::milliseconds(500));
  EXPECT_TRUE(connected_promptly);
  EXPECT_LT(SecondsSince(start), kStarvationTimeoutS / 2);
  ...
}
```

Why `kStarvationTimeoutS = 4.0` rather than the file's `kTimeoutS = 0.5`: with
0.5 s the "blocked" and "prompt" outcomes are only ~0.5 s apart and the test
would be a coin flip under CI load. At 4.0 s the two outcomes differ by 8×.

Why both peers are `SilentProducer`: the assertion is on `accepted()`, i.e. on
*when the consumer's worker got to run at all*. Making the healthy peer answer
correctly would add handshake-semantics variance without strengthening the
signal (A5).

### 5.2 `HealthyPeerProgressesWhileSickPeerBacklogDrains` — the production shape

The same coupling stated as throughput rather than latency, with a backlog of
sick reads deeper than the pool so the queue stays saturated — the issue's "as
long as requests to the dead peer keep arriving" condition. Three healthy reads
are issued behind the backlog; all three should start within 500 ms.

**Sizing matters here and cost one iteration to get right.** The first version
used `2 × kPoolSize = 8` sick reads plus 3 healthy ones, which is 11 reads
against `TestManager`'s `num_slots = 2 * kPoolSize = 8`. The three healthy reads
never reached the pool at all — they failed staging allocation (§2.6) and were
dropped. The test still failed, but for the wrong reason, reporting "0 of 3
healthy handshakes started" rather than measuring queue delay. It is now sized
to `kSickReads = 5`, `kHealthyReads = 3` (total 8, exactly `num_slots`), with
`static_assert`s pinning both bounds and an `ASSERT_TRUE(consumer.has_recv(...))`
precondition per healthy read so the test can never silently degenerate into the
slot-exhaustion case again. The slot exhaustion is now tested separately in §5.3.

### 5.3 `SickPeerStarvesStagingSlotsForHealthyPeer` — finding §2.6

Fills `num_slots` with reads to the sick peer, asserts the precondition
`free_slots() == 0`, then issues one read to the healthy peer and asserts it
produces a live session rather than being rejected. This is the finding that
emerged from debugging §5.2's sizing and is arguably the most severe of the
four: the healthy read does not wait, it fails.

### 5.4 `HandshakeDeadlineBoundsWholeHandshakeNotEachRecv` — finding §2.4

Adds a `DribblingProducer` helper that accepts, reads the request, then emits the
24-byte response header **one byte every `kTimeoutS / 2`**, so no individual
`recv()` ever trips `SO_RCVTIMEO`. Asserts the handshake is abandoned within
`3 × kTimeoutS`. On `origin/main` it runs to ~`24 × (kTimeoutS/2)` instead.

The failure message reports how many header bytes the peer had managed to send,
which makes the per-syscall-vs-total distinction obvious in CI output.

### 5.5 How to run

The repo pins Bazel 8.6.0 (`.bazelversion`). **The system `/usr/bin/bazel` is
9.2.0 and prints a version error while still exiting 0** — do not trust its exit
code. Use the bootstrap binary that `build.sh:86-92` provisions:

The repo's canonical flags matter — building without them produces a different
configuration and a full XLA rebuild (~1 h). Use:

```bash
/tmp/bazel-bootstrap-8.6.0 test \
  --experimental_repo_remote_exec --nocheck_visibility \
  --override_module=torch_tpu=shims/torch_tpu --define with_jax=false \
  //tpu_sync/core:kv_cache_manager_with_transfer_control_test \
  --test_output=all --test_timeout=600 \
  --test_filter='ControlHandshakeTest.SickPeer*:ControlHandshakeTest.HealthyPeerProgressesWhileSickPeerBacklogDrains:ControlHandshakeTest.HandshakeDeadlineBoundsWholeHandshakeNotEachRecv'
```

If `/tmp/bazel-bootstrap-8.6.0` is absent:

```bash
curl -fsSL -o /tmp/bazel-bootstrap-8.6.0 \
  https://storage.googleapis.com/bazel/8.6.0/release/bazel-8.6.0-linux-x86_64
chmod +x /tmp/bazel-bootstrap-8.6.0
```

**BUILD note (measured, not estimated).** The target is `size = "small"`, i.e. a
60 s budget. The full suite now runs **79.3 s**, of which the four new tests are
**14.2 s**; the pre-existing suite was already ~65 s, dominated by
`MidRequestDisconnectDoesNotRaiseSigpipe` at **62.3 s**. So this target is
already over its declared size before any of this work, and the new tests widen
the gap. A fix CL should bump it to `size = "medium"` in
`tpu_sync/core/BUILD:1023-1043`. The runs above used an explicit
`--test_timeout` to sidestep this.

### 5.6 Current status — all four fail on `origin/main`, as intended

Full suite: **20 pre-existing tests pass, 4 new tests fail.** No existing test
was modified or broken.

| Test | Result | Measured evidence |
|---|---|---|
| `SickPeerDoesNotDelayHandshakeToHealthyPeer` | **FAIL** (4.1 s) | `reaching the healthy peer took 4.054s, against a sick-peer handshake timeout of 4s` |
| `HealthyPeerProgressesWhileSickPeerBacklogDrains` | **FAIL** (4.1 s) | `draining 3 healthy handshakes took 4.040s while 5 reads to an unresponsive peer were outstanding` |
| `SickPeerStarvesStagingSlotsForHealthyPeer` | **FAIL** (48 ms) | `has_recv(900)` is `false`; `failed_recving` = `{ "healthy0" }` |
| `HandshakeDeadlineBoundsWholeHandshakeNotEachRecv` | **FAIL** (6.0 s) | `handshake still held a worker after 1.504s with timeout_s=0.5; the peer had sent only 7 of 24 header bytes, each inside SO_RCVTIMEO` |

The first row is the clearest single piece of evidence in this investigation:
**time-to-contact the healthy peer is 4.054 s against a sick-peer timeout of
4.0 s.** The delay does not merely correlate with the sick peer's timeout, it
equals it. That is head-of-line blocking, not jitter or contention.

The second row confirms it is not an artifact of the pool being exactly
saturated: with a backlog deeper than the pool, healthy traffic still waits one
full timeout, and the per-read `has_recv` preconditions prove those reads were
genuinely queued rather than dropped.

The third row shows the failure is not always a delay — past the staging limit
it is an outright rejection in 48 ms.

The fourth row shows the 4 s / 120 s figure is itself a floor, not a ceiling
(§2.4).

---

## 6. Fix menu

Ordered by structural depth, not by preference. §7 gives the recommendation.

### Fix A — Per-peer fair dispatch (admission control + round-robin)

Insert a dispatcher between callers and the pool that (a) caps how many workers
one peer may hold and (b) round-robins across peers with pending work.

```cpp
// Sketch. Wraps a NumaThreadPool; does not add threads.
class PeerFairDispatcher {
 public:
  PeerFairDispatcher(std::shared_ptr<NumaThreadPool> pool, size_t max_per_peer);

  // Runs `task` on the pool, but never lets `peer` hold more than
  // max_per_peer workers at once. Excess work waits in that peer's own
  // queue; queues are drained round-robin, so a saturated peer cannot
  // push another peer's work back in line.
  void Schedule(absl::string_view peer, std::optional<int> numa_node,
                std::function<void()> task);

 private:
  struct PeerQueue {
    std::deque<std::function<void()>> pending;
    size_t in_flight = 0;
  };
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, PeerQueue> peers_ ABSL_GUARDED_BY(mu_);
  std::deque<std::string> round_robin_ ABSL_GUARDED_BY(mu_);
  std::shared_ptr<NumaThreadPool> pool_;
  const size_t max_per_peer_;
};
```

Call site change at `transfer_receive_session.cc:465`:

```cpp
-  base_->push_pool()->Schedule(target_node, [ ... ]);
+  base_->handshake_dispatcher()->Schedule(remote_endpoint, target_node, [ ... ]);
```

Sizing: with `kPoolSize = 4`, `max_per_peer = 2` keeps two workers free for
other peers while still allowing pipelining to a healthy peer. `max_per_peer =
kPoolSize - 1` is the weakest useful setting. This should be a named constant
with a comment tying it to `kPoolSize`, not a bare literal.

- **Pros:** targets the actual defect (isolation) without adding threads. Bounded
  memory. Works for any fan-in. Degrades to current behaviour with one peer.
- **Cons:** an unbounded per-peer `deque` is a memory-growth risk if requests to a
  dead peer keep arriving — pair it with a queue cap that fast-fails (see Fix F).
  Does not help the **producer inbound** side, where the peer is not known until
  the request header has been read; there the key must be the socket's peer
  address (`getpeername`), which is a different and slightly weaker key.
- **Slot reservation — required, not optional.** A thread cap alone does **not**
  fix §2.6. `StartRead` allocates staging *before* it schedules, so a sick peer
  still exhausts `num_slots` and healthy reads are rejected before any dispatcher
  sees them. Fix A must therefore also bound **staging slots per peer**, e.g. by
  moving the `AllocateStagingForLoad` call behind the same admission decision, or
  by reserving a floor of slots that no single peer may consume:

  ```cpp
  // In StartRead, before TransferReceiveSession::Create:
  //   reject early if this peer already holds its share of the staging pool,
  //   so a wedged peer cannot deny slots to peers that are answering.
  if (!staging_admission_.TryAcquire(remote_endpoint)) {
    failed_recving_.insert(req_id);   // fast, explicit, per-peer
    return;
  }
  ```

  `SickPeerStarvesStagingSlotsForHealthyPeer` (§5.3) is the acceptance test for
  this half and will still fail if only the thread cap is implemented.
- **Note:** untracked `tpu_sync/core/peer_fair_dispatcher.{h,cc,_test.cc}` in the
  working tree already prototype something along these lines (A9). Not evaluated
  in this run. Whatever run 2 decides about them, check whether they address the
  staging dimension — a pure thread dispatcher will not.

### Fix B — Give the control plane its own pool

Split handshakes off `push_pool_`/`pull_pool_` onto a dedicated control pool.

- **Pros:** one-line-ish; removes the control↔data coupling in §2.5, which is a
  real and separate problem.
- **Cons:** **does not fix the reported bug.** Four stuck handshakes still starve
  a dedicated control pool of four. Necessary hygiene, not sufficient. Should be
  folded in alongside Fix A rather than offered as an alternative to it.

### Fix C — A separate, short control-plane deadline

Stop letting a 184-byte handshake inherit the bulk-transfer timeout.

```cpp
// kv_cache_manager_with_transfer.h
double timeout_s_ = 120.0;          // bulk transfer
double control_timeout_s_ = 5.0;    // NEW: control handshake
```

```cpp
// transfer_receive_session.cc:484
manager.control_backend_->SendPullRequest(
    remote_endpoint, req_spec, absl::Seconds(manager.control_timeout_s_));
```

- **Pros:** shrinks the blast radius ~24× on its own; small and independently
  reviewable; low risk.
- **Cons:** magnitude only, not isolation — a sick peer still takes the whole pool,
  just for 5 s instead of 120 s. Must be combined with Fix D or the per-syscall
  loophole (§2.4) reopens it. Needs care: `PullAheadOfRegistrationIsAcknowledged`
  (`control_test.cc:300`) shows the producer legitimately holds a pull open until
  registration arrives, so 5 s must be validated against real registration skew.

### Fix D — Make the deadline a total budget

Thread an absolute deadline through `SendPullRequest` and re-arm `SO_RCVTIMEO`
with the *remaining* budget before each syscall, or drive the socket with
`poll()` against the deadline.

```cpp
// Sketch for tcp_control_plane_backend.cc
absl::Status ReadExactBy(int fd, void* buf, size_t len, absl::Time deadline) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  while (len > 0) {
    const absl::Duration left = deadline - absl::Now();
    if (left <= absl::ZeroDuration()) {
      return absl::DeadlineExceededError("control read exceeded total deadline");
    }
    ABSL_RETURN_IF_ERROR(SetSocketTimeouts(fd, absl::ToDoubleSeconds(left)));
    ssize_t n = read(fd, p, len);
    ...
  }
  return absl::OkStatus();
}
```

- **Pros:** closes §2.4 outright; makes every other bound in the system honest.
  Also fixes the mirror case in `HandleControlConnection`.
- **Cons:** touches `ReadExact`/`WriteExact`, which are used by several call sites
  (`SendPullRequest`, `SendAck`, `HandleControlConnection`, `SendErrorResponse`);
  each needs a deadline plumbed or an explicit opt-out. An extra `setsockopt` per
  iteration is negligible next to a syscall that may block for seconds.

### Fix E — Non-blocking handshake on an event loop

Run all control I/O on one `epoll` thread so no pool worker is ever held.

- **Pros:** eliminates the entire bug class; no pool, no fairness policy, no caps.
- **Cons:** substantial refactor of `TcpControlPlaneBackend` and every caller that
  currently assumes a synchronous `absl::StatusOr` return. Not proportionate as a
  fix for #888, but worth recording as the end state if the control plane is ever
  reworked.

### Fix F — Peer circuit breaker

Track consecutive handshake failures per peer; when a peer trips the threshold,
fast-fail new reads to it (returning `Unavailable` immediately) and probe
half-open on a timer.

- **Pros:** the only option that addresses "requests to the dead peer keep
  arriving". Prevents the backlog from forming rather than sharing it fairly.
  Turns a 120 s stall into an immediate, actionable error for the scheduler.
- **Cons:** needs hysteresis and half-open probing or it will flap; a false-positive
  ejection takes a healthy producer out of rotation. Requires a policy decision
  about what the caller should do with a fast failure (§8 Q4). Should land after
  Fix A, not instead of it.

---

## 7. Recommendation

Three CLs, in this order, each independently reviewable and each with the tests
from §5 as acceptance criteria:

1. **CL1 — Fix D + Fix C.** Make the handshake deadline a total budget and give it
   its own short value. Small, low-risk, and turns an unbounded stall into a
   bounded one. Makes `HandshakeDeadlineBoundsWholeHandshakeNotEachRecv` pass.
2. **CL2 — Fix A (+ Fix B folded in), covering both threads and staging slots.**
   Per-peer fair dispatch on the consumer outbound handshake, on a pool that is
   no longer shared with the data path, *plus* per-peer staging admission per
   §2.6. Makes the other three tests pass. **This is the CL that actually fixes
   #888.** Note it is not done until
   `SickPeerStarvesStagingSlotsForHealthyPeer` passes — a thread-only fix leaves
   the worse failure mode in place.
3. **CL3 — Fix F.** Circuit breaker, as defence in depth against sustained traffic
   to a dead peer.

Fix E is recorded as a direction, not proposed for this issue.

Rationale for the ordering: CL1 is cheap and bounds the damage even if CL2 slips.
CL2 is the structural fix but is the largest review surface. CL3 is a policy
change that wants the other two in place first so its thresholds can be tuned
against a system that already fails predictably.

Deliberately **not** recommended: raising `kPoolSize`. It moves the threshold
from 4 concurrent sick handshakes to N, changes no asymptotics, and costs threads
on every node. It is the fix that makes the bug harder to reproduce without
making it less likely in production, where fan-in grows with the fleet.

---

## 8. Open questions for the reviewer

- **Q1 — gRPC backend.** `ResolveControlPlaneBackendType()` can select
  `GrpcControlPlaneBackend` (`grpc_control_plane_backend.cc:277`). Which backend
  do production deployments use? The pool starvation (§2.1–2.3) applies to both,
  but the per-syscall timeout finding (§2.4) is specific to the TCP backend. If
  gRPC is the production path, run 2 should re-verify §2.4 against it.
- **Q2 — Upstream rate limiting (A7).** Is there a scheduler above `StartRead`
  that already bounds in-flight reads per producer? If so, the practical exposure
  is smaller than §5.2 implies and Fix A's `max_per_peer` should be chosen to
  match it.
- **Q3 — The untracked `peer_fair_dispatcher` prototype (A9).** Those three files
  were already in the working tree when this session started and are not on
  `origin/main`. Should run 2 adopt them as the basis for Fix A, or start fresh?
  They have not been built or reviewed here.
- **Q4 — Fast-fail semantics for Fix F.** When the circuit breaker rejects a read,
  what should the caller do — fail the request, re-route to another producer, or
  queue it? This is a scheduler-policy question outside this library.
- **Q5 — `timeout_s` in production (A2).** Confirm the deployed value. Everything
  in §2.4's arithmetic scales linearly with it.
- **Q6 — Test size.** Are ~30 s tests acceptable in this target, or should the
  fault-isolation group move to its own `size = "medium"` target?

---

## 9. Changelog and handoff to run 2

### What run 1 did

- Branched `investigate-888-peer-starvation` from `origin/main` @ `868ae48`.
- Verified every mechanical claim in #888 against current code; recorded five
  corrections (§3) since the issue's line numbers predate a refactor.
- Found three amplifications not in the issue: per-syscall deadline (§2.4),
  control/data pool sharing (§2.5), and staging-slot exhaustion causing outright
  rejection of healthy reads (§2.6).
- Validated the task brief's "Current Knowledge" (§3, second table) — the 4-thread
  handshake claim is right, the "16 threads for data transfer" claim needs
  qualification.
- Added **four** failing tests (§5) and confirmed all 20 pre-existing tests in the
  target still pass.
- Proposed six fixes and a three-CL plan (§6, §7).
- **Implemented no fix.**

### Things that cost time, recorded so run 2 does not repeat them

- `/usr/bin/bazel` is 9.2.0, the repo pins 8.6.0, and **the wrapper prints a
  version error while exiting 0**. A build that "succeeded" may not have run.
  Always use `/tmp/bazel-bootstrap-8.6.0` and check for real output.
- Build without the canonical flags (§5.5) and you get a second configuration and
  a ~1 h XLA rebuild. A stale build from an earlier session was also holding the
  Bazel server lock and had to be killed.
- Test sizing against `num_slots` is easy to get wrong and produces a test that
  fails for the wrong reason (§5.2).

### What run 2 should pick up

1. Resolve Q1–Q6 with the reviewer; amend §4 accordingly.
2. Write the missing test for the *data-path* starvation vector — the second half
   of the brief's "Current Knowledge". Shape: a peer that is slow during H2H chunk
   pulls occupying `pull_pool_` (`kv_cache_manager_base.cc:968`) and thereby
   delaying inbound **control** handling for other peers. This is §2.5's coupling
   and is currently unproven.
3. Consider a test for the producer-side mirror across peers — the cross-peer
   version of `HandlersOutliveConsumersThatNeverSpeak`, keyed on `getpeername`.
4. Decide on the `peer_fair_dispatcher` prototype (Q3).
5. Only then implement CL1.

### Files touched on this branch

| File | Change |
|---|---|
| `tpu_sync/core/kv_cache_manager_with_transfer_control_test.cc` | +4 tests, +1 `DribblingProducer` helper, +1 banner comment. No existing test modified. |
| `issue_888_starvation/run1.md` | This document. |

Untracked and **not** part of this work (pre-existing in the working tree, A9):
`tpu_sync/core/peer_fair_dispatcher.cc`, `.h`, `_test.cc`.
