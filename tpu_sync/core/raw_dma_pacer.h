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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_RAW_DMA_PACER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_RAW_DMA_PACER_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"

namespace raiden {

// Bounds how much raw DMA a process keeps queued on its TPU.
//
// Why: the TPU runtime runs a TensorCore's DMAs and program launches through
// one in-order queue, and a PJRT raw-buffer copy is handed to it as a single
// request for the whole range. Queueing N bytes of raw DMA therefore holds
// every program launched after it until those N bytes have crossed the host
// link (about 19 ms per GiB on TPU7x), and the runtime's completion reads
// wait behind the same queue, so the few launches libtpu admits at a time
// drain and the TensorCore idles ("kernel bubbles"). DMA on the sibling core
// of the same chip delays it too. Tensors copied with `.to()`
// (BufferFromHostBuffer) are also queued as one whole-buffer DMA each and
// delay kernels the same way.
//
// RawDmaPacer covers only tpu_raiden's torch raw transfers, not `.to()`: it
// splits each copy into pieces of at most `chunk_bytes` and keeps at most
// `max_inflight_bytes` of them outstanding per process, first-in first-out
// across callers, so a launch queued behind raw DMA waits for at most one
// window of data.
struct RawDmaPacerOptions {
  // Upper bound on outstanding raw DMA bytes in this process. A value <= 0
  // disables pacing: every copy is issued at once as one request (the
  // pre-pacer behaviour). A single piece is always admitted when nothing is
  // in flight, so progress never depends on the limit.
  int64_t max_inflight_bytes = int64_t{64} << 20;
  // Copies larger than this are split into pieces of at most this size.
  // Normalized to a positive multiple of 4 KiB so every piece of a 4 KiB
  // aligned copy stays 4 KiB aligned.
  int64_t chunk_bytes = int64_t{16} << 20;
};

// Issues one piece covering bytes [offset, offset + size) of a copy (offsets
// relative to the copy's start) and returns its completion future.
using RawDmaIssueFn =
    std::function<xla::Future<>(int64_t offset, int64_t size)>;

struct RawDmaCopy {
  RawDmaIssueFn issue;
  int64_t size_bytes = 0;
};

struct RawDmaPacerStats {
  int64_t inflight_bytes = 0;
  int64_t peak_inflight_bytes = 0;
  int64_t pending_pieces = 0;
  int64_t issued_pieces = 0;
};

class RawDmaPacer {
 public:
  // The process-wide pacer used by the raw transfer APIs, created on first use
  // from OptionsFromEnv() and never destroyed. The environment is read once:
  // if either variable is set to a value that is not an integer, every call
  // returns the same InvalidArgumentError, and the raw transfers and pacing
  // calls that use this pacer raise.
  static absl::StatusOr<RawDmaPacer*> Global();

  // Reads the options from TPU_RAIDEN_RAW_DMA_MAX_INFLIGHT_BYTES and
  // TPU_RAIDEN_RAW_DMA_CHUNK_BYTES (bytes). An unset or empty variable keeps
  // the default above; a value absl::SimpleAtoi cannot parse as an int64_t
  // (e.g. "16M", "1.5") is an InvalidArgumentError naming the variable.
  // Parsed values are returned as is and mean what they mean in SetOptions():
  // a limit <= 0 disables pacing and chunk_bytes is normalized.
  static absl::StatusOr<RawDmaPacerOptions> OptionsFromEnv();

  explicit RawDmaPacer(RawDmaPacerOptions options = {});
  ~RawDmaPacer();

  RawDmaPacer(const RawDmaPacer&) = delete;
  RawDmaPacer& operator=(const RawDmaPacer&) = delete;

  // Queues `copies` behind everything submitted earlier and returns a future
  // that completes once every piece has completed; the first error wins, but
  // the future only resolves after all issued pieces have finished, so no DMA
  // outlives it. `keep_alive` (buffers, host memory) is released only then,
  // even if the caller drops the returned future. Pieces that fit the window
  // are issued on the calling thread; the rest are issued by the pacer thread
  // as earlier pieces complete.
  xla::Future<> Submit(std::vector<RawDmaCopy> copies,
                       std::shared_ptr<void> keep_alive = nullptr);

  RawDmaPacerOptions options() const;
  // Applies to pieces issued from now on; queued pieces keep their split.
  void SetOptions(RawDmaPacerOptions options);

  RawDmaPacerStats stats() const;
  void ResetPeak();

  // Normalizes chunk_bytes as documented on RawDmaPacerOptions.
  static RawDmaPacerOptions Normalize(RawDmaPacerOptions options);

 private:
  struct Job;
  struct Piece {
    std::shared_ptr<Job> job;
    size_t copy_index = 0;
    int64_t offset = 0;
    int64_t size = 0;
  };

  // True when the front pending piece may be issued now.
  bool CanIssueFrontLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool WorkerShouldWakeLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Moves every pending piece that fits the window into `out` and accounts
  // for it as in flight.
  void TakeIssuableLocked(std::vector<Piece>& out)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Issues pieces (never under mu_: a future may complete synchronously).
  void Issue(std::vector<Piece>& pieces);
  void OnPieceDone(const std::shared_ptr<Job>& job, int64_t size,
                   const absl::Status& status);
  void WorkerLoop();

  mutable absl::Mutex mu_;
  RawDmaPacerOptions options_ ABSL_GUARDED_BY(mu_);
  std::deque<Piece> pending_ ABSL_GUARDED_BY(mu_);
  int64_t inflight_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t peak_inflight_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t issued_pieces_ ABSL_GUARDED_BY(mu_) = 0;
  bool shutdown_ ABSL_GUARDED_BY(mu_) = false;
  std::thread worker_;
};

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_RAW_DMA_PACER_H_
