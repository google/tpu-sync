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
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"

namespace raiden {
namespace {

constexpr int64_t kPieceAlignment = 4096;

// Returns the byte count in environment variable `name`, or `default_bytes`
// when the variable is unset or empty.
absl::StatusOr<int64_t> BytesFromEnv(const char* name, int64_t default_bytes) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_bytes;
  }
  int64_t parsed = 0;
  if (!absl::SimpleAtoi(value, &parsed)) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, "=\"", value, "\" is not an integer byte count"));
  }
  return parsed;
}

}  // namespace

// Completion state shared by the pieces of one Submit() call. `issue` is
// immutable once the job is queued; every other field is guarded by the
// owning pacer's mutex.
struct RawDmaPacer::Job {
  std::vector<RawDmaIssueFn> issue;
  int64_t remaining = 0;
  absl::Status status;
  std::optional<xla::Promise<>> promise;
  std::shared_ptr<void> keep_alive;
};

absl::StatusOr<RawDmaPacer*> RawDmaPacer::Global() {
  // Leaked on purpose: static destructors may still await raw copies. Built
  // once, so an invalid environment fails every call with the same error.
  static const absl::NoDestructor<absl::StatusOr<RawDmaPacer*>> pacer(
      []() -> absl::StatusOr<RawDmaPacer*> {
        ABSL_ASSIGN_OR_RETURN(RawDmaPacerOptions options, OptionsFromEnv());
        return new RawDmaPacer(options);
      }());
  return *pacer;
}

absl::StatusOr<RawDmaPacerOptions> RawDmaPacer::OptionsFromEnv() {
  RawDmaPacerOptions options;
  ABSL_ASSIGN_OR_RETURN(options.max_inflight_bytes,
                        BytesFromEnv("TPU_RAIDEN_RAW_DMA_MAX_INFLIGHT_BYTES",
                                     options.max_inflight_bytes));
  ABSL_ASSIGN_OR_RETURN(
      options.chunk_bytes,
      BytesFromEnv("TPU_RAIDEN_RAW_DMA_CHUNK_BYTES", options.chunk_bytes));
  return options;
}

RawDmaPacerOptions RawDmaPacer::Normalize(RawDmaPacerOptions options) {
  int64_t chunk = options.chunk_bytes > 0 ? options.chunk_bytes
                                          : RawDmaPacerOptions{}.chunk_bytes;
  chunk = std::max(kPieceAlignment, chunk / kPieceAlignment * kPieceAlignment);
  options.chunk_bytes = chunk;
  return options;
}

RawDmaPacer::RawDmaPacer(RawDmaPacerOptions options)
    : options_(Normalize(options)) {
  worker_ = std::thread([this] { WorkerLoop(); });
}

RawDmaPacer::~RawDmaPacer() {
  // Pieces never issued are failed; pieces already issued are waited for, so
  // no completion callback can run against a destroyed pacer.
  std::vector<std::pair<xla::Promise<>, absl::Status>> to_resolve;
  std::vector<std::shared_ptr<void>> to_release;
  {
    absl::MutexLock lock(mu_);
    shutdown_ = true;
    for (Piece& piece : pending_) {
      Job& job = *piece.job;
      if (job.status.ok()) {
        job.status = absl::CancelledError("RawDmaPacer destroyed");
      }
      if (--job.remaining == 0 && job.promise.has_value()) {
        to_resolve.emplace_back(std::move(*job.promise), job.status);
        job.promise.reset();
        to_release.push_back(std::move(job.keep_alive));
      }
    }
    pending_.clear();
  }
  worker_.join();
  for (auto& [promise, status] : to_resolve) {
    promise.Set(status);
  }
  absl::MutexLock lock(mu_);
  mu_.Await(absl::Condition(
      +[](int64_t* inflight) { return *inflight == 0; }, &inflight_bytes_));
}

xla::Future<> RawDmaPacer::Submit(std::vector<RawDmaCopy> copies,
                                  std::shared_ptr<void> keep_alive) {
  auto [promise, future] = xla::MakePromise();
  auto job = std::make_shared<Job>();
  job->issue.reserve(copies.size());
  std::vector<Piece> batch;
  int64_t pieces = 0;
  {
    absl::MutexLock lock(mu_);
    const bool paced = options_.max_inflight_bytes > 0;
    for (size_t i = 0; i < copies.size(); ++i) {
      const int64_t size = copies[i].size_bytes;
      job->issue.push_back(std::move(copies[i].issue));
      if (size <= 0) {
        continue;
      }
      // Unpaced, a copy stays one request, exactly as before the pacer.
      const int64_t step = paced ? options_.chunk_bytes : size;
      for (int64_t offset = 0; offset < size; offset += step) {
        pending_.push_back(
            Piece{job, i, offset, std::min(step, size - offset)});
        ++pieces;
      }
    }
    job->remaining = pieces;
    if (pieces > 0) {
      job->promise.emplace(std::move(promise));
      job->keep_alive = std::move(keep_alive);
      TakeIssuableLocked(batch);
    }
  }
  if (pieces == 0) {
    promise.Set();
    return std::move(future);
  }
  Issue(batch);
  return std::move(future);
}

RawDmaPacerOptions RawDmaPacer::options() const {
  absl::MutexLock lock(mu_);
  return options_;
}

void RawDmaPacer::SetOptions(RawDmaPacerOptions options) {
  absl::MutexLock lock(mu_);
  options_ = Normalize(options);
}

RawDmaPacerStats RawDmaPacer::stats() const {
  absl::MutexLock lock(mu_);
  return RawDmaPacerStats{inflight_bytes_, peak_inflight_bytes_,
                          static_cast<int64_t>(pending_.size()),
                          issued_pieces_};
}

void RawDmaPacer::ResetPeak() {
  absl::MutexLock lock(mu_);
  peak_inflight_bytes_ = inflight_bytes_;
}

bool RawDmaPacer::CanIssueFrontLocked() const {
  if (pending_.empty()) {
    return false;
  }
  if (options_.max_inflight_bytes <= 0 || inflight_bytes_ == 0) {
    return true;
  }
  return inflight_bytes_ + pending_.front().size <= options_.max_inflight_bytes;
}

bool RawDmaPacer::WorkerShouldWakeLocked() const {
  return shutdown_ || CanIssueFrontLocked();
}

void RawDmaPacer::TakeIssuableLocked(std::vector<Piece>& out) {
  while (CanIssueFrontLocked()) {
    Piece piece = std::move(pending_.front());
    pending_.pop_front();
    inflight_bytes_ += piece.size;
    peak_inflight_bytes_ = std::max(peak_inflight_bytes_, inflight_bytes_);
    ++issued_pieces_;
    out.push_back(std::move(piece));
  }
}

void RawDmaPacer::Issue(std::vector<Piece>& pieces) {
  for (Piece& piece : pieces) {
    const RawDmaIssueFn& issue = piece.job->issue[piece.copy_index];
    xla::Future<> done = issue
                             ? issue(piece.offset, piece.size)
                             : xla::Future<>(absl::InvalidArgumentError(
                                   "RawDmaPacer: copy has no issue function"));
    if (!done.IsValid()) {
      OnPieceDone(piece.job, piece.size,
                  absl::InternalError(absl::StrCat(
                      "RawDmaPacer: issue function returned an invalid future "
                      "for bytes [",
                      piece.offset, ", ", piece.offset + piece.size, ")")));
      continue;
    }
    done.OnReady([this, job = std::move(piece.job), size = piece.size](
                     absl::Status status) { OnPieceDone(job, size, status); });
  }
  pieces.clear();
}

void RawDmaPacer::OnPieceDone(const std::shared_ptr<Job>& job, int64_t size,
                              const absl::Status& status) {
  // Runs on PJRT completion threads: only bookkeeping here. Freed window space
  // wakes the worker thread (via the mutex condition), which issues the next
  // pieces; issuing from inside a PJRT callback could re-enter the runtime.
  std::optional<xla::Promise<>> promise;
  absl::Status final_status;
  std::shared_ptr<void> keep_alive;
  {
    absl::MutexLock lock(mu_);
    inflight_bytes_ -= size;
    if (!status.ok() && job->status.ok()) {
      job->status = status;
    }
    if (--job->remaining == 0 && job->promise.has_value()) {
      promise = std::move(job->promise);
      job->promise.reset();
      final_status = job->status;
      keep_alive = std::move(job->keep_alive);
    }
  }
  if (promise.has_value()) {
    if (final_status.ok()) {
      promise->Set();
    } else {
      promise->Set(final_status);
    }
  }
}

void RawDmaPacer::WorkerLoop() {
  std::vector<Piece> batch;
  while (true) {
    {
      absl::MutexLock lock(mu_);
      mu_.Await(absl::Condition(this, &RawDmaPacer::WorkerShouldWakeLocked));
      if (shutdown_) {
        return;
      }
      TakeIssuableLocked(batch);
    }
    Issue(batch);
  }
}

}  // namespace raiden
