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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_SESSION_H_

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/transport/transfer_session.h"  // IWYU pragma: export

namespace tpu_raiden {

namespace kv_cache {
class KVCacheManagerBase;
}  // namespace kv_cache

// Manages the host layer staging slots of a transfer session when fewer host
// layer buffers are allocated than layers. A layer leases the slot at the head
// of the free queue when its host staging starts (first bytes arriving on a
// receiver, or D2H dispatch on a sender), keeps it until its host work
// completes (H2D copy on a receiver, or H2H push on a sender), and Release()
// returns the slot to the tail; Acquire() blocks while every slot is leased.
// Layers may acquire and complete in any order.
// Thread-safe.
class StagingLayerManager {
 public:
  // Initializes the manager with free slots [0, |num_slots|).
  explicit StagingLayerManager(size_t num_slots = 0);

  // Returns the slot leased to |unit|, leasing the head free slot if |unit|
  // has none. Waits until |deadline| for a release when every slot is leased;
  // fails kDeadlineExceeded on timeout and kCancelled after Shutdown() or once
  // |*cancelled| is set.
  absl::StatusOr<size_t> Acquire(size_t unit,
                                 std::chrono::steady_clock::time_point deadline,
                                 const std::atomic<bool>* cancelled = nullptr);

  // Returns the slot leased to |unit|, if any.
  std::optional<size_t> Lookup(size_t unit) const;

  // Ends |unit|'s lease, returning its slot to the tail of the free queue and
  // waking one waiter. Returns the released slot; nullopt if |unit| held none.
  std::optional<size_t> Release(size_t unit);

  // Ends every lease.
  void ReleaseAll();

  // Fails every current and future Acquire() with kCancelled.
  void Shutdown();

  size_t num_free() const;

 private:
  mutable absl::Mutex mu_;
  // Free slots in release order: Acquire() pops the head, Release() pushes
  // the tail.
  std::deque<size_t> free_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<size_t, size_t> leases_ ABSL_GUARDED_BY(mu_);
  bool shutdown_ ABSL_GUARDED_BY(mu_) = false;
};

// Host layer staging of a pool-reshard session. A host layer buffer mirrors a
// storage, and several pools can live in one storage (e.g. the FA pool and
// the GDN state pools aliasing it), so pools resolve to buffers by storage:
//  - full mirror (one buffer per storage): a pool's buffer is its own
//    storage's, with no lease and no waiting;
//  - bounded host layers: the storage, not the pool, holds the lease, and
//    every pool of that storage shares it. The lease ends when the last pool
//    holding it releases.
// Leasing per pool instead hands two pools of one storage two buffers (so
// pools beyond the buffer count wait for a release, which can need the very
// pushes that are waiting) and places a pool outside its storage's buffer.
// Thread-safe.
class PoolStagingLayers {
 public:
  // |storage_of_pool[p]| is the storage index of pool p. |bounded| is true
  // when the manager has fewer host layer buffers (|num_host_layers|) than
  // storages.
  PoolStagingLayers(std::vector<size_t> storage_of_pool, bool bounded,
                    size_t num_host_layers);

  // Staging for a session on |base|'s registered pools (none for nullptr).
  static PoolStagingLayers ForManager(const kv_cache::KVCacheManagerBase* base);

  // Returns the buffer of |pool_idx|'s storage, leasing one for the storage
  // when bounded and the storage holds none. Waits until |deadline| for a
  // release when every buffer is leased; fails kDeadlineExceeded on timeout
  // and kCancelled after Shutdown() or once |*cancelled| is set. A pool
  // index past |storage_of_pool| is its own storage (implicit pools).
  absl::StatusOr<size_t> Acquire(size_t pool_idx,
                                 std::chrono::steady_clock::time_point deadline,
                                 const std::atomic<bool>* cancelled = nullptr);

  // Returns the buffer |pool_idx| holds, if any. A full mirror answers with
  // the pool's own storage only while the pool holds it (between Acquire and
  // Release), as a bounded one does.
  std::optional<size_t> Lookup(size_t pool_idx) const;

  // Ends |pool_idx|'s hold; the storage's lease ends with its last holder.
  // Returns the buffer the pool held; nullopt if it held none.
  std::optional<size_t> Release(size_t pool_idx);

  // Ends every hold and lease.
  void ReleaseAll();

  // Fails every future Acquire(), and every bounded one still waiting, with
  // kCancelled.
  void Shutdown();

 private:
  absl::StatusOr<size_t> StorageOf(size_t pool_idx) const;

  const std::vector<size_t> storage_of_pool_;
  const bool bounded_;
  StagingLayerManager storage_leases_;
  mutable absl::Mutex mu_;
  // Pools currently holding each storage's buffer.
  absl::flat_hash_map<size_t, absl::flat_hash_set<size_t>> holders_
      ABSL_GUARDED_BY(mu_);
  // Acquire() calls waiting on |storage_leases_| for each storage; the
  // storage's lease outlives an empty holder set while any is pending.
  absl::flat_hash_map<size_t, int> pending_ ABSL_GUARDED_BY(mu_);
  bool shutdown_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_SESSION_H_
