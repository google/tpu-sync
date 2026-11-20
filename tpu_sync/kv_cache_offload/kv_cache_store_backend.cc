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

#include "tpu_sync/kv_cache_offload/kv_cache_store_backend.h"

namespace tpu_raiden {
namespace kv_cache {

LookupOptions LookupOptions::HostResidentSubset() {
  LookupOptions opts;
  opts.enable_global = false;
  opts.enable_interleaved_lookup = false;
  opts.pin_found = false;
  opts.match_mode = LookupMatchMode::kSubset;
  opts.host_resident_only = true;
  opts.include_eviction_candidates = true;
  return opts;
}

LookupOptions LookupOptions::SnapshotHostResident() {
  LookupOptions opts;
  opts.enable_global = false;
  opts.enable_interleaved_lookup = false;
  opts.pin_found = false;
  opts.match_mode = LookupMatchMode::kSubset;
  opts.host_resident_only = true;
  opts.include_eviction_candidates = false;
  return opts;
}

LookupOptions LookupOptions::PinPresentHostResident() {
  LookupOptions opts;
  opts.enable_global = false;
  opts.enable_interleaved_lookup = false;
  opts.pin_found = true;
  opts.match_mode = LookupMatchMode::kSubset;
  opts.host_resident_only = true;
  opts.include_eviction_candidates = false;
  return opts;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
