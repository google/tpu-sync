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

// The JAX half of the Mooncake TPU buffer provider, the _mooncake_provider
// extension. Mooncake dlopen()s this extension (the path library_path() in
// tpu_sync/solutions/mooncake/jax/mooncake_provider.py returns) and calls the
// mc_tpu_buffer_provider_get defined here, which serves the table of the same
// provider core that exports the handles below.

#include <Python.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>  // IWYU pragma: keep
#include <nanobind/stl/vector.h>  // IWYU pragma: keep
#include "tpu_sync/frameworks/jax/jax_compat.h"
#include "tpu_sync/solutions/mooncake/mooncake_provider.h"

namespace nb = nanobind;

namespace raiden::mooncake {
namespace {

// `shards` holds (shard.data, device_index) pairs. Every buffer is acquired
// before any handle is created, so a shard that fails leaves nothing behind:
// the buffers already acquired are dropped as the exception unwinds.
std::vector<uint64_t> ExportShardBuffers(
    const std::vector<std::pair<nb::object, int>>& shards) {
  std::vector<ExportRequest> requests;
  requests.reserve(shards.size());
  for (const auto& [data, device_index] : shards) {
    // No usage hold: a handle may live as long as the process, and its raw
    // buffer reference alone keeps the HBM allocated (mooncake_provider.h).
    requests.push_back(
        {AcquireShardBuffer(data.ptr(), /*unsafe_skip_buffer_lock=*/true),
         device_index});
  }
  absl::StatusOr<std::vector<BufferHandle>> handles =
      ExportBuffers(std::move(requests));
  if (!handles.ok()) {
    throw std::runtime_error(std::string(handles.status().message()));
  }
  return *std::move(handles);
}

}  // namespace
}  // namespace raiden::mooncake

// Defined here, in the extension's own sources; see mooncake_provider.h.
extern "C" int mc_tpu_buffer_provider_get(
    raiden::mooncake::ProviderTable* out) {
  return raiden::mooncake::GetProviderTable(out);
}

NB_MODULE(_mooncake_provider, m) {
  m.def("export_buffers", &raiden::mooncake::ExportShardBuffers,
        nb::arg("shards"),
        "Exports one Mooncake provider handle per (shard.data, device_index) "
        "pair, in order. All-or-nothing. The caller owns the handles.");
  m.def("live_buffer_count", &raiden::mooncake::LiveBufferCount,
        "Number of exported handles not yet released.");
  m.def("live_dma_mapping_count", &raiden::mooncake::LiveDmaMappingCount,
        "Number of host ranges mapped by the table's dma_map and not yet "
        "unmapped.");
  m.def("mark_process_exiting", &raiden::mooncake::MarkProcessExiting,
        "Declares that interpreter exit has begun: from now on dma_map fails "
        "and dma_unmap skips PJRT, whose client JAX's atexit handler may "
        "destroy. mooncake_provider registers it with atexit.");
}
