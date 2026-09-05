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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_UTILS_OSS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_UTILS_OSS_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef WITHOUT_PYTHON
#include <Python.h>

#include <nanobind/nanobind.h>
#include "tpu_sync/frameworks/jax/jax_compat.h"
#else
#include "tpu_sync/frameworks/jax/mock_nanobind.h"
#endif
#include "xla/pjrt/pjrt_client.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace nb = nanobind;

namespace jax {

#ifndef WITHOUT_PYTHON
// Forward to jax_compat layer which isolates jaxlib's private PyArray layout.
inline xla::PjRtBuffer* GetPjrtBufferFromPyObject(PyObject* obj) {
  return raiden::PjRtBufferFromPyArray(obj);
}

inline xla::ifrt::Array* GetIfrtArrayFromPyObject(PyObject* obj) {
  return raiden::IfrtArrayFromPyArray(obj);
}

// FFI helper to convert nanobind lists to native std::vector
inline std::vector<int64_t> UnpackListToVector(const nb::list& py_list) {
  std::vector<int64_t> result;
  result.reserve(py_list.size());
  for (size_t i = 0; i < py_list.size(); ++i) {
    result.push_back(nb::cast<int64_t>(py_list[i]));
  }
  return result;
}

// FFI helper to extract the underlying C++ PjRtBuffers of a JAX Array
inline std::vector<raiden::RaidenBufferHandle> ExtractPjRtBuffersFromPyArray(
    const nb::object& jax_array, bool unsafe_skip_buffer_lock = false) {
  std::vector<raiden::RaidenBufferHandle> result;
  nb::object addressable_shards = jax_array.attr("addressable_shards");
  size_t num_shards = nb::len(addressable_shards);
  result.reserve(num_shards);

  for (size_t i = 0; i < num_shards; ++i) {
    nb::object shard = addressable_shards[i];
    nb::object shard_data = shard.attr("data");
    xla::PjRtBuffer* buf = GetPjrtBufferFromPyObject(shard_data.ptr());
    auto handle = raiden::RaidenBufferHandle::Acquire(buf, nullptr, nullptr,
                                                      unsafe_skip_buffer_lock);
    if (!handle.ok()) {
      throw std::runtime_error(
          std::string("Failed to acquire buffer handle: ") +
          std::string(handle.status().message()));
    }
    result.push_back(std::move(handle.value()));
  }
  return result;
}
#else   // WITHOUT_PYTHON (Mocks)

inline std::vector<int64_t> UnpackListToVector(const nb::list& py_list) {
  std::vector<int64_t> result;
  result.reserve(py_list.size());
  for (size_t i = 0; i < py_list.size(); ++i) {
    result.push_back(nb::cast<int64_t>(py_list[i]));
  }
  return result;
}

inline std::vector<raiden::RaidenBufferHandle> ExtractPjRtBuffersFromPyArray(
    const nb::object& jax_array, bool unsafe_skip_buffer_lock = false) {
  std::vector<raiden::RaidenBufferHandle> result;
  nb::object addressable_shards = jax_array.attr("addressable_shards");
  size_t num_shards = nb::len(addressable_shards);
  result.reserve(num_shards);

  for (size_t i = 0; i < num_shards; ++i) {
    nb::object shard = addressable_shards[i];
    nb::object shard_data = shard.attr("data");
    // In mock tests, read the PjRtBuffer pointer from the "ptr" attribute.
    xla::PjRtBuffer* buf = reinterpret_cast<xla::PjRtBuffer*>(
        nb::cast<size_t>(shard_data.attr("ptr")));
    auto handle = raiden::RaidenBufferHandle::Acquire(buf, nullptr, nullptr,
                                                      unsafe_skip_buffer_lock);
    if (!handle.ok()) {
      throw std::runtime_error(
          std::string("Failed to acquire buffer handle: ") +
          std::string(handle.status().message()));
    }
    result.push_back(std::move(handle.value()));
  }
  return result;
}
#endif  // WITHOUT_PYTHON

}  // namespace jax

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_UTILS_OSS_H_
