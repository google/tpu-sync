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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef WITHOUT_PYTHON
#include <Python.h>

#include <nanobind/nanobind.h>
#include "tpu_sync/frameworks/jax/jax_compat.h"
#else
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tpu_sync/core/xla_compat.h"
#include "tpu_sync/frameworks/jax/mock_nanobind.h"
#endif
#include "tpu_sync/core/raw_transfer_core.h"

namespace nb = nanobind;

namespace jax {

// FFI helper to convert nanobind lists to native std::vector
inline std::vector<int64_t> UnpackListToVector(const nb::list& py_list) {
  std::vector<int64_t> result;
  result.reserve(py_list.size());
  for (size_t i = 0; i < py_list.size(); ++i) {
    result.push_back(nb::cast<int64_t>(py_list[i]));
  }
  return result;
}

#ifndef WITHOUT_PYTHON

// FFI helper to extract the underlying C++ buffers of a JAX Array, one per
// addressable shard.
inline std::vector<raiden::RaidenBufferHandle> ExtractPjRtBuffersFromPyArray(
    const nb::object& jax_array, bool unsafe_skip_buffer_lock = false) {
  nb::object shards = jax_array.attr("addressable_shards");
  std::vector<raiden::RaidenBufferHandle> result;
  result.reserve(nb::len(shards));
  for (nb::handle shard : shards) {
    nb::object data = shard.attr("data");
    result.push_back(
        raiden::AcquireShardBuffer(data.ptr(), unsafe_skip_buffer_lock));
  }
  return result;
}

#else  // WITHOUT_PYTHON (Mocks)

inline std::vector<raiden::RaidenBufferHandle> ExtractPjRtBuffersFromPyArray(
    const nb::object& jax_array, bool unsafe_skip_buffer_lock = false) {
  std::vector<raiden::RaidenBufferHandle> result;
  nb::object addressable_shards = jax_array.attr("addressable_shards");
  size_t num_shards = nb::len(addressable_shards);
  result.reserve(num_shards);

  for (size_t i = 0; i < num_shards; ++i) {
    nb::object shard = addressable_shards[i];
    nb::object shard_data = shard.attr("data");

    nb::object raw_buf_obj = shard_data.attr("unsafe_raw_buffer")();
    if (raw_buf_obj.is_none()) {
      throw std::runtime_error(
          "shard_data.attr(\"unsafe_raw_buffer\")() returned None");
    }
    std::uintptr_t raw_buf_ptr = 0;
    try {
      nb::object ptr_attr = raw_buf_obj.attr("ptr");
      if (ptr_attr.is_none()) {
        throw std::runtime_error("raw_buf_obj.attr(\"ptr\") is None");
      }
      raw_buf_ptr = nb::cast<std::uintptr_t>(ptr_attr);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string("Failed to get or cast raw_buf_obj.attr(\"ptr\") to "
                      "std::uintptr_t: ") +
          e.what());
    }
    auto* raw_buf = reinterpret_cast<raiden::RawBuffer*>(raw_buf_ptr);

    nb::object shape_obj = shard_data.attr("shape");
    std::vector<int64_t> dims;
    dims.reserve(nb::len(shape_obj));
    for (size_t d = 0; d < nb::len(shape_obj); ++d) {
      dims.push_back(nb::cast<int64_t>(shape_obj[d]));
    }

    // Mock path doesn't have easy access to DtypeToPrimitiveType, use dummy F32
    xla::Shape shape = xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, dims);

    auto handle = raiden::RaidenBufferHandle::AcquireFromRaw(
        raw_buf, shape, unsafe_skip_buffer_lock);
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

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_UTILS_H_
