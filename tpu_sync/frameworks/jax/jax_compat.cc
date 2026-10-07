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

#include "tpu_sync/frameworks/jax/jax_compat.h"

#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>  // IWYU pragma: keep
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/types.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/xla_compat.h"

namespace nb = nanobind;

namespace raiden {
namespace {

#if RAIDEN_JAX >= 1101

// numpy's C API is a table each extension fills for itself with import_array;
// the jaxlib in this process has filled its own, not ours.
void EnsureNumpyImported() {
  [[maybe_unused]] static const bool imported = [] {
    tsl::ImportNumpy();
    if (PyErr_Occurred()) throw nb::python_error();
    return true;
  }();
}

xla::PrimitiveType PrimitiveTypeOf(nb::handle dtype) {
  absl::StatusOr<xla::PrimitiveType> type =
      xla::DtypeToPrimitiveType(nb::cast<xla::nb_dtype>(dtype));
  if (!type.ok()) {
    throw std::runtime_error(
        absl::StrCat("Unsupported dtype: ", type.status().message()));
  }
  return *type;
}

// Rebuilds the XLA layout from jax.Array.format.layout: the three fields jax
// keeps of a PjRtLayout, and all that raiden reads of one.
xla::Layout LayoutOf(nb::handle layout) {
  std::vector<int64_t> minor_to_major =
      nb::cast<std::vector<int64_t>>(layout.attr("major_to_minor"));
  std::reverse(minor_to_major.begin(), minor_to_major.end());
  xla::Layout result(minor_to_major);
  nb::object tiling = layout.attr("tiling");
  if (!tiling.is_none()) {
    for (nb::handle tile : tiling) {
      *result.add_tiles() = xla::Tile(nb::cast<std::vector<int64_t>>(tile));
    }
  }
  result.set_element_size_in_bits(
      nb::cast<int64_t>(layout.attr("sub_byte_element_size_in_bits")));
  return result;
}

#else  // RAIDEN_JAX < 1101

#if RAIDEN_JAX >= 1100
// Returns the runtime jaxlib version as (major * 10000 + minor * 100 + patch),
// e.g., 0.11.0 -> 1100, 0.11.1 -> 1101, 0.11.2 -> 1102.
int GetRuntimeJaxVersion() {
  static const int version = []() -> int {
    if (!Py_IsInitialized()) {
      return 1102;
    }
    PyGILState_STATE gil = PyGILState_Ensure();
    int parsed = 1102;
    PyObject* mod = PyImport_ImportModule("jaxlib.version");
    if (mod == nullptr) {
      PyErr_Clear();
      mod = PyImport_ImportModule("jax.version");
    }
    if (mod != nullptr) {
      PyObject* info = PyObject_GetAttrString(mod, "__version_info__");
      if (info != nullptr && PyTuple_Check(info) && PyTuple_Size(info) >= 3) {
        int64_t major = PyLong_AsLong(PyTuple_GetItem(info, 0));
        int64_t minor = PyLong_AsLong(PyTuple_GetItem(info, 1));
        int64_t patch = PyLong_AsLong(PyTuple_GetItem(info, 2));
        if (!PyErr_Occurred()) {
          parsed = static_cast<int>(major * 10000 + minor * 100 + patch);
        }
      }
      Py_XDECREF(info);
      Py_DECREF(mod);
    }
    PyErr_Clear();
    PyGILState_Release(gil);
    return parsed;
  }();
  return version;
}
#endif  // RAIDEN_JAX >= 1100

// Mirrors jaxlib's private PyArrayObject (py_array.cc) and PyArray::Storage
// (py_array.h) without including them: jaxlib's py_client target is private.
// Only the members read here are named; the rest is padding sized from the
// real headers of every supported jaxlib. `aval` doubles as a runtime layout
// check (see ReadIfrtArray).

// PyArray::Storage in jaxlib 0.10.0 through 0.11.1.
struct PyArrayStorage_0_10_0 {
  PyObject* aval;
  char skipped[72];  // weak_type, dtype, shape, sharding, npy_value, committed,
                     // py_client
  xla::ifrt::ArrayRef ifrt_array;
};
static_assert(offsetof(PyArrayStorage_0_10_0, ifrt_array) == 80);

// PyArray::Storage in jaxlib 0.11.2 and later, which reordered the members and
// added ft_mutex mu.
struct PyArrayStorage_0_11_2 {
  char skipped[32];  // py_client, next, prev, thread_id_bucket, committed,
                     // weak_type, mu
  PyObject* aval;
  char skipped2[48];  // dtype, shape, sharding, npy_value
  xla::ifrt::ArrayRef ifrt_array;
};
static_assert(offsetof(PyArrayStorage_0_11_2, ifrt_array) == 88);

template <typename Storage>
struct PyArrayObject {
  PyObject_HEAD;
#if RAIDEN_JAX < 1100 && PY_VERSION_HEX < 0x030C0000
  PyObject* weakrefs;
  PyObject* dict;
#endif
  bool initialized;
  Storage storage;
};

template <typename Storage>
xla::ifrt::Array* ReadIfrtArray(PyObject* obj) {
  auto* py_array = std::launder(reinterpret_cast<PyArrayObject<Storage>*>(obj));
  if (!py_array->initialized) {
    throw std::runtime_error("PyArrayObject not initialized");
  }
  // Layout self-check: jaxlib serves the public `aval` attribute from this
  // same Storage, so a different object means the mirror does not match the
  // running jaxlib. Compares pointers only; nothing is dereferenced.
  PyObject* aval = PyObject_GetAttrString(obj, "aval");
  const bool layout_matches = aval != nullptr && aval == py_array->storage.aval;
  Py_XDECREF(aval);  // Storage keeps its own reference.
  if (!layout_matches) {
    PyErr_Clear();
    throw std::runtime_error("PyArray layout mismatch with the running jaxlib");
  }
  return py_array->storage.ifrt_array.get();
}

xla::ifrt::Array* GetIfrtArray(PyObject* obj) {
#if RAIDEN_JAX >= 1100
  // Runtime dispatch across 0.11.x jaxlibs; 0.10.x builds run only on 0.10.x.
  if (GetRuntimeJaxVersion() >= 1102) {
    return ReadIfrtArray<PyArrayStorage_0_11_2>(obj);
  }
#endif
  return ReadIfrtArray<PyArrayStorage_0_10_0>(obj);
}

xla::ifrt::PjRtCompatibleArray* CastToPjRtCompatibleArray(
    xla::ifrt::Array* ifrt_array) {
  if (ifrt_array == nullptr) return nullptr;
  // In JAX < 0.11.2, Client's vtable before runtime_type() lacks
  // CopyArraysToHostBufferShards; skip the runtime_type() check there and
  // static_cast directly (cross-DSO dynamic_cast is unavailable in OSS builds
  // because _jax.so hides RTTI symbols).
#if RAIDEN_JAX >= 1100
  if (GetRuntimeJaxVersion() >= 1102 &&
      ifrt_array->client()->runtime_type() != "pjrt_ifrt") {
    return nullptr;
  }
#else
  if (ifrt_array->client()->runtime_type() != "pjrt_ifrt") return nullptr;
#endif
  return static_cast<xla::ifrt::PjRtCompatibleArray*>(ifrt_array);
}

// Returns the PjRtBuffer backing a single-device PyArray. Throws
// std::runtime_error if the array is uninitialized or not PjRt-backed.
xla::PjRtBuffer* PjRtBufferFromPyArray(PyObject* obj) {
  auto* arr = CastToPjRtCompatibleArray(GetIfrtArray(obj));
  if (arr == nullptr) {
    throw std::runtime_error("Not a PjRt compatible array");
  }
  // The extension is built against the selected JAX's own headers, so the
  // direct virtual call uses the correct vtable slot.
  return arr->pjrt_buffers().front().get();
}

#endif  // RAIDEN_JAX >= 1101

}  // namespace

RaidenBufferHandle AcquireShardBuffer(PyObject* shard_data,
                                      bool unsafe_skip_buffer_lock) {
#if RAIDEN_JAX >= 1101
  // jax 0.11.1 added Array.unsafe_raw_buffer(); with it the buffer, shape and
  // layout are all public attributes.
  EnsureNumpyImported();
  nb::handle shard(shard_data);
  nb::object raw = shard.attr("unsafe_raw_buffer")();
  auto* raw_buffer =
      reinterpret_cast<RawBuffer*>(nb::cast<std::uintptr_t>(raw.attr("ptr")));
  xla::Shape shape = xla::ShapeUtil::MakeShape(
      PrimitiveTypeOf(shard.attr("dtype")),
      nb::cast<std::vector<int64_t>>(shard.attr("shape")));
  nb::object layout = shard.attr("format").attr("layout");
  if (!layout.is_none()) {
    *shape.mutable_layout() = LayoutOf(layout);
  }
  absl::StatusOr<RaidenBufferHandle> handle =
      RaidenBufferHandle::AcquireFromRaw(raw_buffer, shape,
                                         unsafe_skip_buffer_lock);
#else
  // Older jaxlibs expose no raw buffer: go through the PjRtBuffer that the
  // PyArray mirror above reaches.
  absl::StatusOr<RaidenBufferHandle> handle =
      RaidenBufferHandle::Acquire(PjRtBufferFromPyArray(shard_data), nullptr,
                                  nullptr, unsafe_skip_buffer_lock);
#endif
  if (!handle.ok()) {
    throw std::runtime_error(absl::StrCat("Failed to acquire buffer handle: ",
                                          handle.status().message()));
  }
  return *std::move(handle);
}

}  // namespace raiden
