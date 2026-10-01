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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>

#include "absl/base/thread_annotations.h"
#include "absl/types/span.h"
#include "jaxlib/py_array.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"

namespace raiden {
namespace {

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

// Mirrors jaxlib's private PyArrayObject from py_array.cc.
struct PyArrayObject {
  PyObject_HEAD;
#if RAIDEN_JAX < 1100 && PY_VERSION_HEX < 0x030C0000
  PyObject* weakrefs;
  PyObject* dict;
#endif
  bool initialized;
  alignas(
      jax::PyArray::Storage) char array_storage[sizeof(jax::PyArray::Storage)];
};

// In JAX 0.11.0 and 0.11.1, PyArray_Storage placed ifrt_array at byte offset 80
// (aval [8B], weak_type + pad [8B], dtype [8B], shape [24B], sharding [8B],
// npy_value [8B], committed + pad [8B], py_client [8B]). In JAX 0.11.2+,
// PyArray_Storage reordered fields and added ft_mutex mu, moving ifrt_array to
// byte offset 88.
struct PyArrayStorage_0_11_0 {
  alignas(void*) char prefix[80];
  xla::ifrt::ArrayRef ifrt_array;
};
static_assert(offsetof(PyArrayStorage_0_11_0, ifrt_array) == 80);
#if RAIDEN_JAX >= 1102
static_assert(offsetof(jax::PyArray::Storage, ifrt_array) == 88);
#else
static_assert(offsetof(jax::PyArray::Storage, ifrt_array) == 80);
#endif

// Vtable layout of xla::ifrt::PjRtCompatibleArray in JAX 0.11.0, where
// xla::ifrt::Value inherited from llvm::RTTIRoot (+3 virtual slots:
// dynamicClassID, isA, anchor) and xla::ifrt::Array did not yet declare
// array_spec() (-1 virtual slot), placing pjrt_buffers() at vtable slot 21
// instead of slot 19 (JAX 0.11.1+).
class PjRtCompatibleArray_0_11_0 {
 public:
  virtual ~PjRtCompatibleArray_0_11_0() = default;
  // llvm::RTTIRoot (3 slots)
  virtual void vslot_02() = 0;
  virtual void vslot_03() = 0;
  virtual void vslot_04() = 0;
  // xla::ifrt::Value (6 slots)
  virtual void vslot_05() = 0;
  virtual void vslot_06() = 0;
  virtual void vslot_07() = 0;
  virtual void vslot_08() = 0;
  virtual void vslot_09() = 0;
  virtual void vslot_10() = 0;
  // xla::ifrt::Array (10 slots in 0.11.0)
  virtual void vslot_11() = 0;
  virtual void vslot_12() = 0;
  virtual void vslot_13() = 0;
  virtual void vslot_14() = 0;
  virtual void vslot_15() = 0;
  virtual void vslot_16() = 0;
  virtual void vslot_17() = 0;
  virtual void vslot_18() = 0;
  virtual void vslot_19() = 0;
  virtual void vslot_20() = 0;
  // xla::ifrt::PjRtCompatibleArray::pjrt_buffers() (slot 21)
  virtual absl::Span<const std::shared_ptr<xla::PjRtBuffer>> pjrt_buffers() = 0;
};

xla::ifrt::Array* GetIfrtArray(PyObject* obj) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  auto* py_array_object = reinterpret_cast<PyArrayObject*>(obj);
  if (!py_array_object->initialized) {
    throw std::runtime_error("PyArrayObject not initialized");
  }
  if (GetRuntimeJaxVersion() < 1102) {
    return std::launder(reinterpret_cast<PyArrayStorage_0_11_0*>(
                            py_array_object->array_storage))
        ->ifrt_array.get();
  }
  return std::launder(reinterpret_cast<jax::PyArray::Storage*>(
                          py_array_object->array_storage))
      ->ifrt_array.get();
}

xla::ifrt::PjRtCompatibleArray* CastToPjRtCompatibleArray(
    xla::ifrt::Array* ifrt_array) {
  if (ifrt_array == nullptr) return nullptr;
  // In JAX < 0.11.2, Client's vtable before runtime_type() lacks
  // CopyArraysToHostBufferShards; skip the runtime_type() check there and
  // static_cast directly (cross-DSO dynamic_cast is unavailable in OSS builds
  // because _jax.so hides RTTI symbols).
  if (GetRuntimeJaxVersion() >= 1102 &&
      ifrt_array->client()->runtime_type() != "pjrt_ifrt") {
    return nullptr;
  }
  return static_cast<xla::ifrt::PjRtCompatibleArray*>(ifrt_array);
}

}  // namespace

xla::PjRtBuffer* PjRtBufferFromPyArray(PyObject* obj)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  auto* arr = CastToPjRtCompatibleArray(GetIfrtArray(obj));
  if (arr == nullptr) {
    throw std::runtime_error("Not a PjRt compatible array");
  }
  if (GetRuntimeJaxVersion() < 1101) {
    return reinterpret_cast<PjRtCompatibleArray_0_11_0*>(arr)
        ->pjrt_buffers()
        .front()
        .get();
  }
  return arr->pjrt_buffers().front().get();
}

xla::ifrt::Array* IfrtArrayFromPyArray(PyObject* obj)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  return GetIfrtArray(obj);
}

}  // namespace raiden
