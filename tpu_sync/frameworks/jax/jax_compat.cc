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

#include <new>
#include <stdexcept>

#include "absl/base/thread_annotations.h"
#include "jaxlib/py_array.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"

namespace raiden {
namespace {

// Mirrors jaxlib's private PyArrayObject layout. Fields must match jaxlib
// exactly across Python versions: Python < 3.12 includes weakrefs and dict.
struct PyArrayObject {
  PyObject_HEAD;
#if PY_VERSION_HEX < 0x030C0000
  PyObject* weakrefs;
  PyObject* dict;
#endif  // PY_VERSION_HEX < 0x030C0000
  bool initialized;
  alignas(
      jax::PyArray::Storage) char array_storage[sizeof(jax::PyArray::Storage)];
};

jax::PyArray::Storage* GetStorage(PyObject* obj) {
  auto* py_array_object = reinterpret_cast<PyArrayObject*>(obj);
  if (!py_array_object->initialized) {
    throw std::runtime_error("PyArrayObject not initialized");
  }
  return std::launder(
      reinterpret_cast<jax::PyArray::Storage*>(py_array_object->array_storage));
}

xla::ifrt::PjRtCompatibleArray* CastToPjRtCompatibleArray(
    xla::ifrt::Array* ifrt_array) {
  if (ifrt_array == nullptr) return nullptr;
  if (ifrt_array->client()->runtime_type() == "pjrt_ifrt") {
    return static_cast<xla::ifrt::PjRtCompatibleArray*>(ifrt_array);
  }
  return nullptr;
}

}  // namespace

xla::PjRtBuffer* PjRtBufferFromPyArray(PyObject* obj)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  auto* arr = CastToPjRtCompatibleArray(GetStorage(obj)->ifrt_array.get());
  if (arr == nullptr) {
    throw std::runtime_error("Not a PjRt compatible array");
  }
  return arr->pjrt_buffers().front().get();
}

xla::ifrt::Array* IfrtArrayFromPyArray(PyObject* obj)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  return GetStorage(obj)->ifrt_array.get();
}

}  // namespace raiden
