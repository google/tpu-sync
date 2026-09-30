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

// Isolates jaxlib PyArray layout mirroring for PjRtBuffer and IFRT array
// access.

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_COMPAT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_COMPAT_H_

#include <Python.h>

// Which JAX this is being built against; see xla_compat.h for encoding.
#ifndef RAIDEN_JAX
#define RAIDEN_JAX 1102
#endif

namespace xla {
class PjRtBuffer;
namespace ifrt {
class Array;
}  // namespace ifrt
}  // namespace xla

namespace raiden {

// Returns the PjRtBuffer backing an addressable shard of a JAX array.
// `obj` must be a jaxlib PyArray. Throws std::runtime_error if uninitialized
// or not backed by a PjRt-compatible IFRT array.
xla::PjRtBuffer* PjRtBufferFromPyArray(PyObject* obj);

// Returns the IFRT array backing a JAX array.
// Throws std::runtime_error if `obj` is an uninitialized PyArray.
xla::ifrt::Array* IfrtArrayFromPyArray(PyObject* obj);

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_COMPAT_H_
