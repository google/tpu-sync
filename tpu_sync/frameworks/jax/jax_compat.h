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

// Reaches the device buffer behind a jaxlib PyArray. How depends on the JAX
// version, so this is the one place that knows: jax >= 0.11.1 exposes it as
// Array.unsafe_raw_buffer(); older jaxlibs need a mirror of PyArray's private
// layout.

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_COMPAT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_COMPAT_H_

#include <Python.h>

#include "tpu_sync/core/raw_transfer_core.h"

// Which JAX this is being built against; see xla_compat.h for encoding.
#ifndef RAIDEN_JAX
#define RAIDEN_JAX 1102
#endif

namespace raiden {

// Acquires the device buffer behind one addressable shard of a jax.Array
// (`shard.data`, a single-device jaxlib PyArray). The GIL must be held.
// Throws std::runtime_error if the shard's buffer cannot be reached.
// From jax 0.11.1 on, unsafe_raw_buffer() takes no usage hold, so
// `unsafe_skip_buffer_lock` only affects older jaxlibs.
RaidenBufferHandle AcquireShardBuffer(PyObject* shard_data,
                                      bool unsafe_skip_buffer_lock);

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FRAMEWORKS_JAX_JAX_COMPAT_H_
