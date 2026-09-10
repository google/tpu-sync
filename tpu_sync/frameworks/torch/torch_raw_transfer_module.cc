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

// Standalone extension module for the torch raw-transfer surface. The
// productized path ships the same bindings inside _tpu_raiden_torch as its
// `raw_transfer` submodule (see torch_raw_transfer_bindings.cc); this module
// exists for tests and source-checkout use without the full torch extension.

#include "nanobind/nanobind.h"
#include "tpu_sync/frameworks/torch/torch_raw_transfer_bindings.h"

namespace raiden {

NB_MODULE(_torch_raw_transfer, m) { BindTorchRawTransfer(m); }

}  // namespace raiden
