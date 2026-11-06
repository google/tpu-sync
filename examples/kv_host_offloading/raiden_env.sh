#!/bin/bash

# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ensure tpu_raiden is importable for BOTH supported install paths. Sourced by
# prefill.sh / decode.sh AFTER RAIDEN_ROOT is set and a venv is active.
#
#  - Wheel install (`pip install --pre tpu-sync-jax`): tpu_raiden is already
#    importable from site-packages (generated protos + the engine .so are bundled
#    inside the package), so we leave PYTHONPATH alone. Prepending the source tree
#    would SHADOW the installed package with a copy lacking the built .so/protos.
#  - Build-from-source (`./build.sh jax`): tpu_raiden is not installed, so expose
#    it from the repo: the root holds the `tpu_raiden` package, bazel-bin holds
#    the generated protobuf modules.
#
# The only remaining requirement is that `import tpu_raiden` works -- which
# is exactly what this file guarantees.
#
_raiden_ready() { python -c "import tpu_sync.rpc.coordination_pb2" 2>/dev/null; }

if ! _raiden_ready; then
  export PYTHONPATH="${RAIDEN_ROOT}:${RAIDEN_ROOT}/bazel-bin:${PYTHONPATH:-}"
fi

if ! _raiden_ready; then
  echo "ERROR: tpu_raiden is not importable. Install it via one of:" >&2
  echo "  - build from source: ./build.sh jax  (from the repo root), or" >&2
  echo "  - wheel:             pip install --pre tpu-sync-jax" >&2
  exit 1
fi