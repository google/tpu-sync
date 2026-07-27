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

# XLA revision the torch extension is compiled against. It must match the
# revision torch_tpu was built with, because the extension calls virtual
# methods on PJRT objects that torch_tpu constructs. Update together with
# torch_tpu.version at the repository root.
XLA_COMMIT = "cc39fef7cae080ae61fc66bfe283bb80abcf409b"
