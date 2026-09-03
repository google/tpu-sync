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
#
# Installs clang-18 into the ml-build container, which raiden's Bazel build
# requires for XLA's .ll codegen targets (CWG2518 needs clang-18). Factored out
# of ci/build_wheel.sh so the GitHub Actions jobs and the wheel build share one
# source of truth. Idempotent: a no-op if clang-18 is already on PATH.

set -exu -o pipefail

if clang --version 2>/dev/null | grep -q 'version 18'; then
  echo "clang-18 already installed."
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget gnupg ca-certificates >/dev/null
# Add the LLVM jammy-18 apt repo manually (the container's add-apt-repository is
# broken: python apt_pkg is missing for python3.12).
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg
echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
  > /etc/apt/sources.list.d/llvm18.list
apt-get update -qq
apt-get install -y -qq clang-18 >/dev/null
ln -sf /usr/bin/clang-18 /usr/bin/clang
ln -sf /usr/bin/clang++-18 /usr/bin/clang++
clang --version | head -1
