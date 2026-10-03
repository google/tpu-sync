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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

# TPUDirect Storage Userspace Library (`libtdsul`).
#
# Vendored copy of the upstream `tpudirect-storage` repository
# (https://cloudrdma-guest.git.corp.google.com/tpudirect-storage, snapshot of
# commit e046a94), which has no permanent google3 home yet. It lives inside
# `//third_party/tpu_raiden` so that `TdsKVBackend` can depend on it without the
# illegal non-experimental -> experimental edge that `//experimental/xpu-rdma/
# tpudirect-storage:tdsul` produced (that edge builds only under
# `--experimental_deps_ok` and can never be submitted).
#
# TODO: replace this vendored copy with a dependency on the canonical
# `libtdsul` target once the TDSUL owners establish one in google3.
#
# Local modifications relative to upstream (re-apply these on any re-sync; a
# plain re-copy will silently drop them):
#   1. `src/util/ThreadPoolConductor.cpp` -- `ThreadPoolConductor::Start()` now
#      pins workers to CPUs only when an explicit core range was configured.
#      Upstream always pinned, deriving core indices from
#      `sysconf(_SC_NPROCESSORS_ONLN)`, which reports machine-wide cores and
#      says nothing about the process's allowed cpuset. Under a restricted
#      cpuset every `pthread_create` failed with EINVAL. Any job restricted to
#      a cpuset can hit this, not just test sandboxes.
#   2. `src/util/ThreadPoolConductor.cpp` -- `WorkerThread::Start()` retries
#      unpinned when an affinity-pinned `pthread_create` fails, instead of
#      logging and leaving a pool that accepts work it will never run.
#   3. `src/util/ThreadPoolConductor.cpp` -- `WorkerThread::Stop()` drains
#      `task_queue_`, failing the futures with `TDS_ERROR_IO` and freeing
#      requests it owns. Upstream leaked every request queued to a worker
#      whose thread never started.
# `tests/thread_pool_conductor_test.cpp` is correspondingly strengthened to
# assert futures actually complete rather than only that nothing crashed.
# Upstream's `kokoro/` CI directory is intentionally not vendored.
#
# These three fixes have not yet been reported upstream.
#
# `libtdsul` is a zero-copy execution engine: it performs POSIX
# `pread`/`pwrite`/`preadv`/`pwritev` between [local-storage: NVMe SSD] /
# [shared-storage: Lustre/NFS] and `TDS_MEM_HOST` buffers (registered handles
# or raw pointers) in [dram: User host_buf]. Per `go/tdsul-api` §1 it holds no
# internal bounce buffer pools; alignment fallbacks belong to the framework
# adapter
# (`//tpu_sync/kv_cache/backends/storage:tds_backend`).
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tdsul",
    srcs = [
        "src/tdsul.cpp",
        "src/util/IoQueue.cpp",
        "src/util/IoUringConductor.cpp",
        "src/util/ThreadPoolConductor.cpp",
    ],
    hdrs = [
        "include/tdsul/def.h",
        "include/tdsul/tdsul.h",
        "src/def_internal.h",
        "src/syscall_internal.h",
        "src/util/IoConductor.h",
        "src/util/IoQueue.h",
        "src/util/IoUringConductor.h",
        "src/util/ThreadPoolConductor.h",
    ],
    # Upstream sources use bare `absl/...` includes rather than google3's
    # `third_party/absl/...` convention, so `third_party` is added as a system
    # include root instead of rewriting every upstream include.
    copts = [
        "-isystem",
        "third_party",
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        # Upstream sources include `absl/...`, `gtest/...` and `gmock/...`
        # directly rather than through google3 module-exporting targets, which
        # `layering_check` cannot resolve. Sibling tpu_raiden targets (e.g.
        # `:posix_backend`) disable it for the same reason.
        "-layering_check",
    ],
    includes = [
        "include",
        "src",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "tdsul_test",
    srcs = ["tests/tdsul_test.cpp"],
    copts = [
        "-isystem",
        "third_party",
        "-isystem",
        "third_party/googletest/googletest/include",
        "-isystem",
        "third_party/googletest/googlemock/include",
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        # Upstream sources include `absl/...`, `gtest/...` and `gmock/...`
        # directly rather than through google3 module-exporting targets, which
        # `layering_check` cannot resolve. Sibling tpu_raiden targets (e.g.
        # `:posix_backend`) disable it for the same reason.
        "-layering_check",
    ],
    deps = [
        ":tdsul",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:initialize",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "thread_pool_conductor_test",
    srcs = ["tests/thread_pool_conductor_test.cpp"],
    copts = [
        "-isystem",
        "third_party",
        "-isystem",
        "third_party/googletest/googletest/include",
        "-isystem",
        "third_party/googletest/googlemock/include",
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        # Upstream sources include `absl/...`, `gtest/...` and `gmock/...`
        # directly rather than through google3 module-exporting targets, which
        # `layering_check` cannot resolve. Sibling tpu_raiden targets (e.g.
        # `:posix_backend`) disable it for the same reason.
        "-layering_check",
    ],
    deps = [
        ":tdsul",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:initialize",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "io_queue_test",
    srcs = ["tests/io_queue_test.cpp"],
    copts = [
        "-isystem",
        "third_party",
        "-isystem",
        "third_party/googletest/googletest/include",
        "-isystem",
        "third_party/googletest/googlemock/include",
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        # Upstream sources include `absl/...`, `gtest/...` and `gmock/...`
        # directly rather than through google3 module-exporting targets, which
        # `layering_check` cannot resolve. Sibling tpu_raiden targets (e.g.
        # `:posix_backend`) disable it for the same reason.
        "-layering_check",
    ],
    deps = [
        ":tdsul",
        "@com_google_googletest//:gtest_main",
    ],
)
