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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "acceptor",
    srcs = ["acceptor.cc"],
    hdrs = ["acceptor.h"],
    deps = [
        ":socket_tcp",
        ":socket_util",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "//tpu_sync/transport/peregrine/src/internal/base:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "acceptor_test",
    srcs = ["acceptor_test.cc"],
    deps = [
        ":acceptor",
        ":socket_tcp",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "//tpu_sync/transport/peregrine/src/internal/util:test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "connector",
    srcs = ["connector.cc"],
    hdrs = ["connector.h"],
    deps = [
        ":socket_tcp",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

cc_test(
    name = "connector_test",
    srcs = ["connector_test.cc"],
    deps = [
        ":acceptor",
        ":connector",
        ":socket_tcp",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "//tpu_sync/transport/peregrine/src/internal/util:test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "socket_util",
    srcs = ["socket_util.cc"],
    hdrs = ["socket_util.h"],
    deps = [
        "//tpu_sync/transport/peregrine/src/internal/base:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "socket_util_test",
    srcs = ["socket_util_test.cc"],
    deps = [
        ":socket_util",
        "//tpu_sync/transport/peregrine/src/internal/base:types",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "socket_base",
    srcs = ["socket_base.cc"],
    hdrs = ["socket_base.h"],
    deps = [
        ":socket_util",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "//tpu_sync/transport/peregrine/src/internal/base:types",
        "//tpu_sync/transport/peregrine/src/util:macro",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "socket_tcp",
    srcs = ["socket_tcp.cc"],
    hdrs = ["socket_tcp.h"],
    deps = [
        ":socket_base",
        ":socket_util",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "//tpu_sync/transport/peregrine/src/internal/base:types",
        "//tpu_sync/transport/peregrine/src/internal/util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "socket_tcp_test",
    srcs = ["socket_tcp_test.cc"],
    deps = [
        ":socket_tcp",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "//tpu_sync/transport/peregrine/src/internal/base:types",
        "//tpu_sync/transport/peregrine/src/internal/util:test_util",
        "//tpu_sync/transport/peregrine/src/util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "socket_udp",
    srcs = ["socket_udp.cc"],
    hdrs = ["socket_udp.h"],
    deps = [
        ":socket_base",
        ":socket_util",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "//tpu_sync/transport/peregrine/src/internal/base:types",
        "//tpu_sync/transport/peregrine/src/internal/util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "socket_udp_test",
    srcs = ["socket_udp_test.cc"],
    deps = [
        ":socket_udp",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "//tpu_sync/transport/peregrine/src/internal/base:types",
        "//tpu_sync/transport/peregrine/src/internal/util:test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "socket_test_util",
    testonly = True,
    srcs = ["socket_test_util.cc"],
    hdrs = ["socket_test_util.h"],
    deps = [
        ":acceptor",
        ":connector",
        ":socket_tcp",
        ":socket_udp",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "//tpu_sync/transport/peregrine/src/internal/util:test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tcp_socket_util",
    srcs = ["tcp_socket_util.cc"],
    hdrs = ["tcp_socket_util.h"],
    deps = [
        ":socket_util",
        "//tpu_sync/transport/peregrine/src/internal/base:types",
        "//tpu_sync/transport/peregrine/src/internal/util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "tcp_socket_util_test",
    srcs = ["tcp_socket_util_test.cc"],
    deps = [
        ":socket_tcp",
        ":tcp_socket_util",
        "//tpu_sync/transport/peregrine/src/internal/base:endpoint",
        "//tpu_sync/transport/peregrine/src/internal/base:types",
        "//tpu_sync/transport/peregrine/src/internal/util:test_util",
        "//tpu_sync/transport/peregrine/src/util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
