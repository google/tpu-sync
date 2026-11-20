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

// Reshard-controller sidecar: a drop-in for the golden
// launcher's inline Python controller. One thin KVCacheStore-owned
// ReshardService per role host, speaking the framed ControllerRequest /
// ControlRequest wire on --port, honoring the ready-file turnup contract:
//
//   1. bind the listener;
//   2. atomically write --ready-file with the raiden_controller_ready JSON
//      (temp file + rename) and print the same line to stdout;
//   3. serve until SIGTERM/SIGINT or COMMAND_SHUTDOWN;
//   4. emit raiden_controller_stopped on exit.

#include <signal.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/kv_cache/reshard/reshard_service.h"
#include "tpu_sync/kv_cache_offload/kv_cache_store.h"

ABSL_FLAG(int, port, -1, "Framed-TCP listener port (0 = ephemeral).");
ABSL_FLAG(std::string, advertise_host, "",
          "Routable host address published in the ready file.");
ABSL_FLAG(double, request_registry_ttl_s, 600.0,
          "Request-block registry TTL in seconds.");
ABSL_FLAG(std::string, ready_file, "",
          "Path receiving the raiden_controller_ready JSON line.");

namespace {

absl::Notification* shutdown_notification = nullptr;

void HandleSignal(int /*signum*/) {
  if (shutdown_notification != nullptr &&
      !shutdown_notification->HasBeenNotified()) {
    shutdown_notification->Notify();
  }
}

std::string RenderHost(const std::string& raw) {
  std::string host = raw;
  while (!host.empty() && (host.front() == '[' || host.front() == ' ')) {
    host.erase(host.begin());
  }
  while (!host.empty() && (host.back() == ']' || host.back() == ' ')) {
    host.pop_back();
  }
  if (absl::StrContains(host, ':')) {
    return absl::StrCat("[", host, "]");
  }
  return host;
}

}  // namespace

int main(int argc, char** argv) {
  // The launcher contract uses dashed flags — the Python
  // controller's argparse spelling. Normalize them to absl's underscore
  // form so the binary is a literal drop-in.
  std::vector<std::string> normalized_storage;
  normalized_storage.reserve(argc);
  std::vector<char*> normalized_argv;
  normalized_argv.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) == 0) {
      const size_t eq = arg.find('=');
      const size_t name_end = eq == std::string::npos ? arg.size() : eq;
      for (size_t j = 2; j < name_end; ++j) {
        if (arg[j] == '-') arg[j] = '_';
      }
    }
    normalized_storage.push_back(std::move(arg));
    normalized_argv.push_back(normalized_storage.back().data());
  }
  absl::ParseCommandLine(static_cast<int>(normalized_argv.size()),
                         normalized_argv.data());
  const int port = absl::GetFlag(FLAGS_port);
  const std::string advertise_host = absl::GetFlag(FLAGS_advertise_host);
  const double ttl_s = absl::GetFlag(FLAGS_request_registry_ttl_s);
  const std::string ready_file = absl::GetFlag(FLAGS_ready_file);
  if (port < 0 || advertise_host.empty() || ready_file.empty()) {
    std::fprintf(stderr,
                 "usage: reshard_sidecar --port N --advertise-host HOST "
                 "--ready-file PATH [--request-registry-ttl-s S]\n");
    return 2;
  }
  if (ttl_s <= 0) {
    std::fprintf(stderr, "request_registry_ttl_s must be positive\n");
    return 2;
  }

  absl::Notification stop;
  shutdown_notification = &stop;
  signal(SIGINT, HandleSignal);
  signal(SIGTERM, HandleSignal);
  signal(SIGPIPE, SIG_IGN);

  // The sidecar hosts the reshard plane as a thin KVCacheStore capability:
  // the store owns the ReshardService; no offload backend, registry,
  // or controller submodule exists in this mode.
  auto store_or =
      tpu_raiden::kv_cache::KVCacheStore::CreateReshardSidecar(port, ttl_s);
  if (!store_or.ok()) {
    std::fprintf(stderr, "Failed to construct reshard sidecar store: %s\n",
                 std::string(store_or.status().message()).c_str());
    return 1;
  }
  tpu_raiden::kv_cache::reshard::ReshardService* service =
      (*store_or)->reshard_service();
  service->set_shutdown_callback([&stop]() {
    if (!stop.HasBeenNotified()) stop.Notify();
  });

  const int bound_port = service->port();
  const std::string rendered_host = RenderHost(advertise_host);
  const std::string address = absl::StrCat(rendered_host, ":", bound_port);
  char ttl_buf[32];
  std::snprintf(ttl_buf, sizeof(ttl_buf), "%.1f", ttl_s);
  char time_buf[40];
  std::snprintf(time_buf, sizeof(time_buf), "%.6f",
                absl::ToUnixMicros(absl::Now()) / 1e6);
  const std::string ready_line = absl::StrCat(
      "{\"schema_version\": 1, \"event\": \"raiden_controller_ready\", ",
      "\"pid\": ", getpid(), ", \"requested_port\": ", port,
      ", \"port\": ", bound_port, ", \"advertise_host\": \"", rendered_host,
      "\", \"address\": \"", address,
      "\", \"request_registry_ttl_s\": ", ttl_buf,
      ", \"time_unix_s\": ", time_buf, "}");

  // Atomic ready-file publication: temp file in the same directory, then
  // rename (the launcher polls for non-empty content).
  {
    const std::string tmp_path = absl::StrCat(ready_file, ".tmp.", getpid());
    FILE* f = std::fopen(tmp_path.c_str(), "w");
    if (f == nullptr) {
      std::fprintf(stderr, "Failed to write ready file %s\n", tmp_path.c_str());
      return 1;
    }
    std::fprintf(f, "%s\n", ready_line.c_str());
    std::fclose(f);
    if (std::rename(tmp_path.c_str(), ready_file.c_str()) != 0) {
      std::fprintf(stderr, "Failed to publish ready file %s\n",
                   ready_file.c_str());
      return 1;
    }
  }
  std::fprintf(stdout, "%s\n", ready_line.c_str());
  std::fflush(stdout);

  stop.WaitForNotification();

  service->StopServer();
  const std::string stopped_line = absl::StrCat(
      "{\"event\": \"raiden_controller_stopped\", \"pid\": ", getpid(),
      ", \"port\": ", bound_port, ", \"address\": \"", address, "\"}");
  std::fprintf(stderr, "%s\n", stopped_line.c_str());
  return 0;
}
