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

#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <memory>
#include <ratio>  // NOLINT(build/c++11)
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/status.h"
#include <openssl/sha.h>
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.grpc.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

using namespace ::tpu_raiden::kv_cache::global_registry;  // NOLINT

namespace {

// RegisterStore runs synchronously inside KVCacheStore construction, so
// it needs a bound: a dead-but-routable registry must not hang construction
// indefinitely.
constexpr absl::Duration kRegisterStoreRpcTimeout = absl::Seconds(10);

void ToProto(const RaidenId& id, ::tpu_sync::rpc::RaidenIdProto* proto) {
  proto->set_job_name(id.job_name);
  proto->set_job_replica_id(id.job_replica_id);
  proto->set_data_name(id.data_name);
  proto->set_data_replica_idx(id.data_replica_idx);
}

void BuildRegisterRequest(const std::vector<Registration>& registrations,
                          RegisterRequest* request) {
  for (const auto& reg : registrations) {
    auto* entry = request->add_entries();
    entry->set_prefix_hash(reg.prefix_hash);
    auto* meta = entry->mutable_metadata();
    ToProto(reg.raiden_id, meta->mutable_raiden_id());
    meta->set_block_id(reg.block_id);
    if (reg.ttl > absl::ZeroDuration()) {
      entry->set_ttl_seconds(absl::ToInt64Seconds(reg.ttl));
    }
  }
}

void BuildUnregisterRequest(const std::vector<std::string>& prefix_hashes,
                            const RaidenId& raiden_id,
                            UnregisterRequest* request) {
  request->mutable_prefix_hashes()->Reserve(prefix_hashes.size());
  for (const auto& hash : prefix_hashes) {
    request->add_prefix_hashes(hash);
  }
  ToProto(raiden_id, request->mutable_raiden_id());
}

void RecordRpcDurationMs(absl::string_view method, absl::StatusCode code,
                         std::chrono::steady_clock::time_point start_time) {
  telemetry::RaidenMetricStore& store =
      telemetry::RaidenMetricStore::GetGlobalMetricStore();
  if (!store.HasBackends()) {
    return;
  }
  const double duration_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start_time)
          .count();
  const telemetry::MetricLabel labels[] = {
      {telemetry::metric_labels::kMethod, method},
      {telemetry::metric_labels::kStatus, absl::StatusCodeToStringView(code)},
  };
  store.ObserveHistogram(telemetry::metric_names::kGlobalRegistryRpcDurationMs,
                         labels, duration_ms);
}

class ScopedRpcTimer {
 public:
  explicit ScopedRpcTimer(absl::string_view method)
      : method_(method),
        enabled_(telemetry::RaidenMetricStore::GetGlobalMetricStore()
                     .HasBackends()),
        start_time_(enabled_ ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{}) {}

  absl::Status Record(absl::Status status) const {
    if (enabled_) {
      RecordRpcDurationMs(method_, status.code(), start_time_);
    }
    return status;
  }

  template <typename T>
  absl::StatusOr<T> Record(absl::StatusOr<T> result) const {
    if (enabled_) {
      RecordRpcDurationMs(method_, result.status().code(), start_time_);
    }
    return result;
  }

 private:
  absl::string_view method_;
  bool enabled_;
  std::chrono::steady_clock::time_point start_time_;
};

}  // namespace

GlobalRegistryClient::GlobalRegistryClient(
    std::shared_ptr<grpc::Channel> channel)
    : stub_(GlobalRegistryService::NewStub(channel)) {}

tsl::Future<> GlobalRegistryClient::RegisterAsync(
    const std::vector<Registration>& registrations, absl::Duration timeout) {
  if (registrations.empty()) {
    // Nothing to publish, so there is no RPC to make and nothing to wait on.
    return tsl::Future<>(absl::OkStatus());
  }
  const bool record_metrics =
      telemetry::RaidenMetricStore::GetGlobalMetricStore().HasBackends();
  const std::chrono::steady_clock::time_point start_time =
      record_metrics ? std::chrono::steady_clock::now()
                     : std::chrono::steady_clock::time_point{};
  if (timeout <= absl::ZeroDuration()) {
    // Already expired. Decided here, not by gRPC: a deadline of exactly "now"
    // races the round trip and on loopback the call usually wins, which would
    // make "zero" silently mean "no timeout" for the callers that matter.
    if (record_metrics) {
      RecordRpcDurationMs("Register", absl::StatusCode::kDeadlineExceeded,
                          start_time);
    }
    return tsl::Future<>(absl::DeadlineExceededError(
        "Registry call timeout is zero or negative, so it expired before it "
        "was dispatched."));
  }

  auto [promise, future] = tsl::MakePromise<>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto request = std::make_shared<RegisterRequest>();
  auto response = std::make_shared<RegisterResponse>();
  BuildRegisterRequest(registrations, request.get());
  if (timeout < absl::InfiniteDuration()) {
    context->set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  }

  stub_->async()->Register(
      context.get(), request.get(), response.get(),
      [context, request, response, stub = stub_, record_metrics, start_time,
       promise = std::move(promise).ToShared()](grpc::Status status) mutable {
        absl::Status result;
        if (!status.ok()) {
          result = absl::InternalError(status.error_message());
        } else if (!response->success()) {
          result = absl::FailedPreconditionError(response->error_message());
        } else {
          result = absl::OkStatus();
        }
        if (record_metrics) {
          RecordRpcDurationMs("Register", result.code(), start_time);
        }
        promise->Set(std::move(result));
      });
  return future;
}

absl::Status GlobalRegistryClient::Register(
    const std::vector<Registration>& registrations, absl::Duration timeout) {
  return RegisterAsync(registrations, timeout).Await();
}

absl::StatusOr<std::vector<KVBlockMetadata>> GlobalRegistryClient::Lookup(
    const std::vector<std::string>& prefix_hashes,
    const RaidenId& client_raiden_id) {
  ScopedRpcTimer timer("Lookup");
  LookupRequest request;
  request.mutable_prefix_hashes()->Reserve(prefix_hashes.size());
  for (const auto& hash : prefix_hashes) {
    request.add_prefix_hashes(hash);
  }
  if (!client_raiden_id.empty()) {
    ToProto(client_raiden_id, request.mutable_client_raiden_id());
  }

  LookupResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->Lookup(&context, request, &response);

  if (!status.ok()) {
    return timer.Record<std::vector<KVBlockMetadata>>(
        absl::InternalError(status.error_message()));
  }

  std::vector<KVBlockMetadata> result;
  result.reserve(response.results_size());
  for (const auto& meta : response.results()) {
    result.push_back(meta);
  }
  return timer.Record(absl::StatusOr<std::vector<KVBlockMetadata>>(
      std::move(result)));
}

absl::StatusOr<std::vector<GlobalRegistryClient::PulledEntry>>
GlobalRegistryClient::PullOwned(const RaidenId& raiden_id) {
  ScopedRpcTimer timer("PullOwned");
  PullOwnedRequest request;
  ToProto(raiden_id, request.mutable_raiden_id());

  grpc::ClientContext context;
  std::unique_ptr<grpc::ClientReader<PullOwnedResponse>> reader =
      stub_->PullOwned(&context, request);

  std::vector<PulledEntry> entries;
  PullOwnedResponse response;
  while (reader->Read(&response)) {
    for (const auto& entry : response.entries()) {
      entries.push_back({entry.prefix_hash(), entry.block_id(),
                         entry.remaining_ttl_seconds()});
    }
  }

  grpc::Status status = reader->Finish();
  if (!status.ok()) {
    return timer.Record<std::vector<PulledEntry>>(
        absl::InternalError(status.error_message()));
  }
  return timer.Record(
      absl::StatusOr<std::vector<PulledEntry>>(std::move(entries)));
}

tsl::Future<> GlobalRegistryClient::UnregisterAsync(
    const std::vector<std::string>& prefix_hashes, const RaidenId& raiden_id,
    absl::Duration timeout) {
  if (prefix_hashes.empty()) {
    // Nothing to withdraw, so there is no RPC to make and nothing to wait on.
    return tsl::Future<>(absl::OkStatus());
  }
  const bool record_metrics =
      telemetry::RaidenMetricStore::GetGlobalMetricStore().HasBackends();
  const std::chrono::steady_clock::time_point start_time =
      record_metrics ? std::chrono::steady_clock::now()
                     : std::chrono::steady_clock::time_point{};
  if (timeout <= absl::ZeroDuration()) {
    // Already expired. Decided here, not by gRPC: a deadline of exactly "now"
    // races the round trip and on loopback the call usually wins, which would
    // make "zero" silently mean "no timeout" for the callers that matter.
    if (record_metrics) {
      RecordRpcDurationMs("Unregister", absl::StatusCode::kDeadlineExceeded,
                          start_time);
    }
    return tsl::Future<>(absl::DeadlineExceededError(
        "Registry call timeout is zero or negative, so it expired before it "
        "was dispatched."));
  }

  auto [promise, future] = tsl::MakePromise<>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto request = std::make_shared<UnregisterRequest>();
  auto response = std::make_shared<UnregisterResponse>();
  BuildUnregisterRequest(prefix_hashes, raiden_id, request.get());
  if (timeout < absl::InfiniteDuration()) {
    context->set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  }

  stub_->async()->Unregister(
      context.get(), request.get(), response.get(),
      [context, request, response, stub = stub_, record_metrics, start_time,
       promise = std::move(promise).ToShared()](grpc::Status status) mutable {
        absl::Status result;
        if (!status.ok()) {
          result = absl::InternalError(status.error_message());
        } else if (!response->success()) {
          result = absl::FailedPreconditionError(response->error_message());
        } else {
          result = absl::OkStatus();
        }
        if (record_metrics) {
          RecordRpcDurationMs("Unregister", result.code(), start_time);
        }
        promise->Set(std::move(result));
      });
  return future;
}

absl::Status GlobalRegistryClient::Unregister(
    const std::vector<std::string>& prefix_hashes, const RaidenId& raiden_id,
    absl::Duration timeout) {
  return UnregisterAsync(prefix_hashes, raiden_id, timeout).Await();
}

absl::Status GlobalRegistryClient::RegisterStore(
    const RaidenId& raiden_id, absl::string_view store_server_address,
    absl::string_view controller_address, absl::Duration ttl,
    absl::string_view kv_pool_group, int32_t evict_tier) {
  ScopedRpcTimer timer("RegisterStore");
  RegisterStoreRequest request;
  StoreInfo* store = request.mutable_store();
  ToProto(raiden_id, store->mutable_raiden_id());
  store->set_store_server_address(std::string(store_server_address));
  store->set_controller_address(std::string(controller_address));
  if (ttl > absl::ZeroDuration()) {
    store->set_ttl_seconds(absl::ToInt64Seconds(ttl));
  }
  store->set_kv_pool_group(std::string(kv_pool_group));
  store->set_evict_tier(evict_tier);

  RegisterStoreResponse response;
  grpc::ClientContext context;
  context.set_deadline(
      absl::ToChronoTime(absl::Now() + kRegisterStoreRpcTimeout));
  grpc::Status status = stub_->RegisterStore(&context, request, &response);

  if (!status.ok()) {
    return timer.Record(absl::InternalError(status.error_message()));
  }
  if (!response.success()) {
    return timer.Record(
        absl::FailedPreconditionError(response.error_message()));
  }
  return timer.Record(absl::OkStatus());
}

absl::StatusOr<StoreInfo> GlobalRegistryClient::ResolveStore(
    const RaidenId& raiden_id) {
  ScopedRpcTimer timer("ResolveStore");
  ResolveStoreRequest request;
  ToProto(raiden_id, request.mutable_raiden_id());

  ResolveStoreResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->ResolveStore(&context, request, &response);

  if (!status.ok()) {
    return timer.Record<StoreInfo>(absl::InternalError(status.error_message()));
  }
  if (!response.found()) {
    return timer.Record<StoreInfo>(absl::NotFoundError(
        absl::StrCat("no store registered for raiden_id ", raiden_id.job_name,
                     "/", raiden_id.job_replica_id, "/", raiden_id.data_name,
                     "/", raiden_id.data_replica_idx)));
  }
  return timer.Record(absl::StatusOr<StoreInfo>(response.store()));
}

absl::Status GlobalRegistryClient::UnregisterStore(const RaidenId& raiden_id) {
  ScopedRpcTimer timer("UnregisterStore");
  UnregisterStoreRequest request;
  ToProto(raiden_id, request.mutable_raiden_id());

  UnregisterStoreResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->UnregisterStore(&context, request, &response);

  if (!status.ok()) {
    return timer.Record(absl::InternalError(status.error_message()));
  }
  // Deliberately not an error when nothing was registered: teardown must be
  // idempotent, and a store that never registered still calls this.
  return timer.Record(absl::OkStatus());
}

absl::Status GlobalRegistryClient::Heartbeat(const RaidenId& raiden_id,
                                             const StoreStatus& status) {
  ScopedRpcTimer timer("Heartbeat");
  HeartbeatRequest request;
  ToProto(raiden_id, request.mutable_raiden_id());
  *request.mutable_status() = status;

  HeartbeatResponse response;
  grpc::ClientContext context;
  context.set_deadline(
      absl::ToChronoTime(absl::Now() + kRegisterStoreRpcTimeout));
  grpc::Status rpc_status = stub_->Heartbeat(&context, request, &response);

  if (!rpc_status.ok()) {
    return timer.Record(absl::InternalError(rpc_status.error_message()));
  }
  if (!response.registered()) {
    return timer.Record(absl::NotFoundError(
        "no live store registration to refresh; RegisterStore again"));
  }
  return timer.Record(absl::OkStatus());
}

absl::StatusOr<std::vector<StoreInfo>>
GlobalRegistryClient::GetPlacementTargets(const RaidenId& raiden_id,
                                          int32_t max_targets) {
  ScopedRpcTimer timer("GetPlacementTargets");
  GetPlacementTargetsRequest request;
  ToProto(raiden_id, request.mutable_raiden_id());
  request.set_max_targets(max_targets);

  GetPlacementTargetsResponse response;
  grpc::ClientContext context;
  context.set_deadline(
      absl::ToChronoTime(absl::Now() + kRegisterStoreRpcTimeout));
  grpc::Status status =
      stub_->GetPlacementTargets(&context, request, &response);

  if (!status.ok()) {
    // Preserve the gRPC code (grpc and absl codes match 1:1): NotFound is the
    // caller's cue to RegisterStore again, not a transport failure.
    return timer.Record<std::vector<StoreInfo>>(
        absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                     status.error_message()));
  }
  // An empty list is an answer (bottom tier), not an error.
  return timer.Record(absl::StatusOr<std::vector<StoreInfo>>(
      std::vector<StoreInfo>(response.targets().begin(),
                             response.targets().end())));
}

absl::Status GlobalRegistryClient::RegisterKVTransferSpec(
    const KVTransferSpec& spec, absl::string_view kv_pool_group) {
  ScopedRpcTimer timer("RegisterKVTransferSpec");
  RegisterKVTransferSpecRequest request;
  *request.mutable_spec() = spec;
  request.set_kv_pool_group(std::string(kv_pool_group));

  RegisterKVTransferSpecResponse response;
  grpc::ClientContext context;
  grpc::Status status =
      stub_->RegisterKVTransferSpec(&context, request, &response);

  if (!status.ok()) {
    // Preserve the gRPC code (grpc and absl codes match 1:1) so callers can
    // tell retryable transport conditions apart.
    return timer.Record(
        absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                     status.error_message()));
  }
  if (!response.success()) {
    return timer.Record(absl::InvalidArgumentError(response.error_message()));
  }
  return timer.Record(absl::OkStatus());
}

absl::StatusOr<KVTransferSpec> GlobalRegistryClient::GetKVTransferSpec(
    absl::string_view kv_pool_group) {
  ScopedRpcTimer timer("GetKVTransferSpec");
  GetKVTransferSpecRequest request;
  request.set_kv_pool_group(std::string(kv_pool_group));

  GetKVTransferSpecResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->GetKVTransferSpec(&context, request, &response);

  if (!status.ok()) {
    // Preserve the gRPC code (grpc and absl codes match 1:1) so callers can
    // tell retryable transport conditions apart.
    return timer.Record<KVTransferSpec>(
        absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                     status.error_message()));
  }
  if (!response.found()) {
    return timer.Record<KVTransferSpec>(
        absl::NotFoundError("no KVTransferSpec published yet"));
  }
  return timer.Record(absl::StatusOr<KVTransferSpec>(response.spec()));
}

std::string CalculatePrefixHash(const std::vector<int64_t>& tokens,
                                absl::string_view parent_hash) {
  SHA256_CTX sha256;
  SHA256_Init(&sha256);

  if (!parent_hash.empty()) {
    SHA256_Update(&sha256, parent_hash.data(), parent_hash.size());
  }

  if (!tokens.empty()) {
    SHA256_Update(&sha256, tokens.data(), tokens.size() * sizeof(int64_t));
  }

  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256_Final(hash, &sha256);

  return absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(hash), SHA256_DIGEST_LENGTH));
}

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden
