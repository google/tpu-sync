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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_LISTENER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_LISTENER_H_

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {

class WeightSynchronizerBase;

// Control-Plane Server Daemon that runs natively in C++ to accept management
// RPC commands (like PushWeights and Shutdown) directly via ControlPipeServer.
//
// The listener also serves the coordinated weight push of
// weight_transfer_worker_service.proto, which moves one weight version from S
// source workers to D destination workers with D + S control RPCs, all sent by
// the coordinator to the workers' listeners:
//   1. A PrepareWeightReceiveRequest to each destination (D RPCs, in
//      parallel), carrying the chunk counts of every source, registers the
//      chunks the destination expects and returns before any data moves.
//   2. Once every destination has answered, a StartWeightPushRequest in mode
//      `coordinated` to each source (S RPCs, in parallel) makes the source
//      stream its scheduled slices straight to the destinations' data
//      endpoints over TCP. The source answers once the destinations have
//      acknowledged every slice.
// A destination's transfer completes once all of its expected chunks have
// arrived. Apart from the PSP-TCP key exchange, which a source runs with each
// destination's listener when PSP-TCP is required, workers send each other no
// control RPCs.
//
// To learn the outcome, the coordinator sets `report_to` in each prepare, and
// each destination will report to it with a ReportWeightReceiveRequest (D more
// RPCs); reporting is not implemented yet, so the listener refuses such
// prepares with UNIMPLEMENTED. The listener also serves
// AwaitWeightReceiveRequest, with which a direct-push source will learn that a
// destination committed; awaiting is not implemented yet either, so the
// listener answers every valid request with UNIMPLEMENTED.
//
// On the gRPC backend the listener also serves the engine's PeregrineService
// (RaidenManagerBase::peregrine_control_service()) on the same port, so one
// client channel reaches both. On any other backend that service cannot be
// served, and constructing the listener CHECK-fails if the engine requires
// PSP-TCP.
//
// Lifetime: the registered service is owned by the engine's transport, so the
// listener must be destroyed before that transport is stopped. A listener
// owned by the engine is destroyed first thing in ~WeightSynchronizerBase,
// ahead of StopTransportServer(); a standalone listener must be destroyed
// before its engine.
//
// The listener serves until its destructor returns. Consequences for an
// engine-owned listener:
// - Its handlers still run after every class derived from
//   WeightSynchronizerBase has been destroyed (that was already so before the
//   listener served PeregrineService); a handler must not rely on derived
//   state.
// - The transport is still running during the whole listener shutdown, so
//   transport receive callbacks (OnDataReceived, OnLayerDataReceived) keep
//   arriving. A derived class with state those callbacks touch must stop the
//   transport in its own destructor (StopTransportServer()).
// - Destruction waits for every in-flight handler. A handler stuck pushing to
//   an unresponsive peer can block it indefinitely: the transport's read and
//   write timeouts are unset by default, and the transport has no public
//   cancel, so nothing ends the operation early.
class WeightSynchronizerListener final {
 public:
  WeightSynchronizerListener(
      WeightSynchronizerBase* engine, int listener_port,
      std::optional<ControlPipeBackendType> backend_type = std::nullopt);
  ~WeightSynchronizerListener();

  WeightSynchronizerListener(const WeightSynchronizerListener&) = delete;
  WeightSynchronizerListener& operator=(const WeightSynchronizerListener&) =
      delete;

  int listener_port() const { return listener_port_; }
  bool is_active() const { return !stopping_.load(); }
  ControlPipeBackendType backend_type() const {
    return pipe_server_ ? pipe_server_->backend_type() : ResolveBackendType();
  }

  // Resolves the control-plane backend type from |backend_type|,
  // TPU_RAIDEN_CONTROL_PLANE_BACKEND, or TPU_RAIDEN_USE_GRPC_CONTROL_PLANE,
  // falling back to the listener default.
  static ControlPipeBackendType ResolveBackendType(
      std::optional<ControlPipeBackendType> backend_type = std::nullopt);

  // Stops the server and waits for in-flight handlers. Must not run
  // concurrently with another Shutdown() or with destruction: only the first
  // caller waits (see GrpcControlPipeServer::Stop()).
  void Shutdown();

  // Executes a single ControlRequest against |engine| and populates |resp|.
  // Invokes |shutdown_callback| if a COMMAND_SHUTDOWN request is processed.
  static void ExecuteControlRequest(
      WeightSynchronizerBase* engine,
      const ::tpu_sync::rpc::ControlRequest& req,
      ::tpu_sync::rpc::ControlResponse* resp,
      std::function<void()> shutdown_callback = nullptr);

 private:
  // Registers `handle` on the pipe server as the handler of `Request` messages
  // of up to `max_payload_bytes` bytes, logging the errors it returns.
  template <typename Request, typename Response>
  void RegisterHandler(absl::StatusOr<Response> (
                           WeightSynchronizerListener::*handle)(const Request&),
                       size_t max_payload_bytes);

  // Rejects COMMAND_START_TRANSFER once the listener is stopping; otherwise
  // runs ExecuteControlRequest().
  absl::StatusOr<::tpu_sync::rpc::ControlResponse> HandleControlRequest(
      const ::tpu_sync::rpc::ControlRequest& req);

  // Step 1 of a coordinated push on a destination worker: registers the
  // incoming transfer with the summed per-layer chunk counts. `req.version`
  // (empty = default version 0) is validated and not recorded yet, a set
  // `req.report_to` is validated and then refused as not implemented yet, and
  // the reply's fields stay empty.
  absl::StatusOr<::tpu_sync::rpc::PrepareWeightReceiveResponse>
  HandlePrepareWeightReceive(
      const ::tpu_sync::rpc::PrepareWeightReceiveRequest& req);

  // Step 2 of a weight push on a source worker: validates the request, returns
  // Unimplemented for mode `direct`, and executes `req.coordinated()` via
  // PushWeightsResharded() (ignoring valid `version` and `timeout_ms` for now).
  absl::StatusOr<::tpu_sync::rpc::StartWeightPushResponse>
  HandleStartWeightPush(const ::tpu_sync::rpc::StartWeightPushRequest& req);

  // Waits for receive `req.uuid` to commit on a destination worker (validates
  // inputs and returns Unimplemented for now).
  absl::StatusOr<::tpu_sync::rpc::AwaitWeightReceiveResponse>
  HandleAwaitWeightReceive(
      const ::tpu_sync::rpc::AwaitWeightReceiveRequest& req);

  WeightSynchronizerBase* engine_;
  int listener_port_ = 0;
  std::atomic<bool> stopping_{false};

  std::unique_ptr<ControlPipeServer> pipe_server_;
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_LISTENER_H_
