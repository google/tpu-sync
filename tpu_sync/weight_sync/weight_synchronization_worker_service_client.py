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

"""Python client for WeightSynchronizationWorkerService."""

from typing import Optional, Sequence
import grpc
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.rpc import raiden_service_pb2_grpc


class WeightSynchronizationWorkerServiceClient:
  """Client for interacting with WeightSynchronizationWorkerService."""

  def __init__(
      self,
      target: Optional[str] = None,
      *,
      channel: Optional[grpc.Channel] = None,
      stub: Optional[
          raiden_service_pb2_grpc.WeightSynchronizationWorkerServiceStub
      ] = None,
      options: Optional[Sequence[tuple[str, object]]] = None,
  ):
    """Initializes the client.

    Args:
      target: Server address string (e.g. 'localhost:50051' or 'ip:port').
      channel: Optional existing grpc.Channel to use.
      stub: Optional existing WeightSynchronizationWorkerServiceStub to use.
      options: Optional list of key-value pairs to configure the grpc channel if
        creating a new channel.
    """
    if stub is not None:
      self._stub = stub
      self._channel = channel
      self._owns_channel = False
    elif channel is not None:
      self._channel = channel
      self._stub = (
          raiden_service_pb2_grpc.WeightSynchronizationWorkerServiceStub(
              channel
          )
      )
      self._owns_channel = False
    elif target is not None:
      self._channel = grpc.insecure_channel(target, options=options)
      self._stub = (
          raiden_service_pb2_grpc.WeightSynchronizationWorkerServiceStub(
              self._channel
          )
      )
      self._owns_channel = True
    else:
      raise ValueError("One of target, channel, or stub must be provided.")

  @property
  def stub(
      self,
  ) -> raiden_service_pb2_grpc.WeightSynchronizationWorkerServiceStub:
    """Returns the underlying gRPC stub."""
    return self._stub

  @property
  def channel(self) -> Optional[grpc.Channel]:
    """Returns the underlying gRPC channel if available."""
    return self._channel

  def handle_control(
      self,
      request: raiden_service_pb2.ControlRequest,
      timeout: Optional[float] = None,
  ) -> grpc.Future:
    """Executes an asynchronous HandleControl RPC.

    Args:
      request: The ControlRequest proto to send.
      timeout: Optional timeout in seconds.

    Returns:
      grpc.Future representing the pending ControlResponse.
    """
    return self._stub.HandleControl.future(request, timeout=timeout)

  def start_transfer(
      self,
      start_transfer_request: raiden_service_pb2.StartTransferRequest,
      peers: Optional[Sequence[str]] = None,
      timeout: Optional[float] = None,
  ) -> grpc.Future:
    """Convenience helper to initiate weight transfer asynchronously.

    Args:
      start_transfer_request: StartTransferRequest proto specifying transfer
        parameters and schedules.
      peers: Optional list of destination peer endpoints.
      timeout: Optional timeout in seconds.

    Returns:
      grpc.Future representing the pending ControlResponse.
    """
    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        start_transfer_request=start_transfer_request,
        peers=peers or [],
    )
    return self.handle_control(req, timeout=timeout)

  def shutdown(self, timeout: Optional[float] = None) -> grpc.Future:
    """Convenience helper to instruct the remote worker to shut down asynchronously.

    Args:
      timeout: Optional timeout in seconds.

    Returns:
      grpc.Future representing the pending ControlResponse.
    """
    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_SHUTDOWN
    )
    return self.handle_control(req, timeout=timeout)

  def close(self) -> None:
    """Closes the underlying gRPC channel if owned by this client."""
    if self._owns_channel and self._channel is not None:
      self._channel.close()

  def __enter__(self) -> "WeightSynchronizationWorkerServiceClient":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()
