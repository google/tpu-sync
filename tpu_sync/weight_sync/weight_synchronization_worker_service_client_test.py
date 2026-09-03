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

"""Unit tests for Python WeightSynchronizationWorkerServiceClient."""

import concurrent.futures
from absl.testing import absltest
import grpc
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.rpc import raiden_service_pb2_grpc
from tpu_sync.weight_sync import weight_synchronization_worker_service_client


class FakeWeightSynchronizationWorkerServicer(
    raiden_service_pb2_grpc.WeightSynchronizationWorkerServiceServicer
):
  """Fake gRPC servicer to test client RPC interactions."""

  def __init__(self):
    self.received_requests: list[raiden_service_pb2.ControlRequest] = []
    self.should_fail = False

  def HandleControl(
      self,
      request: raiden_service_pb2.ControlRequest,
      context: grpc.ServicerContext,
  ) -> raiden_service_pb2.ControlResponse:
    self.received_requests.append(request)
    if self.should_fail:
      context.abort(grpc.StatusCode.INTERNAL, "Simulated RPC failure")

    response = raiden_service_pb2.ControlResponse(
        success=True,
        message="OK",
    )
    if (
        request.command
        == raiden_service_pb2.ControlRequest.COMMAND_GET_METADATA
    ):
      response.get_metadata_response.SetInParent()
    return response


class WeightSynchronizationWorkerServiceClientTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._servicer = FakeWeightSynchronizationWorkerServicer()
    self._server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=2)
    )
    raiden_service_pb2_grpc.add_WeightSynchronizationWorkerServiceServicer_to_server(
        self._servicer, self._server
    )
    self._port = self._server.add_insecure_port("[::]:0")
    self._server.start()
    self._target = f"localhost:{self._port}"

  def tearDown(self):
    self._server.stop(0)
    super().tearDown()

  def test_init_validation(self):
    with self.assertRaises(ValueError):
      weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient()

    # Test initialization with target
    client = weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient(
        target=self._target
    )
    self.assertIsNotNone(client.channel)
    self.assertIsNotNone(client.stub)
    client.close()

    # Test initialization with existing channel
    channel = grpc.insecure_channel(self._target)
    client = weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient(
        channel=channel
    )
    self.assertEqual(client.channel, channel)
    channel.close()

    # Test initialization with existing stub
    channel = grpc.insecure_channel(self._target)
    stub = raiden_service_pb2_grpc.WeightSynchronizationWorkerServiceStub(
        channel
    )
    client = weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient(
        stub=stub
    )
    self.assertEqual(client.stub, stub)
    channel.close()

  def test_context_manager(self):
    with weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient(
        target=self._target
    ) as client:
      self.assertIsNotNone(client.channel)

  def test_handle_control_success(self):
    client = weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient(
        target=self._target
    )
    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_GET_METADATA
    )
    future = client.handle_control(req, timeout=10.0)
    resp = future.result()
    self.assertTrue(resp.success)
    self.assertEqual(resp.message, "OK")
    self.assertLen(self._servicer.received_requests, 1)
    self.assertEqual(
        self._servicer.received_requests[0].command,
        raiden_service_pb2.ControlRequest.COMMAND_GET_METADATA,
    )
    client.close()

  def test_start_transfer_convenience_method(self):
    client = weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient(
        target=self._target
    )
    start_transfer_req = raiden_service_pb2.StartTransferRequest(
        uuid=12345,
        is_sender=True,
    )
    peers = ["127.0.0.1:8001", "127.0.0.1:8002"]
    future = client.start_transfer(
        start_transfer_req, peers=peers, timeout=10.0
    )
    resp = future.result()
    self.assertTrue(resp.success)

    self.assertLen(self._servicer.received_requests, 1)
    received = self._servicer.received_requests[0]
    self.assertEqual(
        received.command,
        raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
    )
    self.assertEqual(received.start_transfer_request.uuid, 12345)
    self.assertTrue(received.start_transfer_request.is_sender)
    self.assertEqual(list(received.peers), peers)
    client.close()

  def test_shutdown_convenience_method(self):
    client = weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient(
        target=self._target
    )
    future = client.shutdown(timeout=10.0)
    resp = future.result()
    self.assertTrue(resp.success)

    self.assertLen(self._servicer.received_requests, 1)
    received = self._servicer.received_requests[0]
    self.assertEqual(
        received.command,
        raiden_service_pb2.ControlRequest.COMMAND_SHUTDOWN,
    )
    client.close()

  def test_rpc_error_propagation(self):
    self._servicer.should_fail = True
    client = weight_synchronization_worker_service_client.WeightSynchronizationWorkerServiceClient(
        target=self._target
    )
    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_SHUTDOWN
    )
    future = client.handle_control(req, timeout=10.0)
    with self.assertRaises(grpc.RpcError) as exc_info:
      future.result()

    self.assertEqual(exc_info.exception.code(), grpc.StatusCode.INTERNAL)
    client.close()


if __name__ == "__main__":
  absltest.main()
