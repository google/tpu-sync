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

"""Unit tests for ControlPipeClient."""

import asyncio
from concurrent import futures
import os
import socket
import struct
import threading
import time
from unittest import mock

from absl.testing import absltest
import grpc
import zmq

from tpu_sync.common.control_pipe import control_pipe_client
from tpu_sync.proto import control_pipe_pb2
from tpu_sync.proto import control_pipe_pb2_grpc
from tpu_sync.rpc import raiden_service_pb2


class _MockControlPipeServicer(
    control_pipe_pb2_grpc.ControlPipeServiceServicer
):
  """Mock gRPC servicer for ControlPipeService."""

  def __init__(self) -> None:
    self.last_envelope: control_pipe_pb2.ControlEnvelope = (
        control_pipe_pb2.ControlEnvelope()
    )
    self.force_error_code: int = 0
    self.force_error_message: str = ""

  def SendControl(
      self,
      request: control_pipe_pb2.ControlEnvelope,
      context: grpc.ServicerContext,
  ) -> control_pipe_pb2.ControlResponseEnvelope:
    del context
    self.last_envelope = request
    if self.force_error_code != 0:
      return control_pipe_pb2.ControlResponseEnvelope(
          request_id=request.request_id,
          status_code=self.force_error_code,
          error_message=self.force_error_message,
      )

    req = raiden_service_pb2.ControlRequest()
    req.ParseFromString(request.payload)
    resp = raiden_service_pb2.ControlResponse(
        success=True,
        message=f"ACK:{req.command}",
    )
    return control_pipe_pb2.ControlResponseEnvelope(
        request_id=request.request_id,
        status_code=0,
        payload=resp.SerializeToString(),
    )


class _TcpTestServer:
  """Lightweight TCP test server supporting CPIP and legacy framing."""

  def __init__(self, use_legacy_framing: bool = False) -> None:
    self._use_legacy_framing = use_legacy_framing
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._sock.bind(("127.0.0.1", 0))
    self._sock.listen(16)
    self.port: int = self._sock.getsockname()[1]
    self._stopping = False
    self._thread = threading.Thread(target=self._serve_loop, daemon=True)
    self._thread.start()

  def _recv_exact(self, conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
      chunk = conn.recv(n - len(buf))
      if not chunk:
        raise EOFError("Client closed connection")
      buf.extend(chunk)
    return bytes(buf)

  def _serve_loop(self) -> None:
    while not self._stopping:
      try:
        conn, _ = self._sock.accept()
      except OSError:
        break
      threading.Thread(
          target=self._handle_client, args=(conn,), daemon=True
      ).start()

  def _handle_client(self, conn: socket.socket) -> None:
    try:
      if self._use_legacy_framing:
        raw_len = self._recv_exact(conn, 4)
        msg_len = int.from_bytes(raw_len, "big")
        payload = self._recv_exact(conn, msg_len)
        req = raiden_service_pb2.ControlRequest()
        req.ParseFromString(payload)
        resp = raiden_service_pb2.ControlResponse(
            success=True, message="LEGACY_OK"
        )
        resp_bytes = resp.SerializeToString()
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
        return

      hdr = self._recv_exact(conn, 8)
      magic, env_len = struct.unpack("!4sI", hdr)
      if magic != control_pipe_client.CPIP_MAGIC:
        return
      env_bytes = self._recv_exact(conn, env_len)
      env = control_pipe_pb2.ControlEnvelope()
      env.ParseFromString(env_bytes)

      req = raiden_service_pb2.ControlRequest()
      req.ParseFromString(env.payload)
      resp = raiden_service_pb2.ControlResponse(success=True, message="CPIP_OK")
      resp_env = control_pipe_pb2.ControlResponseEnvelope(
          request_id=env.request_id,
          status_code=0,
          payload=resp.SerializeToString(),
      )
      resp_env_bytes = resp_env.SerializeToString()
      resp_hdr = struct.pack(
          "!4sI", control_pipe_client.PIPC_MAGIC, len(resp_env_bytes)
      )
      conn.sendall(resp_hdr + resp_env_bytes)
    except Exception:  # pylint: disable=broad-except
      pass
    finally:
      conn.close()

  def close(self) -> None:
    self._stopping = True
    self._sock.close()
    self._thread.join(timeout=2.0)


class _ZmqTestServer:
  """Lightweight ZeroMQ ROUTER test server."""

  def __init__(self) -> None:
    self._ctx = zmq.Context()
    self._sock = self._ctx.socket(zmq.ROUTER)
    self._sock.setsockopt(zmq.LINGER, 0)
    self.port: int = self._sock.bind_to_random_port("tcp://127.0.0.1")
    self.last_envelope = control_pipe_pb2.ControlEnvelope()
    self.force_error_code: int = 0
    self.force_error_message: str = ""
    self.reply_delay: float = 0.0
    self._stopping = False
    self._thread = threading.Thread(target=self._serve_loop, daemon=True)
    self._thread.start()

  def _serve_loop(self) -> None:
    poller = zmq.Poller()
    poller.register(self._sock, zmq.POLLIN)
    while not self._stopping:
      events = dict(poller.poll(20))
      if self._sock in events:
        frames = self._sock.recv_multipart()
        if len(frames) < 2:
          continue
        if self.reply_delay > 0:
          time.sleep(self.reply_delay)
        prefix = frames[:-1]
        payload = frames[-1]
        env = control_pipe_pb2.ControlEnvelope()
        env.ParseFromString(payload)
        self.last_envelope = env
        if self.force_error_code != 0:
          resp_env = control_pipe_pb2.ControlResponseEnvelope(
              request_id=env.request_id,
              status_code=self.force_error_code,
              error_message=self.force_error_message,
          )
        else:
          req = raiden_service_pb2.ControlRequest()
          req.ParseFromString(env.payload)
          resp = raiden_service_pb2.ControlResponse(
              success=True,
              message=f"ZMQ_ACK:{req.command}",
          )
          resp_env = control_pipe_pb2.ControlResponseEnvelope(
              request_id=env.request_id,
              status_code=0,
              payload=resp.SerializeToString(),
          )
        self._sock.send_multipart(prefix + [resp_env.SerializeToString()])

  def close(self) -> None:
    self._stopping = True
    self._thread.join(timeout=2.0)
    self._sock.close()
    self._ctx.term()


class ControlPipeClientTest(absltest.TestCase):

  def test_resolve_control_pipe_backend_precedence(self) -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(
          control_pipe_client.resolve_control_pipe_backend(),
          control_pipe_client.ControlPipeBackendType.TCP,
      )
      self.assertEqual(
          control_pipe_client.resolve_control_pipe_backend(
              control_pipe_client.ControlPipeBackendType.GRPC
          ),
          control_pipe_client.ControlPipeBackendType.GRPC,
      )
      self.assertEqual(
          control_pipe_client.resolve_control_pipe_backend(
              control_pipe_client.ControlPipeBackendType.ZMQ
          ),
          control_pipe_client.ControlPipeBackendType.ZMQ,
      )

    with mock.patch.dict(
        os.environ, {"TPU_RAIDEN_CONTROL_PLANE_BACKEND": "grpc"}, clear=True
    ):
      self.assertEqual(
          control_pipe_client.resolve_control_pipe_backend(),
          control_pipe_client.ControlPipeBackendType.GRPC,
      )

    with mock.patch.dict(
        os.environ, {"TPU_RAIDEN_CONTROL_PLANE_BACKEND": "zmq"}, clear=True
    ):
      self.assertEqual(
          control_pipe_client.resolve_control_pipe_backend(),
          control_pipe_client.ControlPipeBackendType.ZMQ,
      )

    with mock.patch.dict(
        os.environ, {"TPU_RAIDEN_USE_GRPC_CONTROL_PLANE": "1"}, clear=True
    ):
      self.assertEqual(
          control_pipe_client.resolve_control_pipe_backend(),
          control_pipe_client.ControlPipeBackendType.GRPC,
      )

  def test_tcp_cpip_framing(self) -> None:
    server = _TcpTestServer(use_legacy_framing=False)
    endpoint = f"127.0.0.1:{server.port}"
    try:

      async def _run() -> None:
        client = control_pipe_client.ControlPipeClient(
            backend=control_pipe_client.ControlPipeBackendType.TCP
        )
        req = raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER
        )
        resp = await client.call(
            endpoint, req, raiden_service_pb2.ControlResponse, timeout=5.0
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.message, "CPIP_OK")
        await client.aclose()

      asyncio.run(_run())
    finally:
      server.close()

  def test_tcp_legacy_framing(self) -> None:
    server = _TcpTestServer(use_legacy_framing=True)
    endpoint = f"127.0.0.1:{server.port}"
    try:

      async def _run() -> None:
        client = control_pipe_client.ControlPipeClient(
            backend=control_pipe_client.ControlPipeBackendType.TCP,
            use_legacy_tcp_framing=True,
        )
        req = raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_SHUTDOWN
        )
        resp = await client.call(
            endpoint, req, raiden_service_pb2.ControlResponse, timeout=5.0
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.message, "LEGACY_OK")
        await client.aclose()

      asyncio.run(_run())
    finally:
      server.close()

  def test_grpc_backend(self) -> None:
    servicer = _MockControlPipeServicer()
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    control_pipe_pb2_grpc.add_ControlPipeServiceServicer_to_server(
        servicer, grpc_server
    )
    port = grpc_server.add_insecure_port("[::]:0")
    grpc_server.start()
    endpoint = f"localhost:{port}"

    try:

      async def _run() -> None:
        client = control_pipe_client.ControlPipeClient(
            backend=control_pipe_client.ControlPipeBackendType.GRPC
        )
        req = raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER
        )
        resp = await client.call(
            endpoint, req, raiden_service_pb2.ControlResponse, timeout=5.0
        )
        self.assertTrue(resp.success)
        self.assertEqual(
            servicer.last_envelope.message_type,
            "tpu_sync.rpc.ControlRequest",
        )

        # Verify error code propagation
        servicer.force_error_code = 3
        servicer.force_error_message = "invalid block count"
        with self.assertRaisesRegex(RuntimeError, "invalid block count"):
          await client.call(
              endpoint, req, raiden_service_pb2.ControlResponse, timeout=5.0
          )

        await client.aclose()

      asyncio.run(_run())
    finally:
      grpc_server.stop(grace=None)

  def test_zmq_backend(self) -> None:
    server = _ZmqTestServer()
    endpoint = f"127.0.0.1:{server.port}"
    try:

      async def _run() -> None:
        client = control_pipe_client.ControlPipeClient(
            backend=control_pipe_client.ControlPipeBackendType.ZMQ
        )
        req = raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER
        )
        resp = await client.call(
            endpoint, req, raiden_service_pb2.ControlResponse, timeout=5.0
        )
        self.assertTrue(resp.success)
        self.assertEqual(
            server.last_envelope.message_type,
            "tpu_sync.rpc.ControlRequest",
        )

        # Verify remote error code propagation
        server.force_error_code = 3
        server.force_error_message = "zmq invalid block count"
        with self.assertRaisesRegex(RuntimeError, "zmq invalid block count"):
          await client.call(
              endpoint, req, raiden_service_pb2.ControlResponse, timeout=5.0
          )

        # Verify client max_frame_bytes enforcement
        server.force_error_code = 0
        small_client = control_pipe_client.ControlPipeClient(
            backend=control_pipe_client.ControlPipeBackendType.ZMQ,
            max_frame_bytes=16,
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds max_frame_bytes"):
          await small_client.send_raw_bytes(endpoint, b"x" * 100, timeout=5.0)
        await small_client.aclose()

        # Verify async timeout handling
        server.reply_delay = 0.5
        with self.assertRaisesRegex(RuntimeError, "ZMQ ControlPipe call"):
          await client.call(
              endpoint, req, raiden_service_pb2.ControlResponse, timeout=0.1
          )

        await client.aclose()

      asyncio.run(_run())
    finally:
      server.close()


if __name__ == "__main__":
  absltest.main()
