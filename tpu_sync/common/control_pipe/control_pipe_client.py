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

"""Unified asyncio-native ControlPipeClient for Python controller and worker RPCs."""

import asyncio
from concurrent import futures
import enum
import os
import socket
import struct
import threading
from typing import Any, Callable, Optional, Type, TypeVar

from google.protobuf import message as proto_message
import grpc

from tpu_sync.proto import control_pipe_pb2
from tpu_sync.proto import control_pipe_pb2_grpc

ReqT = TypeVar("ReqT", bound=proto_message.Message)
RespT = TypeVar("RespT", bound=proto_message.Message)

CPIP_MAGIC = b"CPIP"
PIPC_MAGIC = b"PIPC"
DEFAULT_MAX_FRAME_BYTES = 64 * 1024 * 1024  # 64 MiB
_HEADER_STRUCT = struct.Struct("!4sI")
_GRPC_CHANNEL_OPTIONS = (
    ("grpc.max_send_message_length", DEFAULT_MAX_FRAME_BYTES),
    ("grpc.max_receive_message_length", DEFAULT_MAX_FRAME_BYTES),
)


class ControlPipeBackendType(enum.Enum):
  """Transport backend type for ControlPipe communication."""

  TCP = "tcp"
  GRPC = "grpc"


def resolve_control_pipe_backend(
    override: Optional[ControlPipeBackendType] = None,
) -> ControlPipeBackendType:
  """Resolves the active ControlPipe backend from override or environment.

  Args:
    override: Optional programmatic backend override.

  Returns:
    The resolved ControlPipeBackendType enum value.
  """
  if override is not None:
    return override
  env_val = os.environ.get("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "").lower()
  if env_val == "grpc":
    return ControlPipeBackendType.GRPC
  if env_val == "tcp":
    return ControlPipeBackendType.TCP
  env_flag = os.environ.get("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "").lower()
  if env_flag in ("1", "true"):
    return ControlPipeBackendType.GRPC
  return ControlPipeBackendType.TCP


def _default_connect_socket(
    endpoint: str, timeout: float = 60.0
) -> socket.socket:
  """Connects a TCP socket to an IPv4 or IPv6 host:port endpoint."""
  rindex = endpoint.rfind(":")
  if rindex == -1:
    raise ValueError(f"Invalid endpoint format (missing port): {endpoint}")
  host = endpoint[:rindex]
  port = int(endpoint[rindex + 1 :])
  if host.startswith("[") and host.endswith("]"):
    host = host[1:-1]

  last_err: Optional[Exception] = None
  for res in socket.getaddrinfo(
      host, port, socket.AF_UNSPEC, socket.SOCK_STREAM
  ):
    af, socktype, proto, _, sa = res
    sock: Optional[socket.socket] = None
    try:
      sock = socket.socket(af, socktype, proto)
      sock.settimeout(timeout)
      sock.connect(sa)
      return sock
    except OSError as err:
      last_err = err
      if sock is not None:
        sock.close()
  raise RuntimeError(
      f"Failed to connect to TCP endpoint {endpoint}: {last_err}"
  )


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
  """Reads exactly num_bytes from sock or raises RuntimeError on EOF."""
  buf = bytearray()
  while len(buf) < num_bytes:
    chunk = sock.recv(num_bytes - len(buf))
    if not chunk:
      raise RuntimeError(
          "Remote peer closed connection while reading frame bytes"
      )
    buf.extend(chunk)
  return bytes(buf)


class ControlPipeClient:
  """Asyncio-native ControlPipe client supporting TCP and gRPC."""

  def __init__(
      self,
      backend: Optional[ControlPipeBackendType] = None,
      use_legacy_tcp_framing: bool = False,
      max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
      name_resolver: Optional[Any] = None,
      socket_connector: Optional[Callable[[str, float], socket.socket]] = None,
      executor: Optional[futures.Executor] = None,
  ) -> None:
    """Initializes ControlPipeClient.

    Args:
      backend: Explicit backend override, or None to resolve from environment.
      use_legacy_tcp_framing: Whether to use raw 4B length-prefixed framing on
        TCP instead of CPIP envelope headers.
      max_frame_bytes: Maximum allowed frame size in bytes.
      name_resolver: Optional coordinate name resolver.
      socket_connector: Optional custom socket connection factory.
      executor: Optional thread pool executor for asynchronous TCP I/O.
    """
    self._backend_override = backend
    self._use_legacy_tcp_framing = use_legacy_tcp_framing
    self._max_frame_bytes = max_frame_bytes
    self._name_resolver = name_resolver
    self._socket_connector = socket_connector or _default_connect_socket
    self._executor = executor
    self._lock = threading.Lock()
    self._next_req_id = 1
    self._aio_channels: dict[str, grpc.aio.Channel] = {}
    self._aio_stubs: dict[str, control_pipe_pb2_grpc.ControlPipeServiceStub] = (
        {}
    )

  @property
  def backend(self) -> ControlPipeBackendType:
    """Returns the currently active ControlPipeBackendType."""
    return resolve_control_pipe_backend(self._backend_override)

  def _allocate_req_id(self) -> int:
    """Allocates a monotonically increasing request ID."""
    with self._lock:
      req_id = self._next_req_id
      self._next_req_id += 1
      return req_id

  def _resolve_address(self, endpoint: str) -> str:
    """Resolves endpoint string via name_resolver if configured."""
    if self._name_resolver is not None:
      try:
        return str(self._name_resolver.resolve(endpoint))
      except Exception:  # pylint: disable=broad-except
        return endpoint
    return endpoint

  def _get_or_create_aio_stub(
      self, endpoint: str
  ) -> control_pipe_pb2_grpc.ControlPipeServiceStub:
    """Returns a cached or newly created async gRPC stub for endpoint."""
    resolved = self._resolve_address(endpoint)
    with self._lock:
      stub = self._aio_stubs.get(resolved)
      if stub is None:
        channel = grpc.aio.insecure_channel(
            resolved, options=_GRPC_CHANNEL_OPTIONS
        )
        stub = control_pipe_pb2_grpc.ControlPipeServiceStub(channel)
        self._aio_channels[resolved] = channel
        self._aio_stubs[resolved] = stub
      return stub

  async def call(
      self,
      endpoint: str,
      request: ReqT,
      response_cls: Type[RespT],
      timeout: float = 600.0,
  ) -> RespT:
    """Dispatches a typed request proto asynchronously and returns response."""
    resp_bytes = await self.send_raw_bytes(
        endpoint,
        request.SerializeToString(),
        timeout=timeout,
        message_type=request.DESCRIPTOR.full_name,
    )
    response = response_cls()
    response.ParseFromString(resp_bytes)
    return response

  async def send_raw_bytes(
      self,
      endpoint: str,
      payload: bytes,
      timeout: float = 600.0,
      message_type: str = "tpu_sync.rpc.ControlRequest",
  ) -> bytes:
    """Dispatches raw payload bytes asynchronously and returns response bytes."""
    if self.backend == ControlPipeBackendType.GRPC:
      return await self._send_raw_grpc_async(
          endpoint, payload, timeout, message_type
      )

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        self._executor,
        self._send_raw_tcp_sync,
        endpoint,
        payload,
        timeout,
        message_type,
    )

  async def _send_raw_grpc_async(
      self,
      endpoint: str,
      payload: bytes,
      timeout: float,
      message_type: str,
  ) -> bytes:
    """Asynchronously dispatches raw request bytes over gRPC ControlPipeService."""
    try:
      stub = self._get_or_create_aio_stub(endpoint)
      env = control_pipe_pb2.ControlEnvelope(
          message_type=message_type,
          request_id=self._allocate_req_id(),
          payload=payload,
      )
      resp_env: control_pipe_pb2.ControlResponseEnvelope = (
          await stub.SendControl(env, timeout=timeout)
      )
    except grpc.RpcError as err:
      raise RuntimeError(
          f"gRPC ControlPipe call to {endpoint} failed: {err}"
      ) from err

    if resp_env.status_code != 0:
      raise RuntimeError(
          f"Remote ControlPipe error (code={resp_env.status_code}): "
          f"{resp_env.error_message}"
      )
    return resp_env.payload

  def _send_raw_tcp_sync(
      self,
      endpoint: str,
      payload: bytes,
      timeout: float,
      message_type: str,
  ) -> bytes:
    """Dispatches raw request bytes over TCP with CPIP or legacy framing."""
    target = (
        self._resolve_address(endpoint)
        if self._socket_connector is _default_connect_socket
        else endpoint
    )
    sock = self._socket_connector(target, timeout)
    try:
      if self._use_legacy_tcp_framing:
        sock.sendall(len(payload).to_bytes(4, "big") + payload)
        resp_len_bytes = _recv_exact(sock, 4)
        resp_len = int.from_bytes(resp_len_bytes, "big")
        return _recv_exact(sock, resp_len)

      env = control_pipe_pb2.ControlEnvelope(
          message_type=message_type,
          request_id=self._allocate_req_id(),
          payload=payload,
      )
      env_bytes = env.SerializeToString()
      header = _HEADER_STRUCT.pack(CPIP_MAGIC, len(env_bytes))
      sock.sendall(header + env_bytes)

      resp_hdr = _recv_exact(sock, _HEADER_STRUCT.size)
      magic, resp_len = _HEADER_STRUCT.unpack(resp_hdr)
      if magic != PIPC_MAGIC:
        raise RuntimeError(f"Invalid ControlPipe response magic: {magic!r}")
      resp_env_bytes = _recv_exact(sock, resp_len)
      resp_env = control_pipe_pb2.ControlResponseEnvelope()
      resp_env.ParseFromString(resp_env_bytes)
      if resp_env.status_code != 0:
        raise RuntimeError(
            f"Remote ControlPipe error (code={resp_env.status_code}): "
            f"{resp_env.error_message}"
        )
      return resp_env.payload
    finally:
      sock.close()

  async def aclose(self) -> None:
    """Asynchronously closes all cached grpc.aio channels."""
    with self._lock:
      channels = list(self._aio_channels.values())
      self._aio_channels.clear()
      self._aio_stubs.clear()
    for channel in channels:
      await channel.close()

  def close(self) -> None:
    """Clears cached client state synchronously."""
    with self._lock:
      self._aio_channels.clear()
      self._aio_stubs.clear()
