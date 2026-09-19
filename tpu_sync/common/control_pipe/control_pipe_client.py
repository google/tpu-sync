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
import time
from typing import Any, Callable, Optional, Type, TypeVar

from google.protobuf import message as proto_message

from tpu_sync.proto import control_pipe_pb2

try:
  import grpc  # pylint: disable=g-import-not-at-top
except ImportError:
  grpc = None  # type: ignore[assignment]

try:
  import zmq  # pylint: disable=g-import-not-at-top
  import zmq.asyncio  # pylint: disable=g-import-not-at-top
except ImportError:
  zmq = None  # type: ignore[assignment]

try:
  from tpu_sync.proto import control_pipe_pb2_grpc  # pylint: disable=g-import-not-at-top
except ImportError:
  control_pipe_pb2_grpc = None  # type: ignore[assignment]

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
  ZMQ = "zmq"


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
  if env_val == "zmq":
    return ControlPipeBackendType.ZMQ
  if env_val == "tcp":
    return ControlPipeBackendType.TCP
  env_flag = os.environ.get("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "").lower()
  if env_flag in ("1", "true"):
    return ControlPipeBackendType.GRPC
  return ControlPipeBackendType.TCP


def _format_zmq_endpoint(endpoint: str) -> str:
  """Formats an endpoint string as a valid ZeroMQ URI."""
  if (
      endpoint.startswith("tcp://")
      or endpoint.startswith("inproc://")
      or endpoint.startswith("ipc://")
  ):
    return endpoint
  return f"tcp://{endpoint}"


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
  """Asyncio-native ControlPipe client supporting TCP, gRPC, and ZMQ."""

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
    self._aio_channels: dict[str, Any] = {}
    self._aio_stubs: dict[str, Any] = {}
    self._legacy_tcp_endpoints: set[str] = set()
    self._zmq_aio_ctx: Optional[Any] = None

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

  def _get_or_create_aio_stub(self, endpoint: str) -> Any:
    """Returns a cached or newly created async gRPC stub for endpoint."""
    if grpc is None or control_pipe_pb2_grpc is None:
      raise RuntimeError("gRPC is required for ControlPipeBackendType.GRPC")
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

  def call_sync(
      self,
      endpoint: str,
      request: ReqT,
      response_cls: Type[RespT],
      timeout: float = 600.0,
  ) -> RespT:
    """Dispatches a typed request proto synchronously and returns response."""
    resp_bytes = self.send_raw_bytes_sync(
        endpoint,
        request.SerializeToString(),
        timeout=timeout,
        message_type=request.DESCRIPTOR.full_name,
    )
    response = response_cls()
    response.ParseFromString(resp_bytes)
    return response

  def send_raw_bytes_sync(
      self,
      endpoint: str,
      payload: bytes,
      timeout: float = 600.0,
      message_type: str = "tpu_sync.rpc.ControlRequest",
  ) -> bytes:
    """Dispatches raw payload bytes synchronously and returns response bytes."""
    if self.backend in (
        ControlPipeBackendType.GRPC,
        ControlPipeBackendType.ZMQ,
    ):
      return asyncio.run(
          self.send_raw_bytes(
              endpoint, payload, timeout=timeout, message_type=message_type
          )
      )
    return self._send_raw_tcp_sync(endpoint, payload, timeout, message_type)

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

    if self.backend == ControlPipeBackendType.ZMQ:
      return await self._send_raw_zmq_async(
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
    if grpc is None or control_pipe_pb2_grpc is None:
      raise RuntimeError("gRPC is required for ControlPipeBackendType.GRPC")
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

  async def _send_raw_zmq_async(
      self,
      endpoint: str,
      payload: bytes,
      timeout: float,
      message_type: str,
  ) -> bytes:
    """Asynchronously dispatches raw request bytes over ZeroMQ."""
    if zmq is None:
      raise RuntimeError("pyzmq is required for ControlPipeBackendType.ZMQ")
    resolved = self._resolve_address(endpoint)
    with self._lock:
      if self._zmq_aio_ctx is None:
        self._zmq_aio_ctx = zmq.asyncio.Context()
      ctx = self._zmq_aio_ctx

    max_bytes = self._max_frame_bytes
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.MAXMSGSIZE, max_bytes)
    timeout_ms = int(timeout * 1000) if timeout > 0 else -1
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    try:
      sock.connect(_format_zmq_endpoint(resolved))
      env = control_pipe_pb2.ControlEnvelope(
          message_type=message_type,
          request_id=self._allocate_req_id(),
          payload=payload,
      )
      env_bytes = env.SerializeToString()
      if len(env_bytes) > max_bytes:
        raise RuntimeError(
            f"ZMQ frame size ({len(env_bytes)}) exceeds max_frame_bytes "
            f"({max_bytes})"
        )
      wait_timeout = timeout if timeout and timeout > 0 else None
      await asyncio.wait_for(sock.send(env_bytes), timeout=wait_timeout)
      resp_bytes = await asyncio.wait_for(sock.recv(), timeout=wait_timeout)
    except (zmq.ZMQError, asyncio.TimeoutError) as err:
      raise RuntimeError(
          f"ZMQ ControlPipe call to {endpoint} failed: {err}"
      ) from err
    finally:
      sock.close(linger=0)

    resp_env = control_pipe_pb2.ControlResponseEnvelope()
    resp_env.ParseFromString(resp_bytes)
    if resp_env.status_code != 0:
      raise RuntimeError(
          f"Remote ControlPipe error (code={resp_env.status_code}): "
          f"{resp_env.error_message}"
      )
    return resp_env.payload

  def _connect_tcp(
      self, endpoint: str, timeout: float
  ) -> tuple[str, socket.socket]:
    """Resolves endpoint and connects TCP socket, retrying until timeout."""
    if self._socket_connector is not _default_connect_socket:
      return endpoint, self._socket_connector(endpoint, timeout)

    start_time = time.monotonic()
    last_err: Optional[Exception] = None
    target = endpoint
    while True:
      target = self._resolve_address(endpoint)
      elapsed = time.monotonic() - start_time
      remaining = timeout - elapsed if timeout > 0 else 10.0
      attempt_timeout = min(10.0, max(0.5, remaining))
      try:
        sock = _default_connect_socket(target, timeout=attempt_timeout)
        sock.settimeout(timeout if timeout > 0 else None)
        return target, sock
      except (OSError, ValueError, RuntimeError) as err:
        last_err = err

      if timeout <= 0 or (time.monotonic() - start_time) >= timeout:
        raise RuntimeError(
            f"Failed to connect to TCP endpoint {target}: {last_err}"
        ) from last_err
      sleep_s = min(1.0, max(0.05, timeout - (time.monotonic() - start_time)))
      time.sleep(sleep_s)

  def _send_legacy_tcp_frame(
      self, endpoint: str, payload: bytes, timeout: float
  ) -> bytes:
    """Sends a raw 4B big-endian length-prefixed frame over TCP."""
    _, sock = self._connect_tcp(endpoint, timeout)
    try:
      sock.sendall(len(payload).to_bytes(4, "big") + payload)
      resp_len_bytes = _recv_exact(sock, 4)
      resp_len = int.from_bytes(resp_len_bytes, "big")
      if resp_len > self._max_frame_bytes:
        raise RuntimeError(
            f"Response frame size ({resp_len}) exceeds max_frame_bytes "
            f"({self._max_frame_bytes})"
        )
      return _recv_exact(sock, resp_len)
    finally:
      sock.close()

  def _send_raw_tcp_sync(
      self,
      endpoint: str,
      payload: bytes,
      timeout: float,
      message_type: str,
  ) -> bytes:
    """Dispatches raw request bytes over TCP with CPIP or legacy framing."""
    with self._lock:
      is_known_legacy = endpoint in self._legacy_tcp_endpoints

    if self._use_legacy_tcp_framing or is_known_legacy:
      return self._send_legacy_tcp_frame(endpoint, payload, timeout)

    _, sock = self._connect_tcp(endpoint, timeout)
    try:
      env = control_pipe_pb2.ControlEnvelope(
          message_type=message_type,
          request_id=self._allocate_req_id(),
          payload=payload,
      )
      env_bytes = env.SerializeToString()
      header = _HEADER_STRUCT.pack(CPIP_MAGIC, len(env_bytes))
      sock.sendall(header + env_bytes)
      if hasattr(sock, "shutdown"):
        try:
          sock.shutdown(socket.SHUT_WR)
        except OSError:
          pass

      resp_hdr = _recv_exact(sock, _HEADER_STRUCT.size)
      magic, resp_len = _HEADER_STRUCT.unpack(resp_hdr)
      if magic != PIPC_MAGIC:
        raise RuntimeError(f"Invalid ControlPipe response magic: {magic!r}")
      resp_env_bytes = _recv_exact(sock, resp_len)
      resp_env = control_pipe_pb2.ControlResponseEnvelope()
      resp_env.ParseFromString(resp_env_bytes)
    except (OSError, RuntimeError) as err:
      sock.close()
      if message_type.startswith("tpu_sync.rpc."):
        with self._lock:
          self._legacy_tcp_endpoints.add(endpoint)
        return self._send_legacy_tcp_frame(endpoint, payload, timeout)
      raise err
    finally:
      sock.close()

    if resp_env.status_code != 0:
      raise RuntimeError(
          f"Remote ControlPipe error (code={resp_env.status_code}): "
          f"{resp_env.error_message}"
      )
    return resp_env.payload

  async def aclose(self) -> None:
    """Asynchronously closes all cached grpc.aio channels and ZMQ context."""
    with self._lock:
      channels = list(self._aio_channels.values())
      self._aio_channels.clear()
      self._aio_stubs.clear()
      if self._zmq_aio_ctx is not None:
        self._zmq_aio_ctx.term()
        self._zmq_aio_ctx = None
    for channel in channels:
      await channel.close()

  def close(self) -> None:
    """Synchronously closes underlying ZMQ context and clears gRPC state."""
    with self._lock:
      self._aio_channels.clear()
      self._aio_stubs.clear()
      if self._zmq_aio_ctx is not None:
        self._zmq_aio_ctx.term()
        self._zmq_aio_ctx = None


def _create_dual_stack_server_socket(port: int) -> socket.socket:
  """Creates an IPv6 dual-stack or IPv4 listening TCP socket."""
  try:
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("::", port))
    sock.listen(128)
    return sock
  except OSError:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(128)
    return sock


class ControlPipeServer:
  """Python TCP ControlPipeServer supporting both CPIP envelope and 4B legacy framing."""

  def __init__(
      self,
      port: int,
      handler: Callable[[bytes], bytes],
      max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
  ) -> None:
    self._handler = handler
    self._max_frame_bytes = max_frame_bytes
    self._sock = _create_dual_stack_server_socket(port)
    self._port = int(self._sock.getsockname()[1]) if port == 0 else port
    self._stopped = False
    self._thread: Optional[threading.Thread] = None

  @property
  def port(self) -> int:
    """Returns the bound TCP port."""
    return self._port

  @property
  def thread(self) -> Optional[threading.Thread]:
    """Returns the background accept loop thread."""
    return self._thread

  def start(self) -> int:
    """Starts the background accept loop and returns the bound port."""
    self._thread = threading.Thread(target=self._accept_loop, daemon=True)
    self._thread.start()
    return self._port

  def stop(self) -> None:
    """Stops the accept loop and closes the listening socket."""
    self._stopped = True
    for host in ("[::1]", "127.0.0.1"):
      try:
        wake_sock = _default_connect_socket(f"{host}:{self._port}", timeout=0.5)
        wake_sock.close()
        break
      except Exception:  # pylint: disable=broad-except
        pass
    try:
      self._sock.shutdown(socket.SHUT_RDWR)
    except OSError:
      pass
    try:
      self._sock.close()
    except OSError:
      pass

  def _accept_loop(self) -> None:
    """Accepts incoming connections and spawns handler threads."""
    while not self._stopped:
      try:
        conn, _ = self._sock.accept()
        if self._stopped:
          conn.close()
          break
        threading.Thread(
            target=self._handle_conn, args=(conn,), daemon=True
        ).start()
      except OSError:
        break

  def _handle_conn(self, conn: socket.socket) -> None:
    """Processes a single CPIP or 4B legacy-framed request on conn."""
    try:
      first_four = _recv_exact(conn, 4)
      if first_four == CPIP_MAGIC:
        len_bytes = _recv_exact(conn, 4)
        env_len = int.from_bytes(len_bytes, "big")
        if env_len > self._max_frame_bytes:
          return
        env_bytes = _recv_exact(conn, env_len)
        env = control_pipe_pb2.ControlEnvelope()
        env.ParseFromString(env_bytes)
        resp_payload = self._handler(env.payload)
        resp_env = control_pipe_pb2.ControlResponseEnvelope(
            request_id=env.request_id,
            status_code=0,
            payload=resp_payload,
        )
        resp_env_bytes = resp_env.SerializeToString()
        header = _HEADER_STRUCT.pack(PIPC_MAGIC, len(resp_env_bytes))
        conn.sendall(header + resp_env_bytes)
      else:
        req_len = int.from_bytes(first_four, "big")
        if req_len > self._max_frame_bytes:
          return
        req_bytes = _recv_exact(conn, req_len)
        resp_bytes = self._handler(req_bytes)
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
    except Exception:  # pylint: disable=broad-except
      pass
    finally:
      conn.close()
