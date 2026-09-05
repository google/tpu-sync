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

"""C++-owned reshard client.

Drop-in for the reshard command surface of
tpu_raiden.rpc.raiden_controller.RaidenControllerClientFacade: identical
method signatures, return shapes, and exception text — the encoding,
presence rules, and transport live in C++ (kv_cache/reshard/reshard_client).
This shim only normalizes Python object shapes (duck-typed RaidenId and
span entries, pb2 variable protos) into plain values for the binding, and
parses get_metadata payloads back into pb2 objects so callers see the
facade's return type.

The legacy relay surface (register_transfer_schedule and dense broadcast) is
deliberately not provided here and remains on the Python facade.
"""

import typing
from typing import Any, Optional

from tpu_sync.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()

# pylint: disable=g-import-not-at-top
from tpu_sync.api.torch import torch_abi

_impl = torch_abi.load_extension(
    "tpu_sync.frameworks.torch",
    "_tpu_raiden_torch",
)
# pylint: enable=g-import-not-at-top

# Capability probe for the vLLM connector: True iff this client (and the
# native extension behind it) accepts the per-tag dst_skip_bytes clip on
# coordinate_transfer. Absent on older shims, so consumers must getattr.
SUPPORTS_DST_SKIP_BYTES = bool(
    getattr(_impl, "reshard_client_supports_dst_skip_bytes", False))
# True iff get_request_block_status (read-only registry lifecycle probe)
# is available; absent on older shims, so consumers must getattr.
SUPPORTS_REQUEST_BLOCK_STATUS = bool(
    getattr(_impl, "reshard_client_supports_request_block_status", False)
)

# GetRequestBlockStatusResponse.Status values returned by
# get_request_block_status.
REQUEST_BLOCK_STATUS_UNKNOWN = 1
REQUEST_BLOCK_STATUS_REGISTERED = 2
REQUEST_BLOCK_STATUS_CLAIMED = 3
REQUEST_BLOCK_STATUS_CANCELLED = 4


def _unit_tuple(unit: Any) -> tuple:
  """Duck-typed: accepts the rpc dataclass, the bound RaidenId, or alikes."""
  return (
      str(getattr(unit, "job_name", "")),
      str(getattr(unit, "job_replica_id", "")),
      str(getattr(unit, "data_name", "")),
      int(getattr(unit, "data_replica_idx", 0)),
  )


def _dst_unit_ordinal(span: Any) -> int:
  """-1 = every destination (absent on the wire); >= 0 = one destination."""
  value = getattr(span, "dst_unit_ordinal", None)
  return -1 if value is None else int(value)


def _span_entry_dict(entry: Any) -> dict:
  return {
      "tag": str(entry.tag),
      "block_ids": [int(block_id) for block_id in entry.block_ids],
      "declared_bytes": int(entry.declared_bytes),
      "dst_space_version": int(getattr(entry, "dst_space_version", 0)),
      "spans": [
          (
              int(span.src_block_ordinal),
              int(span.src_offset_bytes),
              int(span.dst_block_index),
              int(span.dst_offset_bytes),
              int(span.size_bytes),
              int(span.src_stride_bytes),
              int(span.dst_stride_bytes),
              int(span.count),
              _dst_unit_ordinal(span),
          )
          for span in entry.spans
      ],
  }


def _pool_dict(pool: Any) -> dict:
  """Mirrors _coerce_pool_spec_proto's mapping/dataclass tolerance."""
  if isinstance(pool, typing.Mapping):
    get = pool.get
  else:
    get = lambda name, default=None: getattr(pool, name, default)  # noqa: E731
  regions = []
  for region in get("regions", ()) or ():
    if isinstance(region, typing.Mapping):
      rget = region.get
    else:
      rget = lambda name, default=None, r=region: getattr(r, name, default)  # noqa: E731
    regions.append({
        "name": str(rget("name", "")),
        "offset_bytes": int(rget("offset_bytes", 0)),
        "stride_bytes": int(rget("stride_bytes", 0)),
        "unit_bytes": int(rget("unit_bytes", 0)),
        "num_units": int(rget("num_units", 0)),
        "units_per_stride": int(rget("units_per_stride", 1)),
    })
  return {
      "tag": str(get("tag", "")),
      "storage_index": int(get("storage_index", 0)),
      "base_offset_bytes": int(get("base_offset_bytes", 0)),
      "block_stride_bytes": int(get("block_stride_bytes", 0)),
      "num_blocks": int(get("num_blocks", 0)),
      "dtype_tag": str(get("dtype_tag", "")),
      "regions": regions,
  }


class ReshardClient:
  """Facade-compatible reshard client backed by the C++ implementation."""

  def __init__(self, controller_address: str, **_unused: Any):
    # name_resolver/proto_module kwargs of the Python facade are accepted
    # and ignored: name resolution is the transport's job in C++, and the
    # proto modules are fixed.
    self._address = controller_address
    self._impl = _impl.ReshardClient(controller_address=controller_address)

  def register_work_unit(
      self,
      unit: Any,
      shards: list,
      control_plane_rpc_address: Optional[str] = None,
      mesh_shape: Optional[typing.Sequence[int]] = None,
      layout: Optional[typing.Sequence[int]] = None,
      global_shape: Optional[typing.Sequence[int]] = None,
      itemsize: Optional[int] = None,
      pool_manifest: Optional[typing.Sequence[Any]] = None,
      layout_fingerprint: Optional[str] = None,
      page_tokens: Optional[int] = None,
      transfer_parallelism: Optional[int] = None,
      transfer_rank: Optional[int] = None,
      variables: Optional[typing.Sequence[Any]] = None,
  ) -> None:
    self._impl.register_work_unit(
        _unit_tuple(unit),
        [str(s) for s in shards],
        control_plane_rpc_address if control_plane_rpc_address else "",
        [int(v) for v in mesh_shape] if mesh_shape else [],
        [int(v) for v in layout] if layout else [],
        [int(v) for v in global_shape] if global_shape else [],
        int(itemsize) if itemsize else 0,
        pool_manifest is not None,
        [_pool_dict(p) for p in (pool_manifest or ())],
        layout_fingerprint,
        int(page_tokens) if page_tokens is not None else None,
        int(transfer_parallelism)
        if transfer_parallelism is not None else None,
        int(transfer_rank) if transfer_rank is not None else None,
        variables is not None,
        [v.SerializeToString() for v in (variables or ())],
    )

  def register_request_blocks(
      self,
      req_id: str,
      uuid: int,
      unit: Any,
      block_ids: typing.Sequence[int],
      pool_spans: typing.Sequence[Any] = (),
  ) -> None:
    self._impl.register_request_blocks(
        str(req_id), int(uuid), _unit_tuple(unit),
        [int(b) for b in block_ids],
        [_span_entry_dict(entry) for entry in pool_spans])

  def release_request_blocks(self, req_id: str, uuid: int) -> None:
    self._impl.release_request_blocks(str(req_id), int(uuid))

  def complete_request_blocks(self, req_id: str, uuid: int,
                              unit: Any) -> None:
    self._impl.complete_request_blocks(str(req_id), int(uuid),
                                       _unit_tuple(unit))

  def cancel_request_blocks_if_unclaimed(self, req_id: str,
                                         uuid: int) -> bool:
    return bool(
        self._impl.cancel_request_blocks_if_unclaimed(str(req_id), int(uuid)))

  def get_request_block_status(
      self, keys: typing.Sequence[tuple[str, int]]
  ) -> list[int]:
    """Read-only registry lifecycle probe: one REQUEST_BLOCK_STATUS_* value

    per (req_id, uuid) key, in order.
    """
    return [
        int(status)
        for status in self._impl.get_request_block_status(
            [(str(req_id), int(uuid)) for req_id, uuid in keys]
        )
    ]

  def coordinate_transfer(
      self,
      src_units: list,
      dst_units: list,
      req_id: Optional[str] = None,
      use_block_chunks: bool = False,
      is_sender: bool = True,
      expected_block_count: int = 0,
      uuid: int = 0,
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
      shard_push_schedules: Optional[dict] = None,
      dst_mem_type: Any = None,
      dst_device_block_ids: Optional[typing.Sequence[int]] = None,
      num_tokens: Optional[int] = None,
      transfer_pool_tags: Optional[typing.Sequence[str]] = None,
      dst_block_counts: Optional[typing.Sequence[int]] = None,
      dst_skip_bytes: Optional[typing.Sequence[int]] = None,
  ) -> bool:
    # num_tokens accepted for caller compatibility, retired from the wire
    # (facade parity); shard_push_schedules is a legacy-relay parameter the
    # pool path never sets.
    del num_tokens
    if shard_push_schedules:
      raise RuntimeError(
          "shard_push_schedules is a legacy relay parameter; the C++ "
          "reshard client serves the pool path only")
    if dst_mem_type is None:
      mem_type = 1  # RaidenMemoryType.DRAM, the facade default
    else:
      mem_type = int(dst_mem_type)
    return bool(
        self._impl.start_transfer(
            [_unit_tuple(u) for u in src_units],
            [_unit_tuple(u) for u in dst_units],
            req_id if req_id else "",
            bool(use_block_chunks),
            bool(is_sender),
            int(expected_block_count),
            int(uuid),
            dst_controller_address if dst_controller_address else "",
            src_controller_address if src_controller_address else "",
            mem_type,
            dst_device_block_ids is not None,
            [int(b) for b in (dst_device_block_ids or ())],
            transfer_pool_tags is not None,
            [str(t) for t in (transfer_pool_tags or ())],
            [int(c) for c in (dst_block_counts or ())]
            if dst_block_counts else [],
            [int(s) for s in (dst_skip_bytes or ())]
            if dst_skip_bytes and any(int(s) != 0 for s in dst_skip_bytes)
            else [],
        ))

  def start_transfer(self, *args, **kwargs) -> bool:
    """Alias for coordinate_transfer for backward compatibility."""
    return self.coordinate_transfer(*args, **kwargs)

  def get_transfer_status(self, req_id: str, uuid: int = 0) -> int:
    return int(self._impl.get_transfer_status(str(req_id), int(uuid)))

  def get_metadata(self) -> list:
    from tpu_sync.rpc import raiden_service_pb2  # pylint: disable=g-import-not-at-top

    out = []
    for payload in self._impl.get_metadata():
      metadata = raiden_service_pb2.RegisterWorkUnitRequest()
      metadata.ParseFromString(payload)
      out.append(metadata)
    return out

  def shutdown(self) -> bool:
    return bool(self._impl.shutdown())
