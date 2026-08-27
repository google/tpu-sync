import enum
from typing import Any, overload

class BlockStatus(enum.Enum):
  INIT = ...
  REMOTE = ...
  HBM = ...
  HOST = ...
  # A load that sourced from local DRAM ends here; one that sourced from a
  # peer leaves no local entry at all -- its landing blocks are freed and no
  # host copy is kept.
  HOST_AND_HBM = ...
  STORAGE = ...

class RaidenId:
  job_name: str
  job_replica_id: str
  data_name: str
  data_replica_idx: int
  def __init__(
      self,
      job_name: str = '',
      job_replica_id: str = '',
      data_name: str = '',
      data_replica_idx: int = 0,
  ) -> None: ...

class RaidenBlockId:
  raiden_id: RaidenId
  # The local host block when status is HOST or HOST_AND_HBM; the OWNING
  # PEER's host block when status is REMOTE, where it means nothing locally.
  host_block_id: int
  device_block_id: int
  status: BlockStatus
  @overload
  def __init__(
      self,
      raiden_id: RaidenId = ...,
      host_block_id: int = ...,
      status: BlockStatus = ...,
  ) -> None: ...
  @overload
  def __init__(
      self,
      raiden_id: RaidenId = ...,
      host_block_id: int = ...,
      device_block_id: int = ...,
      status: BlockStatus = ...,
  ) -> None: ...

class KVCacheStore:
  def __init__(
      self,
      capacity: int,
      global_registry_address: str = '',
      raiden_id: RaidenId = ...,
      *,
      num_shards: int,
      shard_size_bytes: int = 0,
      store_server_ip: str,
      raiden_controller_port: int = 0,
      expected_worker_count: int = 0,
      kv_pool_group: str = '',
  ) -> None: ...

  @property
  def raiden_id(self) -> RaidenId: ...

  @property
  def raiden_controller_address(self) -> str: ...

  @property
  def store_server_address(self) -> str: ...

  def lookup(
      self,
      block_hashes: list[bytes],
      enable_global: bool = False,
      pin_found: bool = True,
  ) -> list[tuple[bytes, RaidenBlockId]]:
    """Checks the LRU directory for cached block hashes. Returns a list of all
    matched replica pairs prior to the first miss. Pins every local hit unless
    pin_found is False."""
    ...
  def insert(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockId],
      on_host: bool,
  ) -> bool:
    """Caches sharded buffers into host-RAM/HBM backing store, pinning them.

    All-or-nothing, and does NOT evict to make room: without enough free space
    for the new hashes it unpins whatever it already pinned and returns False.
    The pin it grants is what a subsequent save() consumes.

    LOCAL entries only: raises ValueError if any slice has status REMOTE, and
    nothing is inserted or pinned.
    """
    ...
  def capacity(self) -> int:
    """Returns the total manageable pool block capacity of the prefix cache hierarchy."""
    ...
  def release(self, block_hashes: list[bytes]) -> None:
    """Releases previously pinned block hashes, making them eligible for LRU eviction when capacity is exceeded."""
    ...
  def save(
      self,
      block_hashes: list[bytes],
      dst_raiden_id: RaidenId | None = ...,
  ) -> bool:
    """Saves blocks asynchronously, either locally or to a peer.

    Omit dst_raiden_id for a local save (HBM -> host DRAM), which requires each
    block to have status HBM; pass one to offer the blocks to that peer, which
    requires them to be host-resident here already. Poll both with
    poll_save_status.
    """
    ...
  @overload
  def load(self, block_hashes: list[bytes], device_block_ids: list[int]) -> bool:
    """Asynchronously loads KV cache blocks to device (HBM) from local host DRAM.

    LOCAL ONLY: a hash whose cached entry has status REMOTE is refused (returns
    False); the slices overload is the only way to load from a peer.

    `device_block_ids` is the destination and must name one device block per hash.

    NOTE: The block_hashes must be pinned in the LRU cache before calling load.
    A SUCCESSFUL load consumes that pin; do not release afterwards. A failed
    load leaves the pin in place for a retry or a deliberate release.
    """
    ...
  @overload
  def load(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockId],
      device_block_ids: list[int],
  ) -> bool:
    """Asynchronously loads KV cache blocks to device (HBM), either from local
    host DRAM or from a peer.

    ONE CALL IS ONE SOURCE. Every block in a batch must carry the same status,
    and remote blocks must all refer to the same peer.

    `device_block_ids` is the destination and must name one device block per hash.

    If `slices` is provided, the caller's pre-looked up RaidenBlockIds are
    used directly. A LOCAL source requires every hash to be pinned (lookup
    grants that pin) and a successful load consumes it; a REMOTE source needs
    no pin, consumes none, and records nothing locally -- it re-resolves
    hashes at the peer, using `slices` only for the source's identity.
    """
    ...
  def poll_save_status(
      self,
  ) -> tuple[list[bytes], list[bytes], list[bytes], list[bytes], list[bytes]]:
    """Polls status of asynchronous Save operations, local and remote.

    Five lists: (done, failed, pending, existing, unregistered). The last two
    annotate REMOTE failures rather than being outcomes of their own -- a hash
    in `existing` or `unregistered` also appears in `failed`, not instead of
    it -- and are empty for local saves. `existing` is what a destination
    already held when it refused a partial batch; `unregistered` is a
    destination that HAS the bytes but could not publish them, so no lookup
    will find them there.
    """
    ...
  def poll_load_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls status of asynchronous Load operations."""
    ...
  def read_remote(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockId],
      device_block_ids: list[int],
  ) -> bool:
    """Reads REMOTE blocks from their owning peers into local HBM."""
    ...
  def poll_remote_read_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls status of active remote reads."""
    ...
