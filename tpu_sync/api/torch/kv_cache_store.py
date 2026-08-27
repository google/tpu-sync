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

"""Raiden KV Cache Store API for PyTorch."""

from typing import Any

from tpu_sync.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()

# pylint: disable=g-import-not-at-top
from tpu_sync.api.torch import torch_abi

_impl = torch_abi.load_extension(
    "tpu_sync.frameworks.torch",
    "_tpu_raiden_torch",
)
# pylint: enable=g-import-not-at-top


from tpu_sync.api import common

BlockStatus = common.BlockStatus
RaidenId = _impl.RaidenId


class RaidenBlockId:
  """Wrapper around compiled C++ RaidenBlockId."""

  def __init__(
      self,
      raiden_id: RaidenId | None = None,
      host_block_id: int = -1,
      status: BlockStatus = BlockStatus.INIT,
      device_block_id: int = -1,
      impl: Any = None,
  ):
    if impl is not None:
      self._impl = impl
    else:
      if raiden_id is None:
        raiden_id = RaidenId()
      raw_raiden_id = (
          raiden_id._impl if hasattr(raiden_id, "_impl") else raiden_id  # pylint: disable=protected-access
      )
      # Map Python enum to C++ enum
      status_val = getattr(_impl.BlockStatus, status.name)
      self._impl = _impl.RaidenBlockId(
          raw_raiden_id,
          host_block_id,
          device_block_id,
          status_val,
      )

  @property
  def raiden_id(self) -> RaidenId:
    return self._impl.raiden_id

  @property
  def host_block_id(self) -> int:
    return self._impl.host_block_id

  @property
  def device_block_id(self) -> int:
    return self._impl.device_block_id

  @property
  def status(self) -> BlockStatus:
    return BlockStatus[self._impl.status.name]

  @property
  def job_name(self) -> str:
    return self.raiden_id.job_name

  @property
  def job_replica_id(self) -> str:
    return self.raiden_id.job_replica_id

  @property
  def data_name(self) -> str:
    return self.raiden_id.data_name

  @property
  def data_replica_idx(self) -> int:
    return self.raiden_id.data_replica_idx

  def __repr__(self) -> str:
    return (
        f"RaidenBlockId(raiden_id={self.raiden_id},"
        f" host_block_id={self.host_block_id},"
        f" device_block_id={self.device_block_id}, status={self.status})"
    )


class KVCacheStore:
  """Wrapper around compiled C++ KVCacheStore.

  THE THREE WORKFLOWS

  The surface exists to serve exactly three flows. Every pin is granted by one
  call and consumed by one call, so you never balance pins by hand:

    1. Read what is cached:     lookup() -> load()
         lookup() pins each hash it returns; a successful LOCAL load() spends
         that pin. A REMOTE hit is not pinned and a remote load() spends
         nothing -- a registry-only hit never entered the local cache, and a
         peer load records nothing locally either.

    2. Cache something new:     insert() -> save()
         insert() pins what it takes; a successful save() spends it.

    3. Hand a block to a peer:  lookup() -> save(dst)
         The block must already be host-resident here, so lookup() finds it
         locally and pins it, and the successful remote save() spends that pin.

  Flows 1 and 2 never collide over a hash: lookup() returns hits, insert()
  takes the misses, so the sets are disjoint and nothing is pinned twice.

  A FAILED operation does not spend the pin. Retrying, or giving up with
  release(), is your decision -- unpinning under a retry would leave the block
  evictable while you still believed you held it.

  Eviction is not part of any of this. The store reclaims space itself when an
  operation needs a host block; there is no caller-driven removal.
  """

  def __init__(
      self,
      capacity: int,
      global_registry_address: str = "",
      raiden_id: RaidenId | None = None,
      *,
      num_shards: int,
      store_server_ip: str,
      shard_size_bytes: int = 0,
      raiden_controller_port: int = 0,
      expected_worker_count: int = 0,
      kv_pool_group: str = "",
  ):
    """Creates a KVCacheStore.

    Every store has a RaidenController (num_shards >= 1) and a
    store_server_ip; there is no controller-less or unpublished
    configuration. Same-host use: "127.0.0.1".
    num_shards and store_server_ip have no default and must be passed by
    keyword -- there is no value for either that is valid to fall into by
    omission.

    The store monitor (heartbeats to the global registry) and its evict
    sweep (demoting cold blocks to higher-tier peer stores under memory
    pressure) are configured through environment variables, not arguments:
      RAIDEN_ENABLE_STORE_MONITOR       "true"/"1" runs the monitor
      RAIDEN_ENABLE_EVICT_SWEEP         "true"/"1" runs the evict sweep
                                        (requires the monitor)
      RAIDEN_STORE_MONITOR_HEARTBEAT_S  heartbeat period, whole seconds
      RAIDEN_EVICT_SWEEP_PERIOD_S       sweep fallback period, whole seconds
      RAIDEN_EVICT_LOW_WATERMARK        free-block ratio in (0, 1) below
                                        which the sweep starts demoting
      RAIDEN_EVICT_HIGH_WATERMARK       free-block ratio at which it stops
    Unset numeric values mean the built-in defaults. The switches only take
    effect when global_registry_address is set; a registry-less store
    ignores them, so one fleet-wide environment block is safe.

    Args:
      capacity: Maximum number of cache blocks this store can hold.
      global_registry_address: Address of the global registry service; empty
        for a standalone local store.
      raiden_id: Optional RaidenId identifying this store instance.
      num_shards: Shard count for this store's RaidenController; must be
        >= 1.
      store_server_ip: IP that peers use to reach this store. Bind-and-
        advertise: this store's KVCacheStoreService and RaidenController both
        bind it, and it is the host published to the global registry. Must be
        an IP (or hostname) this process can bind -- not empty, not a
        wildcard.
      shard_size_bytes: Size in bytes of each shard buffer; 0 uses default.
      raiden_controller_port: Port for this store's RaidenController; 0 lets
        gRPC choose. Note that the IP address of the controller reuses
        store_server_ip.
      expected_worker_count: When > 0, constructor blocks until that many
        workers have registered with the controller (times out and throws after
        RAIDEN_EXPECTED_WORKERS_TIMEOUT_S, default 120s).
      kv_pool_group: KV pool group this store's KVTransferSpec is published
        under in global registry; empty falls back to raiden_id.job_name.
    """
    raw_raiden_id = _impl.RaidenId()
    if raiden_id is not None:
      raw_raiden_id = (
          raiden_id._impl if hasattr(raiden_id, "_impl") else raiden_id  # pylint: disable=protected-access
      )
    self._impl = _impl.KVCacheStore(
        capacity=capacity,
        global_registry_address=global_registry_address,
        raiden_id=raw_raiden_id,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip=store_server_ip,
        raiden_controller_port=raiden_controller_port,
        expected_worker_count=expected_worker_count,
        kv_pool_group=kv_pool_group,
    )

  @property
  def raiden_id(self) -> RaidenId:
    """Returns the RaidenId associated with this store."""
    return self._impl.raiden_id

  @property
  def raiden_controller_address(self) -> str:
    return self._impl.raiden_controller_address

  @property
  def store_server_address(self) -> str:
    """\"host:port\" published to the global registry, or \"\" if undiscoverable."""
    return self._impl.store_server_address

  def lookup(
      self,
      block_hashes: list[bytes],
      enable_global: bool = False,
      pin_found: bool = True,
  ) -> list[tuple[bytes, RaidenBlockId]]:
    """Checks the LRU directory for cached block hashes.

    PINS every hash it returns by default (pin_found=True), one pin each, so
    the answer cannot be evicted between being given and being used. The
    operation the caller goes on to perform consumes that pin: load() drops it
    on a successful local load, and save() drops it on success.

    A caller that does neither -- one that only wants to OBSERVE what is
    resident -- should pass pin_found=False rather than lookup() followed by
    release(): a pin/unpin round trip moves the entry through the pinned list
    and back, which reorders the LRU.

    Locally cached hits are pinned. Hashes resolved only through the global
    registry are not: they name a block on another node, and there is nothing
    here to hold.

    Args:
      block_hashes: Incoming block hashes to check.
      enable_global: Whether to fallback to global registry on miss. Defaults
        to False; the fallback is a blocking RPC.
      pin_found: Whether to pin every local hit. Defaults to True. Pass False
        to observe residency without acquiring anything.

    Returns:
      A list of tuples containing the block hash and the matching
      RaidenBlockId replica, halting immediately upon the first cache miss.

    Raises:
      RuntimeError: if the underlying lookup failed outright (a backend
        error, not a cache miss).
    """
    raw_res = self._impl.lookup(block_hashes, enable_global, pin_found)
    final_res = []
    for hash_val, raw_slice in raw_res:
      final_res.append((hash_val, RaidenBlockId(impl=raw_slice)))
    return final_res

  def insert(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockId],
      on_host: bool,
  ) -> bool:
    """Caches sharded buffers into host-RAM/HBM backing store, PINNING them.

    Existing hashes are pinned in place; new ones are inserted and pinned. New
    items go in reverse order so that tail blocks are evicted first.

    All-or-nothing, and it does NOT evict to make room: if there is not enough
    free space for the new hashes it unpins whatever it already pinned and
    returns False, leaving the cache exactly as it found it. Making room is the
    store's own business, not something an insert does on your behalf.

    PIN CONTRACT: the pin this grants is what a subsequent save() consumes.
    Call it with hashes lookup() reported as misses -- the two take disjoint
    sets, which is why no hash ends up pinned twice.

    LOCAL entries only. The LRU cache holds blocks resident on THIS node; a
    REMOTE slice names a block on another node and is refused. If any slice
    in the batch has status REMOTE the call raises ValueError and nothing is
    inserted or pinned.

    Args:
      block_hashes: Incoming block hashes to insert.
      slices: List of RaidenBlockId, one for each block hash.
      on_host: Whether the slices are located in host memory.

    Returns:
      bool: whether the whole operation succeeded.

    Raises:
      ValueError: if any slice in the batch has status REMOTE.
    """
    raw_slices = []
    for s in slices:
      if isinstance(s, RaidenId):
        s = RaidenBlockId(raiden_id=s)
      raw_slices.append(s._impl)  # pylint: disable=protected-access
    return self._impl.insert(block_hashes, raw_slices, on_host)

  def capacity(self) -> int:
    """Returns the maximum capacity of the store in blocks."""
    return self._impl.capacity()

  def release(self, block_hashes: list[bytes]) -> None:
    """Releases previously pinned block hashes.

    Released hashes become eligible for LRU eviction when capacity is
    exceeded. The cache holds LOCAL entries only -- insert() enforces that --
    so release needs no status handling.

    Args:
      block_hashes: List of block hashes to release.
    """
    self._impl.release(block_hashes)

  def save(
      self,
      block_hashes: list[bytes],
      dst_raiden_id: RaidenId | None = None,
  ) -> bool:
    """Saves blocks asynchronously, either locally or to a peer.

    Where to depends on dst_raiden_id:
      omitted -- LOCAL save, device (HBM) to host (DRAM). Each block must have
                 status HBM here -- one already on host has nothing to save.
      given   -- REMOTE save: offer the blocks to that peer, which takes its
                 own copy. Each block must already be HOST-resident here,
                 because the bytes are pulled out of host DRAM. There is no
                 two-hop: a block only in HBM must be saved locally first.

    Both kinds are polled with poll_save_status().

    PIN CONTRACT: every hash must be pinned on entry -- lookup() grants that
    pin for a remote save, insert() for a local one -- and a SUCCESSFUL save
    consumes exactly one pin per hash. Do not release afterwards. A FAILED
    save does not consume it: the entry stays pinned so you can retry or
    release deliberately.

    A remote save also takes a SEPARATE internal pin for as long as the
    destination may still be reading. That one is not yours.

    A remote save requires a global registry at both ends -- that is how the
    destination is resolved, and how the blocks become reachable once they
    land.

    IMPORTANT: a remote destination allocates landing blocks from FREE blocks
    only and never evicts to make room, so a peer whose cache is warm refuses
    every offer. Destination-side eviction is separate, unimplemented work.

    Args:
      block_hashes: List of block hashes to save.
      dst_raiden_id: Omit for a local save; pass a peer to offer them to it.

    Returns:
      True if successfully launched, False if validation failed. For a remote
      save a True return does NOT mean the peer has the blocks -- poll for
      that.
    """
    raw_dst = (
        None
        if dst_raiden_id is None
        else (
            dst_raiden_id._impl
            if hasattr(dst_raiden_id, "_impl")
            else dst_raiden_id
        )
    )
    return self._impl.save(block_hashes, raw_dst)

  def load(
      self,
      block_hashes: list[bytes],
      device_block_ids: list[int],
      slices: list[RaidenBlockId] | None = None,
  ) -> bool:
    """Asynchronously loads KV cache blocks to device (HBM), either from local
    host DRAM or from a peer (if `slices` is provided).

    If `slices` is None, this is LOCAL-ONLY: a hash whose cached entry has
    status REMOTE is refused (the call returns False) -- passing `slices` is
    the only way to load from a peer.
    If `slices` is provided, it handles loading from local host DRAM or from a peer.

    ONE CALL IS ONE SOURCE. Every block in a batch must carry the same status,
    and remote blocks must all refer to the same peer.

    `device_block_ids` is the destination and must name one device block per hash.

    PIN CONTRACT:
      local source  -- every hash must be pinned on entry (lookup() is what
                       normally grants that pin), and a SUCCESSFUL load
                       consumes exactly one pin per hash. Do not release
                       afterwards. A FAILED load does not consume it: the entry
                       stays pinned so you can retry, or release it
                       deliberately. Giving up is your decision, not the
                       store's.
      remote source -- no pin is required and none is consumed. A hash resolved
                       only through the registry never entered the local cache,
                       so there is nothing here to have pinned.

    A load from a peer records NOTHING locally: no host copy is kept, so a
    later lookup() of that hash is still a miss. Your own block manager is what
    remembers you already own the device block.

    Remote loads re-resolve hashes at the peer, ignoring the rest of `slices`.

    Args:
      block_hashes: List of block hashes to load.
      device_block_ids: Destination device block IDs, one per hash.
      slices: Optional pre-resolved RaidenBlockIds.

    Returns:
      True if successfully launched, False if validation failed. Launching is
      not completion: poll_load_status() reports whether the transfer landed.
    """
    if slices is None:
      return self._impl.load(block_hashes, device_block_ids)
    raw_slices = []
    for s in slices:
      if isinstance(s, RaidenId):
        s = RaidenBlockId(raiden_id=s)
      raw_slices.append(s._impl)  # pylint: disable=protected-access
    return self._impl.load(block_hashes, raw_slices, device_block_ids)

  def poll_save_status(
      self,
  ) -> tuple[list[bytes], list[bytes], list[bytes], list[bytes], list[bytes]]:
    """Polls the status of all active Save operations, local and remote.

    For completed local transfers it advances the LRU block states to
    HOST_AND_HBM and updates their host block locations; for failed ones it
    releases the allocated host blocks. For remote saves it asks each
    destination for a verdict.

    Returns:
      A tuple of (done, failed, pending, existing, unregistered). The last two
      annotate REMOTE failures rather than being outcomes of their own, and
      are empty for local saves:
        done: Hashes whose save completed. For a remote save that means the
          destination now holds AND has published them, including the case
          where it already had them -- hashes are content-addressed, so a peer
          having them is the post-condition the caller wanted. This is the
          only result that makes it safe to drop a local copy.
        failed: Hashes whose save did not succeed.
        pending: Hashes whose save is still in progress. Note this means
          different things on the two paths: for a local save it is bounded by
          a DMA, for a remote one by the destination answering or the ~30s
          HOLD window elapsing.
        existing: Hashes a destination already held when it refused a PARTIAL
          batch. This store does not reissue with the remainder -- doing so is
          the caller's decision, and this is the list it needs to make it.
        unregistered: The destination HAS these -- the transfer succeeded --
          but could not publish them, so no lookup will find them there. They
          are reported failed because the safe default is to keep your own
          copy: freeing it would move the block from findable-here to
          findable-nowhere. A caller that only needed the peer to hold the
          bytes may treat this as success; a caller that was about to free its
          copy must not.

    A hash in `existing` or `unregistered` also appears in `failed`, not
    instead of it.
    """
    return self._impl.poll_save_status()

  def poll_load_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls the status of all active asynchronous Load operations.

    For completed LOCAL transfers, it advances the LRU block states to
    HOST_AND_HBM and updates their device block locations. A completed REMOTE
    load is only reported here -- it records nothing locally.

    Returns:
      A tuple of (done, failed, pending), where:
        done: List of block hashes whose Load transfer successfully completed.
        failed: List of block hashes whose Load transfer failed.
        pending: List of block hashes whose Load transfer is still in progress.
    """
    return self._impl.poll_load_status()

  def read_remote(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockId],
      device_block_ids: list[int],
  ) -> bool:
    """Reads REMOTE blocks from their owning peers straight into local HBM.

    Returns as soon as the reads are issued; poll with
    poll_remote_read_status().

    This store's cache is neither consulted nor modified. The hashes need not
    be present locally and need not be pinned; nothing is inserted on success
    and nothing is left behind on failure. The bytes land ONLY in the given
    device blocks -- no local host copy is kept, so a later local load() of the
    same hash is still a miss.

    Compare with load(): both bring a peer's block into local HBM. Use load()
    when the hash may be resident locally and you want the store to decide; use
    read_remote() when you already hold the source coordinates and want no
    local record of the transfer.

    Requires a global registry: it is what maps the owning peer to the
    controller address this store acquires its read lease from. A store built
    without a global_registry_address fails every read.

    Args:
      block_hashes: Block hashes to read.
      slices: One REMOTE RaidenBlockId per hash, naming where to read from. Only
        two fields are used -- raiden_id (the owning peer) and host_block_id
        (the block on that peer) -- so a lookup() answer can be passed straight
        through.
      device_block_ids: One local device block id per hash. Mandatory. On
        FAILURE their contents are UNDEFINED: they are written before the
        source's verdict is known. Treat them as scratch until the read reports
        success.

    Returns:
      True if successfully launched.
    """
    raw_slices = []
    for s in slices:
      if isinstance(s, RaidenId):
        s = RaidenBlockId(raiden_id=s)
      raw_slices.append(s._impl)  # pylint: disable=protected-access
    return self._impl.read_remote(block_hashes, raw_slices, device_block_ids)

  def poll_remote_read_status(
      self,
  ) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls the status of all active remote reads.

    Returns:
      A tuple of (done, failed, pending), where:
        done: List of block hashes whose remote read successfully completed.
        failed: List of block hashes whose remote read failed.
        pending: List of block hashes whose remote read is still in progress.
    """
    return self._impl.poll_remote_read_status()
