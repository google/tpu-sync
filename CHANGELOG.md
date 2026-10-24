# Changelog

## [Unreleased]

### Added

- Bounded-staging sub-transfer chains for pool resharding.
  `CoordinateTransferRequest.dst_window_blocks` (per tag; 0 = unbounded) caps
  the destination blocks one transfer may carry: a request over its window is
  coordinated as a chain of sub-transfers, each an ordinary pool-reshard plan
  over a page-aligned slice of the request (derived engine uuid, own receiver
  arm and sender dispatch, own staging lease) under the request's single
  registry claim; tags with fewer windows ride in the last sub-transfers so a
  single-block state tag still lands after the paged tags. Sub-transfer 0
  runs inside the coordination RPC; the rest run on a coordinator thread, and a
  chained arm refused by the receiver's staging lease wait is retried while the
  arena drains. `StartTransferRequest.sub_plan_index` / `sub_plan_count` carry
  the chain position (zero for single-plan requests, wire-compatible), and
  `KVCacheManagerWithTransfer` reports one send/receive terminal per request
  once every sub-plan has settled, at the first sub-plan failure, or when a
  chain stalls for the transfer timeout. The torch reshard client accepts
  `dst_window_blocks` and exports `SUPPORTS_DST_WINDOW_BLOCKS`.
- `TPU_RAIDEN_DATA_CONN_REUSE=per_transfer`: a sender closes its idle
  data-plane TCP connections to a peer when a send session to that peer
  settles (pool reshard and legacy pull-serve), so open connections follow the
  transfers in flight rather than every (rank, receiver) pair ever pushed to;
  the receiver's per-connection thread ends on the resulting EOF. Default
  `pooled` keeps the historical behaviour; any value other than `pooled` or
  `per_transfer` (case-insensitive) fails send session creation with
  `InvalidArgument`. `BlockTransport` / `RawBufferTransport` gained
  `CloseIdleConnections(peer)` and `IdleConnectionCount(peer)`; `ConnPool`
  gained `CloseIdle` / `IdleCount`.
- `TPU_RAIDEN_TRANSFER_STREAMS`: the pool-reshard planner's default stream
  count per push when the consumer requests none. A positive integer, clamped
  to the admitted transfer parallelism; unset, empty or `0` keeps one stream
  per source rank; any other value fails the plan with `InvalidArgument`.

[Unreleased]: https://github.com/tpu-raiden/tpu_raiden/compare/v0.1.0...HEAD
