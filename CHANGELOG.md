# Changelog

## [Unreleased]

### Added

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
- `TPU_RAIDEN_ROLLBACK_FAILED_TRANSFERS=1` (set on the source controller):
  when a pool-reshard coordination fails, the controller cancels the receivers
  it prepared (new `COMMAND_CANCEL_TRANSFER`, at most 5 s) and sets
  `ControllerResponse.transfer_released` when no receive is left on any
  destination unit. The torch client then raises `TransferReleasedError`, a
  `RuntimeError` with the unchanged text. When a sender fails to start, the
  controller also stops the senders that did start (at most 5 s more) and then
  cancels the registry claim instead of leaving it until its TTL. Off by
  default.

[Unreleased]: https://github.com/tpu-raiden/tpu_raiden/compare/v0.1.0...HEAD
