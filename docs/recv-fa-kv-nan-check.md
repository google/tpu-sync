# Received KV non-finite diagnostics

Set `TPU_NAN_LOG_LEVEL=3` or `4` on the receiver to check received KV
data for NaN or infinity. The default is `0`; levels `0`, `1`, and `2`
leave this receive checker disabled. Values outside `0` through `4`
are rejected when a receive session is created.

The checker covers `fa`, `fa.l<N>`, `gdn.conv`, and `gdn.ssm` pools.
GDN tags also accept `.g<N>` and `.l<N>` suffixes in that order.
Supported dtype tags are:

| Dtype tags | Non-finite classification |
| --- | --- |
| `bfloat16`, `bf16`, `float16`, `f16`, `float32`, `f32` | All exponent bits set |
| `float8_e4m3fn` | `(byte & 0x7f) == 0x7f` |
| `float8_e5m2` | `(byte & 0x7c) == 0x7c` |
| `float8_e4m3fnuz`, `float8_e5m2fnuz` | `byte == 0x80` |

Unknown dtype tags, including ambiguous `fp8` and integer storage tags,
are skipped. FP8 detection uses the declared logical pool dtype, even
when several pools share one physical byte buffer.

Checks read the live ranges of the received destination blocks from HBM
after host-to-device copies. Padding and bytes outside those ranges are
masked out. Packed U32 classifiers process four FP8 bytes per word without
float conversion. Compatible pools share scan groups, while results remain
separate for each logical pool. Physical storage arguments are deduplicated.

The receiver submits one program per device before publishing receive
completion. A bounded worker handles compilation, submission, and asynchronous
boolean readback; receive completion does not wait for the readback. Programs
are cached by client, buffer geometry, and classifier semantics. The first
check for a new geometry may compile a program. Enabling diagnostics adds
device work and can add receive latency; it is not a zero-overhead mode.

Logs identify the request, transfer UUID, pool, dtype, and affected shards:

```text
NaN/Inf detected in received KV cache: req_id=example uuid=transfer pool=2 tag=fa dtype=float8_e4m3fn shards=[0] source=device_hbm
```

Each pool reports at most three affected receives per window of 100
successful checks. Tensor contents are not copied to the host. An unavailable
check or a full diagnostic queue produces a warning and allows the transfer
to complete. Static PJRT C API buffers with a registered typed buffer owner
are required; the Torch manager supplies this owner and drains pending checks
before releasing its buffers.

Native regression targets:

```sh
bazel test //tpu_sync/core:nan_log_level_test \
  //tpu_sync/core:recv_nan_check_test \
  //tpu_sync/core:recv_nan_check_fp8_test \
  //tpu_sync/core:reshard_receive_session_test \
  //tpu_sync/core:kv_cache_manager_with_transfer_pool_reshard_test
```
