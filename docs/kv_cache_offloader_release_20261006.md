# release 分支的整页 KV offloader

## TL;DR

`alibaba/release-0.0.1` 移植了上游
[PR #851](https://github.com/google/tpu-sync/pull/851) 的
`d069a36ed0829d46cae7ecb831bd324835106bea`。Torch wheel 提供
`tpu_sync.api.torch.kv_cache_offloader.KVCacheOffloader`，供外部共享内存池
注册一次映射后执行完整 scheduler page 的 H2D / D2H。
真实两 rank 整页 DMA，以及 TP2/PCP1/MTP3 的真实模型 / V6D HTTP
保存、重载和流断开恢复均已通过；生产接口保持整页字节池契约。

## 来源与 release 适配

- release 基线：`3d8cc9c8233e5fe59d57f7ab819ef43ca01fcdf1`。
- cherry-pick：`3d3ea116`，保留原提交作者和 `cherry picked from` 信息。
- 合入构建期间 release 新增的 `d2c7b49d`、`ab81a432`，保留其
  ControlPipe / reshard 修复，并重新构建完整 wheel。
- 三处合并冲突保留 release 已有的 ABI loader、wheel 文件和新增 offloader
  文件：`ci/wheel/BUILD.bazel`、`tpu_sync/api/torch/BUILD`、Torch binding。
- 设备 buffer 的 owning handle 使用 release 已公开的 `TensorBufferHandle`。
- production target 复用 `xla_raw_transfer_headers`，与 release 其他 Torch
  target 的动态 glue 边界一致，避免额外链接整份 PjRt client。
- libzmq 的 CMake 安装目录默认设为 `lib`。目标主机的默认目录是 `lib64`，
  实际编译成功后 Bazel 因找不到声明的 `lib/libzmq.a` 失败；显式目录后越过
  该失败点。这是构建适配，不改变 DMA 算法。

## 整页契约

设备 buffer 是按 `page_nbytes` 等分的原始物理字节池，不在 offloader 中解释
attention 或 Mamba 的逻辑 shape。调用方保证所有注册 buffer 等大，且
buffer 大小可以被 `page_nbytes` 整除。

主机对象为 contiguous CPU `int8` tensor，排列为
`[num_ranks, num_tensors, page_nbytes]`。第 `b` 个 scheduler page 的设备
起始位置为 `b * page_nbytes`；主机对象中 rank `r`、tensor `l` 的位置为
`(r * num_tensors + l) * page_nbytes`。每次复制长度是完整
`page_nbytes`，即使一个 scheduler page 包含多个 kernel blocks。
这是旧 standalone offloader 的契约，不能与 manager 的
`[rank, slice, tensor, native_slice_bytes]` 排列混用。

外部池负责分配、锁页、对象之间不重叠和生命周期；offloader 只负责对调用方
现有映射做一次 DMA 注册。正常传输完成后才能 unmap / 释放池。
注册时持有 device buffer handle，后台 D2H 不重新获取可能已 donation 的 tensor。

## 构建与验证

标准容器构建入口如下；需可读取 `torch_tpu.version` 指定的 wheel registry：

```bash
RAIDEN_TORCH_ABIS="2.13.0" WHEEL_VERSION_EXTRAS="" ci/build_wheel.sh torch
python -m pip install --no-deps dist/tpu_sync_torch-*.whl
python -m unittest tpu_sync.api.torch.kv_cache_offloader_test
```

本次远端使用已有 torch 2.13.0 / torch-tpu release 环境，直接构建完整
`//ci/wheel:raiden_torch_wheel`，随后执行官方构建脚本同样的 ABI 重命名及
`libpywrap_2_13_0_common.so` NEEDED 处理。未用独立 `.so` 替代 wheel，
也未修改共享运行环境中的安装。

远端路径为
`/ssd/1/yanzecheng_yzc/tpu-sync/kvs-release-20261006`。
该目录的 `build-release.sh`、`install-release.sh`、`logs/` 保留具体命令及输出。
完整 wheel 构建和真实两 rank 的整页 DMA 校验已通过：BF16 / FP8，
每 page 含 1 / 3 个 kernel blocks，2 / 24 个 tensors，非连续 page IDs，
每组 3 次往返，退出码 0、`RELEASE_WHOLE_PAGE_DMA_OK`。
随后真实 Qwen3.5-35B-A3B-FP8 模型 / V6D HTTP E2E 退出码 0、
`V6D_HTTP_FAILOVER_E2E_OK`：清空本地 prefix cache 后重载命中 10368
tokens，20 个输出 token IDs 一致，warm/cold 流断开后可恢复。
补充 logprobs 均有限，重载前后最大差为 0.0。模型实际 page 为 1179648
bytes、每 rank 11 个物理 tensors；这些 geometry 由平台和 KVS 建立，
offloader 按 page 字节数复制。运行栈延迟 PjRt 初始化通过独立 vLLM
worker extension 验证适配，未写入 offloader 或 KVS 生产代码。
完整环境、wheel 校验值、真实 TPU 整页验证和模型/V6D 复现见
[KVS 集成记录](https://github.com/aios-tpu-infra/kvs_connector/blob/main/docs/release_offloader_20261006.md)。

## 能力边界

本次验证覆盖正常完成和 KVS HTTP 取消后的回收。底层 raw-transfer helper
既有的“同一批次部分 native 提交成功后，后续提交失败”路径没有在本次扩展：
它可能遗漏先前接受的 event。KVS 将 native DMA 错误作为 fatal，保留所有
对象、buffer 和映射 owner，拒绝正常关闭，交由进程退出回收；不能把正常
unmap 测试理解成对任意 native 失败的安全恢复证明。
