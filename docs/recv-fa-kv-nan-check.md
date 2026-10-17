# Decode 接收后 full attention KV NaN 检测

## TL;DR

在 decode worker 启动前设置 `TPU_RAIDEN_CHECK_RECV_FA_NAN=1`，即可在 pool-reshard
接收路径中，每个 full attention pool 的网络接收完成后、H2D 之前，直接检查 host staging KV。
发现 NaN 时记录 `ERROR`，包括传输计划的 `req_id`、UUID、pool/tag、block、shard
和字节偏移。仅记录诊断，不修改 cache、采样结果或请求成功状态。

默认关闭。开启后只增加 CPU 扫描和地址查询开销，没有额外 D2H；此实现尚未通过真实
P-D E2E 验收。本检查说明 decode 收到的 host 字节是否包含 NaN，不证明 H2D 后 HBM 的内容。

## 范围和时序

- 入口是 `ReshardReceiveSession::ExecuteEligiblePoolH2ds` 中已收到完整 pool 的 H2D
  提交之前；不检查发送端、未收齐的 pool 或 legacy 非 pool-reshard 路径。
- 仅识别 vLLM manifest 的 `fa`、`fa.l<数字>` 标签；不扫描 GDN 状态。
- 按 manifest 的真实 dtype 解码 BF16、FP16、FP32，保留 `bf16` / `f16` / `f32`
  别名。未知 dtype 或按字节描述的整数 carrier 不猜测解码方式，记录 `skipped`。
- 直接解释 host staging 中收到的 raw transfer 字节，避免 Python CPU carrier 的逻辑
  布局改变字节解释。
- 按 pool 的 live regions 排除间隙，再以该请求传输计划的
  `dst_expected_extent_bytes` 截断每页，排除尾页未写区域。
- 通过 `PoolHostBaseAddrs(uuid, pool)` 取得各 shard 的 host 地址，已包含 pool 的
  storage/base offset。bounded staging 使用 `host_slot_by_block` 找到 leased slot，
  full mirror 使用原 block ID，然后按 pool stride 定位该页。
- 扫描在已有 H2D 提交线程同步执行，其原 in-flight 引用维持 staging lease 的生命周期。
  扫描只读，不新增 DMA、缓存副本、后台任务或完成回调。
- 每个异常 FA pool 最多记录一条，定位其第一个 NaN；同一请求的多个异常 pool 可产生多条日志。
  日志 request ID 是 native transfer plan 的 ID；上层如使用独立 decode ID，按现有
  connector 映射关联。

日志示例：

```text
NaN detected in received full-attention KV cache: req_id=<request-id> uuid=<uuid> pool=3 tag=fa.l3 block_id=42 shard=0 byte_offset=16 source=host_staging
```

地址查询失败时输出带 `req_id` 的 `unavailable` 警告，并继续原接收完成流程；该次
诊断不构成“没有 NaN”的证据。该检查只回答有效 FA KV 是否有 NaN，不检测 Inf。

## 验证和复现

开发基线为 `partner/release-0.0.1` 的 `bf9db1b8e8c61ea2a6166e02b13da848d16627e9`。
原接收计划校验已保证 pool/block 范围、完整有效前缀和 extent 数量；诊断额外检查数值元素对齐。
扫描的 domain 是 little-endian、完整 BF16/FP16/FP32 元素；每轮都消耗正长度的 live run
或在 `live_bytes == 0` 时退出，不会把物理 padding 计入有效字节预算。

配置好仓库的 Bazel/XLA/native 构建依赖后，在源码根目录运行：

```bash
bazel test --config=oss \
  //tpu_sync/core:recv_nan_check_test \
  //tpu_sync/core:reshard_receive_session_test \
  //tpu_sync/core:kv_cache_manager_with_transfer_pool_reshard_test
```

新增长期测试覆盖 NaN 与 Inf/有限值区分、region 间隙和未传输尾部不误报、非对齐地址读取
及空前缀。已使用 GoogleTest v1.17.0 和 Apple clang 17.0.0 直接编译运行，3 项全部通过。
已有该版本 GoogleTest 源码时，可绕过 native 依赖复现纯扫描测试：

```bash
gtest_root=/tmp/googletest-1.17.0
clang++ -std=c++20 -Wall -Wextra -Werror -pthread -I. \
  -I"$gtest_root/googletest/include" -I"$gtest_root/googletest" \
  tpu_sync/core/recv_nan_check_test.cc \
  "$gtest_root/googletest/src/gtest-all.cc" \
  "$gtest_root/googletest/src/gtest_main.cc" -o .recv_nan_check_gtest
./.recv_nan_check_gtest
```

开发时另用 C++20 临时脚本枚举 BF16/FP16 全部 65,536 种位模式，与浮点
`isnan` 比较，并枚举不同有效前缀和 NaN 位置；共 131,120 个数值/范围检查通过，
AddressSanitizer/UndefinedBehaviorSanitizer 同样通过。临时脚本不入库。

当前 macOS 没有 Bazel/Torch native 依赖，未运行上述 Bazel 测试；项目
`bash run_tests.sh torch` 在首项测试前因缺少 `python` 命令退出。
真实验收还需在独立 TPU 实验环境开启开关，以有限 KV 和受控有效 token NaN 两组请求
验证日志归属，再用仅 padding 含 NaN 的对照验证不误报，并确认两组请求接收生命周期正常。

## 共享 staging 的只读约束

代码审查确认，pool 的 `order_rank` 只限制 H2D 顺序，并不限制网络接收顺序。
同一 storage 的别名 pool 可以有不同 block stride：例如 FA 的 `block=1, stride=128`
与状态 pool 的 `block=2, stride=64` 都指向 offset 128。FA H2D 后，状态数据可已经
进入 staging；此时若把 FA 读回原 staging，会在后续状态 H2D 前覆盖状态数据。
因此本实现按用户要求只读 host staging，没有 D2H 回写。扫描放在原 H2D 提交之前，
保持原接收/上传顺序；不尝试通过诊断解决传输布局本身的别名或并发覆盖问题。
