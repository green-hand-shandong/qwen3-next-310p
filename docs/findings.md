# 排查记录 / Troubleshooting Log

时间线记录 Qwen3-Next-80B-A3B 在 310P 上的完整 bring-up 过程（2026-11-19 ~ 11-20）。

## 环境探测

- 硬件：Atlas 300I Duo，`npu-smi` 显示 8 卡 16 芯片（310P3，每卡双芯片，~43GB/芯片）
- 历史包袱：triton-ascend 不支持 310P，旧路线（`main-310p` 镜像 + 手装 triton_ascend wheel）不可行
- 现役路线：`vllm-ascend:v0.30.0rc2-310p-torchmamba` 镜像已验证「无 Triton、AscendC + PyTorch 回退」路线可行（Nemotron-3-Nano 混合架构跑通）
- 选择 `v0.32.0rc1-310p`（更新两个大版本，Qwen3Next 模型代码 + `_310p` GDN 算子齐全）

## 问题 1：HCCL 初始化失败（error 19）

```
RuntimeError: createHCCLCommOrigin:... hcclGetRootInfo(&hcclID), error code is 19 (call network api fail)
```

- `--net=host` 必现；bridge 网络正常
- 二分定位：同镜像同卡，仅网络模式不同 → host 网络下 HCCL 网络接口探测失败
- **结论：310P 容器必须用 bridge 网络**

## 问题 2：设备映射与显存冲突

- `--device=/dev/davinci8..15` 挂载后容器内仍可见 16 卡（privileged）
- `docker exec` 传入的 `ASCEND_RT_VISIBLE_DEVICES` 不会传给 vllm worker 子进程
- worker 直接用 device 0-7 = davinci0-7，撞上其他容器占用的 davinci3（24GB）
- **结论：`ASCEND_RT_VISIBLE_DEVICES` 必须用 `docker run -e` 容器级注入**

## 问题 3：MoE gating AICore 越界（核心 bug）

```
errorStr: The address for the MTE instruction to write on-chip buffer is out of bounds.
error code = 0x200000000, device(8), core list[0-7]
```

- 崩溃点：`profile_run` 的 dummy run，`qwen3_next.py:265` → `self.experts(...)` → `grouped_topk_router._compute_routing` → `torch_npu.npu_moe_gating_top_k_softmax`
- 二分定位 expert 数上限：
  - E=64 ✅，E=65 ❌，E=128/256/512 全崩
  - k（top-k）不影响：k=1..10 均同
  - tokens 数不影响：8/64/256 均同
- **结论：`npu_moe_gating_top_k_softmax` 在 310P 上 expert 数硬上限 64**（UB 容量限制）
- Qwen3-Next 512 experts → 原生算子不可用
- **修复**：纯 torch `softmax + topk` 替换，数值对照一致（单段 E=64 时与原算子 id match 100%）
- 注意：不能简单分段调用原算子再合并——原算子输出是「段内 softmax」权重，分段后分母不含全局，数值不等价

## 问题 4：图模式失败

```
rtStreamEndCapture execution failed, reason=an error has invalidated the stream capture sequence
```

- 去掉 `--enforce-eager` 后 aclgraph capture 崩溃
- 与 v0.30 镜像 nemotron 日志一致：`NPU Triton causal_conv1d_update is unavailable ... falling back to the PyTorch implementation, which syncs per request and therefore stalls ACL graph capture at decode-FULL`
- **结论：GDN conv1d 的 PyTorch 回退路径与 aclgraph 不兼容，必须 enforce-eager**

## 问题 5：MTP 投机解码（待解决）

- `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` 启动正常，首个请求崩
- `call aclnnRecurrentGatedDeltaRuleV310 failed ... Failed to execute tiling`
- 对应 0001 补丁（PR #18730）修复的 2D state indices / accepted tokens 路径
- kernel `.h` 已热同步到 `_cann_ops_custom/.../ascendc/recurrent_gated_delta_rule_v310/`，但该算子是预编译 .o（staticKey 路由），需要重编 host 库
- `csrc/build.sh --pkg --soc=ascend310p` 在容器内失败：CMake symbol 生成阶段 opgraph 目标缺失、VERSION 空、protobuf ExternalProject 下载问题
- **待办：在 CI 环境构建 `cann-ops-transformer-custom` run 包，或修复容器内构建链**

## 多轮正确性验证（无 MTP）

```
轮1: 我叫小明，最喜欢的数字是7
轮2: 我最喜欢的数字乘以6是多少？ → 42 ✅
轮3: 再乘以2？ → 84 ✅
```

无 MTP 时 accepted-tokens 异常路径不触发，GDN 状态跨轮传递正常。

## 性能基线

| 场景 | 数据 |
|---|---|
| 单请求 decode | ~3 tok/s |
| 8 并发 × 100tok | 26.7 tok/s 聚合 |
| 16 并发 × 200tok | 51.1 tok/s 聚合（引擎峰值 56） |
| 长输入 1462pt prefill+60tok gen | 11.2s |
| prefix cache 二次请求 | 8.3s → 5.8s |

## 性能剖析（2026-11-20）

### 现象
单请求 decode 288 ms/token（3.5 tok/s），但 **AICore 平均利用率仅 1.8%（峰值 7%）** —— NPU 几乎空转。

### 微基准分解（单 rank，TP8 分片后形状）

| 组件 | 单层耗时 | 层数 | 小计 |
|---|---|---|---|
| GDN recurrent（elemwise+sum） | 0.11 ms | 36 | 4.0 ms |
| 标准注意力 decode @8K | 0.32 ms | 12 | 3.8 ms |
| MoE 10-expert GEMM | 0.56 ms | 48 | 26.9 ms |
| MoE gating（softmax+topk 512） | 0.17 ms | 48 | 8.2 ms |
| Dense GEMM 2048² | 0.09 ms | 48 | 4.3 ms |
| **理想合计** | | | **~47 ms** |

组合 48 层 step 实测：**39.7 ms**（其中 CPU 提交 38.8ms，占 98%）。

### 真实服务 vs 复现的差距

复现只要 40ms，真实服务 288ms，差 7 倍。py-spy 采样（7971 样本）定位：

```
MoE 块           40.5%   （gating/dispatch/gmm1/gmm2/combine/sharedExpert/allreduce）
GDN 块           40.8%   （l2norm/gating_pytorch/recurrent AscendC/norm/fix_qkv_order）
all_reduce        5.2%   （每层 2 次 × 48 层）
其余 linear/norm  ~14%
```

热点本质：**CPU 侧算子分发与框架开销**，不是算子执行。
- 每 token 约 1200-1700 次 python→C++ 算子分发（48 层 × 25-35 算子）
- 分发+launch+view/dtype 转换 ≈ 0.17-0.24 ms/次 → 288 ms
- 310P 驱动栈的单次 launch 开销（~0.04ms 纯 launch + dispatcher 开销）放大了这一切

### 为什么 Mixtral-8x7B 快

| 维度 | Mixtral-8x7B | Qwen3-Next-80B |
|---|---|---|
| 推理框架 | MindIE 2.1.RC1（ATB 图引擎） | vLLM-Ascend eager |
| 调度方式 | 全图编译、算子融合、极少 CPU 调度 | 每 token ~1500 次 python 算子分发 |
| 架构 | 纯 attention + 8 experts | GDN 混合 + 512 experts |
| MoE 复杂度 | 8 experts，dispatch 轻 | 512 experts，dispatch/permute/combine 重 |
| 定制算子 | ATB 内置优化 | GDN AscendC 定制路径 + python wrapper 层厚 |

**结论：不是某几个算子慢，而是 eager 模式下 CPU 调度开销主导（AICore 1.8% 证实）。**
优化路径优先级：
1. 算子级图优化 / torch.compile（受 aclgraph 与 GDN 回退冲突阻塞）
2. MindIE 支持此模型后迁移（需 MindIE 适配 Qwen3-Next 混合架构）
3. 减少 per-token 分发数：合并小算子（gating+l2norm+norm 融合）、C++ 化 GDN wrapper
4. MTP 投机解码（需 0001 补丁重编）

## 补充剖析：算子实现层确认（2026-11-20 下午）

### 问题澄清：当前实现是 PyTorch 还是 AscendC？

**混合**。逐层确认：

| 组件 | 实现 | 调用方式 |
|---|---|---|
| GDN recurrent 核心计算 | **AscendC**（v310 定制算子） | `torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310` → aclnnRecurrentGatedDeltaRuleV310 |
| GDN causal_conv1d | **AscendC**（v310） | `torch.ops._C_ascend.npu_causal_conv1d_310` |
| GDN chunk prefill (fwd_h) | **AscendC**（vllm 定制） | `torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h_vllm` |
| GDN chunk_fwd_o | **AscendC** | `torch.ops._C_ascend.chunk_fwd_o_vllm` |
| GDN gating (decode) | **纯 PyTorch**（本项目替换） | `softmax+topk` |
| GDN l2norm | 310P NPU 算子 | `torch_npu.npu_rms_norm`（借道实现） |
| MoE gating | **纯 PyTorch** | softmax+topk |
| MoE expert GEMM | 310P NPU 算子 | `torch_npu.npu_grouped_matmul` (aclnnGroupedMatmulV5) |
| MoE token dispatch/combine | NPU 算子 | npu_moe_token_permute/unpermute |
| attention | 310P NPU 算子 | paged attention (Compatibility backend) |
| 各种 norm | NPU 算子 | rms_norm 系列 |

`torch.ops._C_ascend` 经 `vllm_ascend_C.so` (TORCH_LIBRARY) 注册 **7 个 AscendC 算子**：
npu_recurrent_gated_delta_rule_310 / npu_causal_conv1d_310 / chunk_gated_delta_rule_fwd_h_vllm / chunk_fwd_o_vllm / kda_gate_cumsum_vllm / kda_layout_swap12 / get_npu_view_from_cpu_tensor

### msprof/torch_npu profiler 器械级数据（单算子 CPU/NPU 分解）

**AscendC recurrent（单层 decode，50 次采样）**：
- CPU wrapper（aclnn enqueue/host tiling/dispatch）: 3.0 ms → 60 us/次
- device kernel 真实计算: 0.93 ms → **18.6 us/次**
- 即 0.252 ms/次总耗时中 device 只占 7%，93% 是 CPU 侧！

**grouped_matmul gmm1（10 experts, 100 次采样）**：
- 真实 GEMM 计算: 4.37 ms → **43.7 us/次**
- TransData 格式转换（ND→NZ）: 14.32 ms → **143 us/次** ← 比真计算大 3 倍！
- CPU wrapper: 4.9 ms → 49 us/次
- 310P 的 GroupedMatmulV5 强制走 TransData 预处理权重格式

### 最终瓶颈链（每 token 288ms 的构成）

1. **~1500 次 python→C++ 算子分发**（48 层 × 25-35 op）：调度+launch 为主
2. **TransData 格式转换**：grouped_matmul 每层 2 次（gmm1/gmm2）× 48 层 ≈ 14ms/token 纯浪费
3. **CPU 侧 aclnn wrapper**：每个 AscendC/aclnn 算子 50-60us 的 enqueue/host tiling
4. AICore 利用率仅 1.8% —— device 算力严重过剩，纯 CPU 调度瓶颈

### 与 Mixtral-8x7B(MindIE) 差距的本质

MindIE/ATB 在加载时做**离线图编译 + 权重 NZ 预转换 + 算子融合**，运行时 CPU 只做极少量调度。
vLLM-Ascend eager 模式则每 token 在 CPU 上走完整个 dispatcher→aclnn→enqueue 链。
**这不是换一两个算子能解决的，是执行模型差异。**

## 重大修正：npu_moe_gating_top_k_softmax 的真实边界（2026-11-23）

### 修正之前结论

之前二分测试（在同一进程内连续测试）得出"E=65 崩"，推断 64 上限。**这次用子进程隔离重测发现那是在受污染进程里的误报**。

### 精确边界（子进程隔离，k=10）

```
E=64 OK | E=65~78 FAIL | E=80 OK | E=84~92 FAIL | E=96 OK
E=104 FAIL | E=112 OK | E=120 FAIL | E=128 OK | E=160/192/256/320/384/448 OK
E=512 FAIL（tokens=1/8 均崩）
```

**真实规律**：
- E ≥ 80 且为 16 的倍数 → OK（80/96/112/128/160/192/256/320/384/448）
- E ≤ 78 非 16 倍数 → FAIL（对齐要求）
- **E=512 恰好超过 tile 容量硬上限（448 < 512）**

### 对 Qwen3.5/3.8 系"支持好"的解释

本机实际模型（README 记录）：
- Qwen3-30B-A3B：**128 experts** → 安全区，vllm-ascend 原生路径直接跑通（w8a8 单芯，PIECEWISE graph）
- Qwen3.5-35B-A3B：**256 experts** → 安全区
- 这些模型与 Qwen3-Next **共享同一架构**（Qwen3.5 直接复用 `Qwen3NextSparseMoeBlock`，GDN 参数完全一致：32 v-heads/16 k-heads/256 dim/4 层一全注意力）

**Qwen3-Next-80B 的 512 experts 是全系唯一超出 310P gating tile 容量的**。这解释了"为什么同架构的兄弟模型都好好的"。

### 其他收获（来自 Qwen3-30B-A3B 调优记录）

- 30B-A3B 在 310P 上用 **PIECEWISE graph 模式**跑通（`cudagraph_mode: PIECEWISE`），非 FULL —— Qwen3-Next 的 GDN conv1d 回退冲突可能与 FULL 模式相关，**PIECEWISE 值得试**
- 它们同样遇到 TransData ~10%、glue 算子 2/3 的问题 —— 与我们的剖析一致
- E=128 grouped GEMM 在 decode 小 batch 时只有 0.3~0.6 TFLOPS —— A3B 模型 decode 的共性问题
- torch_npu profiler 在长 prefill 下会 segfault（CANN 9.1）—— 我们用 micro-bench 绕过了

### 可再尝试的优化（参考 Qwen3.5/30B 路径）

1. **PIECEWISE graph 模式**：30B-A3B 的生产配置，Qwen3-Next 值得试（绕 FULL capture 对 conv1d 回退的敏感）
2. **w8a8 量化**：Qwen3-30B-A3B-w8a8 单芯放得下，Qwen3-Next-80B w8a8 后 ~45GB，TP4 即可，还能减 TransData
3. **MTP**：Qwen3-Next 权重自带 1553 个 mtp tensors（config 无显式字段，qwen3_next_mtp.py 直接读），仍是单请求延迟的正统解法
