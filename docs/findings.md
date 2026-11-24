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

## 补充：E=512 崩溃的根因定位（2026-11-23 续）

### 不是"硬件锁"，是 310P 专用 kernel 的实现缺陷

读了 CANN 自带的 kernel 源码（`opp/built-in/.../moe_gating_top_k_softmax/op_kernel/moe_gating_top_k_softmax_310p.h`）：

- 整个 kernel 把**一整行（全部 expert logits）一次性搬进 UB**（`CopyIn: DataCopy(xLT, gmX_, rowWork_ * numCol_)`），无分块循环
- `MAX_UB_SIZE = 245760`（240KB），减去 workspace 和 softmax/topk 中间缓冲后，单行容量上限实测在 **E≈466** 附近（464 OK，468 FAIL）
- E=512 时一行 1KB fp16，加中间缓冲超 UB → MTE 写越界 → AICore 异常

### 为什么说是"实现缺陷"而非硬件限制

1. 310P 的 UB 物理上足够（240KB 远大于 1KB 一行），**只是这个 kernel 没写分行/分块逻辑**
2. CANN 已有支持大 E 的 **v2 版本**（`moe_gating_top_k_softmax_v2`，含 REPEAT_MAX 分块检查），但 **op-plugin 未在 310P 上适配 v2**（调用即报错，与 Qwen3-30B-A3B 调优记录中"v2 未适配 310P"一致）
3. 对比：910B 系列 UB 更大且 kernel 有分块，E=512 无压力

### 结论

- **软件层面被 CANN 的 310P kernel 实现卡住，不是硬件能力边界**
- 修复路径（按成本）：
  1. 给 CANN 提 issue：310P 适配 `moe_gating_top_k_softmax_v2`（v2 kernel 已有，只差 op-plugin 分发）—— 最正统
  2. 自行修改 310P kernel 加行分块循环，重编 built-in 算子包 —— 可行但要维护
  3. 我们已用的纯 torch softmax+topk 替代 —— 已生效，性能损耗可忽略（gating 只占 0.17ms/层）

## 整体分析定稿：单请求慢的完整解释（2026-11-23）

### 结论先行

**单请求 3.7 tok/s 不是 Qwen3-Next 特有问题，是 vllm-ascend eager 模式在 310P 上的普遍水平。**
参考：Qwen3-30B-A3B-w8a8（TP1，PIECEWISE graph）当年 TPOT 也有 163.7ms/token——换算每层 4.1ms，
我们 48 层 270ms 每层 5.6ms，同一量级（我们多两次 all_reduce/层）。

### 每 token 270ms 的完整构成（py-spy 60s@300Hz，7971+ 样本）

| 类别 | 耗时 | 占比 | 说明 |
|---|---|---|---|
| 算子分发（OpOverloadPacket.__call__） | 69ms | 25.6% | ~1500 次 python→C++ |
| GDN 全链 | 52ms | 19.1% | l2norm+gating+recurrent+conv+norm |
| MoE 全链 | 32ms | 12.0% | gating+dispatch+gmm1/2+combine+shared |
| linear 投影（QKV/O/gate/shared） | 26ms | 9.5% | 含 unquantized_gemm 7.4% |
| all_reduce ×96 | 13ms | 4.8% | TP8 每层 2 次 |
| norm ×144 | 12ms | 4.5% | GemmaRMSNorm310 |
| 纯框架开销 | ~30ms | ~11% | Module._call_impl/__getattr__/dataclass init 等 |
| 其他杂项 | ~35ms | ~13% | rope/split/sigmoid/silu/cache wrapper |

### NPU 侧实测（decode 持续负载下）

- AICore 利用率：5.3%（峰值 6%）——95% 时间在等 CPU
- Memory BW：~2%——访存也远未饱和

### 与 Qwen3-30B-A3B 当年 msprof 结论交叉验证（完全吻合）

1. **host 下发是结构性热点**：它实测 launch node 1.94M 次，每层 host 下发 0.61ms 占 TPOT 18%；
   单 stream 忙碌占空比仅 12.3%，AICore 排队空转远大于计算
2. **decode 小 kernel 核并行度极低**：Block Num 只有 1~8（310P 有 20 个 AI Core），
   glue kernel 串行占 1 个核跑 20~35us
3. **所有 decode kernel 都是 MTE bound**：mac 利用率 2~3%，cube 几乎闲置
4. **每个 aclnn 调用都要先查 workspace**（GetWorkspaceSize 成对出现）——graph 模式能省的大头

### 为什么 micro-bench 只要 40ms 而真实服务 270ms

micro-bench 每层只放了 ~10 个"干净"算子（无 view/dtype/contiguous/元数据构建）；
真实模型每层 25-35 个算子 + 大量 python glue（qkv 重排、fix_query_key_value_ordering、
mixed_qkv 拆分、tensor cache wrapper、dataclass 构建）。差值 = 这些 glue 的 CPU 时间。

### 修复优先级（不考虑 MTP）

1. **PIECEWISE graph 模式**（30B-A3B 生产配置验证过）：把 96 次 all_reduce、144 次 norm、
   全部 aclnn workspace 查询烧进图，预计砍掉 40-60% CPU 开销 → **TPOT 有望从 270ms 降到 110-160ms**
   - 风险：GDN conv1d 回退路径在 capture 时的兼容性（FULL 失败的原因），PIECEWISE 只 capture
     attention+MoE 子图，GDN 段保持 eager，可能绕开
2. **算子融合**：gating 链（当前 5-6 个算子）融合为 1 个 AscendC 算子
3. **w8a8 量化**：权重搬运量减半，缓解 MTE bound

## 突破：PIECEWISE graph 模式跑通，decode 3.7 → 13 tok/s（3.5x）（2026-11-23）

### 配置

```bash
--compilation-config '{"cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1,2,4]}'
--max-num-seqs 4
```

关键：vllm 0.32 的 `splitting_ops` 已内置 `qwen_gdn_attention_core` / `linear_attention`，
GDN 核心被切出图外保持 eager，只有 attention+MoE 子图被 capture——这正是 30B-A3B 的生产模式。

### 实测

| 配置 | decode |
|---|---|
| eager（旧） | 3.7 tok/s |
| **PIECEWISE batch≤4** | **13.0 tok/s（3.5x）** |
| 首 token 延迟 | 858ms（与 eager 持平） |
| 4 并发 | 42.4 tok/s |
| ERROR | 0 |

### batch>=8 capture 崩溃的限制

`cudagraph_capture_sizes=[1,2,4,8,16]` 时 batch=8 的 capture 稳定触发 507903
（stream capture 序列失效）。与 max-num-batched-tokens 无关（256/1024 都崩）。
batch≤4 全部正常。疑似 310P driver 对 capture 图规模的限制（30B-A3B 记录里
"TP>1 graph 捕获 event 资源不足" 同源）。

### 对"怎么不用 eager"的完整回答

310P 上 non-eager 的三个层级：
1. **PIECEWISE cudagraph**（已跑通）✅ — vllm-ascend 原生支持，只 capture 非注意力子图
2. FULL cudagraph — 被 GDN conv1d 回退路径阻塞（capture 时同步），除非 conv1d 出 AscendC 版
3. torchair/npugraph_ex — 310P hardware profile 未支持（`npugraph_ex` capability 缺失）

结论：PIECEWISE 是当前唯一可用且效果最好的 non-eager 方案。

## FULL graph 模式修正与最终模式选择（2026-11-24）

### 修正一个错误结论

之前记录"FULL 失败是因为 GDN conv1d PyTorch 回退破坏 capture"——**这个判断错了**。
检查 gdn_310.py 发现 vllm-ascend 0.32 的 GDN decode 路径 conv1d **已经用 AscendC 算子**
（`torch.ops._C_ascend.npu_causal_conv1d_310`），不存在回退。

FULL_AND_PIECEWISE 模式失败的真实原因（serve_full2.log 实锤）：
```
Create capture event failed, stream_id=[9->372], error=117571609
```
**310P driver 的 capture event 资源耗尽**——FULL 模式每层一个图 × 48 层 + 多 stream 交互，
event 数量爆炸。48 次 capture event 创建失败。与 30B-A3B 记录"FULL/TP=2 捕获 event
资源不足"完全一致。

### FULL 模式实测（capture_sizes=[1] 时能启动）

| 模式 | 单请求 decode | 4 并发 |
|---|---|---|
| eager | 3.7 tok/s | ~13 tok/s |
| PIECEWISE [1,2,4] | 13.2 tok/s | **41.7 tok/s** |
| FULL [1] | **38.7 tok/s（10.5x）** | 13.5 tok/s（batch>1 无图回退） |

- FULL[1] 只 capture batch=1：单请求 38.7 tok/s（比 PIECEWISE 快 3 倍）
- 但 batch=2/4 无图命中回退 eager，2 并发反而掉到 6.6 tok/s
- capture_sizes=[1,2] 就 event 耗尽（48 次 fail）——FULL 模式 event 预算只够 1 个图

### 最终选择建议

| 场景 | 推荐模式 |
|---|---|
| 单请求/低并发流式（≤2路） | FULL_AND_PIECEWISE + capture_sizes=[1]（38.7 tok/s） |
| 并发服务（3-4路） | PIECEWISE + [1,2,4]（41.7 tok/s @ 4并发） |
| 更高并发 | 多实例数据并行（TP4×2） |

### event 资源问题可尝试的缓解

- `HCCL_OP_EXPANSION_MODE` / driver event 上限调优（需查 310P driver 文档）
- 向 CANN 提 issue：请求增大 capture event 资源池
- 上游 vllm-ascend issue：FULL 模式减少 per-layer 图数量（合并 capture）

## 评审修正（2026-11-24，外部评审意见采纳）

### 表述修正

1. **"gating 上限约 466"** → 精确表述：实测 E=464 通过、E=468 失败（k=10, tokens=8），
   边界在 464~468 之间；其他 k/tokens 组合未完整扫描，不排除边界随参数浮动。
   E=80/96/112/128/160/192/256/320/384/448 通过（16 对齐），65~78、84~92、104、120 失败（非 16 对齐）。

2. **"AICore 5.3% = 95% 时间等 CPU"** → 低利用率是必要非充分证据。
   与 30B-A3B msprof 的 host 下发占比 18%、单 stream 忙碌占空比 12.3% 交叉印证了
   CPU 主导，但 5.3% 与 94.7% 之间还包含 kernel 执行、排队、DMA 等其他成分，
   精确比例需要 Host/Device Timeline 对齐分析（待做）。

3. **"docker exec 环境变量不传给子进程"** → 准确描述：vllm 的 multiproc worker 使用
   spawn 方式启动，`docker exec -d` 里手动 export 的变量在 nohup+bash -c 链路中
   未能可靠传递到 EngineCore 子进程；`docker run -e` 注入则稳定。根因是
   vllm worker 的环境继承链，不是 Docker 通用行为。

4. **"310P 必须 bridge 网络"** → 仅在本机验证了 host 网络失败（HCCL error 19）。
   未排查具体 HCCL 接口/IP 配置（如 hccn.conf 为空、网卡路由），不能外推为
   310P 通用结论。

5. **"v2 算子只差 op-plugin 分发"** → 需补充确认：310P 的 kernel 二进制、tiling
   实现是否已在 libcust 包中就位。目前仅验证了 op-plugin 调用层报错。

6. **"FULL 崩溃是驱动缺陷"** → 修正为：driver capture event 资源耗尽（117571609）。
   是"资源硬限制/框架重复申请"还是"驱动泄漏"未区分——FULL_AND_PIECEWISE 双模式
   并存时 event 消耗更大，切换到纯 FULL_DECODE_ONLY 可能降低占用（待实验）。

7. **"稳定运行 4 天 0 ERROR"** → 补充：期间为低强度功能验证流量（数十次请求），
   非持续压测；未监控显存增长曲线。生产结论需补压测。

8. **"38.7 tok/s 可服务"** → 区分：FULL[1] 是单请求实验配置（max_num_seqs=1 语义）；
   生产服务配置为 PIECEWISE[1,2,4]（13.2 tok/s 单请求 / 41.7 tok/s @ 4 并发）。

### 启动脚本已同步

仓库 `scripts/start_serve.sh` 此前遗漏更新（仍是 eager 版），已同步为 PIECEWISE 生产配置。

## P0 实验：FULL_DECODE_ONLY 完整矩阵（2026-11-24，采纳评审建议）

### 实验矩阵结果

| 实验 | 模式 | capture_sizes | 启动 | event fail | 单请求 | 并发吞吐 |
|---|---|---|---|---|---|---|
| A | FULL_DECODE_ONLY | [1] | ✅ | 0 | 38.8 | 16并发 49.3 |
| E | FULL_DECODE_ONLY | [1,2] | ✅ | 0 | 38.9 | 2并发 49.6（延迟6s） |
| **F** | **FULL_DECODE_ONLY** | **[1,2,4]** | ✅ | **0** | **38.2** | **4并发 78.8（延迟8s）** |
| H | FULL_DECODE_ONLY | [1,2,4,8] | ❌ | 48 | - | - |
| （对照）FULL_AND_PIECEWISE | [1] | ✅ | 0 | 38.7 | 4并发 13.5（回退） |
| （对照）FULL_AND_PIECEWISE | [1,2] | ❌ | 48 | - | - |

### 关键结论

1. **评审判断证实**：FULL_AND_PIECEWISE 双模式并存是 event 耗尽的主因——
   decode FULL 图 + prefill PIECEWISE 图各占一份 event 预算。
   纯 FULL_DECODE_ONLY（prefill 走 eager）资源占用减半，能容纳 3 个 capture size。
2. **当前最优生产配置**：`FULL_DECODE_ONLY, capture_sizes=[1,2,4], max_num_seqs=4`
   - 单请求 38.2 tok/s
   - 4 并发 78.8 tok/s（平均延迟 8s）—— **比此前所有配置都快**
   - batch=4 图命中；batch≥8（capture 未覆盖）回退 eager，且 [1,2,4,8] event 耗尽
3. **FULL_DECODE_ONLY 语义**：decode 走 FULL 图，prefill 走 eager。
   prefill 图不 capture 反而省 event；prefill 的 849ms TTFT 与其他模式持平。

### 与旧配置对比

| 配置 | 单请求 | 4并发 | 16并发 |
|---|---|---|---|
| eager（最初） | 3.7 | ~13 | ~13 |
| PIECEWISE[1,2,4] | 13.2 | 41.7 | — |
| FULL_AND_PIECEWISE[1] | 38.7 | 13.5 | 49.3 |
| **FULL_DECODE_ONLY[1,2,4]** | **38.2** | **78.8** | 48.7（回退） |

### event 资源机制推断

- FULL_AND_PIECEWISE[1]：decode FULL 图 1 个 + prefill PW 图 1 个 = 2 套 event
- FULL_DECODE_ONLY[1,2,4]：decode FULL 图 3 个 + prefill 无 = 3 套 event ≈ 前者 1.5 倍？
  实际 [1,2] 就 0 fail 而 FAP[1,2] 48 fail —— 说明 PW prefill 图的 event 消耗
  远大于 FULL decode 图（PW 图按 splitting_ops 切成 ~96 个子图，每子图独立 event）
- 精确 event 计数需要 driver 层工具，当前为行为推断

## FULL Graph 模式 msprof 剖析（2026-11-24，50 tok decode，rank0）

### 采集方法

vllm 0.32 内置 profiler（`--profiler-config '{"profiler":"torch","torch_profiler_dir":"...","torch_profiler_activities":["CPU","PrivateUse1"]}'`）
+ `msprof --export=on` 解析 rank 目录的 aicore 原始数据 → op_statistic / api_statistic CSV。
（msprof application 模式对 multiproc worker 不可用：dynamic profiling socket 连接超时；
msprof --pid 在容器内也报 no valid pid。torch profiler HTTP 接口是可行路径。）

### NPU 利用率对比（npu-smi watch，decode 持续负载）

| 模式 | AICore avg | 峰值 | MemBW |
|---|---|---|---|
| eager | 5.3% | 7% | ~2% |
| **FULL_DECODE_ONLY** | **9.8~13.1%** | **61%** | **最高 20.7%** |

### rank0 device 算子统计（50 tok，总计 1202ms）

| 算子 | 次数 | 总ms | 均us | 占比 |
|---|---|---|---|---|
| GroupedMatmul | 4800 | 486.6 | 101.4 | 40.5% |
| MatMulV2 | 16250 | 232.8 | 14.3 | 19.4% |
| **TransData** | **23174** | **82.3** | 3.6 | 6.8% |
| Cast | 23135 | 54.6 | 2.4 | 4.5% |
| RmsNorm | 11450 | 35.4 | 3.1 | 2.9% |
| RecurrentGatedDeltaRuleV310 | 1764 | 20.9 | 11.8 | 1.7% |
| MoeInitRouting/Unpermute | 4800 | 38.7 | - | 3.2% |

**关键数字**：
- GroupedMatmul 4800 次 = 50 tok × 48 层 × 2 (gmm1+gmm2)，单次 101us
- TransData 23174 次 / 82ms —— graph 模式下仍在发生（每层多个），但单次仅 3.6us
  （eager 微基准里 143us 是含同步的端到端时间；graph 内批量流水后单次开销大降）
- RecurrentGatedDeltaRuleV310 单次 11.8us —— AscendC 算子本身极快

### rank0 host API 统计（总计 2460ms）

| API | 次数 | 总ms | 均us |
|---|---|---|---|
| aclrtSynchronizeEvent | 50 | 733.6 | 14673 |
| launch（kernel 下发） | 25452 | 360.1 | 14.2 |
| aclnnInplaceCopy | 9173 | 177.1 | 19.3 |
| aclrtSynchronizeStream | 36 | 162.7 | 4519 |
| hcom_allReduce | 97 | 62.7 | 646.6 |

**FULL graph 下的剩余瓶颈**：
1. aclrtSynchronizeEvent 50 次 733ms —— 每 token 一次大同步（14.7ms/次），
   这是调度器级别的强制同步（每 token 等待图执行完），不是图内部开销
2. launch 25452 次 360ms —— 图 replay 时 host 仍需逐 kernel 下发（310P 不支持
   整图 device 端自执行），但单次降到 14.2us（eager 时约 180us）
3. allReduce 97 次 646us/次 —— TP8 通信延迟，96 次/token 预期吻合

### 结论

FULL graph 把 per-op 分发从 180us 压到 14us（13 倍），AICore 利用率翻倍。
剩余 270ms→25.8ms/token 的构成中，最大的单项是每 token 一次的调度器同步
（14.7ms），其次是 launch（7.2ms/token）和 allReduce（13ms/token）。
进一步优化空间：调度器同步与 pipeline 重叠、allReduce 与计算重叠。

## 10 并发场景实测（2026-11-24）

### 当前配置（FULL_DECODE_ONLY[1,2,4], max_num_seqs=4）下 10 并发

| 场景 | 总吞吐 | 每请求平均 | 说明 |
|---|---|---|---|
| 10 并发 × 短上下文 (200tok out) | 74.6 tok/s | 7.5 tok/s | batch=4 图命中 + 排队轮转 |
| 10 并发 × 混合上下文 (690~5458pt) | 29.1 tok/s | 2.9 tok/s | prefill 占比上升 |

### 上下文长度支持

- **硬件/服务上限**：max_model_len=8192（可配置到更大，KV cache 535K tok 可支撑 65 路 8K）
- **实测通过**：单请求 5458pt prefill 正常（还能更大）
- **10 并发下的上下文**：每个请求都可以独立使用最大 8192 token 上下文，
  KV cache 完全够（10×8192=82K << 535K）。限制不在上下文长度，在 decode 并发

### 10 并发的结构性限制

capture_sizes 最大 [1,2,4]（batch≥8 的 graph capture 触发 driver event 耗尽），
因此 10 并发时 vllm 调度器把请求分成 batch≤4 的批次轮转处理：
- 吞吐不会因并发数增加而线性提升（4 并发 78.8 → 10 并发 74.6，略降）
- 每请求分到的 decode 带宽约 7.5 tok/s（短上下文）
- 长上下文请求 prefill 耗时显著（5458pt prefill 约数秒）

### 提升并发容量的路径

1. **多实例 DP**：起 2-3 个 TP2/TP4 实例（需 W8A8 减小权重），每实例独立 capture
2. **driver event 资源扩展**：向 CANN 确认 310P capture event 上限能否调大
3. **上限估算**：若 capture 支持到 batch=16，10 并发可全部命中图，预期总吞吐 100+ tok/s
