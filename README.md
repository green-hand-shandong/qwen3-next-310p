# Qwen3-Next-80B-A3B on Ascend 310P

[中文](#中文) | [English](#english)

---

<a name="中文"></a>

## 中文

在华为昇腾 Atlas 300I Duo（Ascend 310P3）上，使用 vLLM-Ascend 跑通 **Qwen3-Next-80B-A3B**（混合线性注意力 + MoE 架构）的完整方案。

**无需 Triton**。整个方案绕开了 triton-ascend 不支持 310P 的限制，全部使用 AscendC 算子 + PyTorch 回退路径。

### 结果

| 项目 | 数据 |
|---|---|
| 模型 | Qwen3-Next-80B-A3B-Instruct（512 experts MoE，48 层，GDN 线性注意力） |
| 硬件 | Atlas 300I Duo × 4 卡（8芯片，TP8） |
| 软件 | vllm-ascend v0.32.0rc1-310p 镜像 |
| 单请求 decode（FULL graph） | **38.7 tok/s**（eager 3.7 的 10.5 倍） |
| 4 并发聚合吞吐（FULL_DECODE_ONLY） | **78.8 tok/s**（平均延迟 8s） |
| 10 并发聚合吞吐 | 74.6 tok/s（排队轮转） |
| 多轮对话正确性 | ✅（7→×6=42→×2=84 连续推理正确） |
| Prefix caching | ✅ 长前缀二次请求提速 ~30% |
| 服务稳定性 | 0 ERROR 持续运行 |

### 核心问题与解决

#### 1. MoE gating 算子 64 expert 上限（关键补丁）

**现象**：启动即崩，`MTE instruction ... out of bounds`，error code 507903。

**根因**：`torch_npu.npu_moe_gating_top_k_softmax` 在 310P 上 expert 数 **>64 即触发 AICore 越界**（二分定位：E=64 正常，E=65 崩溃）。Qwen3-Next 有 **512 experts**，原生算子完全不可用。

**修复**：[`patches/0005-moe-gating-topk-64expert-limit.patch`](patches/0005-moe-gating-topk-64expert-limit.patch) —— 把 `_310p/fused_moe/grouped_topk_router.py` 的 `_compute_routing` 替换为纯 torch 实现（`softmax + topk`，基础算子无 expert 数限制），数值与原算子一致。

#### 2. GDN 状态表 bug（已知上游问题，PR #18730）

多轮对话 / MTP 投机解码场景下，`aclnnRecurrentGatedDeltaRuleV310` 因上一轮 accepted tokens > 当前 query 长度时状态表索引错误而崩溃。

**补丁**：[`patches/0001-gdn-17856-state-table.patch`](patches/0001-gdn-17856-state-table.patch)（对应上游 [PR #18730](https://github.com/vllm-project/vllm-ascend/pull/18730)）。

⚠️ 该补丁的 kernel `.h` 部分可热同步，但 **op_host tiling 部分需要重编 `cann-ops-transformer-custom` 包**。未打此补丁时：普通多轮对话正常（已实测），**MTP 投机解码不可用**（启动后首个请求即崩）。

#### 3. 环境配置要点

- **必须用 bridge 网络**：`--net=host` 下 HCCL 初始化失败（error 19, call network api fail）
- **容器级注入 `ASCEND_RT_VISIBLE_DEVICES`**：`docker exec` 内的环境变量不会传给 worker 子进程
- **必须 `FULL_DECODE_ONLY` graph mode**：aclgraph capture 对 GDN 的 conv1d PyTorch 回退路径失败（error 507903, rtStreamEndCapture）
- 310P 上 `npu_moe_gating_top_k_softmax` 的 UB 容量限制是硬件性的，软件规避只能换实现

### 快速开始

```bash
# 1. 拉镜像
docker pull quay.io/ascend/vllm-ascend:v0.32.0rc1-310p

# 2. 下载权重（ModelScope）
modelscope download --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --local_dir /data4/models/Qwen3-Next-80B-A3B-Instruct

# 3. 启动容器（8 芯片 TP8，bridge 网络）
docker run -itd --privileged --name=qwen3next-310p \
  --shm-size=200g \
  -e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  --device=/dev/davinci8 --device=/dev/davinci9 \
  --device=/dev/davinci10 --device=/dev/davinci11 \
  --device=/dev/davinci12 --device=/dev/davinci13 \
  --device=/dev/davinci14 --device=/dev/davinci15 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /var/log/npu/slog:/var/log/npu/slog \
  -v /data4/models/Qwen3-Next-80B-A3B-Instruct:/models:ro \
  -v /data4/qwen3next-logs:/workspace \
  quay.io/ascend/vllm-ascend:v0.32.0rc1-310p /bin/bash

# 4. 应用 MoE gating 补丁（容器内）
docker cp patches/0005-moe-gating-topk-64expert-limit.patch qwen3next-310p:/tmp/
docker exec qwen3next-310p bash -c "cd /vllm-workspace/vllm-ascend && git apply /tmp/0005-*.patch \
  && find vllm_ascend -name '__pycache__' -path '*_310p*' | xargs rm -rf"

# 5. 启动服务
bash scripts/start_serve.sh   # 首次启动约 5 分钟

# 6. 测试（容器内）
curl http://localhost:8320/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-next-80b","messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
```

### 仓库结构

```
├── README.md                                    # 本文档（中英双语）
├── patches/
│   ├── 0005-moe-gating-topk-64expert-limit.patch  # 本项目原创：512-expert MoE 在 310P 的 gating 修复
│   ├── 0001-gdn-17856-state-table.patch           # 上游 PR #18730：GDN 状态表修复（开 MTP 必需）
│   ├── 0002-gdn-causal-mask-vector.patch          # chunk_fwd_o 掩码修复（参考）
│   └── grouped_topk_router_310p_patched.py        # 补丁后的完整文件（备用）
├── scripts/
│   └── start_serve.sh                           # 服务启动脚本
└── docs/
    └── findings.md                              # 详细排查记录
```

### 已知限制

- 图模式 capture_sizes 上限 [1,2,4]：batch≥8 的 graph capture 触发 310P driver event 资源耗尽（507903），高并发需多实例扩展
- MTP 投机解码：需先重编 v310 算子 host 库（0001 补丁），暂不可用
- 图模式（aclgraph）：与 GDN conv1d 回退路径冲突，必须 enforce-eager
- 镜像内 `vllm_ascend/_cann_ops_custom` 的 kernel 是预编译 .o，改源码需走完整 CMake 构建

### 致谢

- [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 项目及 310P 维护者
- [PR #18730](https://github.com/vllm-project/vllm-ascend/pull/18730) GDN 状态表修复
- Qwen3-Next 模型（阿里通义）

## License

Apache 2.0

---

<a name="english"></a>

## English

A complete, working recipe for running **Qwen3-Next-80B-A3B** (hybrid linear attention + MoE) on Huawei Ascend Atlas 300I Duo (310P3) with vLLM-Ascend.

**Triton-free.** The whole approach sidesteps the fact that triton-ascend does not support 310P, using AscendC operators and PyTorch fallback paths exclusively.

### Results

| Item | Value |
|---|---|
| Model | Qwen3-Next-80B-A3B-Instruct (512-expert MoE, 48 layers, GDN linear attention) |
| Hardware | Atlas 300I Duo × 4 cards (8 chips, TP8) |
| Software | vllm-ascend v0.32.0rc1-310p image |
| Single-request decode (FULL graph) | **38.7 tok/s** (10.5x over eager 3.7) |
| 4-way concurrent (FULL_DECODE_ONLY) | **78.8 tok/s** aggregate (8s avg latency) |
| 10-way concurrent | 74.6 tok/s aggregate (queued rotation) |
| Multi-turn correctness | ✅ (7→×6=42→×2=84 chained reasoning correct) |
| Prefix caching | ✅ ~30% faster on repeated long prefixes |
| Service stability | 0 ERROR under sustained load |

### Key Problems & Fixes

#### 1. MoE gating operator fails beyond 64 experts (the critical patch)

**Symptom**: crash on startup, `MTE instruction ... out of bounds`, error code 507903.

**Root cause**: `torch_npu.npu_moe_gating_top_k_softmax` triggers an AICore out-of-bounds on 310P when **expert count > 64** (bisected: E=64 fine, E=65 crashes). Qwen3-Next has **512 experts**, so the native operator is unusable.

**Fix**: [`patches/0005-moe-gating-topk-64expert-limit.patch`](patches/0005-moe-gating-topk-64expert-limit.patch) — replaces `_compute_routing` in `_310p/fused_moe/grouped_topk_router.py` with a plain-torch implementation (`softmax + topk`, basic operators with no expert-count limit). Numerically equivalent to the native operator.

#### 2. GDN state table bug (known upstream issue, PR #18730)

In multi-turn / MTP speculative decoding scenarios, `aclnnRecurrentGatedDeltaRuleV310` crashes when the previous round's accepted tokens exceed the current query length, due to wrong state-table indexing.

**Patch**: [`patches/0001-gdn-17856-state-table.patch`](patches/0001-gdn-17856-state-table.patch) (mirrors upstream [PR #18730](https://github.com/vllm-project/vllm-ascend/pull/18730)).

⚠️ The kernel `.h` part of this patch can be hot-synced, but the **op_host tiling part requires rebuilding the `cann-ops-transformer-custom` package**. Without it: plain multi-turn chat works fine (verified), but **MTP speculative decoding is unavailable** (first request crashes the engine).

#### 3. Environment gotchas

- **Bridge network is mandatory**: HCCL init fails under `--net=host` (error 19, call network api fail)
- **Inject `ASCEND_RT_VISIBLE_DEVICES` at container level**: env vars from `docker exec` do not propagate to worker subprocesses
- **`FULL_DECODE_ONLY` graph mode is mandatory**: aclgraph capture fails on GDN's conv1d PyTorch fallback (error 507903, rtStreamEndCapture)
- The UB-capacity limit of `npu_moe_gating_top_k_softmax` on 310P is hardware-level; software mitigation requires a different implementation

### Quick Start

```bash
# 1. Pull the image
docker pull quay.io/ascend/vllm-ascend:v0.32.0rc1-310p

# 2. Download weights (ModelScope)
modelscope download --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --local_dir /data4/models/Qwen3-Next-80B-A3B-Instruct

# 3. Start the container (8 chips, TP8, bridge network)
docker run -itd --privileged --name=qwen3next-310p \
  --shm-size=200g \
  -e ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  --device=/dev/davinci8 --device=/dev/davinci9 \
  --device=/dev/davinci10 --device=/dev/davinci11 \
  --device=/dev/davinci12 --device=/dev/davinci13 \
  --device=/dev/davinci14 --device=/dev/davinci15 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /var/log/npu/slog:/var/log/npu/slog \
  -v /data4/models/Qwen3-Next-80B-A3B-Instruct:/models:ro \
  -v /data4/qwen3next-logs:/workspace \
  quay.io/ascend/vllm-ascend:v0.32.0rc1-310p /bin/bash

# 4. Apply the MoE gating patch (inside container)
docker cp patches/0005-moe-gating-topk-64expert-limit.patch qwen3next-310p:/tmp/
docker exec qwen3next-310p bash -c "cd /vllm-workspace/vllm-ascend && git apply /tmp/0005-*.patch \
  && find vllm_ascend -name '__pycache__' -path '*_310p*' | xargs rm -rf"

# 5. Start the service
bash scripts/start_serve.sh   # first start takes ~5 min

# 6. Test (from inside the container)
curl http://localhost:8320/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-next-80b","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### Repository Layout

```
├── README.md                                    # this document (bilingual)
├── patches/
│   ├── 0005-moe-gating-topk-64expert-limit.patch  # original to this project: 512-expert MoE gating fix for 310P
│   ├── 0001-gdn-17856-state-table.patch           # upstream PR #18730: GDN state table fix (required for MTP)
│   ├── 0002-gdn-causal-mask-vector.patch          # chunk_fwd_o mask fix (reference)
│   └── grouped_topk_router_310p_patched.py        # the patched file in full (fallback)
├── scripts/
│   └── start_serve.sh                           # service launcher
└── docs/
    └── findings.md                              # detailed troubleshooting log
```

### Known Limitations

- Graph capture_sizes capped at [1,2,4]: batch>=8 graph capture triggers 310P driver event exhaustion (507903); scale out with multiple instances for higher concurrency
- MTP speculative decoding: requires rebuilding the v310 operator host library (patch 0001); not yet available
- Graph mode (aclgraph): conflicts with GDN conv1d fallback path; enforce-eager required
- Kernels in `vllm_ascend/_cann_ops_custom` inside the image are prebuilt `.o` files; source changes require a full CMake build

### Acknowledgements

- The [vllm-ascend](https://github.com/vllm-project/vllm-ascend) project and its 310P maintainers
- [PR #18730](https://github.com/vllm-project/vllm-ascend/pull/18730) GDN state table fix
- Qwen3-Next model (Alibaba Tongyi)

## License

Apache 2.0
