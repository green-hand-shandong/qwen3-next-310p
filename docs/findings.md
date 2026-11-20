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
