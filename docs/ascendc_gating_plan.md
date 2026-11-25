# 自定义 gating AscendC 算子：最终评估

## 分段调用方案（已否决）
E=448+64 两段各自调原生算子再合并 → 不可行：
1. 原生算子输出段内 softmax，两段概率不可直接比（12% index 一致率）
2. 2 次算子调用 0.616ms，比 torch 版 0.201ms 更慢

## 三条路线的最终对比

### 路线A：改 CANN built-in kernel（加行分块）
- kernel 源可读（244 行，结构清晰），但 host tiling 编在 .so 里
- E=468 崩而 UB 只用 4% → 崩因是 tiling 的 SoftMax/TopK API 参数 bug，
  不是真容量不够。改 kernel 不改 tiling 可能仍崩
- 需要重编 CANN built-in 包并污染全局环境

### 路线B：vllm-ascend 内写全新 AscendC 算子
- 走 recurrent_v310 模式（JIT），构建链已打通（pip install -e 验证过）
- 工作量四件套：kernel .h + host tiling .cpp + dynamic JIT py + torch_binding 注册
- TopK tiling 逻辑复杂，需要从 CANN 源码借鉴
- 收益：10.2→2.8ms/token，TPOT 25.8→18.4，38.7→54 tok/s (+40%)
- 且可提交 upstream，一劳永逸

### 路线C：找 built-in tiling bug
- E=468 崩但 UB 只用 4% → 疑似 SoftMax/TopK API tiling 参数 bug
- tiling 在 .so 里，需要 CANN 源码或逆向——不可控

## 建议
路线B 是正解，但工作量大（估计 2-3 天含调试）。作为独立任务排期。
短期维持 torch 版（39% TPOT 占比但功能正确），把精力先给 W8A8（收益更直接）。
