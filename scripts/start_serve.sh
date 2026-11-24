#!/bin/bash
# Qwen3-Next-80B-A3B on 310P TP8 — FULL_DECODE_ONLY（当前最优生产配置，2026-11-24）
# 单请求 38.2 tok/s；4 并发 78.8 tok/s（延迟 8s）
# 限制: capture_sizes 上限 [1,2,4]；batch>=8 回退 eager（driver event 资源限制）
docker start qwen3next-310p 2>/dev/null
sleep 5
docker exec -d qwen3next-310p bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh; cd /workspace && nohup vllm serve /models \
  --served-model-name qwen3-next-80b \
  --host 0.0.0.0 --port 8320 \
  --tensor-parallel-size 8 \
  --dtype float16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code \
  --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [1,2,4]}' \
  > /workspace/serve_final.log 2>&1 &"
echo "vllm FULL_DECODE_ONLY launching on port 8320 (container-internal), 启动约 7 分钟"
