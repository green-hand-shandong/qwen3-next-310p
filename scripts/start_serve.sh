#!/bin/bash
# Qwen3-Next-80B-A3B on 310P TP8 — PIECEWISE graph 模式（推荐配置）
# 性能: decode 13 tok/s（eager 3.7 的 3.5 倍）
# 注意: cudagraph_capture_sizes 最大到 4；batch>=8 的 capture 会触发 310P driver bug (507903)
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
  --compilation-config '{\"cudagraph_mode\": \"PIECEWISE\", \"cudagraph_capture_sizes\": [1,2,4]}' \
  > /workspace/serve_final.log 2>&1 &"
echo "vllm PIECEWISE launching on port 8320 (container-internal), 启动约 6 分钟"
