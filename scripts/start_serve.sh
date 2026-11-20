#!/bin/bash
# Qwen3-Next-80B-A3B on 310P TP8 服务启动脚本
docker start qwen3next-310p 2>/dev/null
sleep 5
docker exec -d qwen3next-310p bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh; cd /workspace && nohup vllm serve /models \
  --served-model-name qwen3-next-80b \
  --host 0.0.0.0 --port 8320 \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --dtype float16 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code \
  > /workspace/serve_final.log 2>&1 &"
echo "vllm launching on port 8320 (container-internal), 首次启动约5分钟"
