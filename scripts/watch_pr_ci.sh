#!/bin/bash
# 监控 PR #23076 CI 状态
GH_TOKEN_LINE=$(grep 'x-access-token' ~/.git-credentials | head -1)
GH_TOKEN=$(echo "$GH_TOKEN_LINE" | sed 's|https://x-access-token:||; s|@github.com||')
SHA=$(cd /data4/vllm-ascend-pr2 && git rev-parse HEAD)
while true; do
  R=$(curl -s -m 30 -H "Authorization: token $GH_TOKEN" "https://api.github.com/repos/vllm-project/vllm-ascend/commits/$SHA/check-runs")
  FAILS=$(echo "$R" | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for r in d.get('check_runs',[]) if r.get('conclusion')=='failure'))")
  PEND=$(echo "$R" | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for r in d.get('check_runs',[]) if r['status'] not in ('completed','skipped')))")
  echo "$(date +%H:%M) fails=$FAILS pending=$PEND"
  if [ "$FAILS" -gt 0 ]; then echo "CI FAILED"; break; fi
  if [ "$PEND" -eq 0 ]; then echo "ALL CHECKS DONE"; break; fi
  sleep 300
done
