#!/bin/bash
# 等贝叶斯剩余臂训完 -> 拉权重 -> 6 分片评测 -> 合并。
# 完成判据用 sentinel 文件与 episode 校验, 不用进程存在性 (CLAUDE.md 17.1)。
set -u
W=/home/abc/temp/cc_6env_20261007
P='/home/abc/workspace/CrowdNav(20270731_backup2)/CrowdNav'
R=root@REMOTE_HOST_REDACTED
RB=/root/bayes-vl-optimized-20261007/crowd_nav/runs/optimized-10000-r1
PY=/root/miniconda3/envs/mamba/bin/python
log(){ echo "[$(date +%H:%M:%S)] $*" >> $W/watcher.log; }

declare -A BB=( [formal-bayes]=bayes [formal-bayes_mean]=bayes_mean )
PENDING="formal-bayes_mean formal-bayes"

log "watcher 启动, 待评: $PENDING"
while [ -n "$PENDING" ]; do
  NEXT=""
  for arm in $PENDING; do
    [ -f "$W/bayes/$arm.json" ] && { log "$arm 已完成评测, 跳过"; continue; }
    # 远端 checkpoint 是否存在且 episode==10000 (完整性校验, 避免拷到半写文件)
    ok=$(timeout 120 ssh -o ConnectTimeout=20 $R "cd /root/bayes-vl-optimized-20261007 && \
      [ -f $RB/$arm/rl_model_ep10000.pth ] && $PY -c \"
import torch,sys
try:
    d=torch.load('$RB/$arm/rl_model_ep10000.pth',map_location='cpu',weights_only=False)
    print('READY' if d.get('episode')==10000 else 'PARTIAL')
except Exception as e:
    print('PARTIAL')
\" || echo MISSING" 2>/dev/null | tail -1)
    if [ "$ok" != "READY" ]; then
      log "$arm 还没好 ($ok)"; NEXT="$NEXT $arm"; continue
    fi
    log "$arm checkpoint READY, 开始拉取"
    mkdir -p $W/arms/$arm $W/shards/$arm
    timeout 600 scp -q $R:$RB/$arm/{config.ini,rl_model_ep10000.pth} $W/arms/$arm/ || {
      log "$arm scp 失败, 下轮重试"; NEXT="$NEXT $arm"; continue; }
    rsha=$(timeout 120 ssh -o ConnectTimeout=20 $R "sha256sum $RB/$arm/rl_model_ep10000.pth" 2>/dev/null | awk '{print $1}')
    lsha=$(sha256sum $W/arms/$arm/rl_model_ep10000.pth | awk '{print $1}')
    if [ "$rsha" != "$lsha" ]; then log "$arm sha 不符($rsha vs $lsha), 下轮重试"; NEXT="$NEXT $arm"; continue; fi
    log "$arm sha 校验通过, 起 6 分片"
    for i in 0 1 2 3 4 5; do
      python3 $W/eval6_bayes.py --project "$P" --arm-dir $W/arms/$arm \
        --backbone ${BB[$arm]} --out $W/shards/$arm/cell$i.json --cells $i \
        > $W/shards/$arm/cell$i.log 2>&1 &
    done
    wait
    if python3 $W/merge.py $W/shards/$arm $W/bayes/$arm.json >> $W/watcher.log 2>&1; then
      log "$arm 评测完成 -> bayes/$arm.json"
      touch $W/DONE.$arm
    else
      log "$arm 合并失败, 见 shards/$arm/*.log"; NEXT="$NEXT $arm"
    fi
  done
  PENDING=$(echo $NEXT | xargs 2>/dev/null || true)
  [ -n "$PENDING" ] && sleep 180
done
log "全部完成"
touch $W/DONE.ALL
