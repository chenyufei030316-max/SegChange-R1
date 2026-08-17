#!/bin/bash
# 用法: run_with_retry.sh <config.yaml> <max_retries>
# SegChange的resume是yaml字段,不是CLI参数，每次重试前检查latest.pth是否存在，
# 存在的话生成一个带resume的临时config再跑。
set -u
CONFIG=$1
MAX_RETRIES=${2:-3}
cd /ssd4Tb/chenyf/SegChange-R1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate segchange

NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['name'])")
CKPT="work_dirs/${NAME}/checkpoints/latest.pth"
TMPCFG="/tmp/$(basename $CONFIG .yaml)_retry.yaml"

attempt=0
while [ $attempt -le $MAX_RETRIES ]; do
  echo "[retry-wrapper] $NAME attempt $attempt: $(date)"
  if [ -f "$CKPT" ]; then
    python3 -c "
import yaml
d = yaml.safe_load(open('$CONFIG'))
d['resume'] = '$CKPT'
yaml.safe_dump(d, open('$TMPCFG', 'w'))
"
    RUNCFG=$TMPCFG
    echo "[retry-wrapper] $NAME resuming from $CKPT"
  else
    RUNCFG=$CONFIG
  fi
  python train.py -c $RUNCFG
  code=$?
  if [ $code -eq 0 ]; then
    echo "[retry-wrapper] $NAME succeeded on attempt $attempt: $(date)"
    rm -f $TMPCFG
    exit 0
  fi
  echo "[retry-wrapper] $NAME FAILED (exit $code) on attempt $attempt: $(date)"
  attempt=$((attempt+1))
  sleep 5
done
echo "[retry-wrapper] $NAME exhausted $MAX_RETRIES retries, giving up: $(date)"
rm -f $TMPCFG
exit 1
