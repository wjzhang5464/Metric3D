#!/bin/bash
# Fine-tune Metric3D v2-S on TartanAir v2 (reports/finetune_v2s_tartanair/PROTOCOL.md). Started by hand on a
# compute node; progress is printed live (every 100 iterations: losses, LR, time per iteration, ETA) and logged.
#
#   bash tartanair_finetune/run_finetune.sh a     # run (a): depth_normalize=(0.1, 700)  -> ft_v2s_ta2_dn700
#   bash tartanair_finetune/run_finetune.sh b     # run (b): depth_normalize=(0.1, 200)  -> ft_v2s_ta2_dn200
#
# Run it inside tmux (or screen) on the node so a dropped connection does not kill it:
#   tmux new -s ft_a      ... bash tartanair_finetune/run_finetune.sh a      (detach: Ctrl-b d; back: tmux attach -t ft_a)
#
# If a run stops early, re-running the same command resumes from its newest checkpoint (one per epoch) via the
# official --resume-from. Without a checkpoint it starts from the official v2-S weights (--load-from).
# 20 epochs = 87,220 iterations; about 1.2 s per iteration on one A100 (smoke test), i.e. about 29 h per run.
set -euo pipefail

RUN=${1:-}
case "$RUN" in
  a) CFG=mono/configs/RAFTDecoder/vit.raft5.small.tartanair.dn700.py; NAME=ft_v2s_ta2_dn700 ;;
  b) CFG=mono/configs/RAFTDecoder/vit.raft5.small.tartanair.dn200.py; NAME=ft_v2s_ta2_dn200 ;;
  *) echo "usage: $0 a|b" >&2; exit 2 ;;
esac

case "$(hostname)" in
  gilbreth-fe*|*login*) echo "refusing to train on a login node ($(hostname)); use your interactive compute node" >&2; exit 1 ;;
esac
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=/depot/chan129/users/zhan5056/virtual_env/Metric3D_ViT/bin/python
SCRATCH=${RCAC_SCRATCH:-/scratch/gilbreth/zhan5056}
WEIGHTS=$SCRATCH/metric3d/weights/metric_depth_vit_small_800k.pth
WORK=${FT_WORK:-$SCRATCH/metric3d/checkpoints/$NAME}   # FT_WORK / FT_OPTIONS: for smoke tests only
OPTIONS=(${FT_OPTIONS:-})
SEED=3047
FINAL_STEP=87220   # 20 epochs x 4,361 iterations (see the config)
mkdir -p "$WORK"

LATEST=$(ls "$WORK"/ckpt/step*.pth 2>/dev/null | sort | tail -n 1 || true)
if [ -n "$LATEST" ] && [ -z "${FT_OPTIONS:-}" ] && [ "$(basename "$LATEST")" = "$(printf 'step%08d.pth' $FINAL_STEP)" ]; then
  echo "run $RUN is already complete: $LATEST"; exit 0
fi
if [ -n "$LATEST" ]; then
  INIT=(--resume-from "$LATEST"); echo "resuming from $LATEST"
else
  INIT=(--load-from "$WEIGHTS"); echo "starting from the official weights $WEIGHTS"
fi

{
  echo "=== $(date '+%F %T') run $RUN ($NAME) on $(hostname), commit $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD -- training tartanair_finetune || echo '+uncommitted')"
  echo "config $CFG; seed $SEED; ${INIT[*]}"
} | tee -a "$WORK/run_history.txt"

cd "$REPO/training"
# -u: unbuffered, so the log shows progress immediately. The official logger also writes <work-dir>/<timestamp>.log.
"$PY" -u -B mono/tools/train.py "$CFG" \
    --launcher None \
    --use-tensorboard \
    --seed "$SEED" \
    "${INIT[@]}" \
    --work-dir "$WORK" \
    --experiment_name "$NAME" \
    ${OPTIONS[@]+--options "${OPTIONS[@]}"} \
    2>&1 | grep --line-buffered -v -e 'pkg_resources' -e 'declare_namespace' -e 'warnings.warn' \
         | tee -a "$WORK/console.log"

echo "=== $(date '+%F %T') finished; checkpoints in $WORK/ckpt" | tee -a "$WORK/run_history.txt"
