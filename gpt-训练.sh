#!/usr/bin/env bash
# 仓库默认 Go2 主方法；参数：物理 GPU 编号、随机种子、迭代数、并行环境数。
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GPU_ID=${1:-0}
TRAIN_SEED=${2:-46}
TRAIN_ITERATIONS=${3:-30000}
TRAIN_ENVS=${4:-7500}
for value in "$GPU_ID" "$TRAIN_SEED" "$TRAIN_ITERATIONS" "$TRAIN_ENVS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo '参数必须为非负整数。' >&2; exit 2; }
done
SESSION_NAME="gpt-locomposition-s${TRAIN_SEED}"
if [[ -z "${TMUX:-}" ]]; then
    printf -v RUN_COMMAND 'bash %q %q %q %q %q' "$PROJECT_DIR/gpt-训练.sh" "$GPU_ID" "$TRAIN_SEED" "$TRAIN_ITERATIONS" "$TRAIN_ENVS"
    tmux new-session -d -s "$SESSION_NAME" "$RUN_COMMAND"
    echo "已创建训练会话：tmux attach -t $SESSION_NAME"
    exit 0
fi
export ENV_NAME="gpt-go2-default-seed${TRAIN_SEED}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=Y OMNICLIENT_HUB_MODE=disabled WANDB_MODE=disabled
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd "$PROJECT_DIR/locomposition"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG_PATH="$PROJECT_DIR/gpt-复现记录/gpt-正式训练-seed${TRAIN_SEED}-${STAMP}.log"
STATUS_PATH="$PROJECT_DIR/gpt-复现记录/gpt-训练状态-seed${TRAIN_SEED}-${STAMP}.txt"
mkdir -p "$PROJECT_DIR/gpt-复现记录"
echo "正在训练，日志：$LOG_PATH" | tee "$STATUS_PATH"
set +e
"$PROJECT_DIR/.venv/bin/python" scripts/clean_rl/train.py \
    --task=LoComposition-Go2-Rough-Terrain-Joint-State-History-v0 \
    --seed="$TRAIN_SEED" --headless --num_envs="$TRAIN_ENVS" \
    --num_iterations="$TRAIN_ITERATIONS" --logger=tensorboard --device=cuda:0 \
    env.scene.ray_caster.debug_vis=false \
    env.scene.ray_caster_height_constraints.debug_vis=false \
    env.commands.base_velocity.debug_vis=false 2>&1 | tee "$LOG_PATH"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
if ((TRAIN_EXIT != 0)); then
    echo "训练失败，退出码：$TRAIN_EXIT；日志：$LOG_PATH" | tee "$STATUS_PATH"
    exit "$TRAIN_EXIT"
fi
RUN_DIR=$(sed -n 's/^\[INFO\] Reserved run directory: //p' "$LOG_PATH" | tail -1)
[[ -f "$RUN_DIR/model_${TRAIN_ITERATIONS}.pt" ]] || { echo '最终模型缺失。' | tee "$STATUS_PATH"; exit 1; }
echo "训练完成，开始无视频评估：$RUN_DIR" | tee "$STATUS_PATH"
set +e
bash "$PROJECT_DIR/gpt-评估.sh" "$RUN_DIR" "$GPU_ID" "$TRAIN_SEED" \
    2>&1 | tee "$PROJECT_DIR/gpt-复现记录/gpt-正式评估-seed${TRAIN_SEED}-${STAMP}.log"
EVAL_EXIT=${PIPESTATUS[0]}
set -e
echo "训练完成；评估退出码：$EVAL_EXIT；模型目录：$RUN_DIR" | tee "$STATUS_PATH"
exit "$EVAL_EXIT"
