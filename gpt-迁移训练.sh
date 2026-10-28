#!/usr/bin/env bash
# 参数：检查点绝对路径、物理 GPU 编号、种子、总迭代预算、并行环境数。
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT=${1:?请提供已有检查点的绝对路径}
GPU_ID=${2:-2}
TRAIN_SEED=${3:-46}
TOTAL_ITERATIONS=${4:-30000}
TRAIN_ENVS=${5:-7500}
[[ "$CHECKPOINT" = /* && -f "$CHECKPOINT" ]] || { echo '检查点必须为存在的绝对路径。' >&2; exit 2; }
for value in "$GPU_ID" "$TRAIN_SEED" "$TOTAL_ITERATIONS" "$TRAIN_ENVS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo '数值参数必须为非负整数。' >&2; exit 2; }
done
SESSION_NAME="gpt-locomposition-s${TRAIN_SEED}-gpu${GPU_ID}"
if [[ -z "${TMUX:-}" ]]; then
    printf -v RUN_COMMAND 'bash %q %q %q %q %q %q' "$PROJECT_DIR/gpt-迁移训练.sh" "$CHECKPOINT" "$GPU_ID" "$TRAIN_SEED" "$TOTAL_ITERATIONS" "$TRAIN_ENVS"
    tmux new-session -d -s "$SESSION_NAME" "$RUN_COMMAND"
    echo "已创建续训会话：tmux attach -t $SESSION_NAME"
    exit 0
fi
GPU_NAME=$(nvidia-smi --id="$GPU_ID" --query-gpu=name --format=csv,noheader)
if [[ "$GPU_NAME" = *V100* ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv-v100/bin/python"
else
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
fi
[[ -x "$PYTHON_BIN" ]] || { echo "缺少兼容环境：$PYTHON_BIN" >&2; exit 2; }
export LOCOMPOSITION_PYTHON="$PYTHON_BIN"
export ENV_NAME="gpt-go2-default-seed${TRAIN_SEED}"
export CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=Y OMNICLIENT_HUB_MODE=disabled WANDB_MODE=disabled
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd "$PROJECT_DIR/locomposition"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG_PATH="$PROJECT_DIR/gpt-复现记录/gpt-迁移训练-seed${TRAIN_SEED}-gpu${GPU_ID}-${STAMP}.log"
STATUS_PATH="$PROJECT_DIR/gpt-复现记录/gpt-迁移状态-seed${TRAIN_SEED}-gpu${GPU_ID}-${STAMP}.txt"
mkdir -p "$PROJECT_DIR/gpt-复现记录"
echo "从 $CHECKPOINT 续训；物理 GPU $GPU_ID；日志：$LOG_PATH" | tee "$STATUS_PATH"
RESUME_ARGS=(--resume --checkpoint="$CHECKPOINT")
if [[ "$(basename -- "$CHECKPOINT")" = model_* ]]; then
    # 旧模型没有优化器和地形分配；本次迁移的约束课程已在旧日志中确认达到 1.0。
    RESUME_ARGS+=(--allow_partial_resume --resume_constraint_progress=1.0)
fi
set +e
"$PYTHON_BIN" scripts/clean_rl/train.py \
    --task=LoComposition-Go2-Rough-Terrain-Joint-State-History-v0 \
    --seed="$TRAIN_SEED" --headless --num_envs="$TRAIN_ENVS" \
    --num_iterations="$TOTAL_ITERATIONS" --logger=tensorboard --device=cuda:0 \
    "${RESUME_ARGS[@]}" \
    env.scene.ray_caster.debug_vis=false \
    env.scene.ray_caster_height_constraints.debug_vis=false \
    env.commands.base_velocity.debug_vis=false 2>&1 | tee "$LOG_PATH"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
if ((TRAIN_EXIT != 0)); then
    echo "续训退出码：$TRAIN_EXIT；日志：$LOG_PATH" | tee "$STATUS_PATH"
    exit "$TRAIN_EXIT"
fi
RUN_DIR=$(sed -n 's/^\[INFO\] Reserved run directory: //p' "$LOG_PATH" | tail -1)
[[ -f "$RUN_DIR/model_${TOTAL_ITERATIONS}.pt" ]] || { echo '最终模型缺失。' | tee "$STATUS_PATH"; exit 1; }
echo "训练完成，开始无视频评估：$RUN_DIR" | tee "$STATUS_PATH"
set +e
bash "$PROJECT_DIR/gpt-评估.sh" "$RUN_DIR" "$GPU_ID" "$TRAIN_SEED" \
    2>&1 | tee "$PROJECT_DIR/gpt-复现记录/gpt-迁移后评估-seed${TRAIN_SEED}-${STAMP}.log"
EVAL_EXIT=${PIPESTATUS[0]}
set -e
echo "训练完成；评估退出码：$EVAL_EXIT；模型目录：$RUN_DIR" | tee "$STATUS_PATH"
exit "$EVAL_EXIT"
