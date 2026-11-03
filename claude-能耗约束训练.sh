#!/usr/bin/env bash
# 能耗拉格朗日约束微调：与四组微调 A–E 同起点（model_30000）、同种子、同预算。
# 参数：物理 GPU 编号、迭代数、并行环境数、实验标签（区分不同版本，避免覆盖）。会自动在 tmux 会话中运行。
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GPU_ID=${1:-4}
TRAIN_ITERATIONS=${2:-5000}
TRAIN_ENVS=${3:-7500}
TAG=${4:-v1-20261029}
for value in "$GPU_ID" "$TRAIN_ITERATIONS" "$TRAIN_ENVS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo '参数必须为非负整数。' >&2; exit 2; }
done
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo '实验标签只能包含字母、数字、点、下划线和连字符。' >&2; exit 2; }
SESSION_NAME="claude-energy-constraint-${TAG}-${TRAIN_ITERATIONS}"
if [[ -z "${TMUX:-}" ]]; then
    printf -v RUN_COMMAND 'bash %q %q %q %q %q' "$PROJECT_DIR/claude-能耗约束训练.sh" "$GPU_ID" "$TRAIN_ITERATIONS" "$TRAIN_ENVS" "$TAG"
    tmux new-session -d -s "$SESSION_NAME" "$RUN_COMMAND"
    echo "已创建训练会话：tmux attach -t $SESSION_NAME"
    exit 0
fi
if [[ "$TAG" == "v1-20261029" ]]; then OUT_DIR="$PROJECT_DIR/claude-能耗约束实验-20261029"; else OUT_DIR="$PROJECT_DIR/claude-能耗约束实验-${TAG}"; fi
BASE_CHECKPOINT="$PROJECT_DIR/locomposition/logs/clean_rl/gpt-go2-default-seed46/2026-10-10-09-56-51/model_30000.pt"
REFERENCE="$PROJECT_DIR/gpt-四组微调实验-20261026/gpt-固定参考尺度.json"
# A100（gpu0/1）用 .venv，V100（gpu2-5）用 .venv-v100，与四组微调一致
if ((GPU_ID <= 1)); then PYTHON="$PROJECT_DIR/.venv/bin/python"; else PYTHON="$PROJECT_DIR/.venv-v100/bin/python"; fi
if [[ "$TAG" == "v1-20261029" ]]; then export ENV_NAME="claude-energy-constraint-seed46-20261029"; else export ENV_NAME="claude-energy-constraint-seed46-${TAG}-${TRAIN_ITERATIONS}"; fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=Y OMNICLIENT_HUB_MODE=disabled WANDB_MODE=disabled
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG_PATH="$OUT_DIR/claude-训练-${TRAIN_ITERATIONS}次-${STAMP}.log"
echo "正在训练，日志：$LOG_PATH"
cd "$PROJECT_DIR/locomposition"
echo "代码版本：$(git rev-parse HEAD)$(git diff --quiet HEAD || echo ' (含未提交改动)')" | tee "$LOG_PATH"
set +e
"$PYTHON" scripts/clean_rl/train.py \
    --task=LoComposition-Go2-Rough-Terrain-Joint-State-History-v0 \
    --seed=46 --headless --num_envs="$TRAIN_ENVS" \
    --num_iterations="$TRAIN_ITERATIONS" --logger=tensorboard --device=cuda:0 \
    --finetune --checkpoint="$BASE_CHECKPOINT" \
    --energy_constraint --energy_reference="$REFERENCE" \
    env.scene.ray_caster.debug_vis=false \
    env.scene.ray_caster_height_constraints.debug_vis=false \
    env.commands.base_velocity.debug_vis=false 2>&1 | tee -a "$LOG_PATH"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
echo "训练结束，退出码 $TRAIN_EXIT，日志：$LOG_PATH"
exit "$TRAIN_EXIT"
