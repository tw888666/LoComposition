#!/usr/bin/env bash
# 参数：输出目录、物理显卡编号、评估种子；其余参数透传。
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:?请提供独立输出目录}
GPU_ID=${2:-2}
EVAL_SEED=${3:-46}
shift "$(( $# < 3 ? $# : 3 ))"
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
if compgen -G "$RUN_DIR/eval_*/metrics_summary.json" >/dev/null; then
    echo '输出目录已有评估结果，请使用新的目录。' >&2
    exit 1
fi
MODEL="$PROJECT_DIR/locomposition/sim2real/traced_checkpoints/2025-06-28-17-13-04_21349_traced_deterministic.pt"
exec bash "$PROJECT_DIR/gpt-评估.sh" "$RUN_DIR" "$GPU_ID" "$EVAL_SEED" \
    --torchscript_path "$MODEL" --torchscript_action_scale 0.8 "$@"
