#!/usr/bin/env bash
# 参数：训练目录、物理 GPU 编号、评估种子；之后可附加作者评估脚本的参数。
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${1:?请提供训练输出目录的绝对路径}
GPU_ID=${2:-0}
EVAL_SEED=${3:-46}
shift "$(( $# < 3 ? $# : 3 ))"
export CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=Y OMNICLIENT_HUB_MODE=disabled WANDB_MODE=disabled
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
if [[ -n "${LOCOMPOSITION_PYTHON:-}" ]]; then
    PYTHON_BIN="$LOCOMPOSITION_PYTHON"
elif [[ "$(nvidia-smi --id="$GPU_ID" --query-gpu=name --format=csv,noheader)" = *V100* ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv-v100/bin/python"
else
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
fi
cd "$PROJECT_DIR/locomposition"
exec "$PYTHON_BIN" scripts/eval.py \
    --run_dir="$RUN_DIR" \
    --task=LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0 \
    --seed="$EVAL_SEED" --headless --no_video --skip_plots "$@"
