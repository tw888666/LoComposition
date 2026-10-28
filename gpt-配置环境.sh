#!/usr/bin/env bash
# 本机依赖覆盖环境的重建脚本；要求现有基础环境及两份源码仍然存在。
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_PYTHON=${LOCOMPOSITION_BASE_PYTHON:-/home/xy.chen/miniconda3/envs/env_isaaclab/bin/python}
VENV_DIR=${LOCOMPOSITION_VENV_DIR:-$PROJECT_DIR/.venv}
ENV_SUFFIX=${LOCOMPOSITION_ENV_SUFFIX:-}
cd "$PROJECT_DIR"
[[ -x "$BASE_PYTHON" ]] || { echo "基础 Python 不存在：$BASE_PYTHON" >&2; exit 1; }
[[ -d locomposition/exts/locomposition && -d dependencies/IsaacLab/source ]] || {
    echo '请先准备复现说明中指定提交的项目源码和 Isaac Lab 源码。' >&2; exit 1;
}
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install --no-deps --no-build-isolation \
    -e dependencies/IsaacLab/source/isaaclab \
    -e dependencies/IsaacLab/source/isaaclab_assets \
    -e dependencies/IsaacLab/source/isaaclab_tasks \
    -e dependencies/IsaacLab/source/isaaclab_rl \
    -e locomposition/exts/locomposition
"$VENV_DIR/bin/python" -m pip install \
    'rich==15.0.0' 'memory-profiler==0.61.0' 'SciencePlots==2.2.2' \
    'daqp==0.7.2' 'flatdict==4.0.0' 'psutil==5.9.8' \
    'typing-extensions==4.12.2' 'packaging==23.0' 'wheel==0.45.1' \
    'ipython==8.30.0' 'onnx==1.18.0' 'pyzmq==27.2.0'
mkdir -p gpt-复现记录
"$VENV_DIR/bin/python" -m pip freeze > "gpt-复现记录/gpt-运行环境依赖${ENV_SUFFIX}.txt"
CHECK_PATH="gpt-复现记录/gpt-依赖一致性检查${ENV_SUFFIX}.txt"
if "$VENV_DIR/bin/python" -m pip check > "$CHECK_PATH" 2>&1; then
    cat "$CHECK_PATH"
else
    cat "$CHECK_PATH"
    "$VENV_DIR/bin/python" - "$CHECK_PATH" <<'PY'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().strip().splitlines()
known = 'fastapi 0.115.7 has requirement starlette<0.46.0,>=0.40.0, but you have starlette 0.49.1.'
if lines != [known]:
    raise SystemExit('发现作者已知问题以外的依赖冲突。')
print('仅保留作者安装脚本明确允许的 FastAPI/Starlette 元数据冲突。')
PY
fi
