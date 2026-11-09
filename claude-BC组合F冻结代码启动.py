"""在子进程中选择 F 组的冻结实现，并记录实际导入来源。与 gpt-CE冻结代码启动.py 相同，只换了冻结目录。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'claude-BC组合F多训练种子-20261105'
FROZEN = OUT/'冻结实现'
PACKAGE = FROZEN/'exts/locomposition'


def audit(path):
    rows = []
    for name, module in list(sys.modules.items()):
        if name == 'locomposition' or name.startswith('locomposition.'):
            file = getattr(module, '__file__', None)
            if file:
                file = Path(file).resolve()
                if not file.is_relative_to(PACKAGE):
                    raise RuntimeError(f'发现未隔离的项目模块：{name} -> {file}')
                rows.append({'模块': name, '来源': str(file), 'SHA256': hashlib.sha256(file.read_bytes()).hexdigest()})
    if not any(r['模块'] == 'locomposition.tasks.utils.cleanrl.ppo' for r in rows):
        return False
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'状态': '通过', '实际项目模块全部来自冻结目录': True, '模块': rows}, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'eval'))
    parser.add_argument('--audit', type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == '--':
        forwarded.pop(0)
    target = FROZEN/'scripts/clean_rl/train.py' if args.mode == 'train' else OUT/'claude-冻结评估入口.py'
    sys.path[:0] = [str(PACKAGE), str(FROZEN/'scripts'), str(FROZEN/'scripts/clean_rl')]
    sys.argv = [str(target)] + forwarded
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    finished = threading.Event()

    def watch():
        while not finished.wait(5):
            try:
                if audit(args.audit):
                    return
            except Exception as error:
                args.audit.write_text(json.dumps({'状态': '失败', '错误': str(error)}, ensure_ascii=False)+'\n')
                os._exit(3)

    worker = threading.Thread(target=watch, daemon=True)
    worker.start()
    try:
        runpy.run_path(str(target), run_name='__main__')
    finally:
        finished.set()
        worker.join(timeout=10)
        audit(args.audit)


if __name__ == '__main__':
    main()
