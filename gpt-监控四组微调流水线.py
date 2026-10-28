"""监控本次流水线进程退出，防止失败后总状态仍显示执行中。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pid', required=True, type=int)
args = parser.parse_args()
root = Path(__file__).resolve().parent / 'gpt-四组微调实验-20261026'
path = root / 'gpt-总状态.json'


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


while alive(args.pid):
    time.sleep(30)

status = json.loads(path.read_text())
if status['状态'] == '执行中':
    status['状态'] = '流水线异常退出，需检查日志'
    status['异常发现时间'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    status['异常说明'] = '主进程已退出但没有完成记录；检查gpt-流水线.log以及各组状态，不能视为实验完成。'
    status['仍存活训练子进程'] = []
    for label in 'ABCD':
        state_path = root / f'gpt-{label}-状态.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            pid = state.get('进程PID')
            if pid and alive(pid):
                status['仍存活训练子进程'].append({'组别': label, 'PID': pid})
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)
print(json.dumps({'监测结束': status['状态']}, ensure_ascii=False), flush=True)
