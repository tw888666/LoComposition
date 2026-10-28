"""保留已核验的三维回放画面，仅重新排版指标与俯视图，避免重复渲染三维场景。"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bc_video', ROOT/'gpt-B与C方向指标对比视频.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
v = m.video
v.load_rendering_dependencies()
v.WIDTH, v.HEIGHT = 800, 450
m.FONTS = {s: v.ImageFont.truetype(str(v.FONT), s) for s in (16, 19, 21, 24, 27, 30)}
out = v.DEFAULT_ROOT/'B与C方向对比视频'
marker = out/'gpt-方向视频完成核验.json'
result = v.read_json(marker)
original_script_hash = result['脚本SHA256']

for record in result['场景']:
    scene = record['场景']
    target = Path(record['视频核验']['视频'])
    temporary = target.with_suffix('.排版临时.mp4')
    panels = []
    for group in ('B', 'C'):
        clip = v.load_clip(v.choose_run(v.DEFAULT_ROOT, group, scene, 5000), group, scene)
        with np.load(clip.source) as source:
            data = {key: source[key].copy() for key in ('机身位置', '机身姿态偏航俯仰横滚',
                    '机身线速度机体系', '关节力矩', '关节角速度')}
        final = m.report.direction_metrics(data, len(clip.positions), clip.cfg)
        heading = np.deg2rad(final['初始目标航向度'])
        delta = clip.positions[:, :2].astype(float)-np.array(clip.cfg['起点'][:2])
        panels.append(SimpleNamespace(clip=clip,
            forward=delta@np.array([np.cos(heading), np.sin(heading)]),
            side=delta@np.array([-np.sin(heading), np.cos(heading)]),
            current_metrics=lambda i, data=data, clip=clip: m.report.direction_metrics(data, 2*i+2, clip.cfg)))
    decoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(target),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '2', 'pipe:1'], stdout=subprocess.PIPE)
    encoder = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1600x900', '-r', '25', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4',
        '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
    try:
        for i in range(record['视频核验']['帧数']):
            raw = decoder.stdout.read(1600*900*3)
            if len(raw) != 1600*900*3:
                raise RuntimeError(f'{scene} 第{i}帧解码不完整')
            source_frame = v.Image.frombytes('RGB', (1600, 900), raw)
            for p, panel in enumerate(panels):
                panel.frame = lambda index, p=p: source_frame.crop((p*800, 70, (p+1)*800, 520))
            canvas = m.compose(panels, scene, i)
            encoder.stdin.write(np.asarray(canvas).tobytes())
            if i == min(199, record['视频核验']['帧数']-1):
                canvas.save(out/f'gpt-B与C方向对比预览-{scene}.png')
        assert not decoder.stdout.read(1), '解码器返回额外帧'
        encoder.stdin.close()
        assert decoder.wait() == 0 and encoder.wait() == 0
        record['排版前视频SHA256'] = record['视频核验']['SHA256']
        record['视频核验'] = v.verify_video(temporary, record['视频核验']['时长秒'])
        temporary.replace(target)
        record['视频核验']['视频'] = str(target)
        print(f'{scene}：标尺修正与解码核验完成', flush=True)
    finally:
        for process in (decoder, encoder):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        temporary.unlink(missing_ok=True)

target = Path(result['合成视频']['视频'])
temporary = target.with_suffix('.排版临时.mp4')
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(out/'gpt-视频拼接清单.txt'),
                '-c', 'copy', '-movflags', '+faststart', str(temporary)], check=True)
result['合成视频'] = v.verify_video(temporary, 55)
temporary.replace(target)
result['合成视频']['视频'] = str(target)
result['原三维渲染脚本SHA256'] = original_script_hash
result['脚本SHA256'] = v.sha256(ROOT/'gpt-B与C方向指标对比视频.py')
result['排版修正'] = {'说明': '保留已核验的三维回放画面，重绘指标与俯视图，将标尺限制在绘图区内。',
                    '脚本SHA256': v.sha256(Path(__file__))}
v.write_json(marker, result)
print('全部视频排版修正完成，55秒、1375帧，完整解码通过。', flush=True)
