"""回放种子46的B/C正式评估，叠加与复评报告同口径的方向和能耗指标。"""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


video = load_module('gpt_bc_video_base', 'gpt-四组微调对比视频.py')
report = load_module('gpt_bc_metrics', 'gpt-四组微调结果汇总.py')
SCENES = ('楼梯8.5厘米', '平地0.4', '平地0.8', '平地1.0', '崎岖1.0', '坡道0.8')
COLORS = {'B': '#ffba70', 'C': '#75e2c1'}
BG, MUTED = '#111d2c', '#b6c6d9'
W, H = 1600, 900


class DirectionReplay(video.Replay):
    def __init__(self, clip, cache):
        super().__init__(clip, video.DEFAULT_MODEL, cache)
        with np.load(clip.source, allow_pickle=False) as source:
            self.metrics_data = {key: source[key].copy() for key in (
                '机身位置', '机身姿态偏航俯仰横滚', '机身线速度机体系', '关节力矩', '关节角速度')}
        self.final_metrics = report.direction_metrics(self.metrics_data, len(clip.positions), clip.cfg)
        heading = np.deg2rad(self.final_metrics['初始目标航向度'])
        displacement = clip.positions[:, :2].astype(float) - np.array(clip.cfg['起点'][:2])
        self.forward = displacement @ np.array([np.cos(heading), np.sin(heading)])
        self.side = displacement @ np.array([-np.sin(heading), np.cos(heading)])

    def current_metrics(self, index):
        return report.direction_metrics(self.metrics_data, min(2*index+2, len(self.clip.positions)), self.clip.cfg)


def text(draw, xy, value, size=24, color='white'):
    draw.text(xy, value, font=FONTS[size], fill=color)


def draw_tracks(draw, replays, frame):
    text(draw, (28, 691), '俯视轨迹 · 横纵同尺度 · 箭头表示当前机头方向', 21)
    left, top, right, bottom = 80, 742, 900, 863
    all_x = np.concatenate(([0], *(r.forward for r in replays)))
    all_y = np.concatenate(([0], *(r.side for r in replays)))
    xmin, xmax = min(-.3, all_x.min()-.2), max(1, all_x.max()+.5)
    ymin, ymax = min(-.35, all_y.min()-.2), max(.35, all_y.max()+.2)
    scale = min((right-left)/(xmax-xmin), (bottom-top)/(ymax-ymin))
    # 将绘图区留白居中，两个方向始终共用一个米到像素比例。
    xoffset = left + ((right-left)-(xmax-xmin)*scale)/2
    yoffset = top + ((bottom-top)-(ymax-ymin)*scale)/2

    def project(x, y):
        return xoffset+(x-xmin)*scale, yoffset+(ymax-y)*scale

    origin = project(0, 0)
    end = project(xmax, 0)
    for x in range(int(origin[0]), int(end[0]), 12):
        draw.line((x, origin[1], min(x+6, end[0]), origin[1]), fill='#728396', width=2)
    for mark in range(int(np.ceil(xmin)), int(np.floor(xmax))+1):
        px, py = project(mark, 0)
        draw.line((px, py-3, px, py+3), fill=MUTED, width=1)
        text(draw, (px-5, py+4), str(mark), 16, MUTED)
    # 米标尺兼作横向比例参照。
    sx, sy = right-scale-10, top+4
    draw.line((sx, sy, sx+scale, sy), fill=MUTED, width=2)
    text(draw, (sx, sy+3), '1 米', 16, MUTED)
    for replay in replays:
        i = min(2*frame+1, len(replay.forward)-1)
        points = [origin] + [project(x, y) for x, y in zip(replay.forward[:i+1], replay.side[:i+1])]
        color = COLORS[replay.clip.group]
        draw.line(points, fill=color, width=3)
        px, py = points[-1]
        yaw = np.deg2rad(replay.current_metrics(frame)['终点有符号航向偏差度'])
        tip = (px+21*np.cos(yaw), py-21*np.sin(yaw))
        draw.line((px, py, *tip), fill=color, width=4)
        draw.polygon([tip, (tip[0]-9*np.cos(yaw-.45), tip[1]+9*np.sin(yaw-.45)),
                      (tip[0]-9*np.cos(yaw+.45), tip[1]+9*np.sin(yaw+.45))], fill=color)
        draw.ellipse((px-4, py-4, px+4, py+4), fill=color)
    text(draw, (28, 870), '虚线：初始前进方向；刻度：有效前进距离（米）', 16, MUTED)


def compose(replays, scene, frame):
    canvas = video.Image.new('RGB', (W, H), BG)
    draw = video.ImageDraw.Draw(canvas)
    speed = replays[0].clip.cfg['指令速度米每秒']
    text(draw, (25, 12), f'B / C 方向保持与能耗对比  |  {scene}', 30)
    text(draw, (1030, 16), f'同种子 46  ·  指令 {speed:.1f} 米/秒  ·  {(frame+1)/25:.2f} 秒', 21, MUTED)
    for panel, replay in enumerate(replays):
        x = panel * video.WIDTH
        canvas.paste(replay.frame(frame), (x, 70))
        color = COLORS[replay.clip.group]
        draw.rectangle((x, 70, x+video.WIDTH-1, 75), fill=color)
        m = replay.current_metrics(frame)
        values = (
            f"航向偏差  {m['终点有符号航向偏差度']:+.2f}°",
            f"横向偏移  {m['终点绝对横向偏移米']:.3f} 米",
            f"有效前进  {m['沿初始方向有效前进距离米']:.3f} 米",
        )
        text(draw, (x+22, 535), values[0], 27, color)
        text(draw, (x+410, 535), values[1], 27, color)
        text(draw, (x+22, 580), values[2], 27)
        energy = m['每米有效前进机械功焦耳']
        energy_label = '暂不可定义' if energy is None else f'{energy:.2f} 焦耳/米'
        text(draw, (x+410, 583), f'机械功/有效米  {energy_label}', 21)
        q = m['每米有效前进力矩平方积分牛米平方秒每米']
        q_label = '暂不可定义' if q is None else f'{q:.2f} 牛米²·秒/米'
        text(draw, (x+22, 630), f'力矩平方积分/有效米  {q_label}', 21, MUTED)
    draw.line((800, 76, 800, 665), fill='#394b61', width=2)
    draw.line((25, 678, 1575, 678), fill='#394b61', width=2)
    draw_tracks(draw, replays, frame)
    notes = (
        '橙色 B：增大功率惩罚    绿色 C：混合能耗',
        '航向偏差：相对初始方向，左转为正。',
        '横向偏移：偏离初始直线的距离。',
        '有效前进：位移在初始方向上的净投影。',
        '能耗指标：累计至当前时刻的仿真代理值。',
        '原评估轨迹回放 · 正常速度 · 结论以多种子汇总为准',
    )
    for i, note in enumerate(notes):
        text(draw, (970, 698+i*31), note, 19 if i < 5 else 16, MUTED)
    return canvas


def render_scene(scene, out, preview):
    replays, encoder = [], None
    target = out / f'gpt-B与C方向对比-{scene}.mp4'
    temporary = target.with_suffix('.临时.mp4')
    started = time.monotonic()
    try:
        for group in ('B', 'C'):
            run = video.choose_run(video.DEFAULT_ROOT, group, scene, 5000)
            clip = video.load_clip(run, group, scene)
            if clip.terminated or not np.isclose(clip.duration, clip.cfg['时长秒']):
                raise ValueError(f'{group}/{scene} 必须为完整未提前终止回合')
            replays.append(DirectionReplay(clip, out / '回放缓存' / scene / group))
        seconds = replays[0].clip.duration
        if seconds != replays[1].clip.duration:
            raise ValueError('B/C首回合窗口不一致')
        frames = round(seconds * video.FPS)
        with (out / f'gpt-{scene}-编码.log').open('w') as log:
            if not preview:
                encoder = subprocess.Popen(['ffmpeg', '-y', '-v', 'warning', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', f'{W}x{H}', '-r', str(video.FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                    '-threads', '4', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE, stderr=log)
            for index in ([min(199, frames-1)] if preview else range(frames)):
                canvas = compose(replays, scene, index)
                if preview or index == min(199, frames-1):
                    canvas.save(out / f'gpt-B与C方向对比预览-{scene}.png')
                if encoder:
                    encoder.stdin.write(np.asarray(canvas).tobytes())
                if not preview and (index+1) % 50 == 0:
                    print(f'{scene}：{index+1}/{frames}帧，耗时{time.monotonic()-started:.1f}秒', flush=True)
            if encoder:
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError(f'{scene} 编码失败')
                temporary.replace(target)
        evidence = [dict(r.evidence(), 最终方向指标=r.final_metrics) for r in replays]
        return {'场景': scene, '轨迹核验': evidence,
                '视频核验': None if preview else video.verify_video(target, seconds)}
    finally:
        for replay in replays:
            replay.close()
        if encoder and encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=30)
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--scene', choices=SCENES, action='append')
    args = parser.parse_args()
    video.require_completed_summary(video.DEFAULT_ROOT)
    video.load_rendering_dependencies()
    video.WIDTH, video.HEIGHT = 800, 450
    global FONTS
    FONTS = {s: video.ImageFont.truetype(str(video.FONT), s) for s in (16, 19, 21, 24, 27, 30)}
    out = video.DEFAULT_ROOT / 'B与C方向对比视频'
    out.mkdir(parents=True, exist_ok=True)
    records = [render_scene(scene, out, args.preview) for scene in (args.scene or SCENES)]
    result = {'状态': '预览完成' if args.preview else '视频完成', '种子': 46, '每秒帧数': video.FPS,
              '说明': '原正式评估轨迹的运动学回放，未重新运行物理或策略；方向指标直接调用复评报告函数。',
              '场景': records, '脚本SHA256': video.sha256(Path(__file__)),
              '评估指标脚本SHA256': video.sha256(ROOT / 'gpt-四组微调结果汇总.py')}
    if not args.preview:
        target = out / 'gpt-B与C六场景方向与能耗同步对比.mp4'
        listing = out / 'gpt-视频拼接清单.txt'
        listing.write_text(''.join(f"file '{Path(r['视频核验']['视频']).name}'\n" for r in records))
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
                        '-c', 'copy', '-movflags', '+faststart', str(target)], check=True)
        result['合成视频'] = video.verify_video(target, sum(r['视频核验']['时长秒'] for r in records))
        cursor = 0
        result['场景时间轴'] = []
        for record in records:
            end = cursor+record['视频核验']['时长秒']
            result['场景时间轴'].append({'场景': record['场景'], '起始秒': cursor, '结束秒': end})
            cursor = end
    video.write_json(out / ('gpt-方向视频预览核验.json' if args.preview else 'gpt-方向视频完成核验.json'), result)
    print(json.dumps({'状态': result['状态'], '输出目录': str(out)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
