"""将三个同种子楼梯案例与真实统计合成为同步对比视频。"""
from pathlib import Path
import json
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-30000模型对照评估-20261021'
data=json.loads((OUT/'阶梯评估/gpt-通过率统计.json').read_text())
assert all(s['完成次数']==10 for s in data['统计'])
card=Image.new('RGB',(960,540),(19,29,43))
draw=ImageDraw.Draw(card)
font=lambda n:ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',n)
draw.text((35,22),'自训 30000 · 同协议阶梯评估',font=font(32),fill='white')
y=84
for s in data['统计']:
    draw.text((35,y),f"{s['台阶厘米']} 厘米：{s['通过次数']} / 10 通过（{s['通过率']:.0%}）",font=font(30),fill=(255,195,150));y+=48
low=json.loads((OUT/'阶梯评估/8.5厘米/种子40/gpt-通过判定.json').read_text())
lowtext=f"{low['通过时间秒']:.2f} 秒通过" if low['通过'] else '未通过'
lines=['作者 21349 参考：15 / 20 厘米均为 0 / 10',
       '种子 40–49；七级台阶；指令 1.0 米/秒',
       '四足越阶并直立 0.2 秒；每次最多 15 秒',
       f'左上：8.5 厘米额外对照，{lowtext}',
       '右上 / 左下：15 / 20 厘米，均为种子 40',
       '真实仿真轨迹回放；取消质心随机偏移',
       '一个训练模型，多个评估种子；不代表论文复现']
for line in lines:
    draw.text((35,y),line,font=font(22),fill=(194,213,230));y+=43
cardpath=OUT/'gpt-对比视频结果卡.png';card.save(cardpath)
videos=[]
for h in (8.5,15,20):
    run=OUT/f'阶梯评估/{h:g}厘米/种子40'
    meta=json.loads((run/'录像/gpt-录像核验.json').read_text())
    v=Path(meta['视频']);assert v.exists() and meta['帧数']==375
    videos.append(v)
target=OUT/'gpt-自训30000阶梯同步对比.mp4'
cmd=['ffmpeg','-y','-loglevel','warning','-filter_complex_threads','2']
for v in videos:cmd+=['-i',str(v)]
cmd+=['-loop','1','-framerate','25','-i',str(cardpath),'-filter_complex',
      '[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|960_0|0_540|960_540:shortest=1[v]',
      '-map','[v]','-t','15','-r','25','-an','-c:v','libx264','-threads','4','-crf','20',
      '-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
with (OUT/'gpt-同步对比编码.log').open('w') as f:subprocess.run(cmd,stderr=f,check=True)
print(target)
