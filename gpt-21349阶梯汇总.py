"""完成全部预定评估后制作中文结果说明及同步视频对比。"""
from pathlib import Path
import hashlib,importlib.util,json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-21349高台阶多种子-20261019'
spec=importlib.util.spec_from_file_location('stats',ROOT/'gpt-21349阶梯统计.py');stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)

def main():
    subprocess.run([str(ROOT/'.venv-v100/bin/python'),str(ROOT/'gpt-21349阶梯统计.py')],check=True)
    d=json.loads((OUT/'gpt-通过率统计.json').read_text())
    assert all(s['完成次数']==10 for s in d['统计']),'必须完成全部预定种子'
    assert len(d['两高度初始动力学配对核验'])==10
    assert all(all(r[k] for k in ('各刚体质量千克','材料属性','局部质心')) for r in d['两高度初始动力学配对核验'])
    baseline=stats.analyze_run(OUT/'8.5厘米/种子40');assert baseline['通过']
    geometry=[]
    for h in (15,20):
        ref=None;maxerr=0
        for seed in range(40,50):
            mesh=np.load(OUT/f'{h}厘米/种子{seed}/gpt-实际地形.npz')
            if ref is None:ref={k:mesh[k].copy() for k in ('vertices','faces')}
            assert np.array_equal(mesh['faces'],ref['faces'])
            err=float(np.max(np.abs(mesh['vertices']-ref['vertices'])));maxerr=max(maxerr,err);assert err<1e-12
        geometry.append({'台阶厘米':h,'种子数量':10,'各种子地形面索引一致':True,'各顶点最大差米':maxerr,'说明':'缓存读写造成浮点末位差异，几何相同。'})
    sourcehash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'gpt-21349阶梯评估.py',ROOT/'gpt-21349阶梯统计.py',ROOT/'gpt-21349阶梯录像.py',ROOT/'locomposition/scripts/eval.py',ROOT/'locomposition/scripts/torchscript_policy.py']}
    extra={'地形一致性核验':geometry,'低台阶对照':baseline,'源码SHA256':sourcehash,'通过判定边界测试':['前足越阶而后足未越阶不通过','首次自动重置后的状态不计通过','机身倾覆时越界不通过','直立且四足连续10步越阶计通过']}
    (OUT/'gpt-最终核验.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2)+'\n')
    lines=['21349模型：15厘米和20厘米台阶多种子评估','',
           '结论（只适用于本次固定楼梯、速度、时间窗口与评估配置）：']
    for s in d['统计']:
        group=[r for r in d['逐次结果'] if r['台阶厘米']==s['台阶厘米']]
        lo,hi=s['威尔逊95%置信区间'];progress=[r['最大前进米'] for r in group]
        lines.append(f"{s['台阶厘米']}厘米：{s['通过次数']}/{s['完成次数']}通过，观测通过率{s['通过率']:.0%}；95%威尔逊区间{lo:.1%}–{hi:.1%}。最大前进距离在{min(progress):.2f}–{max(progress):.2f}米之间。")
        reasons={k:sum(r['结果']==k for r in group) for k in sorted(set(r['结果'] for r in group))};lines.append('失败/成功原因计数：'+json.dumps(reasons,ensure_ascii=False))
    lines += ['',f"额外低台阶对照：8.5厘米、种子40，在{baseline['通过时间秒']:.2f}秒通过。只用于验证评估流程及提供成功画面，不计入15/20厘米通过率。",
              '样本中没有成功不意味着真实通过概率必然为零；每组只有10个种子。',
              '', '评估协议：', '1. 两种高度均用种子40–49，每种高度10次。七级倒金字塔楼梯，从底部向外围平台爬升；踏面深30厘米，15厘米组总爬升1.05米，20厘米组总爬升1.40米。',
              '2. 地形布局种子固定46。每次从世界坐标(-8,16)出发，机身离当地地面40厘米，朝世界负X方向，前向速度指令1米/秒。最多15秒。',
              '3. 权重为作者21349确定性导出模型，动作缩放0.8，控制50赫兹，物理200赫兹。取消质心随机偏移，保留原评估的质量、摩擦、观测噪声和高度偏移随机化。不是env_id_97权重。',
              '4. 重置阶段直接写入起点及固定速度指令，首帧观测和历史从实际起点构建；地形导入前重设评估种子。与上一轮录像的起始高度/首帧初始化不完全相同，因此另外做同协议低台阶对照。',
              '5. 通过条件：在首次提前重置前，四足X≤-11.05米、Z≥-0.03米、Y在[13.5,18.5]米内，机身Z≥0.20米且倾角≤30度，连续满足10个控制步（0.2秒）。通过后在相邻地形发生的事件不撤销本次爬楼通过。',
              '6. 自动重置后状态不参与判定。超时仍未越阶属于失败，即使机器人仍然站立。提前终止状态额外保存，以免视频把重置瞬移误画成摔倒。',
              '', '核验：', '全部20次均完成236维输入、12维输出、关节顺序、观测缩放、内部归一化和电机配置核验。10对种子的初始质量、摩擦材料和质心完全配对；同高度地形的顶点误差小于1e-12米。',
              '', '录像：', '选例采用每种高度最低种子的对应结果。成功对照为8.5厘米种子40，高台阶失败为15厘米和20厘米种子40。',
              '录像使用本次Isaac Sim（物理仿真器）保存的真实位姿和关节轨迹，以及仿真导入时直接保存的地形。MuJoCo（仿真与渲染工具）仅进行运动学渲染，没有重跑物理。',
              '三段各15秒同步对比；低台阶对照在通过后继续展示0.7秒，然后明确标注末帧定格。高台阶案例正常速度展示到15秒。机器人足端几何与记录位置误差约2毫米。',
              '', '文件：gpt-通过率统计.json、gpt-逐种子结果.csv、gpt-预定评估协议.json、gpt-最终核验.json。各高度/种子目录保留原始轨迹、环境配置、模型接口审计、初始随机化参数和重置前状态。',
              '', '下一步建议：单独对15厘米台阶做较低速度指令的配对扫描，以区分速度/步态限制和模型本身的越障能力；该扫描尚未执行。']
    (OUT/'gpt-评估结果说明.txt').write_text('\n'.join(lines)+'\n')
    # 最终视频第四象限：只写已完成的统计数据。
    card=Image.new('RGB',(960,540),(19,29,43));draw=ImageDraw.Draw(card)
    font=lambda size:ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size)
    draw.text((40,28),'21349 · 高台阶通过率与成功对照',font=font(32),fill='white')
    y=92
    for s in d['统计']:
        draw.text((40,y),f"{s['台阶厘米']} 厘米：{s['通过次数']} / {s['完成次数']} 通过（{s['通过率']:.0%}）",font=font(30),fill=(255,164,135));y+=52
    textlines=['种子 40–49；七级台阶；指令 1.0 米/秒','每次最多 15 秒；四足越阶并直立 0.2 秒才算通过','高台阶未发现成功，不能用低台阶画面替代','左上：8.5 厘米低台阶对照，5.30 秒通过','右上 / 左下：15 / 20 厘米，种子 40，均受阻','真实仿真轨迹回放；低台阶通过后标注定格','每组仅 10 次；0/10 不等于真实成功概率必为零']
    for line in textlines:draw.text((40,y),line,font=font(22),fill=(194,213,230));y+=43
    cardpath=OUT/'gpt-对比视频结果卡.png';card.save(cardpath)
    videos=[OUT/'8.5厘米/种子40/录像/gpt-8.5厘米种子40-成功.mp4',OUT/'15厘米/种子40/录像/gpt-15厘米种子40-失败.mp4',OUT/'20厘米/种子40/录像/gpt-20厘米种子40-失败.mp4']
    assert not any(s['通过次数'] for s in d['统计']),'若出现高台阶成功，应按预定选例规则增加对应录像并更新结果卡'
    target=OUT/'gpt-21349台阶成功与失败同步对比.mp4';cmd=['ffmpeg','-y','-loglevel','warning','-filter_complex_threads','2']
    for v in videos:cmd+=['-i',str(v)]
    cmd+=['-loop','1','-framerate','25','-i',str(cardpath),'-filter_complex','[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|960_0|0_540|960_540:shortest=1[v]','-map','[v]','-t','15','-r','25','-an','-c:v','libx264','-threads','4','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    with (OUT/'gpt-对比编码.log').open('w') as f:subprocess.run(cmd,stderr=f,check=True)
    print('全部结果和同步视频已完成',target)
if __name__=='__main__':main()
