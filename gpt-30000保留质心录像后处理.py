"""核验默认质心偏移的真实生效值，汇总对应轨迹并生成录像。"""
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-30000保留质心随机偏移录像-20261026'
PREV=ROOT/'gpt-复现记录/gpt-30000模型对照评估-20261021'
os.environ.setdefault('MPLCONFIGDIR','/tmp/gpt-保留质心绘图缓存')

def load_config(path):
    return yaml.safe_load(re.sub(r'!!python\S*','',path.read_text()))

def diffs(a,b,path=''):
    if type(a)!=type(b):return [{'配置':path,'本次':a,'上次':b}]
    if isinstance(a,dict):
        return [item for k in sorted(a.keys()|b.keys()) for item in diffs(a.get(k),b.get(k),path+'/'+k)]
    return [] if a==b else [{'配置':path,'本次':a,'上次':b}]

def audit():
    run=OUT/'录像评估'
    a=load_config(run/'gpt-构造环境配置.yaml')
    b=load_config(PREV/'完整评估/gpt-构造环境配置.yaml')
    changes=diffs(a,b)
    assert {d['配置'] for d in changes}=={'/events/randomize_com/params/com_range/'+axis for axis in 'xyz'},changes
    actual=json.loads((run/'gpt-运行时质量与质心.json').read_text())
    previous=json.loads((PREV/'完整评估/gpt-运行时质量与质心.json').read_text())
    assert actual['刚体名称']==previous['刚体名称']
    np.testing.assert_allclose(actual['各刚体质量千克'],previous['各刚体质量千克'],rtol=0,atol=0)
    com=np.array(actual['各刚体局部质心位置与旋转'])
    reference=np.array(previous['各刚体局部质心位置与旋转'])
    delta=com-reference
    base=actual['刚体名称'].index('base')
    other=np.arange(len(com))!=base
    assert np.all(delta[other]==0) and np.all(delta[:,3:]==0)
    shift=delta[base,:3]
    assert np.max(np.abs(shift))<=.030001 and np.max(np.abs(shift))>1e-6
    result={'唯一环境配置差异':changes,'质量与上次一致':True,
            '实际机身质心偏移厘米':(shift*100).tolist(),
            '采样方式':'启动时采样一次，整段运行使用该偏移，不逐帧重采样',
            '其余刚体质心与旋转不变':True}
    (OUT/'gpt-质心与配置核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

def render(script):
    log=OUT/f'{script}-运行.log'
    with log.open('w') as f:
        r=subprocess.run([sys.executable,str(ROOT/f'{script}.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'录像生成失败：{log}')
    print(script,'完成',flush=True)

def main():
    deadline=time.monotonic()+3600
    while True:
        candidates=list((OUT/'录像评估').glob('eval_*/metrics_summary.json'))
        try:
            if candidates and json.loads(candidates[0].read_text())['total_sim_steps']==2000:
                break
        except (json.JSONDecodeError,KeyError):pass
        if time.monotonic()>deadline:raise TimeoutError('等待评估结果超时；未终止仿真进程。')
        time.sleep(5)
    proof=audit()
    summary=json.loads(candidates[0].read_text())
    data=np.load(candidates[0].parent/'plots/sim_data.npz')
    assert summary['random_sim_steps']==0 and summary['fixed_command_sim_steps']==500
    for key in ('base_position_array','base_orientation_array','joint_positions_array','base_linear_velocity_body_array'):
        assert len(data[key])==2000 and np.isfinite(data[key]).all()
    records=[]
    for index,(name,*_) in enumerate(summary['fixed_command_scenarios']):
        sl=slice(index*500,index*500+499)
        pos=data['base_position_array'][sl]
        records.append({'场景':name,'无提前重置':summary['fixed_command_scenario_completion'][name],
            '平均前向速度米每秒_不含末尾超时重置步':float(data['base_linear_velocity_body_array'][sl,0].mean()),
            '水平首尾位移米':float(np.linalg.norm(pos[-1,:2]-pos[0,:2]))})
    (OUT/'gpt-评估结果.json').write_text(json.dumps({'场景结果':records,'总控制步数':2000,
        '说明':'专门录像运行，跳过原完整评估的随机前缀和其他场景；不把这四段指标与上次全窗口平均直接比较。'},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(records,ensure_ascii=False),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(render,script) for script in ('gpt-30000保留质心行走视频','gpt-30000保留质心崎岖楼梯视频')]
        for future in concurrent.futures.as_completed(futures):future.result()
    videos=[]
    for path in sorted(OUT.glob('*/gpt-视频来源与核验.json')):
        meta=json.loads(path.read_text());video=Path(meta['视频'])
        assert hashlib.sha256(Path(meta['来源轨迹']).read_bytes()).hexdigest()==meta['来源SHA256']
        check=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
        assert check.returncode==0 and not check.stderr,check.stderr
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
             'format=duration:stream=codec_name,width,height,nb_frames','-of','json',str(video)],text=True))
        videos.append({'视频':str(video),'完整解码':'通过','来源校验':'通过','SHA256':hashlib.sha256(video.read_bytes()).hexdigest(),**info})
    assert len(videos)==2
    origin=json.loads((OUT/'gpt-录像条件与来源.json').read_text())
    assert hashlib.sha256(Path(origin['模型']).read_bytes()).hexdigest()==origin['模型SHA256']
    (OUT/'gpt-录像完成核验.json').write_text(json.dumps({'状态':'评估与录像完成','视频':videos,
         '模型未改变':True,'质心核验':proof},ensure_ascii=False,indent=2)+'\n')
    lines=['30000轮模型：保留默认质心随机偏移的行走录像','日期：2026-10-26','',
        '本轮重新运行仿真。权重与种子46不变；只保留楼梯、平地1.0/1.6米每秒和崎岖四场景。',
        '实际环境配置核验：只有质心偏移范围从三轴0恢复为各±0.03米；各刚体质量不变。',
        '实际机身质心偏移（相对上一轮零偏移，局部XYZ，厘米）：'+json.dumps(proof['实际机身质心偏移厘米']),
        '质心在环境启动时随机采样，整个运行保留此值；不是每帧随机抖动。',
        '录像沿用真实轨迹运动学回放：没有额外物理积分，也没有修正失败动作。',
        '平地录像包含1.0、1.6米/秒和半速回放，共28秒；崎岖与原始楼梯各10秒，共20秒。',
        '本轮未重复15/20厘米高台阶多种子测试。',
        '四场景各500控制步，共2000步；跳过随机前缀，因此不是上次完整评估的逐步复跑。', '', '本次四场景结果：']
    lines.extend(json.dumps(row,ensure_ascii=False) for row in records)
    lines+=['','视频：']+[v['视频'] for v in videos]
    lines+=['','全部视频通过完整解码和来源核验。任务已完成。',
            '下一步可直接对照两轮同场景视频；若要评价质心鲁棒性，再增加评估种子与偏移方向。']
    (OUT/'gpt-录像与评估说明.txt').write_text('\n'.join(lines)+'\n')
    print('全部评估与录像核验完成',flush=True)

if __name__=='__main__':main()
