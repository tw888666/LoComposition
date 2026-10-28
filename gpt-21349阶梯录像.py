"""从本轮评估真实轨迹和直接导出的地形录制台阶案例，保留失败终止前状态。"""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.gpt-video-deps'))
os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',EGL_PLATFORM='surfaceless',LIBGL_ALWAYS_SOFTWARE='1',MESA_SHADER_CACHE_DIR='/tmp/gpt-mesa-video',LP_NUM_THREADS='8')
import argparse,hashlib,json,subprocess,time,xml.etree.ElementTree as ET
import numpy as np
import trimesh,mujoco
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
MODEL=ROOT/'gpt-复现记录/gpt-21349行走视频-20261019/机器人模型'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--preview',action='store_true');a=p.parse_args()
    run=a.run.resolve();rec=json.loads((run/'gpt-通过判定.json').read_text());cond=json.loads((run/'gpt-评估条件.json').read_text())
    out=run/'录像';out.mkdir(exist_ok=True)
    data=np.load(rec['轨迹']);end=rec['首次重置前记录步数'];height=cond['台阶高度米']*100;seed=cond['评估种子']
    success=rec['通过'];pass_step=rec['首个通过步']
    positions=data['base_position_array'][:end].copy();angles=data['base_orientation_array'][:end].copy();joints=data['joint_positions_array'][:end].copy();feet=data['foot_positions_world_frame_array'][:end].copy();vel=data['base_linear_velocity_body_array'][:end,0].copy()
    records=json.loads((run/'gpt-重置前真实状态.json').read_text())
    # 环境自动重置前截取的状态补在轨迹末尾，避免把下一回合瞬移误画成摔倒。
    if not success and records and records[0]['全局步']==end+1:
        last=records[0]
        positions=np.vstack([positions,last['机身位置']]);angles=np.vstack([angles,last['机身角度偏航俯仰横滚']]);joints=np.vstack([joints,last['关节角']]);feet=np.concatenate([feet,np.array(last['足端世界坐标'])[None]],axis=0);vel=np.r_[vel,np.nan]
    last_index=min(len(positions)-1,pass_step+35) if success else len(positions)-1
    origin=positions[0].copy();origin[2]=0;rot=Rotation.from_euler('z',-angles[0,0])
    terrain=np.load(run/'gpt-实际地形.npz');vertices=terrain['vertices'];faces=terrain['faces'];tri=vertices[faces]
    # 保留起点、完整七级楼梯和平台的上下文；不根据输赢修改障碍几何。
    lo=np.minimum(positions[:last_index+1,:2].min(axis=0),[-12.,14.])-2
    hi=np.maximum(positions[:last_index+1,:2].max(axis=0),[-8.,18.])+2
    keep=np.all(tri[:,:,:2].max(axis=1)>=lo,axis=1)&np.all(tri[:,:,:2].min(axis=1)<=hi,axis=1)
    mesh=trimesh.Trimesh(vertices=vertices.copy(),faces=faces[keep],process=False);mesh.remove_unreferenced_vertices();mesh.vertices=rot.apply(mesh.vertices-origin)
    terrain_obj=out/'gpt-真实地形.obj';mesh.export(terrain_obj)
    tree=ET.parse(MODEL/'scene.xml');root=tree.getroot();root.find('include').set('file',str(MODEL/'go2.xml'))
    visual=root.find('visual');visual.find('global').set('offwidth','960');visual.find('global').set('offheight','540');visual.find('headlight').set('ambient','0.35 0.35 0.35');ET.SubElement(visual,'quality',shadowsize='2048',offsamples='4')
    for tex in root.find('asset').findall('texture'):
        if tex.get('type')=='skybox':tex.set('rgb1','0.72 0.80 0.88');tex.set('rgb2','0.90 0.94 0.97')
    ET.SubElement(root.find('asset'),'mesh',name='stairs',file=str(terrain_obj))
    world=root.find('worldbody');world.remove(world.find("geom[@name='floor']"));world.find('light').set('pos','-3 -4 8');world.find('light').set('dir','0.3 0.4 -1')
    ET.SubElement(world,'geom',name='stairs',type='mesh',mesh='stairs',rgba='0.49 0.55 0.60 1',contype='0',conaffinity='0')
    scene=MODEL/f'gpt-高台阶{height:g}种子{seed}.xml';tree.write(scene,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(scene));state=mujoco.MjData(model)
    addresses=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,str(n))] for n in data['joint_names']]
    footids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('FL','FR','RL','RR')]
    renderer=mujoco.Renderer(model,height=540,width=960);camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=2.7;camera.azimuth=75;camera.elevation=-20
    opts=mujoco.MjvOption();opts.geomgroup[3]=0
    fonts={s:ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',s) for s in (17,20,25)}
    tag='成功' if success else '失败';target=out/f'gpt-{height:g}厘米种子{seed}-{tag}.mp4'
    encoder=None;errors=[];frames=0;started=time.monotonic();last_frame=None
    if not a.preview:
        encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=(out/'gpt-编码.log').open('w'))
    offsets=sorted(set([min(100,last_index),min(300,last_index),last_index])) if a.preview else range(375)
    try:
        for output_index in offsets:
            idx=output_index if a.preview else min(output_index*2,last_index)
            frozen=(not a.preview and output_index*2>last_index)
            # 终态定格可直接复用上一帧，后处理标题仍明确说明，避免额外渲染数百次。
            if frozen and last_frame is not None:
                frame=last_frame.copy()
            else:
                state.qpos[:3]=rot.apply(positions[idx]-origin);yaw,pitch,roll=angles[idx];q=(rot*Rotation.from_euler('xyz',[roll,pitch,yaw])).as_quat();state.qpos[3:7]=q[[3,0,1,2]];state.qpos[addresses]=joints[idx];mujoco.mj_forward(model,state)
                err=np.linalg.norm(state.geom_xpos[footids]-rot.apply(feet[idx]-origin),axis=1);errors.extend(err.tolist());assert err.max()<.015,'足端位置与真实数据不一致'
                camera.lookat[:]=state.qpos[:3]+[.2,0,-.04];renderer.update_scene(state,camera=camera,scene_option=opts);frame=Image.fromarray(renderer.render());last_frame=frame.copy()
            draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,91),fill=(19,29,43));draw.text((24,10),f'21349  |  {height:g} 厘米 × 7 级  |  种子 {seed}',font=fonts[25],fill='white')
            label='低台阶对照，不计入高台阶通过率' if height<15 else '高台阶正式评估'
            draw.text((24,51),f'{label} · 指令 1.0 米/秒 · 真实仿真轨迹回放',font=fonts[17],fill=(194,213,230))
            status=('已通过' if success and idx>=pass_step else '最终未通过' if not success else '成功案例')
            draw.text((790,14),status,font=fonts[20],fill=(116,229,187) if success else (255,164,135))
            draw.rectangle((0,477,960,540),fill=(19,29,43))
            velocity='终止前状态' if not np.isfinite(vel[idx]) else f'前向速度 {vel[idx]:.2f} 米/秒'
            draw.text((24,484),f'时间 {(idx+1)*.02:5.2f} 秒     {velocity}     前进 {-8-positions[idx,0]:.2f} 米',font=fonts[20],fill='white')
            note=('已通过，末帧定格' if success else '已终止，末帧定格') if frozen else '正常速度；无额外物理仿真；台阶与关节运动均来自本次评估'
            if not success and idx>=last_index-1:note=rec['结果']+'；最终状态'+('定格' if frozen else '')
            draw.text((24,515),note,font=fonts[17],fill=(194,213,230))
            if a.preview:frame.save(out/f'gpt-预览-{idx}.png')
            else:
                encoder.stdin.write(np.asarray(frame).tobytes());frames+=1
                if frames%100==0:print(f'{height:g}厘米种子{seed} 已渲染 {frames}/375 帧，用时 {time.monotonic()-started:.1f}秒',flush=True)
        if encoder:encoder.stdin.close();assert encoder.wait()==0
    finally:
        renderer.close()
        if encoder and encoder.poll() is None:encoder.stdin.close();encoder.wait()
    info={'台阶厘米':height,'种子':seed,'结果':tag,'来源轨迹':rec['轨迹'],'来源SHA256':rec['轨迹SHA256'],'地形SHA256':hashlib.sha256((run/'gpt-实际地形.npz').read_bytes()).hexdigest(),'仅运动学回放':True,'足端最大误差米':max(errors),'实际播放至控制步':last_index,'实际轨迹时长秒':(last_index+1)*.02,'定格说明':'通过后继续展示0.7秒再定格；失败展示至首次终止或15秒时限。余下时间保持末帧并标注。','输出时长秒':15,'帧数':frames,'视频':str(target)}
    (out/('gpt-预览核验.json' if a.preview else 'gpt-录像核验.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');print(json.dumps(info,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
