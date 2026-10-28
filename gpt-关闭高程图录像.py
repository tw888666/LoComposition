"""将高程遮蔽实验的实际轨迹与实际地形渲染为视频，首次终止后定格。"""
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
OUT=ROOT/'gpt-复现记录/gpt-关闭高程图对照-20261022'
MODEL=OUT/'机器人模型'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--preview',action='store_true');p.add_argument('--model-dir',type=Path,default=MODEL);a=p.parse_args()
    model_dir=a.model_dir.resolve()
    run=a.run.resolve();rec=json.loads((run/'gpt-场景结果.json').read_text());cfg=json.loads((run/'gpt-评估条件.json').read_text())
    out=run/'录像';out.mkdir(exist_ok=True)
    data=np.load(rec['轨迹']);end=rec['首次重置前记录步数'];condition=cfg['条件'];scene_name=cfg['场景'];speed=cfg['指令速度米每秒']
    positions=data['base_position_array'][:end].copy();angles=data['base_orientation_array'][:end].copy();joints=data['joint_positions_array'][:end].copy();feet=data['foot_positions_world_frame_array'][:end].copy();vel=data['base_linear_velocity_body_array'][:end,0].copy()
    records=json.loads((run/'gpt-重置前真实状态.json').read_text()) if (run/'gpt-重置前真实状态.json').exists() else []
    if records and records[0]['全局步']==end+1:
        last=records[0]
        positions=np.vstack([positions,last['机身位置']]);angles=np.vstack([angles,last['机身角度偏航俯仰横滚']]);joints=np.vstack([joints,last['关节角']]);feet=np.concatenate([feet,np.array(last['足端世界坐标'])[None]],axis=0);vel=np.r_[vel,last['前向速度']]
    last_index=len(positions)-1
    origin=positions[0].copy();origin[2]=0;rot=Rotation.from_euler('z',-angles[0,0])
    terrain=np.load(run/'gpt-实际地形.npz');vertices=terrain['vertices'];faces=terrain['faces'];tri=vertices[faces]
    lo=positions[:,:2].min(axis=0)-3;hi=positions[:,:2].max(axis=0)+3
    if scene_name=='8.5厘米楼梯':lo=np.minimum(lo,[-14.,12.]);hi=np.maximum(hi,[-6.,20.])
    keep=np.all(tri[:,:,:2].max(axis=1)>=lo,axis=1)&np.all(tri[:,:,:2].min(axis=1)<=hi,axis=1)
    assert keep.any(),'地形裁剪为空'
    mesh=trimesh.Trimesh(vertices=vertices.copy(),faces=faces[keep],process=False);mesh.remove_unreferenced_vertices();mesh.vertices=rot.apply(mesh.vertices-origin)
    terrain_obj=out/'gpt-真实地形.obj';mesh.export(terrain_obj)
    tree=ET.parse(model_dir/'scene.xml');root=tree.getroot();root.find('include').set('file',str(model_dir/'go2.xml'))
    visual=root.find('visual');visual.find('global').set('offwidth','960');visual.find('global').set('offheight','540');visual.find('headlight').set('ambient','0.35 0.35 0.35');ET.SubElement(visual,'quality',shadowsize='2048',offsamples='4')
    for tex in root.find('asset').findall('texture'):
        if tex.get('type')=='skybox':tex.set('rgb1','0.72 0.80 0.88');tex.set('rgb2','0.90 0.94 0.97')
    ET.SubElement(root.find('asset'),'mesh',name='actual_terrain',file=str(terrain_obj),inertia='shell')
    world=root.find('worldbody');world.remove(world.find("geom[@name='floor']"));world.find('light').set('pos','-3 -4 8');world.find('light').set('dir','0.3 0.4 -1')
    ET.SubElement(world,'geom',name='actual_terrain',type='mesh',mesh='actual_terrain',rgba='0.49 0.55 0.60 1',contype='0',conaffinity='0')
    scene=model_dir/f'gpt-{cfg.get("迭代",30000)}-{condition}-{scene_name}.xml';tree.write(scene,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(scene));state=mujoco.MjData(model)
    addresses=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,str(n))] for n in data['joint_names']]
    footids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('FL','FR','RL','RR')]
    renderer=mujoco.Renderer(model,height=540,width=960);camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=2.7;camera.azimuth=75;camera.elevation=-20
    opts=mujoco.MjvOption();opts.geomgroup[3]=0
    fonts={s:ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',s) for s in (17,20,25)}
    target=out/f'gpt-{condition}-{scene_name}.mp4';encoder=None;errors=[];frames=0;started=time.monotonic();last_frame=None;last_rendered_index=None
    if not a.preview:
        encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=(out/'gpt-编码.log').open('w'))
    offsets=sorted(set([min(100,last_index),min(300,last_index),last_index])) if a.preview else range(250)
    try:
        for output_index in offsets:
            idx=output_index if a.preview else min(output_index*2,last_index)
            frozen=not a.preview and output_index*2>last_index
            if frozen and last_frame is not None and last_rendered_index==idx:frame=last_frame.copy()
            else:
                state.qpos[:3]=rot.apply(positions[idx]-origin);yaw,pitch,roll=angles[idx];q=(rot*Rotation.from_euler('xyz',[roll,pitch,yaw])).as_quat();state.qpos[3:7]=q[[3,0,1,2]];state.qpos[addresses]=joints[idx];mujoco.mj_forward(model,state)
                err=np.linalg.norm(state.geom_xpos[footids]-rot.apply(feet[idx]-origin),axis=1);errors.extend(err.tolist());assert err.max()<.015,'足端位置与真实数据不一致'
                camera.lookat[:]=state.qpos[:3]+[.2,0,-.04];renderer.update_scene(state,camera=camera,scene_option=opts);frame=Image.fromarray(renderer.render());last_frame=frame.copy();last_rendered_index=idx
            draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,91),fill=(19,29,43));draw.text((24,10),f'自训 {cfg.get("迭代",30000)}  |  {condition}  |  {scene_name}',font=fonts[25],fill='white')
            map_label='143维固定 −0.30 米' if condition=='关闭高程图' else '143维真实高程输入'
            draw.text((24,51),f'{map_label} · 指令 {speed:g} 米/秒 · 种子 {cfg["种子"]} · 真实轨迹回放',font=fonts[17],fill=(194,213,230))
            draw.rectangle((0,477,960,540),fill=(19,29,43))
            draw.text((24,484),f'时间 {(idx+1)*.02:5.2f} 秒     前向速度 {vel[idx]:.2f} 米/秒     前进 {positions[0,0]-positions[idx,0]:.2f} 米',font=fonts[20],fill='white')
            note='已提前终止，末帧定格' if frozen else '正常速度；实际地形与关节运动来自本次仿真'
            if scene_name=='8.5厘米楼梯':
                passed=rec['楼梯通过时间秒'] is not None and (idx+1)*.02>=rec['楼梯通过时间秒']
                note+=('；已通过七级楼梯' if passed else '；七级楼梯尚未通过')
            draw.text((24,515),note,font=fonts[17],fill=(255,174,145) if frozen else (194,213,230))
            if a.preview:frame.save(out/f'gpt-预览-{idx}.png')
            else:
                encoder.stdin.write(np.asarray(frame).tobytes());frames+=1
                if frames%100==0:print(f'{condition}/{scene_name} {frames}/250帧，{time.monotonic()-started:.1f}秒',flush=True)
        if encoder:encoder.stdin.close();assert encoder.wait()==0
    finally:
        renderer.close()
        if encoder and encoder.poll() is None:encoder.stdin.close();encoder.wait()
    info={'条件':condition,'场景':scene_name,'来源轨迹':rec['轨迹'],'来源SHA256':rec['轨迹SHA256'],'地形SHA256':hashlib.sha256((run/'gpt-实际地形.npz').read_bytes()).hexdigest(),'仅运动学回放':True,'足端最大误差米':max(errors),'实际播放至控制步':last_index,'实际轨迹时长秒':(last_index+1)*.02,'定格说明':'首次提前终止后定格，禁止显示自动重置后轨迹。','输出时长秒':10,'帧数':frames,'视频':str(target)}
    (out/('gpt-预览核验.json' if a.preview else 'gpt-录像核验.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');print(json.dumps(info,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
