"""将真实评估轨迹及同配置地形渲染为崎岖与楼梯视频，不进行物理积分。"""
from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.gpt-video-deps'))
os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',EGL_PLATFORM='surfaceless',LIBGL_ALWAYS_SOFTWARE='1',MESA_SHADER_CACHE_DIR='/tmp/gpt-mesa-video',LP_NUM_THREADS='8')
import argparse,json,hashlib,subprocess,time,xml.etree.ElementTree as ET
import numpy as np
import trimesh,mujoco,yaml
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
OUT=ROOT/'gpt-复现记录/gpt-30000模型对照评估-20261021/崎岖楼梯视频'
MODEL=ROOT/'gpt-复现记录/gpt-30000模型对照评估-20261021/机器人模型'
SOURCE=next((ROOT/'gpt-复现记录/gpt-30000模型对照评估-20261021/完整评估').glob('eval_*/plots/sim_data.npz'))

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preview',action='store_true');args=parser.parse_args()
 data=np.load(SOURCE);summary=json.loads((SOURCE.parent.parent/'metrics_summary.json').read_text())
 terrainpath=SOURCE.parent.parent.parent/'gpt-实际地形.npz'
 terrain=np.load(terrainpath);vertices=terrain['vertices'];faces=terrain['faces'];tri=vertices[faces]
 fonts={s:ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',s) for s in (17,20,25)}
 target=OUT/'gpt-自训30000崎岖地形与楼梯.mp4';encoder=None;frames=0;records=[];errors=[]
 if not args.preview:
  encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=(OUT/'gpt-视频编码.log').open('w'))
 started=time.monotonic()
 for title,index in [('崎岖地形',7),('上楼梯',1)]:
  assert summary['fixed_command_scenario_completion'][summary['fixed_command_scenarios'][index][0]], '此视频入口要求完整场景，提前失败应另行截断'
  start=4000+index*500;positions=data['base_position_array'][start:start+499]
  origin=positions[0].copy();origin[2]=0
  rot=Rotation.from_euler('z',-data['base_orientation_array'][start,0])
  lo=positions[:,:2].min(axis=0)-3;hi=positions[:,:2].max(axis=0)+3
  keep=np.all(tri[:,:,:2].max(axis=1)>=lo,axis=1)&np.all(tri[:,:,:2].min(axis=1)<=hi,axis=1)
  mesh=trimesh.Trimesh(vertices=vertices.copy(),faces=faces[keep],process=False);mesh.remove_unreferenced_vertices()
  # 将接触足端与原世界坐标地面射线交点比对，验证地形没有放错位置。
  feet=data['foot_positions_world_frame_array'][start:start+499:2].reshape(-1,3)
  contact=data['contact_state_array'][start:start+499:2].reshape(-1).astype(bool)
  rayorig=feet.copy();rayorig[:,2]+=3
  hits,rayid,_=mesh.ray.intersects_location(rayorig,np.tile([0,0,-1],(len(feet),1)),multiple_hits=False)
  ground=np.full(len(feet),np.nan);ground[rayid]=hits[:,2]
  gaps=feet[:,2]-ground;active=gaps[contact&np.isfinite(gaps)]
  if not len(active) or np.median(np.abs(active))>.06:raise RuntimeError('接触足端与地形不匹配')
  mesh.vertices=rot.apply(mesh.vertices-origin)
  terrain_obj=OUT/f'gpt-{title}真实地形.obj';mesh.export(terrain_obj)
  tree=ET.parse(MODEL/'scene.xml');root=tree.getroot();root.find('include').set('file',str(MODEL/'go2.xml'))
  # include内meshdir相对于主XML；主XML放在模型目录，保留原机器人资源解析。
  visual=root.find('visual');visual.find('global').set('offwidth','960');visual.find('global').set('offheight','540')
  visual.find('headlight').set('ambient','0.35 0.35 0.35');ET.SubElement(visual,'quality',shadowsize='2048',offsamples='4')
  for tex in root.find('asset').findall('texture'):
   if tex.get('type')=='skybox':tex.set('rgb1','0.72 0.80 0.88');tex.set('rgb2','0.90 0.94 0.97')
  ET.SubElement(root.find('asset'),'mesh',name='actual_terrain',file=str(terrain_obj))
  world=root.find('worldbody');world.remove(world.find("geom[@name='floor']"))
  light=world.find('light');light.set('pos','-3 -4 8');light.set('dir','0.3 0.4 -1')
  ET.SubElement(world,'geom',name='actual_terrain',type='mesh',mesh='actual_terrain',rgba='0.49 0.55 0.60 1',contype='0',conaffinity='0')
  scene=MODEL/f'gpt-{title}回放.xml';tree.write(scene,encoding='unicode')
  model=mujoco.MjModel.from_xml_path(str(scene));state=mujoco.MjData(model)
  addresses=[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,str(n))] for n in data['joint_names']]
  footids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('FL','FR','RL','RR')]
  renderer=mujoco.Renderer(model,height=540,width=960);camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
  camera.distance=2.65;camera.azimuth=75;camera.elevation=-23
  opts=mujoco.MjvOption();opts.geomgroup[3]=0
  offsets=[50,200,400,498] if args.preview else np.arange(0,500,2)
  records.append({'场景':title,'原场景名称':summary['fixed_command_scenarios'][index][0],'起始控制步':start,'末控制步':start+498,'视频起始秒':frames/25,'时长秒':10,'前进位移米':float(rot.apply(positions[-1]-positions[0])[0]),'接触足端与地面高度差分位数米':np.quantile(active,[0,.05,.5,.95,1]).tolist(),'接触射线覆盖率':float(np.isfinite(ground[contact]).mean()),'完成信息':summary['fixed_command_scenario_completion'][summary['fixed_command_scenarios'][index][0]]})
  for off in offsets:
   idx=start+off;state.qpos[:3]=rot.apply(data['base_position_array'][idx]-origin)
   yaw,pitch,roll=data['base_orientation_array'][idx];q=(rot*Rotation.from_euler('xyz',[roll,pitch,yaw])).as_quat();state.qpos[3:7]=q[[3,0,1,2]]
   state.qpos[addresses]=data['joint_positions_array'][idx];mujoco.mj_forward(model,state)
   err=np.linalg.norm(state.geom_xpos[footids]-rot.apply(data['foot_positions_world_frame_array'][idx]-origin),axis=1);errors.extend(err.tolist())
   assert err.max()<.015,'足端运动学不一致'
   camera.lookat[:]=state.qpos[:3]+[.15,0,-.06]
   renderer.update_scene(state,camera=camera,scene_option=opts)
   frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
   draw.rectangle((0,0,960,91),fill=(19,29,43));draw.text((24,10),f'自训 30000  |  {title}  |  指令 1.0 米/秒',font=fonts[25],fill='white')
   draw.text((24,51),'真实评估轨迹与同配置地形回放 · 仅取消质心随机偏移 · 种子 46',font=fonts[17],fill=(194,213,230));draw.text((830,14),'正常速度',font=fonts[20],fill=(116,229,187))
   draw.rectangle((0,477,960,540),fill=(19,29,43))
   vx=data['base_linear_velocity_body_array'][idx,0]
   draw.text((24,485),f'时间 {off*.02:4.2f} 秒     实际前向速度 {vx:.2f} 米/秒     前进 {state.qpos[0]:.2f} 米',font=fonts[20],fill='white')
   draw.text((24,515),'完整 10 秒场景；独立渲染真实机身与关节运动，未重新进行物理仿真',font=fonts[17],fill=(194,213,230))
   if args.preview:frame.save(OUT/f'gpt-{title}预览-{off}.png')
   else:
    encoder.stdin.write(np.asarray(frame).tobytes());frames+=1
    if frames%100==0:print(f'已渲染 {frames}/500 帧，用时 {time.monotonic()-started:.1f} 秒',flush=True)
  renderer.close()
 if encoder:
  encoder.stdin.close();assert encoder.wait()==0
 info={'来源轨迹':str(SOURCE),'来源SHA256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'地形配置完全一致':True,'地形种子':46,'地形SHA256':hashlib.sha256(terrainpath.read_bytes()).hexdigest(),'说明':'本轮环境导入时直接保存的真实仿真地形；原轨迹运动学回放，MuJoCo仅作渲染，没有物理积分。仅裁剪远离轨迹的地形，统一刚体变换。各10秒片段显示第0至498步，第499步为回合时限重置步。','足端最大误差米':max(errors),'片段':records,'视频帧数':frames,'视频':str(target)}
 (OUT/('gpt-预览核验.json' if args.preview else 'gpt-视频来源与核验.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
