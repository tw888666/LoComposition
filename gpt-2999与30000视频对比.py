"""生成2999次权重四场景录像，以及左2999、右30000的同步对照。"""
from pathlib import Path
import concurrent.futures,hashlib,importlib.util,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-2999与30000权重对照-20261022'
PREVIOUS=ROOT/'gpt-复现记录/gpt-关闭高程图对照-20261022'
SCENES=('平地1.0','平地1.6','崎岖地形','8.5厘米楼梯')
MODEL=OUT/'机器人模型'
spec=importlib.util.spec_from_file_location('weight_analysis',ROOT/'gpt-2999与30000对比分析.py');analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
def render(scene):
    run=OUT/'2999次/开启高程图'/scene;marker=run/'录像/gpt-录像核验.json'
    if marker.exists():return json.loads(marker.read_text())
    with (run/'gpt-渲染.log').open('w') as f:
        subprocess.run([sys.executable,str(ROOT/'gpt-关闭高程图录像.py'),'--run',str(run),'--model-dir',str(MODEL)],stdout=f,stderr=subprocess.STDOUT,check=True)
    info=json.loads(marker.read_text());info['迭代']=2999;print('录像完成',scene,flush=True);return info

def main():
    analysis.main();assert json.loads((OUT/'gpt-权重对比汇总.json').read_text())['完整']
    MODEL.mkdir(exist_ok=True);source=PREVIOUS/'机器人模型'
    for name in ('scene.xml','go2.xml','LICENSE','gpt-模型资源来源.json'):
        if (source/name).exists():shutil.copy2(source/name,MODEL/name)
    if not (MODEL/'assets').exists():(MODEL/'assets').symlink_to((source/'assets').resolve(),target_is_directory=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(render,SCENES))
    early=OUT/'gpt-2999次-开启高程图-四场景行走.mp4';late=OUT/'gpt-30000次-开启高程图-四场景行走.mp4'
    clips=[OUT/'2999次/开启高程图'/s/'录像'/f'gpt-开启高程图-{s}.mp4' for s in SCENES]
    listing=OUT/'gpt-2999次-拼接清单.txt';listing.write_text(''.join("file '"+str(p)+"'\n" for p in clips))
    subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(early)],check=True)
    old_video=PREVIOUS/'gpt-开启高程图-四场景行走.mp4';shutil.copy2(old_video,late)
    assert hashlib.sha256(old_video.read_bytes()).digest()==hashlib.sha256(late.read_bytes()).digest()
    compare=OUT/'gpt-2999与30000次-开启高程图-同步对照.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(early),'-i',str(late),'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-an','-c:v','libx264','-threads','8','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(compare)],check=True)
    checked=[]
    for p in clips+[early,late,compare]:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames,codec_name:format=duration','-of','json',str(p)]))
        stream=probe['streams'][0];duration=float(probe['format']['duration']);expected=10 if p in clips else 40
        assert duration==expected and int(stream['nb_read_frames'])==25*expected
        subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        checked.append({'视频':str(p),'时长秒':duration,'帧数':int(stream['nb_read_frames']),'宽':stream['width'],'高':stream['height'],'完整解码':'通过','SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for sec,name in ((4,'平地'),(26,'崎岖终止后'),(36,'楼梯')):
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(sec),'-i',str(compare),'-frames:v','1','-update','1',str(OUT/f'gpt-同步对照预览-{name}.png')],check=True)
    result={'状态':'四场景评估与录像全部完成','同步对照布局':'左2999次，右30000次，两组开启高程图','视频':checked,'2999次渲染核验':records,'复用30000次视频来源':str(old_video),'说明':'提前终止的片段仅显示首次终止前真实轨迹及捕获的终态，剩余时间明确标注定格；不显示重置后的续跑。'}
    (OUT/'gpt-任务完成.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('全部录像和合成视频完整解码核验通过',flush=True)
if __name__=='__main__':main()
