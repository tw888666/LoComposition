"""等待本次配对评估，渲染每组真实轨迹，合成40秒关闭视频和同步对照。"""
from pathlib import Path
import concurrent.futures,hashlib,importlib.util,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-关闭高程图对照-20261022'
SCENES=('平地1.0','平地1.6','崎岖地形','8.5厘米楼梯')
CONDITIONS=('开启高程图','关闭高程图')
spec=importlib.util.spec_from_file_location('analysis',ROOT/'gpt-关闭高程图分析.py');analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
def render(run):
    analysis.analyze(run)
    marker=run/'录像/gpt-录像核验.json'
    if marker.exists():return json.loads(marker.read_text())
    with (run/'gpt-渲染.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'gpt-关闭高程图录像.py'),'--run',str(run)],stdout=log,stderr=subprocess.STDOUT,check=True)
    r=json.loads(marker.read_text());print('录像完成',r['条件'],r['场景'],flush=True);return r

def main():
    todo={OUT/c/s for c in CONDITIONS for s in SCENES};jobs=[];start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while todo:
            if time.monotonic()-start>1200:raise TimeoutError('评估尚未完成，保留现有录像')
            for run in sorted(todo.copy()):
                if (run/'gpt-评估完成.json').exists():
                    jobs.append(pool.submit(render,run));todo.remove(run);print('开始录像',run.parent.name,run.name,flush=True)
            for f in jobs:
                if f.done():f.result()
            if todo:time.sleep(5)
        records=[f.result() for f in jobs]
    analysis.main()
    assert json.loads((OUT/'gpt-评估汇总.json').read_text())['完整']
    clips=[]
    for c in CONDITIONS:
        paths=[OUT/c/s/'录像'/f'gpt-{c}-{s}.mp4' for s in SCENES]
        lst=OUT/f'gpt-{c}-拼接清单.txt';lst.write_text(''.join("file '"+str(p)+"'\n" for p in paths))
        target=OUT/f'gpt-{c}-四场景行走.mp4'
        subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(lst),'-c','copy','-movflags','+faststart',str(target)],check=True)
        clips.append(target)
    compare=OUT/'gpt-高程图开关-同步对照.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(clips[0]),'-i',str(clips[1]),'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-an','-c:v','libx264','-threads','8','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(compare)],check=True)
    allvideos=[Path(r['视频']) for r in records]+clips+[compare];checked=[]
    for p in allvideos:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames,codec_name:format=duration','-of','json',str(p)]))
        stream=probe['streams'][0];duration=float(probe['format']['duration']);expected=40 if p in clips+[compare] else 10
        assert duration==expected and int(stream['nb_read_frames'])==25*expected
        subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        checked.append({'视频':str(p),'时长秒':duration,'帧数':int(stream['nb_read_frames']),'宽':stream['width'],'高':stream['height'],'完整解码':'通过','SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for sec,name in ((4,'平地'),(24,'崎岖'),(36,'楼梯')):
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(sec),'-i',str(compare),'-frames:v','1','-update','1',str(OUT/f'gpt-同步对照预览-{name}.png')],check=True)
    result={'状态':'全部评估和录像完成','实际策略输入与配对条件':'八组核验通过','录像':checked,'轨迹渲染核验':records}
    (OUT/'gpt-任务完成.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('八组录像、三个合成视频全部核验通过',flush=True)
if __name__=='__main__':main()
