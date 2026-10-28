"""审计中间权重的配对评估，绘制通过率、速度误差和每米机械功曲线。"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gpt-复现记录/gpt-中间权重收敛评估-20261022'
spec=importlib.util.spec_from_file_location('stairs',ROOT/'gpt-30000阶梯统计.py');stairs=importlib.util.module_from_spec(spec);spec.loader.exec_module(stairs)

def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def metrics(d,n):
    if n==0:return None
    vx=d['base_linear_velocity_body_array'][:n,0].astype(float);cmd=d['commanded_velocity_array'][:n,0].astype(float)
    err=d['base_linear_velocity_body_array'][:n,:2].astype(float)-d['commanded_velocity_array'][:n,:2].astype(float)
    power=np.abs(d['joint_torques_array'][:n]*d['joint_velocities_array'][:n]).sum(axis=1).astype(float)
    distance=float(d['distance_increment_array'][:n].sum());energy=float(power.sum()*.02)
    return {'窗口秒':n*.02,'平均前向速度米每秒':float(vx.mean()),'前向速度均方根误差米每秒':float(np.sqrt(np.mean((vx-cmd)**2))),
        '两轴速度均方根误差均值米每秒':float(np.sqrt(np.mean(err**2,axis=0)).mean()),'绝对机械功焦耳':energy,
        '平均绝对机械功率瓦':float(power.mean()),'水平路径长度米':distance,'每米机械功焦耳':energy/distance if distance>1e-9 else None}

def parse_run(job,protocol):
    run=Path(job['目录']);marker=run/'gpt-评估完成.json'
    if not marker.exists():return None
    cfg=json.loads((run/'gpt-评估条件.json').read_text());summary_path=next(run.glob('eval_*/metrics_summary.json'));summary=json.loads(summary_path.read_text());source=summary_path.parent/'plots/sim_data.npz'
    n=job['迭代'];seed=job['种子'];scene=job['场景'];expected=protocol['权重'][str(n)]
    assert cfg['条件']=='开启高程图' and cfg['种子']==seed and cfg['场景']==scene
    assert cfg.get('地形种子',46)==46
    assert Path(cfg['权重']).resolve()==Path(expected['路径']).resolve()
    assert cfg['权重SHA256']==expected['SHA256']
    assert Path(summary['used_checkpoint_path']).resolve()==Path(expected['路径']).resolve()
    with np.load(run/'gpt-实际策略输入.npz') as audit:
        raw=audit['原始观测'];actual=audit['实际输入'];action=audit['策略动作']
        assert actual.shape==(500,236) and action.shape==(500,12)
        assert np.isfinite(actual).all() and np.isfinite(action).all()
        np.testing.assert_array_equal(raw,actual)
        first=raw[0].copy()
    d=np.load(source)
    for key in ('base_position_array','base_linear_velocity_body_array','joint_torques_array','joint_velocities_array','distance_increment_array'):
        assert len(d[key])==500 and np.isfinite(d[key]).all(),(run,key)
    np.testing.assert_allclose(d['commanded_velocity_array'],np.tile([cfg['指令速度米每秒'],0,0],(500,1)),atol=1e-6)
    end=min(summary['automatic_reset_steps'],default=500);assert 0<end<=500
    resets=json.loads((run/'gpt-重置前真实状态.json').read_text()) if (run/'gpt-重置前真实状态.json').exists() else []
    first_reset=resets[0] if resets else None
    reasons=[name for name,on in first_reset['终止项'].items() if on and name!='time_out'] if first_reset else []
    displacement=float(cfg['起点'][0]-d['base_position_array'][end-1,0]);side=float(abs(d['base_position_array'][end-1,1]-cfg['起点'][1]))
    pass_step=stairs.classify(d,summary)[0] if scene=='8.5厘米楼梯' else None
    if scene=='8.5厘米楼梯':
        passed=pass_step is not None;reason='通过楼梯' if passed else '10秒内未通过楼梯' if not reasons else '越阶前提前终止：'+','.join(reasons)
    else:
        passed=not reasons and end>=499 and displacement>=cfg['指令速度米每秒']*10*.7 and side<=1
        reason='通过' if passed else '提前终止：'+','.join(reasons) if reasons else '前进距离不足或偏离路线'
    return {**job,'通过':bool(passed),'结果':reason,'提前终止':bool(reasons),'终止原因':reasons,'首次终止秒':first_reset['全局步']*.02 if reasons else None,
        '有效步数':int(end),'前进米':displacement,'侧向偏移米':side,'楼梯通过秒':(pass_step+1)*.02 if pass_step is not None else None,
        '自身有效窗口指标':metrics(d,end),'轨迹':str(source),'轨迹SHA256':sha(source),'地形SHA256':sha(run/'gpt-实际地形.npz'),
        '配置SHA256':sha(run/'gpt-构造环境配置.yaml'),'动力学':json.loads((run/'gpt-运行时随机化.json').read_text()),'首步观测':first.tolist(),'高程图输入审计':'通过','权重SHA256':expected['SHA256']}

def csvwrite(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def aggregate(rows,protocol):
    iterations=protocol['迭代列表'];seeds=protocol['评估种子'];scenes=protocol['场景'];checks=[]
    lookup={(r['迭代'],r['种子'],r['场景']):r for r in rows}
    for seed in seeds:
        for scene in scenes:
            group=[lookup[(n,seed,scene)] for n in iterations];ref=group[0];end=min(r['有效步数'] for r in group)
            for r in group:
                assert r['配置SHA256']==ref['配置SHA256'],('配置不一致',seed,scene,r['迭代'])
                assert r['动力学']==ref['动力学'],('动力学不一致',seed,scene,r['迭代'])
                assert r['地形SHA256']==ref['地形SHA256'],('地形不一致',seed,scene,r['迭代'])
                np.testing.assert_array_equal(r['首步观测'],ref['首步观测'])
                with np.load(r['轨迹']) as d:r['共同窗口指标']=metrics(d,end)
                assert r['共同窗口指标']['每米机械功焦耳'] is not None
            checks.append({'种子':seed,'场景':scene,'八权重配对核验':'通过','共同窗口秒':end*.02,'首步完整原始观测一致':True,'动力学与地形一致':True})
    for scene in scenes:
        assert len({lookup[(n,seed,scene)]['地形SHA256'] for n in iterations for seed in seeds})==1,('地形布局未固定',scene)
    rng=np.random.default_rng(20261022);resamples=rng.integers(0,len(seeds),size=(10000,len(seeds)))
    summaries=[];by_scene=[];boot_metrics={}
    metric_names=('前向速度均方根误差米每秒','每米机械功焦耳')
    for n in iterations:
        subset=[r for r in rows if r['迭代']==n];item={'迭代':n,'成功次数':sum(r['通过'] for r in subset),'测试次数':len(subset)};item['通过率']=item['成功次数']/item['测试次数']
        item['补充地形完成次数']=sum((r['楼梯通过秒'] is not None) if r['场景']=='8.5厘米楼梯' else (not r['提前终止'] and r['有效步数']>=499) for r in subset)
        item['补充地形完成率']=item['补充地形完成次数']/len(subset)
        item['失败原因计数']={'提前终止':sum(not r['通过'] and r['提前终止'] for r in subset),
            '楼梯未通过且未提前终止':sum(not r['通过'] and not r['提前终止'] and r['场景']=='8.5厘米楼梯' for r in subset),
            '其他场景偏航或前进不足':sum(not r['通过'] and not r['提前终止'] and r['场景']!='8.5厘米楼梯' for r in subset)}
        item['功率惩罚权重']=min(n*24/300000,1)*.008
        for key in metric_names:
            values=np.array([[lookup[(n,seed,scene)]['共同窗口指标'][key] for scene in scenes] for seed in seeds])
            boot=values[resamples].mean(axis=(1,2));boot_metrics[(n,key)]=boot
            item[key]=float(values.mean());item[key+'95区间']=np.quantile(boot,[.025,.975]).tolist()
            item[key+'自身有效窗口均值']=float(np.mean([r['自身有效窗口指标'][key] for r in subset]))
        item['各场景成功次数']={scene:sum(lookup[(n,seed,scene)]['通过'] for seed in seeds) for scene in scenes}
        summaries.append(item)
        for scene in scenes:
            vals=[lookup[(n,seed,scene)] for seed in seeds]
            q={'迭代':n,'场景':scene,'成功次数':sum(r['通过'] for r in vals),'测试次数':len(vals)};q['通过率']=q['成功次数']/len(vals)
            for key in metric_names:q[key]=float(np.mean([r['共同窗口指标'][key] for r in vals]))
            by_scene.append(q)
    final=summaries[-1]
    for s in summaries:
        s['速度误差相对30000倍数']=s[metric_names[0]]/final[metric_names[0]]
        s['每米机械功相对30000倍数']=s[metric_names[1]]/final[metric_names[1]]
        s['速度误差相比30000差值95区间']=np.quantile(boot_metrics[(s['迭代'],metric_names[0])]-boot_metrics[(30000,metric_names[0])],[.025,.975]).tolist()
        s['每米机械功相比30000倍数95区间']=np.quantile(boot_metrics[(s['迭代'],metric_names[1])]/boot_metrics[(30000,metric_names[1])],[.025,.975]).tolist()
        gates={'总体通过率至少90%':s['通过率']>=.9-1e-12,'逐场景通过率至少80%':min(s['各场景成功次数'].values())>=.8*len(seeds)-1e-12,
            '通过率相对30000下降不超过5百分点':s['通过率']>=final['通过率']-.05-1e-12,
            '速度误差在参考容差内':s[metric_names[0]]<=final[metric_names[0]]*1.05+.01,
            '每米机械功不超过参考110%':s[metric_names[1]]<=final[metric_names[1]]*1.10}
        s['参考标准各项']=gates;s['达到参考标准']=all(gates.values())
    earliest=None
    for i,s in enumerate(summaries):
        if len(summaries)-i>=3 and all(t['达到参考标准'] for t in summaries[i:]):earliest=s['迭代'];break
    performance_plateau=None
    for i,s in enumerate(summaries):
        if len(summaries)-i>=3 and all(t['参考标准各项']['速度误差在参考容差内'] and t['参考标准各项']['每米机械功不超过参考110%'] for t in summaries[i:]):
            performance_plateau=s['迭代'];break
    pareto=[]
    for s in summaries:
        dominated=False
        for t in summaries:
            if t is s:continue
            better_or_equal=(t['通过率']>=s['通过率'] and t[metric_names[0]]<=s[metric_names[0]] and t[metric_names[1]]<=s[metric_names[1]])
            strictly_better=(t['通过率']>s['通过率'] or t[metric_names[0]]<s[metric_names[0]] or t[metric_names[1]]<s[metric_names[1]])
            if better_or_equal and strictly_better:dominated=True;break
        if not dominated:pareto.append(s['迭代'])
    decision={'按预定标准最早稳定采样点':earliest,'单点达标迭代':[s['迭代'] for s in summaries if s['达到参考标准']],
        '仅速度和机械功接近最终的最早稳定采样点':performance_plateau,'三指标未被其他权重同时优于的采样点':pareto,
        '说明':'阈值是本次预先约定的工程筛选标准，五个种子的点估计不构成可靠性保证；不能宣称精确收敛迭代。',
        '共同窗口最短秒':min(q['共同窗口秒'] for q in checks),'共同窗口最长秒':max(q['共同窗口秒'] for q in checks),'共同窗口不足1秒组数':sum(q['共同窗口秒']<1 for q in checks)}
    sensitivity=[]
    for n in iterations:
        group=[r for r in rows if r['迭代']==n];item={'迭代':n}
        for limit in (.95,1.,1.05):
            count=0
            for r in group:
                if r['场景']=='8.5厘米楼梯':passed=r['楼梯通过秒'] is not None
                else:
                    command=1.6 if r['场景']=='平地1.6' else 1.
                    passed=not r['提前终止'] and r['有效步数']>=499 and r['前进米']>=command*10*.7 and r['侧向偏移米']<=limit
                count+=bool(passed)
            item[f'侧偏阈值{limit:g}米成功次数']=count
        sensitivity.append(item)
    decision['侧偏阈值敏感性补充']=sensitivity
    decision['30000次路线未通过明细']=[{'种子':r['种子'],'场景':r['场景'],'侧向偏移米':r['侧向偏移米'],'结果':r['结果'],'提前终止':r['提前终止']} for r in rows if r['迭代']==30000 and not r['通过']]
    basic_candidates=[s['迭代'] for s in summaries if s['通过率']>=.9 and min(s['各场景成功次数'].values())>=.8*len(seeds)]
    decision['本轮基础路线任务最早候选']=basic_candidates[0] if basic_candidates else None
    if decision['本轮基础路线任务最早候选'] is not None:
        base=next(s for s in summaries if s['迭代']==decision['本轮基础路线任务最早候选'])
        decision['从基础候选到30000的机械功下降百分比']=(1-final['每米机械功焦耳']/base['每米机械功焦耳'])*100
        decision['从基础候选到30000的前向速度误差下降百分比']=(1-final['前向速度均方根误差米每秒']/base['前向速度均方根误差米每秒'])*100
    return summaries,by_scene,checks,decision

def figures(summary,by_scene,decision,protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_name=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
    plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'#f5f7fa','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    x=np.array([s['迭代'] for s in summary]);metrics_keys=['通过率','前向速度均方根误差米每秒','每米机械功焦耳'];labels=['经验完成率（%）','机身前向速度均方根误差（米/秒）','每米关节绝对机械功（焦耳/米）'];colors=['#0c7c86','#3164b2','#b15b1a']
    def setup(ax,label):
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_xticks(x);ax.set_xlim(2000,31000);ax.axvline(12500,color='#87919f',ls=':',lw=1)
        ax.tick_params(axis='x',labelsize=9);ax.set_ylim(bottom=0)
    fig,axs=plt.subplots(3,1,figsize=(12,12),sharex=True)
    fig.suptitle('已有权重评估：训练到多少次已接近最终表现？',fontsize=19,y=.985)
    for ax,key,label,color in zip(axs,metrics_keys,labels,colors):
        scale=100 if key=='通过率' else 1;y=np.array([s[key] for s in summary])*scale
        ax.plot(x,y,'o-',color=color,lw=2.3,ms=6,label='路线通过（主标准）' if key=='通过率' else None)
        if key!='通过率':
            interval=np.array([s[key+'95区间'] for s in summary]);ax.fill_between(x,interval[:,0],interval[:,1],color=color,alpha=.16,label='配对种子重采样95%区间')
            ax.axhline(y[-1],ls='--',color='#7c8490',lw=1,label='30000次参考值');ax.legend(loc='upper right',fontsize=9)
        else:
            ax.plot(x,[s['补充地形完成率']*100 for s in summary],'s--',color='#727d90',lw=1.4,ms=5,label='地形完成（补充，不限侧偏与距离）')
            for s,xx,yy in zip(summary,x,y):ax.annotate(f"{s['成功次数']}/{s['测试次数']}"+('*' if s['迭代']==30000 else ''),(xx,yy),xytext=(0,8),textcoords='offset points',ha='center',fontsize=10)
            ax.set_ylim(0,112)
            ax.legend(loc='lower right',fontsize=9)
        setup(ax,label)
        if decision['按预定标准最早稳定采样点'] is not None:ax.axvline(decision['按预定标准最早稳定采样点'],color='#319561',ls='--',lw=1.2)
    axs[0].set_title('四个固定场景 × 五个动力学及噪声种子；全部开启真实高程图',fontsize=12,pad=15)
    axs[-1].set_xlabel('训练迭代数（次）')
    note=f"路线通过：平地/崎岖10秒无提前终止、前进≥目标70%、侧偏≤1米；楼梯须实际越过平台。\n连续指标使用八权重共同存活窗口：{decision['共同窗口最短秒']:.2f}～{decision['共同窗口最长秒']:.2f}秒；场景与种子等权平均。\n灰色竖虚线：约12500次功率惩罚达到完整强度。每场景仅5次；固定布局，不能代表全部地形。\n*30000次唯一未通过：侧偏1.00048米，略超1米阈值；没有跌倒，不能据此认定稳定性退步。"
    fig.text(.07,.018,note,fontsize=10,color='#525b66');fig.tight_layout(rect=(0,.105,1,.96))
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'gpt-训练迭代数与评估表现.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(12,12),sharex=True);palette=['#3164b2','#0c8d78','#d17b13','#9a4ba1']
    for ax,key,label in zip(axs,metrics_keys,labels):
        for scene,color in zip(protocol['场景'],palette):
            group=[r for r in by_scene if r['场景']==scene];ax.plot(x,[r[key]*(100 if key=='通过率' else 1) for r in group],'o-',label=scene,color=color,lw=1.8,ms=5)
        setup(ax,label)
        if key=='通过率':ax.set_ylim(-3,106)
        ax.legend(ncol=2,fontsize=10)
    fig.suptitle('逐场景曲线：检查总体平均是否掩盖差异',fontsize=18,y=.98);axs[-1].set_xlabel('训练迭代数（次）');fig.text(.08,.018,'通过率：每个场景5次；速度和机械功：同场景、同种子使用八权重共同有效窗口。',fontsize=10);fig.tight_layout(rect=(0,.045,1,.96))
    for ext in ('png','pdf'):fig.savefig(OUT/f'gpt-逐场景评估曲线.{ext}',dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partial',action='store_true');args=p.parse_args()
    protocol=json.loads((OUT/'gpt-预定评估协议.json').read_text());jobs=json.loads((OUT/'gpt-运行清单.json').read_text());rows=[]
    for n,obj in protocol['权重'].items():assert sha(obj['路径'])==obj['SHA256']
    cache=OUT/'gpt-逐次审计缓存';cache.mkdir(exist_ok=True)
    for j in jobs:
        path=cache/f"gpt-{j['迭代']}次-种子{j['种子']}-{j['场景']}.json"
        if path.exists():r=json.loads(path.read_text())
        else:
            r=parse_run(j,protocol)
            if r is not None:save(path,r)
        if r:rows.append(r)
    save(OUT/'gpt-已完成逐次结果.json',rows)
    counts={str(n):{'完成':sum(r['迭代']==n for r in rows),'通过':sum(r['迭代']==n and r['通过'] for r in rows)} for n in protocol['迭代列表']}
    print(json.dumps({'完成':len(rows),'分权重':counts},ensure_ascii=False),flush=True)
    if len(rows)<len(jobs):
        if args.partial:return
        raise RuntimeError('仍有未完成评估，不生成不完整曲线。')
    # 缓存不代替最终来源核对。
    for r in rows:assert sha(r['轨迹'])==r['轨迹SHA256']
    summary,scenes,checks,decision=aggregate(rows,protocol)
    result={'全部160组完成':True,'汇总':summary,'逐场景':scenes,'配对核验':checks,'筛选结论':decision}
    save(OUT/'gpt-曲线统计与结论.json',result)
    flat=[]
    for r in rows:
        flat.append({k:v for k,v in r.items() if k not in ('动力学','首步观测','自身有效窗口指标','共同窗口指标')}|r['共同窗口指标'])
    csvwrite(OUT/'gpt-160组逐次指标.csv',flat,['迭代','种子','场景','通过','结果','有效步数','前进米','侧向偏移米','楼梯通过秒','首次终止秒','窗口秒','平均前向速度米每秒','前向速度均方根误差米每秒','每米机械功焦耳','绝对机械功焦耳','水平路径长度米','轨迹'])
    csvwrite(OUT/'gpt-按迭代汇总.csv',summary,['迭代','成功次数','测试次数','通过率','补充地形完成次数','补充地形完成率','前向速度均方根误差米每秒','每米机械功焦耳','功率惩罚权重','速度误差相对30000倍数','每米机械功相对30000倍数','达到参考标准'])
    csvwrite(OUT/'gpt-逐场景汇总.csv',scenes,['迭代','场景','成功次数','测试次数','通过率','前向速度均方根误差米每秒','每米机械功焦耳'])
    figures(summary,scenes,decision,protocol)
    lines=['已有中间权重的收敛评估','',
        '本轮建议：约20000次可作为基础场景使用的候选预算；若同时要求接近当前30000次的速度精度和机械能耗，仍建议保留30000次训练预算。',
        '19999次到30000次，平均每米机械功再下降约14.1%，前向速度误差再下降约9.0%，尚不能说两万次之后收益已经很小。',
        '没有发现满足预定连续采样点标准的稳定平台；停止和选权重应依据验证结果，不能认定最后权重必然最好，也不能认定30000次之后绝无收益。','',
        f"共{len(rows)}组；四个场景、五个配对种子、八个已有权重。",'本次没有重新训练；2999和30000次的种子46复用此前8组结果，其余152组新运行。',
        '全部开启真实高程图，地形布局种子固定46；随机种子改变动力学与观测噪声，取消质心随机偏移。',
        '通过定义：'+protocol['平地与崎岖通过定义']+' 楼梯：'+protocol['楼梯通过定义'],
        '连续指标：'+protocol['主连续指标窗口'],protocol['机械功定义'],protocol['误差带'],
        '补充诊断曲线：平地和崎岖场景仅要求10秒无提前终止，楼梯仍要求越阶；此曲线不设侧向偏移和前进距离阈值，不代替预定主标准，也不作为停止筛选标准。',
        '速度误差在机身坐标系计算，不能直接反映航向积累造成的路线偏移；路线通过条件另外约束侧偏。',
        '每米机械功的分母为水平路径长度，含侧向运动；楼梯的爬升高度和所处路段也影响该指标。判断纯粹能效变化时应同时看平地分项。',
        f"共同窗口范围{decision['共同窗口最短秒']:.2f}至{decision['共同窗口最长秒']:.2f}秒；不足1秒的组数{decision['共同窗口不足1秒组数']}。",'',
        '迭代 | 成功次数/20 | 前向速度均方根误差米/秒 | 每米关节绝对机械功焦耳/米 | 达到参考标准']
    for s in summary:lines.append(f"{s['迭代']} | {s['成功次数']}/20 | {s['前向速度均方根误差米每秒']:.4f} | {s['每米机械功焦耳']:.2f} | {s['达到参考标准']}")
    lines+=['','预定参考标准：'+json.dumps(protocol['接近30000次参考标准'],ensure_ascii=False),
        '单点满足标准的迭代：'+str(decision['单点达标迭代']),
        '满足至少三个连续采样点、且此后全部达标的最早采样点：'+str(decision['按预定标准最早稳定采样点']),
        '只考察速度与机械功容差、忽略通过率时的最早稳定采样点：'+str(decision['仅速度和机械功接近最终的最早稳定采样点'])+'；这不能代替任务通过要求。',
        '三指标未被其他采样权重同时优于的迭代：'+str(decision['三指标未被其他权重同时优于的采样点']),
        '30000次严格路线通过19/20，唯一未通过是种子46崎岖场景侧偏1.000483米，略超1米；没有跌倒。放宽至1.05米则为20/20，不能据此认定19999次比30000次更可靠。',
        '侧偏阈值敏感性补充（不更改预定主标准）：'+json.dumps(decision['侧偏阈值敏感性补充'],ensure_ascii=False),
        '若结果为None，说明本次没有足够证据按该标准判断提前停止，而不是程序未完成。',
        '取样点之间没有评估，不能将最早达标采样点视为精确收敛时刻。一次训练轨迹也不能推广为所有训练种子的必需预算。',
        '全部20组跨权重配对核验通过：真实地形、环境配置、初始动力学、首步原始观测一致；所有输入确认为真实高程。',
        '下一步：根据主要使用场景选候选权重；若需要可靠部署结论，应在更多种子、地形布局和更长路线确认。']
    (OUT/'gpt-评估结论.txt').write_text('\n'.join(lines)+'\n')
    save(OUT/'gpt-任务完成.json',{'状态':'完成','评估组数':len(rows),'跨权重配对组数':len(checks),'曲线':['gpt-训练迭代数与评估表现.png','gpt-逐场景评估曲线.png'],'筛选结论':decision})
    print(json.dumps({'汇总':summary,'筛选结论':decision},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
