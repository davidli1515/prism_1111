"""Complete new-root report using only labels collected by the three searches."""
from collections import Counter
import json
import os
from pathlib import Path
import numpy as np
from newroots_support import ARMS,config_key,read


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def report(out):
    out=Path(out);plan=read(out/'plan.json');summary=read(out/'summary.json')
    budgets=[2,4,6,8,10,12];rows=[];details=[];failures=[];mode_totals=Counter()
    comparisons={b:Counter() for b in ('scalar_excess','hybrid')}
    for ctx in plan['contexts']:
        cid=ctx['id'];root_cpi=ctx['root_cpi'];rounds={};results={};curves={};known={}
        for arm in ARMS:
            directory=out/'walks'/cid/arm
            rounds[arm]=[read(p) for p in sorted(directory.glob('round_*.json'))]
            results[arm]=read(directory/'summary.json')
            curve=[]
            for budget in budgets:
                done=[r for r in rounds[arm] if max(t['query'] for t in r['trials'])<=budget]
                curve.append(done[-1]['next_cpi'] if done else root_cpi)
            curves[arm]=curve
            for r in rounds[arm]:
                for t in r['trials']:
                    key=config_key(t['config'])
                    if key in known:
                        assert np.isclose(t['cpi'],known[key]['cpi'],rtol=3e-6)
                    if key not in known or t['area'] is not None:
                        known[key]=t
                    if t['area'] is None:
                        # Missing area cannot help a candidate already no better
                        # than this round's chosen next state, including tolerance.
                        cannot_change=(t['cpi']>=r['root_cpi']*(1-1e-6) or t['cpi']>=r['next_cpi'])
                        failures.append(dict(context=cid,arm=arm,round=r['step']+1,
                            config=t['config'],cpi=t['cpi'],root_cpi=r['root_cpi'],next_cpi=r['next_cpi'],
                            cannot_affect_acceptance=cannot_change,error=t.get('area_error')))
        target=results['scalar_excess']['final_cpi']
        first={a:next((b for b,c in zip([0]+budgets,[root_cpi]+curves[a])
                            if c<=target*(1+1e-6)),None) for a in ARMS}
        modes=Counter();informative=0;changed_hybrid=0;changed_scalar=0
        for r in rounds['adaptive']:
            d=read(out/r['decision_file']);mode=d['allocation_mode'];modes[mode]+=1
            informative+=r['ranking_feedback']['discriminating_comparisons']
            so=np.argsort(d['scalar_predicted_cpis'],kind='stable').tolist()
            mo=np.argsort(d['mscs_predicted_cpis'],kind='stable').tolist()
            hybrid=[so[0]]+[j for j in mo if j!=so[0]][:1]
            diff_h=set(d['chosen'])!=set(hybrid);diff_s=set(d['chosen'])!=set(so[:2])
            changed_hybrid+=diff_h;changed_scalar+=diff_s
            audit={}
            for name,choices in [('scalar_excess',so[:2]),('hybrid',hybrid),('mscs_logit',mo[:2])]:
                trials=[known.get(config_key(d['candidates'][j]['config'])) for j in choices]
                missing=sum(t is None or (t['area'] is None and t['cpi']<r['root_cpi']*(1-1e-6)) for t in trials)
                value=None
                if missing==0:
                    value=min([r['root_cpi']]+[t['cpi'] for t in trials if t['area'] is not None
                        and t['area']<=ctx['area_cap']+1e-9 and t['cpi']<r['root_cpi']*(1-1e-6)])
                audit[name]=dict(missing_required_labels=missing,best_accepted_cpi=value,
                    gain_pct=None if value is None else 100*(1-value/r['root_cpi']))
            details.append(dict(context=cid,round=r['step']+1,mode=mode,
                state_before=d['allocator_before'],state_after=r['allocator_after'],
                changed_batch_vs_fixed_hybrid=diff_h,changed_batch_vs_scalar=diff_s,
                ranking_feedback=r['ranking_feedback'],same_state_existing_label_audit=audit,
                actual_step_gain_pct=100*(1-r['next_cpi']/r['root_cpi']),
                verified=[dict(parameter=t['parameter'],old=r['root'][t['parameter']],
                    new=t['config'][t['parameter']],scalar_gain_pct=100*(1-t['scalar_predicted_cpi']/r['root_cpi']),
                    mscs_gain_pct=100*(1-t['mscs_predicted_cpi']/r['root_cpi']),actual_gain_pct=100*t['actual_gain'],
                    feasible=t['feasible'],accepted=t['accepted']) for t in r['trials']]))
        mode_totals.update(modes)
        for baseline in comparisons:
            diff=(results[baseline]['final_cpi']-results['adaptive']['final_cpi'])/root_cpi
            comparisons[baseline]['win' if diff>1e-6 else 'loss' if diff< -1e-6 else 'tie']+=1
        rows.append(dict(context=cid,rank=ctx['rank'],budgets=budgets,
            final_gain_pct={a:100*results[a]['gain'] for a in ARMS},
            final_area={a:results[a]['final_area'] for a in ARMS},
            gain_pct_by_budget={a:[100*(1-c/root_cpi) for c in curves[a]] for a in ARMS},
            first_budget_matching_scalar_final=first,allocation_rounds=dict(modes),
            discriminating_comparisons=informative,changed_batches_vs_hybrid=changed_hybrid,
            changed_batches_vs_scalar=changed_scalar,allocator_final=results['adaptive']['allocator_final']))
    compute={a:{key:sum(w['timing_seconds'][key] for w in summary['walks'] if w['arm']==a)
                   for key in ('feature_build','root_forward','jacobian','ranking','candidate_forward','area','diagnostics')}
             for a in ARMS}
    analysis=dict(rows=rows,adaptive_vs_baselines={a:dict(c) for a,c in comparisons.items()},
        allocation_rounds=dict(mode_totals),area_failures=failures,
        area_failures_cannot_change_results=all(f['cannot_affect_acceptance'] for f in failures),
        candidate_queries={a:sum(w['counts']['candidate_forward_calls'] for w in summary['walks'] if w['arm']==a) for a in ARMS},
        backward_outputs={a:sum(w['counts']['backward_scalar_outputs'] for w in summary['walks'] if w['arm']==a) for a in ARMS},
        timing_seconds=compute)
    write(out/'analysis.json',analysis);write(out/'decision_audit.json',details)
    lines=['固定规则的新起点验证：同一checkpoint、seed42，scalar / 固定1+1 / 自适应',
        '起点来自同四个benchmark的原有档案；排除前四轮greedy开发JSON中出现过的配置。',
        '这不是新benchmark、独立硬件测试集或训练分布留出测试。',
        '选择规则：前沿按面积排序，取最接近75%排序位置的未涉及合法配置；未按效果挑选。','',
        '起点 | scalar CPI降幅% | 固定1+1 CPI降幅% | 自适应 CPI降幅%']
    for row in rows:
        lines.append(row['context']+' | '+' | '.join(f"{row['final_gain_pct'][a]:.6f}" for a in ARMS))
    lines+=['',f"自适应相对scalar：{dict(comparisons['scalar_excess'])}",
        f"自适应相对固定1+1：{dict(comparisons['hybrid'])}",'',
        '首次达到scalar终点的候选验证预算（0=起点已满足；None=预算内未达到）：',
        '起点 | scalar | 固定1+1 | 自适应']
    for row in rows:
        lines.append(row['context']+' | '+' | '.join(str(row['first_budget_matching_scalar_final'][a]) for a in ARMS))
    lines+=['',f'自适应每轮选择模式：{dict(mode_totals)}。']
    for row in rows:
        lines.append(f"{row['context']}: 有区分力的比较{row['discriminating_comparisons']}次；相同状态候选相对固定1+1改变{row['changed_batches_vs_hybrid']}/6批，相对scalar改变{row['changed_batches_vs_scalar']}/6批；累计损失(乘2){row['allocator_final']['loss2']}。")
    lines+=['',f"候选验证数：{analysis['candidate_queries']}；反向标量输出数：{analysis['backward_outputs']}。",
        f"运行耗时（报告前）：{summary['wall_seconds']:.2f}秒；面积计数：{summary['area_counts']}。",
        f"面积失败数：{summary['area_failures']}；全部不影响接受决定：{analysis['area_failures_cannot_change_results']}。",
        '全部预算曲线、模式计数和计算成本见analysis.json；逐步解释及已有标签反事实见decision_audit.json。',
        '只验证冻结surrogate上的搜索；不代表gem5性能收益，不证明原版保护/解耦机制。',
        '一次排序领先不代表统计置信度；共享面积缓存不支持端到端时间加速主张。']
    with (out/'FINDINGS_ZH.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/mscs_newroots_1010_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,row in zip(axes.flat,rows):
        for arm,label in zip(ARMS,('Scalar','Fixed 1+1','Adaptive')):
            ax.plot([0]+budgets,[0]+row['gain_pct_by_budget'][arm],label=label,marker='o',markersize=3,
                    linestyle='--' if arm=='adaptive' else '-',alpha=.85)
        ax.set_title(row['context']);ax.set_xlabel('Candidate verifications')
        ax.set_ylabel('CPI reduction from root (%)');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Unchanged policies, four new greedy roots; frozen surrogate, seed 42')
    fig.tight_layout();fig.savefig(out/'newroots_paths.png',dpi=160);fig.savefig(out/'newroots_paths.pdf')
    plt.close(fig)


if __name__=='__main__':
    import sys
    report(Path(sys.argv[1]))
