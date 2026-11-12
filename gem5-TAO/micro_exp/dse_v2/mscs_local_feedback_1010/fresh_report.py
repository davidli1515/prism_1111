"""Report the complete predeclared regression gate and all budgets."""
from collections import Counter
import json
import os
from pathlib import Path
import numpy as np
from local_support import COMPARISON_ARMS


def read(p):return json.loads(Path(p).read_text())


def report(out):
    out=Path(out);plan=read(out/'plan.json');summary=read(out/'summary.json')
    budgets=[2,4,6,8,10,12];rows=[];failures=[];details=[]
    counts={a:Counter() for a in ('scalar_excess','hybrid','adaptive')}
    for ctx in plan['contexts']:
        results={};rounds={};curves={};root_cpi=ctx['root_cpi']
        for arm in COMPARISON_ARMS:
            directory=out if arm=='local' else Path(ctx['controls'][arm])
            path=directory/'walks'/ctx['id']/arm
            results[arm]=read(path/'summary.json')
            rounds[arm]=[read(p) for p in sorted(path.glob('round_*.json'))]
            curves[arm]=[]
            for budget in budgets:
                done=[r for r in rounds[arm] if max(t['query'] for t in r['trials'])<=budget]
                curves[arm].append(done[-1]['next_cpi'] if done else root_cpi)
        target=results['scalar_excess']['final_cpi']
        first={a:next((b for b,c in zip([0]+budgets,[root_cpi]+curves[a]) if c<=target*(1+1e-6)),None)
               for a in COMPARISON_ARMS}
        changed=0;scored_mscs_rounds=0
        for r in rounds['local']:
            d=read(out/r['decision_file'])
            so=np.argsort(d['scalar_predicted_cpis'],kind='stable').tolist()
            mo=np.argsort(d['mscs_predicted_cpis'],kind='stable').tolist()
            hybrid=[so[0]]+[i for i in mo if i!=so[0]][:1]
            changed+=set(hybrid)!=set(d['chosen'])
            scored_mscs_rounds+='mscs' in d['local_ranking']['anchor_models']
            details.append(dict(context=ctx['id'],round=r['step']+1,radius=r['radius'],
                changed_batch_vs_same_state_hybrid=set(hybrid)!=set(d['chosen']),
                state_before=d['allocator_before'],state_after=r['allocator_after'],
                actual_step_gain_pct=100*(1-r['next_cpi']/r['root_cpi']),
                trials=[dict(parameter=t['parameter'],sign=t['sign'],stride=t['stride'],
                    old=r['root'][t['parameter']],new=t['config'][t['parameter']],
                    proposed_by=t['proposed_by'],scalar_gain_pct=100*(1-t['scalar_predicted_cpi']/r['root_cpi']),
                    mscs_gain_pct=100*(1-t['mscs_predicted_cpi']/r['root_cpi']),actual_gain_pct=100*t['actual_gain'],
                    feasible=t['feasible'],accepted=t['accepted']) for t in r['trials']]))
            for t in r['trials']:
                if t['area'] is None:
                    failures.append(dict(context=ctx['id'],round=r['step']+1,config=t['config'],
                        error=t.get('area_error'),cannot_affect_acceptance=t['cpi']>=r['root_cpi']*(1-1e-6) or t['cpi']>=r['next_cpi']))
        for arm in counts:
            diff=(results[arm]['final_cpi']-results['local']['final_cpi'])/root_cpi
            counts[arm]['win' if diff>1e-6 else 'loss' if diff< -1e-6 else 'tie']+=1
        rows.append(dict(context=ctx['id'],budgets=budgets,
            final_gain_pct={a:100*results[a]['gain'] for a in COMPARISON_ARMS},
            gain_pct_by_budget={a:[100*(1-c/root_cpi) for c in curves[a]] for a in COMPARISON_ARMS},
            first_budget_matching_scalar_final=first,changed_batches_vs_same_state_hybrid=changed,
            rounds_using_local_mscs_scores=scored_mscs_rounds,final_state=results['local']['allocator_final']))
    shadow={'rows':[]}  # No historical shadow path for fresh starts.
    analysis=dict(rows=rows,gate_passed=summary['gate_passed'],comparisons={a:dict(c) for a,c in counts.items()},
        new_candidate_queries=sum(w['counts']['candidate_forward_calls'] for w in summary['walks']),
        area_failures=failures,area_failures_cannot_change_results=all(f['cannot_affect_acceptance'] for f in failures),
        shadow_changed_decisions=sum(r['changed_candidate_set'] for r in shadow['rows']),
        shadow_missing_label_decisions=sum(r['missing_required_labels']>0 for r in shadow['rows']))
    for name,value in [('analysis.json',analysis),('decision_audit.json',details)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    lines=['冻结局部规则：两个新增greedy起点；seed42、固定checkpoint',
        '四个方法全部重新运行，每方法每起点12次候选验证。','',
        '起点 | scalar降幅% | 固定1+1降幅% | 全局adaptive降幅% | local降幅%']
    for row in rows:lines.append(row['context']+' | '+' | '.join(f"{row['final_gain_pct'][a]:.6f}" for a in COMPARISON_ARMS))
    lines+=['',f"相对各基线胜平负：{analysis['comparisons']}",
        f"新起点是否全部达到开发阶段的同一比较标准：{summary['gate_passed']}。",'达到scalar终点所需候选预算：']
    for row in rows:lines.append(row['context']+' | '+' | '.join(str(row['first_budget_matching_scalar_final'][a]) for a in COMPARISON_ARMS))
    lines+=['',f"新增候选查询：{analysis['new_candidate_queries']}；运行耗时（报告前）：{summary['wall_seconds']:.2f}秒。",
        f"面积计数：{summary['area_counts']}；失败：{summary['area_failures']}。",
        f"已有轨迹shadow改变选择{analysis['shadow_changed_decisions']}次，其中所有shadow决策中缺少必要反事实标签{analysis['shadow_missing_label_decisions']}次；不当作在线结果。",
        '全局/局部反馈差异必须结合实际选择和radius变化解释，不能仅凭loss变化归因。',
        '所有结果都是冻结surrogate验证，不证明gem5收益或原版Protect/Decouple。']
    lines.append('这是预定的最后阶段；保留全部结果，不再扩展或调规则。')
    with (out/'FINDINGS_ZH.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/mscs_local_feedback_1010_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(rows),figsize=(10,4))
    for ax,row in zip(axes,rows):
        for arm,label in zip(COMPARISON_ARMS,('Scalar','Fixed 1+1','Global adaptive','Local feedback')):
            ax.plot([0]+budgets,[0]+row['gain_pct_by_budget'][arm],label=label,marker='o',markersize=3,
                    linestyle='--' if arm=='local' else '-',alpha=.85)
        ax.set_title(row['context']);ax.set_xlabel('Candidate verifications')
        ax.set_ylabel('CPI reduction (%)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'local_paths.png',dpi=160);fig.savefig(out/'local_paths.pdf');plt.close(fig)
