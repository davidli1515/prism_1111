"""Descriptive trajectory report; no further inference or retrospective tuning."""
import csv
import json
from pathlib import Path
import numpy as np
from hybrid_support import ARMS


def read(p):
    return json.loads(Path(p).read_text())


def report(out):
    out = Path(out); summary = read(out/'summary.json')
    lines = ['多步 greedy：scalar 保留一个名额，MSCS 提供另一个候选', '',
             '固定 checkpoint，seed 42，面积上限=起点面积×1.10。每法每起点 6 轮，每轮最多 2 次候选验证。',
             '三组均使用原有完整特征、15 资源、1/2/4 步长、相同接受和缩步规则。',
             '两个旧胜负案例单独作回归，另外四个起点已被上一轮单步实验检查，不能当作未看过的确认集。', '',
             '组别 | 起点 | scalar降幅% | MSCS降幅% | hybrid降幅% | hybrid对scalar优势(百分点)']
    for r in summary['paired']:
        lines.append(r['group']+' | '+r['context']+' | '+' | '.join(f"{r['gain_pct'][a]:.5f}" for a in ARMS)
                     +f" | {r['hybrid_vs_scalar_gain_pp']:+.5f}")
    lines += ['']
    for group in ('regression', 'extension'):
        rows = [r for r in summary['paired'] if r['group'] == group]
        for comparator, key in (('scalar','hybrid_vs_scalar_gain_pp'),('MSCS','hybrid_vs_mscs_gain_pp')):
            w = sum(r[key]>.001 for r in rows); l = sum(r[key]<-.001 for r in rows)
            lines.append(f'{group}: hybrid 对 {comparator}: {w} 胜 / {len(rows)-w-l} 平 / {l} 负，容差 0.001 个百分点。')
    lines += ['', '逐步候选名额使用（以 hybrid 自己所在状态的 scalar 排序为比较对象）：']
    all_trials, choice_rows = [], []
    for pair in summary['paired']:
        ctx = pair['context']; root = out/'walks'/ctx/'hybrid'
        differs, accepted_alt, novel_accepted, prefix_losses = 0,0,0,0
        for file in sorted(root.glob('round_*.json')):
            rr = read(file); dd = read(out/rr['decision_file'])
            scalar_two = np.argsort(dd['scalar_predicted_cpis'],kind='stable')[:2].tolist()
            different = set(dd['chosen']) != set(scalar_two)
            differs += different
            accepted = rr['accepted_trial']
            if accepted is not None and rr['trials'][accepted]['proposed_by'] == 'mscs_alternative':
                accepted_alt += 1
                novel_accepted += dd['chosen'][accepted] not in scalar_two
            row = {'context':ctx,'step':rr['step']+1,'different_from_scalar_top2':bool(different),
                   'accepted_source':rr['trials'][accepted]['proposed_by'] if accepted is not None else None,
                   'accepted_outside_scalar_top2': accepted is not None and dd['chosen'][accepted] not in scalar_two,
                   'root_cpi':rr['root_cpi'],'next_cpi':rr['next_cpi']}
            choice_rows.append(row)
            for j,t in enumerate(rr['trials']):
                e=t['explanation']; pred=np.array(e['predicted_pressure_cpi_change']); actual=np.array(e['actual_pressure_cpi_change'])
                all_trials.append({'context':ctx,'step':rr['step']+1,'query':t['query'],
                                   'proposed_by':t['proposed_by'],'parameter':t['parameter'],
                                   'from':rr['root'][t['parameter']],'to':t['config'][t['parameter']],
                                   'feasible':t['feasible'],'accepted':t['accepted'],
                                   'scalar_predicted_gain_pct':100*(1-t['scalar_predicted_cpi']/rr['root_cpi']),
                                   'mscs_predicted_gain_pct':100*(1-t['mscs_predicted_cpi']/rr['root_cpi']),
                                   'actual_gain_pct':100*t['actual_gain'],
                                   'component_change_MAE':float(np.mean(abs(pred-actual)))})
        lines.append(f'{ctx}: 改变scalar前二候选集合 {differs} 轮；接受MSCS名额 {accepted_alt} 步；其中超出scalar前二 {novel_accepted} 步。')
    with (out/'proposal_diagnostics.json').open('x') as f:
        json.dump(choice_rows,f,indent=2)
    with (out/'hybrid_explanations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_trials[0])); w.writeheader(); w.writerows(all_trials)
    lines += ['', '成本（不以共享面积缓存下的运行时声称方法加速）：']
    for arm in ARMS:
        rows = [r for r in summary['walks'] if r['arm'] == arm]
        calls = sum(r['counts']['candidate_forward_calls'] for r in rows)
        roots = sum(r['counts']['root_forward_calls'] for r in rows)
        backs = sum(r['counts']['backward_scalar_outputs'] for r in rows)
        compute = sum(r['compute_seconds_excluding_area']-r['timing_seconds']['diagnostics'] for r in rows)
        features = sum(r['timing_seconds']['feature_build'] for r in rows)
        jac = sum(r['timing_seconds']['jacobian'] for r in rows)
        lines.append(f'{arm}: 候选前向 {calls}，根节点前向 {roots}，反向标量 {backs}，搜索计算(不含面积/JVP诊断) {compute:.3f}s，其中特征 {features:.3f}s、Jacobian {jac:.3f}s。')
    lines += [f"面积调用 {summary['area_counts']}；失败 {summary['area_failures']}。",
              f"实验耗时 {summary['wall_seconds']:.1f} 秒（写 summary 时），无额外候选池穷举。", '',
              '结论边界：',
              '  保留 scalar 首选只保证该状态下验证这个候选，不保证胜过 scalar 的两个名额或后续路径。',
              '  解释、CPI 改善均基于固定 surrogate；没有新 gem5 证据。',
              '  Protect/Decouple 本轮没有参与决策；分量相反变化不直接命名为保护或解耦。',
              '  两个旧案例是回归检查，四个扩展起点不是完全独立确认集。单 seed、小样本，不作总体优势结论。',
              f"原来源文件 SHA256 保持不变: {summary['source_preservation_passed']}",
              f"全部逐轮预测与选择的独立重放: {read(out/'validation.json')['passed']}"]
    with (out/'FINDINGS_ZH.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')
    plot(out,summary)


def plot(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(10,9),sharex=True)
    for ax,pair in zip(axes.flat,summary['paired']):
        for arm,color,style,marker in zip(ARMS,['#0072B2','#009E73','#D55E00'],['-','--',':'],['o','s','x']):
            rr=[read(p) for p in sorted((out/'walks'/pair['context']/arm).glob('round_*.json'))]
            c0=rr[0]['root_cpi']; xx=[0]+[r['trials'][-1]['query'] for r in rr]
            yy=[0.]+[100*(1-r['next_cpi']/c0) for r in rr]
            ax.plot(xx,yy,label=arm,color=color,linestyle=style,marker=marker,markersize=4)
        ax.set_title(pair['context']+' ('+pair['group']+')')
        ax.set_xlabel('Verified candidates'); ax.set_ylabel('CPI reduction (%)')
        ax.set_xticks(range(0,13,2)); ax.grid(alpha=.2)
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,ncol=3,loc='upper center',frameon=False)
    fig.suptitle('Frozen surrogate; seed 42; six rounds; area cap = 1.10 x root',y=.967,fontsize=10)
    fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(out/'hybrid_paths.png',dpi=180); fig.savefig(out/'hybrid_paths.pdf'); plt.close(fig)
