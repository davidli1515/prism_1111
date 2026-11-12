"""Report all predeclared endpoints; do not change the allocation policy."""
from collections import Counter
import json
import os
from pathlib import Path
import numpy as np
from adaptive_support import COMPARISON_ARMS, config_key


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def report(out):
    out = Path(out)
    plan, summary = read(out/'plan.json'), read(out/'summary.json')
    baseline = Path(plan['baseline_directory'])
    budgets = [2,4,6,8,10,12]
    rows, details, failures = [], [], []
    mode_totals = Counter()
    for ctx in plan['contexts']:
        cid = ctx['id']; root_cpi = ctx['root_cpi']
        rounds, curves, final = {}, {}, {}
        # The historical results below are reporting-only, never allocator inputs.
        known = {}
        for arm in COMPARISON_ARMS:
            directory = (out if arm == 'adaptive' else baseline)/'walks'/cid/arm
            rounds[arm] = [read(p) for p in sorted(directory.glob('round_*.json'))]
            final[arm] = read(directory/'summary.json')
            curve = []
            for budget in budgets:
                done = [r for r in rounds[arm] if max(t['query'] for t in r['trials']) <= budget]
                curve.append(done[-1]['next_cpi'] if done else root_cpi)
            curves[arm] = curve
            for r in rounds[arm]:
                for t in r['trials']:
                    key = config_key(t['config'])
                    if key in known:
                        assert np.isclose(t['cpi'],known[key]['cpi'],rtol=3e-6)
                    if key not in known or t['area'] is not None:
                        known[key] = t
        target = final['scalar_excess']['final_cpi']
        first = {a: next((b for b,c in zip(budgets,curves[a]) if c <= target*(1+1e-6)),None)
                 for a in COMPARISON_ARMS}
        modes = Counter(); changed_batches = 0; discriminating = 0
        for r in rounds['adaptive']:
            d = read(out/r['decision_file']); mode = d['allocation_mode']; modes[mode] += 1
            discriminating += r['ranking_feedback']['discriminating_comparisons']
            so = np.argsort(d['scalar_predicted_cpis'],kind='stable').tolist()
            mo = np.argsort(d['mscs_predicted_cpis'],kind='stable').tolist()
            hybrid = [so[0]]+[i for i in mo if i != so[0]][:1]
            differs = set(d['chosen']) != set(hybrid)
            changed_batches += differs
            same_state = {}
            for label, choices in [('scalar_excess',so[:2]),('hybrid',hybrid),('mscs_logit',mo[:2])]:
                values = [known.get(config_key(d['candidates'][i]['config'])) for i in choices]
                missing = sum(t is None or (t['area'] is None and t['cpi'] < r['root_cpi']*(1-1e-6)) for t in values)
                value = None
                if not missing:
                    value = min([r['root_cpi']]+[t['cpi'] for t in values
                        if t['area'] is not None and t['area'] <= ctx['area_cap']+1e-9
                        and t['cpi'] < r['root_cpi']*(1-1e-6)])
                same_state[label] = dict(missing_required_labels=missing,
                    best_accepted_cpi=value, gain_pct=None if value is None else 100*(1-value/r['root_cpi']))
            details.append(dict(context=cid, round=r['step']+1, mode=mode,
                state_before=d['allocator_before'], state_after=r['allocator_after'],
                changed_batch_vs_same_state_fixed_hybrid=differs,
                actual_step_gain_pct=100*(1-r['next_cpi']/r['root_cpi']),
                same_state_existing_label_audit=same_state,
                verified=[dict(parameter=t['parameter'], old=r['root'][t['parameter']],
                    new=t['config'][t['parameter']], scalar_gain_pct=100*(1-t['scalar_predicted_cpi']/r['root_cpi']),
                    mscs_gain_pct=100*(1-t['mscs_predicted_cpi']/r['root_cpi']),
                    actual_gain_pct=100*t['actual_gain'], feasible=t['feasible'], accepted=t['accepted'])
                    for t in r['trials']]))
            for t in r['trials']:
                if t['area'] is None:
                    failures.append(dict(context=cid,round=r['step']+1,config=t['config'],
                        cpi=t['cpi'],root_cpi=r['root_cpi'],error=t.get('area_error'),
                        cannot_affect_acceptance=t['cpi'] >= r['root_cpi']*(1-1e-6)))
        mode_totals.update(modes)
        gains = {a: [100*(1-c/root_cpi) for c in curves[a]] for a in COMPARISON_ARMS}
        rows.append(dict(context=cid,group=ctx['group'],budgets=budgets,gain_pct_by_budget=gains,
            final_gain_pct={a:100*final[a]['gain'] for a in COMPARISON_ARMS},
            first_budget_matching_scalar_final=first,allocation_rounds=dict(modes),
            changed_batches_vs_fixed_hybrid_at_same_state=changed_batches,
            discriminating_comparisons=discriminating,allocator_final=final['adaptive']['allocator_final']))
    wins = Counter()
    for ctx,row in zip(plan['contexts'],rows):
        delta = summary['paired'][plan['contexts'].index(ctx)]['adaptive_vs_scalar_gain_pp']
        wins['win' if delta > 1e-4 else 'loss' if delta < -1e-4 else 'tie'] += 1
    report_data = dict(rows=rows,adaptive_vs_scalar=wins,allocation_rounds=dict(mode_totals),
                       new_candidate_queries=sum(w['counts']['candidate_forward_calls'] for w in summary['walks']),
                       reused_baseline_queries=216,area_failures=failures)
    write(out/'analysis.json',report_data); write(out/'decision_audit.json',details)
    lines = ['固定 checkpoint、seed42：排序反馈自适应 greedy（六个已检查过的开发起点）',
             '三个基线复用只读历史轨迹，运行前独立重放验证；本次只新增自适应查询。', '',
             '相对共同起点的最终 CPI 降幅（%，越大越好）：',
             '起点 | scalar | MSCS | 固定1+1 | 自适应']
    for row in rows:
        lines.append(row['context']+' | '+' | '.join(f"{row['final_gain_pct'][a]:.6f}" for a in COMPARISON_ARMS))
    lines += ['',f"自适应相对 scalar：{dict(wins)}；新候选验证 {report_data['new_candidate_queries']} 次。",
              '每轮两次查询；首次达到 scalar 最终 CPI 的预算（允许相对1e-6误差）：',
              '起点 | scalar | MSCS | 固定1+1 | 自适应']
    for row in rows:
        lines.append(row['context']+' | '+' | '.join(str(row['first_budget_matching_scalar_final'][a]) for a in COMPARISON_ARMS))
    lines += ['',f'自适应的36轮名额分配：{dict(mode_totals)}。',
              '改变名额不一定改变候选集合；下面报告与同状态固定1+1相比实际换了多少批次：']
    for row in rows:
        lines.append(f"{row['context']}: 换批 {row['changed_batches_vs_fixed_hybrid_at_same_state']}/6；能区分两种预测器对错的比较 {row['discriminating_comparisons']} 次；最终累计损失(乘2) {row['allocator_final']['loss2']}。")
    lines += ['',f"新运行耗时（含运行前基线重放，不含最终报告）：{summary['wall_seconds']:.2f} 秒。",
              f"面积计数：{summary['area_counts']}；失败数 {summary['area_failures']}。",
              f"失败是否全部不可能影响接受决定：{all(f['cannot_affect_acceptance'] for f in failures)}。",
              '结果只针对冻结 surrogate；没有新增 gem5 证据，没有独立泛化验证。',
              '累计排序领先不是统计置信度，少量、经过选择的比较可能不足以决定全候选池的可信度。',
              '没有证明原文 Protect/Decouple；名额分配的收益不能改名为保护或解耦收益。',
              '历史对照、共享面积缓存不支持端到端运行时间加速主张。',
              '完整偶数预算曲线：analysis.json；每一步反馈及有限同状态对照：decision_audit.json。']
    with (out/'FINDINGS_ZH.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/mscs_adaptive_1010_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(12,7))
    labels = ('Scalar','MSCS','Fixed 1+1','Adaptive')
    for ax,row in zip(axes.flat,rows):
        for arm,label in zip(COMPARISON_ARMS,labels):
            ax.plot([0]+budgets,[0]+row['gain_pct_by_budget'][arm],marker='o',markersize=3,label=label,
                    linestyle='--' if arm == 'adaptive' else '-',alpha=.85)
        ax.set_title(row['context']); ax.set_xlabel('Candidate verifications')
        ax.set_ylabel('CPI reduction from root (%)'); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Frozen surrogate, seed 42; six previously inspected development roots',fontsize=12)
    fig.tight_layout()
    fig.savefig(out/'adaptive_paths.png',dpi=160); fig.savefig(out/'adaptive_paths.pdf')
    plt.close(fig)


if __name__ == '__main__':
    import sys
    report(Path(sys.argv[1]))
