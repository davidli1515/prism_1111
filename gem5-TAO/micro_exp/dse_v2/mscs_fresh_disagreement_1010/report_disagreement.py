"""Report all screened roots and all selected outcomes, including failures."""
from fresh_disagreement_support import *


def report(out):
    plan = read(out/'confirmation_plan.json'); data = read(out/'labels.json')
    labels = {r['id']:r for r in data['records']}; rows = []
    for cell in plan['cells']:
        d = read(cell['decision_file']); width = compute_w_min(d['root'])
        with np.load(out/d['prediction_file']) as arr:
            pc0 = pressure(arr['root_m'], width); pc = pressure(arr['predicted_m'], width)
        results = {}; unconstrained = {}
        for arm, ids in cell['chosen'].items():
            recs = [labels[cell['label_ids'][str(i)]] for i in ids]
            results[arm] = outcome(cell['root_cpi'], cell['area_cap'], recs)
            unconstrained[arm] = outcome(cell['root_cpi'], cell['area_cap'], recs, False)
        actions = []
        for si, lid in cell['label_ids'].items():
            i = int(si); a = d['candidates'][i]; r = labels[lid]; p = a['parameter']
            actions.append(dict(index=i, parameter=p, sign=a['sign'], stride=a['stride'],
                before=d['root'][p], after=a['config'][p], cpi=r['cpi'], area=r['area'],
                area_cap=cell['area_cap'], area_status=r['area_status'],
                feasible=None if r['area'] is None else r['area'] <= cell['area_cap']+1e-9,
                scalar_gain_pct=100*(1-d['scalar_predicted_cpis'][i]/cell['root_cpi']),
                mscs_gain_pct=100*(1-d['mscs_predicted_cpis'][i]/cell['root_cpi']),
                actual_gain_pct=100*(1-r['cpi']/cell['root_cpi']),
                predicted_pressure_cpi_change=(pc[i]-pc0).tolist(),
                actual_pressure_cpi_change=(np.array(r['pressure_cpi'])-pc0).tolist(),
                selected_by=[arm for arm in ARMS if i in cell['chosen'][arm]],
                in_prior_exposure_inventory=r['in_prior_exposure_inventory']))
        rows.append(dict(cell_id=cell['cell_id'], context=cell['context'], pool=cell['pool'],
                         primary=cell['primary'], root_cpi=cell['root_cpi'], chosen=cell['chosen'],
                         outcomes=results, unconstrained=unconstrained, actions=actions))
    summaries = {}
    for primary in (True, False):
        subset = [r for r in rows if r['primary'] == primary]; comparisons = {}
        for arm in ARMS[1:]:
            counts = dict(win=0, tie=0, loss=0, unresolved=0)
            for row in subset:
                a, s = row['outcomes'][arm], row['outcomes']['scalar']
                if not a['resolved'] or not s['resolved']:
                    counts['unresolved'] += 1; continue
                gap = (s['best_cpi']-a['best_cpi'])/row['root_cpi']
                counts['win' if gap > 1e-6 else 'loss' if gap < -1e-6 else 'tie'] += 1
            comparisons[arm+'_vs_scalar'] = counts
        summaries['primary_full' if primary else 'secondary_fixed_stride'] = dict(cells=len(subset), comparisons=comparisons)
    summary = dict(scope=plan['scope'], screened_roots=len(plan['screened']),
        selected_roots=plan['selected_roots'], unfilled_root_quota=plan['unfilled_root_quota'],
        selected_cells=len(rows), screening_root_forwards=plan['root_forward_calls'],
        screening_backward_outputs=plan['backward_scalar_outputs'],
        diagnostic_root_forwards=plan['diagnostic_root_forwards'],
        diagnostic_backward_outputs=plan['diagnostic_backward_outputs'],
        diagnostic_jvp_calls=plan['diagnostic_jvp_calls'],
        candidate_forwards=data['new_candidate_forwards'], reused_candidate_labels=0,
        prior_exposed_candidates=sum(r['in_prior_exposure_inventory'] for r in data['records']),
        logical_queries_per_arm=2*len(rows), logical_queries_all_arms=8*len(rows),
        area_counts=data['area_counts'], area_failures=data['area_failures'],
        screening_seconds=plan['wall_seconds'], evaluation_seconds=data['wall_seconds'],
        groups=summaries, cells=rows, cold_local_equals_hybrid=True)
    save(out/'summary.json', summary)
    lines = ['新起点、预测分歧条件下的最小验证：固定checkpoint和原策略',
        '新起点仅指未进入已有1010实验的根/候选池；原档案已评估过这些配置，不是训练或硬件留出。',
        '每点只做单步、每方法两个候选；冷启动local严格等于固定1+1，不检验反馈迁移。',
        f"筛查{len(plan['screened'])}点，选中{len(plan['contexts'])}点；未满足名额{plan['unfilled_root_quota']}。",
        f"筛查成本{plan['root_forward_calls']}次根前向、{plan['backward_scalar_outputs']}个反向输出；候选新前向{len(labels)}次。",
        f"候选中此前出现在排除清单的有{summary['prior_exposed_candidates']}个；未复用候选CPI标签。",
        '标量使用相同root、slice和输入位移，独立标量反向验证其梯度与MSCS恢复的梯度一致。',
        '全池为主比较；若分歧首次出现在固定stride池，则额外报告该池，不能合并成独立样本。', '', '筛查记录:']
    for r in plan['screened']:
        lines.append(f"  {r['context']['id']}: 第一个分歧池={r['first_disagreement']}; 入选={r['selected']}")
    for name, group in summaries.items():
        lines += ['', f"{name}: {group['cells']}组"]
        for arm, cc in group['comparisons'].items():
            lines.append(f"  {arm}: {cc['win']}胜 {cc['tie']}平 {cc['loss']}负 {cc['unresolved']}未知")
    lines += ['', '单步CPI降幅%：scalar / MSCS / hybrid / cold local']
    for row in rows:
        vs = ['未知' if row['outcomes'][a]['gain_pct'] is None else f"{row['outcomes'][a]['gain_pct']:.6f}" for a in ARMS]
        lines.append(row['cell_id']+'  '+' / '.join(vs))
        for a in row['actions']:
            lines.append(f"  {a['parameter']} {a['before']}->{a['after']} ({','.join(a['selected_by'])}): "
                f"scalar预测{a['scalar_gain_pct']:+.6f}%, MSCS预测{a['mscs_gain_pct']:+.6f}%, "
                f"完整模型{a['actual_gain_pct']:+.6f}%, 面积可行{a['feasible']}")
    lines += ['', '所有验证值来自同一冻结surrogate；没有新gem5。',
        '按预测分歧选点，且样本极小：不外推总体搜索胜率、硬件收益或运行时间加速。',
        '筛查和验证费用分别列出，不把提前筛查成本当作免费在线信息。',
        '没有验证原版Protect的未来能力保护，也没有验证Decouple的联合动作收益。']
    with (out/'FINDINGS_ZH.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')
