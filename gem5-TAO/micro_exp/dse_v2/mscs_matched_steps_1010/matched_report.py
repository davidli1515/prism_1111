"""Immediate equal-budget results. No feedback update or path concatenation."""
from matched_support import *


def report(out):
    plan = read(out/'plan.json'); lab = read(out/'labels.json')
    labels = {r['id']: r for r in lab['records']}; rows = []
    for cell in plan['cells']:
        d = read(cell['decision_file']); width = compute_w_min(d['root'])
        directory = next(Path(c['source_directory']) for c in plan['contexts'] if c['id'] == cell['context'])
        with np.load(directory/d['prediction_file']) as arr:
            pc0 = pressure(arr['root_m'], width)
            predictions = pressure(arr['predicted_m'], width)
        outcomes = {}; unconstrained = {}; actions = []
        for arm in ARMS:
            records = [labels[cell['label_ids'][str(i)]] for i in cell['chosen'][arm]]
            outcomes[arm] = outcome(cell['root_cpi'], cell['area_cap'], records)
            unconstrained[arm] = outcome(cell['root_cpi'], cell['area_cap'], records, False)
        for index, lid in cell['label_ids'].items():
            i = int(index); a = d['candidates'][i]; rec = labels[lid]; p = a['parameter']
            actions.append(dict(index=i, parameter=p, sign=a['sign'], stride=a['stride'],
                before=d['root'][p], after=a['config'][p], cpi=rec['cpi'], area=rec['area'],
                area_cap=cell['area_cap'], area_status=rec['area_status'],
                feasible=None if rec['area'] is None else rec['area'] <= cell['area_cap']+1e-9,
                scalar_gain_pct=100*(1-d['scalar_predicted_cpis'][i]/cell['root_cpi']),
                mscs_gain_pct=100*(1-d['mscs_predicted_cpis'][i]/cell['root_cpi']),
                actual_gain_pct=100*(1-rec['cpi']/cell['root_cpi']),
                predicted_pressure_cpi_change=(predictions[i]-pc0).tolist(),
                actual_pressure_cpi_change=None if rec['pressure_cpi'] is None else
                    (np.array(rec['pressure_cpi'])-pc0).tolist(),
                selected_by=[arm for arm in ARMS if i in cell['chosen'][arm]]))
        rows.append(dict(cell_id=cell['cell_id'], context=cell['context'], step=cell['step'],
                         pool=cell['pool'], triggers=cell['triggers'], root_cpi=cell['root_cpi'],
                         chosen=cell['chosen'], outcomes=outcomes, unconstrained=unconstrained,
                         actions=actions))
    summaries = {}
    for group in ('full', 'fixed_stride'):
        subset = [r for r in rows if (r['pool'] == 'full') == (group == 'full')]
        group_summary = {}
        for comparator in ('scalar', 'hybrid'):
            for arm in ('mscs', 'hybrid', 'local'):
                if arm == comparator:
                    continue
                counts = dict(win=0, tie=0, loss=0, unresolved=0)
                diffs = []
                for r in subset:
                    a, b = r['outcomes'][arm], r['outcomes'][comparator]
                    if a['best_cpi'] is None or b['best_cpi'] is None:
                        counts['unresolved'] += 1; continue
                    diff = (b['best_cpi']-a['best_cpi'])/r['root_cpi']
                    counts['win' if diff > 1e-6 else 'loss' if diff < -1e-6 else 'tie'] += 1
                    diffs.append(100*diff)
                group_summary[f'{arm}_vs_{comparator}'] = dict(counts,
                    gain_differences_pp=diffs, descriptive_mean_pp=float(np.mean(diffs)) if diffs else None)
        summaries[group] = dict(cells=len(subset), comparisons=group_summary)
    summary = dict(scope=plan['scope'], cells=rows, groups=summaries,
        surveyed_decisions=len({r['decision_file'] for r in plan['survey']}),
        surveyed_pools=len(plan['survey']), selected_cells=len(rows),
        new_forward_calls=lab['new_forward_calls'], reused_labels=lab['historical_records'],
        unique_labels=lab['unique_records'], area_counts=lab['area_counts'],
        area_failures=lab['area_failures'], logical_queries_per_arm=2*len(rows),
        logical_queries_all_arms=8*len(rows), evaluation_seconds=lab['wall_seconds'])
    save(out/'summary.json', summary)
    lines = ['相同状态/候选池/两候选预算：已检查轨迹的条件机制诊断',
             '不是新根节点泛化，不是完整搜索，不是 gem5。',
             f"筛查 {summary['surveyed_decisions']} 个旧状态、{summary['surveyed_pools']} 个分层池；按预测分歧预选 {len(rows)} 个单步比较。",
             f"每方法每单步2次逻辑验证；去重需 {lab['unique_records']} 个标签，复用 {lab['historical_records']}，新前向 {lab['new_forward_calls']}。",
             '固定stride仅保留未裁剪、相同生成倍率的动作；不同资源的基准量不同。',
             'full 与 stride 行可能同根且相关；不作统计显著性或整体搜索胜率推断。', '']
    for group, value in summaries.items():
        lines.append(f"{group}: {value['cells']} 个比较")
        for name, comp in value['comparisons'].items():
            lines.append(f"  {name}: {comp['win']}胜 {comp['tie']}平 {comp['loss']}负 {comp['unresolved']}未知")
    lines += ['', '单步CPI降幅%（scalar / MSCS / hybrid / local）:']
    for row in rows:
        vals = ['未知' if row['outcomes'][a]['gain_pct'] is None else
                f"{row['outcomes'][a]['gain_pct']:.6f}" for a in ARMS]
        lines.append(row['cell_id']+'  '+' / '.join(vals))
        for a in row['actions']:
            lines.append(f"  {a['parameter']} {a['before']}->{a['after']} ({','.join(a['selected_by'])}): "
                         f"预测scalar {a['scalar_gain_pct']:+.5f}%, MSCS {a['mscs_gain_pct']:+.5f}%, "
                         f"完整模型 {a['actual_gain_pct']:+.5f}%, 面积可行 {a['feasible']}")
    lines += ['', '所有新代码和结果放在独立目录；原策略、原结果与checkpoint未修改。',
              '原版Protect需要未来投资的条件反事实，Decouple需要联合动作检验；本轮均未证明。']
    with (out/'FINDINGS_ZH.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')
