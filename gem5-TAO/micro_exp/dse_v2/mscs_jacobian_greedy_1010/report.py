"""Fixed descriptive report; no selection of winners or follow-up experiments."""
import csv
import json
from pathlib import Path
import numpy as np
from core import ARMS


def read(p):
    return json.loads(Path(p).read_text())


def report(out):
    out = Path(out); summary = read(out/'summary.json')
    lines = ['完整特征 Jacobian greedy：固定 checkpoint，seed 42，4 个配对起点',
             '', '每种方法每个起点最多验证 12 个候选，面积上限为起点面积的 110%。',
             '数字为相对各自共同起点的 CPI 降幅；比较单轮搜索，不是完整 Pareto DSE。', '',
             '起点 | scalar_linear 降幅% | scalar_excess 降幅% | MSCS 降幅% | MSCS 对较强 scalar 的优势(百分点)']
    for r in summary['paired']:
        lines.append(r['context']+' | '+' | '.join(f"{r['gain_pct'][a]:.4f}" for a in ARMS)
                     +f" | {r['mscs_vs_best_scalar_gain_pp']:+.4f}")
    diffs = [r['mscs_vs_best_scalar_gain_pp'] for r in summary['paired']]
    wins = sum(d > .001 for d in diffs); losses = sum(d < -.001 for d in diffs)
    lines += ['', f'对每个起点较强的 scalar：{wins} 胜 / {4-wins-losses} 平 / {losses} 负（平局容差 0.001 个百分点）。',
              '这是探索性小样本，不能由其中一次胜出推断跨 benchmark 优势。', '',
              '成本：特征生成、求导和预测均计时；MSCS 每次 Jacobian 12 个反向标量，scalar 为 4 个。',
              '面积缓存随顺序复用，因此不以总墙钟时间宣称 MSCS 加速。']
    for arm in ARMS:
        rows = [r for r in summary['walks'] if r['arm'] == arm]
        q = sum(r['counts']['candidate_forward_calls'] for r in rows)
        seconds = sum(r['compute_seconds_excluding_area'] for r in rows)
        jac = sum(r['timing_seconds']['jacobian'] for r in rows)
        forward = sum(r['timing_seconds']['candidate_forward'] for r in rows)
        lines.append(f'{arm}: 候选验证 {q}, 搜索计算(不含面积) {seconds:.3f}s, 求导 {jac:.3f}s, 验证前向 {forward:.3f}s')
    lines += [f"面积调用: {summary['area_counts']}; 失败 {summary['area_failures']}", '',
              '搜索结束后审计初始候选池（无面积筛选，标签没有回流到搜索）：',
              '起点 | 初始 CPI 预测 MAE / 起点 CPI (%) [linear, excess, MSCS] | top2 捕获可用收益比例 [linear, excess, MSCS]']
    extra = 0
    for a in summary['audit_metrics']:
        ms = a['metrics']; extra += a['counts'].get('extra_forward_calls', 0)
        mae = ', '.join(f"{100*ms[k]['relative_cpi_MAE']:.3f}" for k in ARMS)
        capture = ', '.join(f"{100*ms[k]['top2_gain_capture']:.1f}%" if ms[k]['top2_gain_capture'] is not None else 'N/A' for k in ARMS)
        lines.append(f"{a['context']} | {mae} | {capture}")
    lines += [f'独立初始候选池审计额外前向: {extra}。此数不包含在搜索预算内，也未用于排名或接受决策。', '',
              '解释质量：每个 trial 保存预测/实际 4×3 分量 CPI 变化；stage 为不同记账视角，不能直接相加。',
              '解释只经固定 surrogate 验证，不等于真实 gem5 因果归因。']
    explanation_rows = []
    for p in sorted((out/'walks').glob('*/mscs_logit/round_*.json')):
        r = read(p)
        for t in r['trials']:
            e = t['explanation']; pred = np.array(e['predicted_pressure_cpi_change'])
            actual = np.array(e['actual_pressure_cpi_change'])
            mask = abs(actual) > 1e-5
            agreement = float(np.mean(np.sign(pred[mask]) == np.sign(actual[mask]))) if mask.any() else None
            explanation_rows.append({'context': p.parent.parent.name, 'step': r['step'], 'parameter': t['parameter'],
                                     'from': r['root'][t['parameter']], 'to': t['config'][t['parameter']],
                                     'accepted': t['accepted'], 'feasible': t['feasible'],
                                     'predicted_gain_pct': 100*t['predicted_gain'], 'actual_gain_pct': 100*t['actual_gain'],
                                     'component_change_MAE': float(np.mean(abs(pred-actual))),
                                     'component_sign_agreement_abs_gt_1e_minus5': agreement})
    with (out/'explanations.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(explanation_rows[0])); writer.writeheader(); writer.writerows(explanation_rows)
    accepted = [r for r in explanation_rows if r['accepted']]
    lines.append(f'MSCS 接受 {len(accepted)} 步，全部已在 explanations.csv 和原始 round JSON 中保留。')
    for r in accepted:
        lines.append(f"  {r['context']} 步{r['step']+1}: {r['parameter']} {r['from']}→{r['to']}; "
                     f"预测 CPI 收益 {r['predicted_gain_pct']:+.3f}%, 实际 {r['actual_gain_pct']:+.3f}%, "
                     f"分量变化 MAE {r['component_change_MAE']:.5f}")
    lines += ['', '判断口径：',
              '  有限步长 MSCS 预测是否更准，与相同预算是否找到更低 CPI，分别报告。',
              '  起点一阶 CPI 导数一致；差别是 component log-odds 外推隐含的有限步长曲率。',
              '  完整特征生成有成本；输入切换 ReLU 区域后局部 logits 外推可能不准。',
              '  Scalar 二阶或其他非线性方法也能获取曲率，本实验不证明 MSCS 在原理上独占它。',
              '  架构宽度、L1 trace、BP 类型固定，仅测试合法的单资源调整。',
              '  每步解释是可被反驳的预测，不是仅凭标签就确认瓶颈来源。',
              '  初始审计标签仅限离线诊断，未测全部候选面积，不声称知道可行最优动作。', '',
              f"原文件 SHA256 保持不变: {summary['source_preservation_passed']}",
              f"独立重放验证: {read(out/'validation.json')['passed']}",
              f"运行秒数（写 summary 时）: {summary['wall_seconds']:.1f}"]
    with (out/'FINDINGS_ZH.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import sys
    report(sys.argv[1])
