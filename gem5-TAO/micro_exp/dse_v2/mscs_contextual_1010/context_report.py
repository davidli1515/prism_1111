import json
from pathlib import Path
import numpy as np
from mechanism import MODES


def read(p):
    return json.loads(Path(p).read_text())


def report(out):
    out = Path(out); summary = read(out/'summary.json'); plan = read(out/'plan.json')
    lines = ['条件保护 / 互补解耦：最小确认实验', '',
             '固定 checkpoint、seed 42、4 个不同于上一轮的起点。每个方法每个起点只作一次 greedy 决策。',
             '每法 2 次候选验证，第一名额共用 scalar 提案，第二名额使用本方法排名；相同面积约束。',
             '这不是完整搜索轨迹或 gem5 验证。以下收益均来自固定 surrogate。', '',
             '起点 | scalar | full MSCS | 无解耦 | 无保护 | 无条件交互  （CPI 降幅%）']
    for rr in summary['results']:
        lines.append(rr['context']+' | '+' | '.join(f"{rr['outcomes'][m]['gain_pct']:.5f}" for m in MODES))
    differences = [r['mscs_vs_scalar_gain_pp'] for r in summary['results']]
    w = sum(d>.001 for d in differences); l = sum(d<-.001 for d in differences)
    lines += ['', f'MSCS 对 scalar: {w} 胜 / {len(differences)-w-l} 平 / {l} 负。平局容差 0.001 个百分点。', '',
              '如何绑定论文（修订后的可操作机制，不是复现原 Algorithm 2）：',
              '  解耦：明确两资源联合调整的额外收益，而不把大 Scancel 直接当成有益。',
              '  保护：明确某项投入之后缩减另一资源的代价，而不只看当前分量变化范数。',
              '  无解耦/无保护：分别去掉 up/up 和 up/down 的非加性交互；无保护还关闭相应的提案延期。',
              '  保护不否决 scalar 保底名额或已验证的 CPI 改善。Scalar 也使用自己的条件回收预测。', '',
              '机制见证（预测先冻结，再审计单独 A、单独 B、联合 AB；没有回流搜索）：']
    for rr in summary['results']:
        ctx = next(c for c in plan['contexts'] if c['id'] == rr['context'])
        pool = read(out/'roots'/rr['context']/'pool.json')
        for kind in ('decouple', 'protect'):
            if kind not in rr['witness_results']:
                lines.append(f"{rr['context']} {kind}: 没有满足预设条件的预测见证；不强制补选。")
                continue
            r = rr['witness_results'][kind]
            acts = [pool[i] for i in r['parts']]
            actions = ' + '.join(f"{a['parameter']} {ctx['root'][a['parameter']]}->{a['config'][a['parameter']]}" for a in acts)
            lines += [f"{rr['context']} {kind}: {actions}",
                      f"  交互项 / 起点 CPI：scalar预测 {100*r['predictions']['scalar']['interaction']:+.5f}%，MSCS预测 {100*r['predictions']['mscs']['interaction']:+.5f}%，验证 {100*r['actual_interaction']:+.5f}%",
                      f"  第二动作单独代价 {100*r['actual_alone_cost']:+.5f}%，投入第一资源之后的代价 {100*r['actual_after_investment_cost']:+.5f}%",
                      f"  数值机制成立: {r['mechanism_verified']}; 联合动作可行: {r['joint_feasible']}; 第一投入可行: {r['investment_feasible']}"]
            if kind == 'protect':
                lines.append(f"  缩减确实省面积: 单独 {r['area_saved_alone']}, 投入之后 {r['area_saved_after']}")
    search_physical = sum(r['counts'].get('search_physical_forward_calls',0) for r in summary['results'])
    audit_physical = sum(r['counts'].get('audit_physical_forward_calls',0) for r in summary['results'])
    lines += ['', f'搜索逻辑候选验证 40；去重后物理候选前向 {search_physical}；独立审计额外前向 {audit_physical}。',
              f"面积调用 {summary['area_counts']}; 失败 {summary['area_failures']}。",
              f"总耗时 {summary['elapsed_seconds']:.1f} 秒（写 summary 时）。",
              '共同候选特征和物理标签复用仅用于快速实验；不据此声称某一方法加速。',
              '所有候选的完整特征都重建；联合特征没有近似为两单独位移相加。',
              '有限步长局部预测仍可能跨越 ReLU 区域；一个有效见证不能确立整个模型的预测可靠性。',
              '只审计预设少量见证，不能估计全集精度或声称找到候选池最优。',
              f"原来源文件哈希保持不变: {summary['source_preservation_passed']}",
              f"独立 NumPy 重放通过: {read(out/'validation.json')['passed']}"]
    with (out/'FINDINGS_ZH.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')
