"""Explain both differing outcomes from the frozen initial-pool audits, no queries."""
import json
from pathlib import Path
import sys
import numpy as np


def read(p):
    return json.loads(Path(p).read_text())


def main(out):
    out = Path(out); cases = []
    for ctx in ('602.gcc_s_middle', '605.mcf_s_middle'):
        root = out/'walks'/ctx
        ds = {a: read(root/a/'decision_00.json') for a in ('scalar_linear', 'scalar_excess', 'mscs_logit')}
        audit = read(out/'audits'/f'{ctx}.json'); c0 = ds['mscs_logit']['root_cpi']
        arr = np.load(root/'mscs_logit/initial_predictions.npz')
        p0 = (arr['root_cs'][..., None]*arr['root_m'][..., 1:]).mean(0)
        for i, c in enumerate(audit['candidates']):
            if c['parameter'] != 'LQEntries' or c['sign'] != 1:
                continue
            dp = np.array(audit['predicted_pressure_cpi'][i])-p0
            da = np.array(audit['actual_pressure_cpi'][i])-p0
            np.testing.assert_allclose(dp.sum(-1).mean(), ds['mscs_logit']['predicted_cpis'][i]-c0, atol=5e-6)
            np.testing.assert_allclose(da.sum(-1).mean(), audit['actual_cpis'][i]-c0, atol=5e-6)
            cases.append({'context': ctx, 'parameter': 'LQEntries', 'from': ds['mscs_logit']['root']['LQEntries'],
                          'to': c['config']['LQEntries'], 'configuration': c['config'],
                          'predicted_gain_pct': {a: 100*(1-d['predicted_cpis'][i]/c0) for a, d in ds.items()},
                          'actual_gain_pct': 100*(1-audit['actual_cpis'][i]/c0),
                          'predicted_pressure_cpi_change_4x3': dp.tolist(),
                          'actual_pressure_cpi_change_4x3': da.tolist(),
                          'predicted_stage_average_FP_BP_SP': dp.mean(0).tolist(),
                          'actual_stage_average_FP_BP_SP': da.mean(0).tolist(),
                          'scope': 'initial predictions frozen before labels; exact frozen surrogate, not gem5'})
    with (out/'case_study.json').open('x') as f:
        json.dump(cases, f, indent=2)
    lines = ['成对案例：同一机制何时帮助搜索、何时过度保守', '',
             '此分析只读既有的冻结预测和离线审计，没有新查询或搜索规则变化。',
             '正的收益表示 CPI 降低。分量单位为 CPI，先逐 stage 记账，再平均；不把四个 stage 当独立开销相加。', '',
             '起点 | LQ 调整 | scalar_excess 预测收益% | MSCS 预测收益% | 固定 surrogate 实际收益%']
    for r in cases:
        lines.append(f"{r['context']} | {r['from']}->{r['to']} | {r['predicted_gain_pct']['scalar_excess']:+.5f} | "
                     f"{r['predicted_gain_pct']['mscs_logit']:+.5f} | {r['actual_gain_pct']:+.5f}")
    lines += ['', 'gcc：scalar 反复高估加 LQ 的收益；MSCS 预测 FP 增加抵消 BP 减少，避开了这些变慢候选。',
              '例如 24->32：scalar 预测改善 0.891%，MSCS 预测恶化 0.033%，验证为恶化 0.113%。',
              'MSCS 提前转向 l1d_assoc，最终在同样 12 次候选验证下多降低 0.347 个百分点的 CPI。',
              '大的 LQ 步长上 MSCS 仍明显高估恶化幅度；这例支持有用的方向区分，不支持幅度准确。', '',
              'mcf：同样的非线性外推过度保守，把 LQ 79->87 的收益从 scalar 的 0.100% 压到 0.002%。',
              '验证实际有 0.074% 收益；MSCS 改选 L2，后续路径最终少改善 0.093 个百分点。',
              '三类分量变化互相抵消时，各分量不大的误差也会成为净收益的重要误差。', '',
              '结论范围：这个小实验有可复核的搜索收益案例，也有同机制导致的失败。',
              '不能写成稳定优于 scalar，或已验证了真实硬件瓶颈因果关系。',
              '下一步问题应是如何在有限步长下判断分量外推何时可信；不据这四点调参再把同四点当确认集。']
    with (out/'CASE_STUDY_ZH.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main(sys.argv[1])
