"""Post-run counterfactual checks using ONLY already evaluated configurations."""
import json
from pathlib import Path
import sys
import numpy as np
from hybrid_support import config_key


def read(p):
    return json.loads(Path(p).read_text())


def inspect(out):
    out=Path(out); plan=read(out/'plan.json'); rows=[]
    for ctx in plan['contexts']:
        root=out/'walks'/ctx['id']; labels={}
        for path in root.glob('*/round_*.json'):
            r=read(path)
            for t in r['trials']:
                k=config_key(t['config'])
                if k in labels:
                    np.testing.assert_allclose(labels[k]['cpi'],t['cpi'],rtol=3e-6)
                labels[k]=t
        for path in sorted((root/'hybrid').glob('round_*.json')):
            r=read(path); d=read(out/r['decision_file'])
            scalar=np.argsort(d['scalar_predicted_cpis'],kind='stable')[:2].tolist()
            if set(scalar)==set(d['chosen']):
                continue
            trial_labels=[labels.get(config_key(d['candidates'][i]['config'])) for i in scalar]
            have=all(t is not None and t['area'] is not None for t in trial_labels)
            rr={'context':ctx['id'],'group':ctx['group'],'step':r['step']+1,
                'scalar_top2':scalar,'hybrid_choices':d['chosen'],
                'scalar_counterfactual_fully_observed':have,
                'scope':'same hybrid state; reuses existing exact outputs; not a counterfactual full trajectory'}
            if have:
                valid=[t['cpi'] for t in trial_labels if t['area']<=ctx['area_cap']+1e-9
                       and t['cpi']<r['root_cpi']*(1-1e-6)]
                scalar_next=min(valid) if valid else r['root_cpi']
                rr['hybrid_vs_scalar_step_gain_pp']=100*(scalar_next-r['next_cpi'])/r['root_cpi']
                rr['scalar_next_cpi']=scalar_next; rr['hybrid_next_cpi']=r['next_cpi']
            rr['hybrid_actions']=[{'parameter':t['parameter'],'from':r['root'][t['parameter']],
                                   'to':t['config'][t['parameter']],'source':t['proposed_by'],
                                   'accepted':t['accepted'],'feasible':t['feasible'],
                                   'scalar_predicted_gain_pct':100*(1-t['scalar_predicted_cpi']/r['root_cpi']),
                                   'mscs_predicted_gain_pct':100*(1-t['mscs_predicted_cpi']/r['root_cpi']),
                                   'actual_gain_pct':100*t['actual_gain']} for t in r['trials']]
            rows.append(rr)
    with (out/'matched_state_audit.json').open('x') as f:
        json.dump({'new_model_queries':0,'rows':rows},f,indent=2)
    lines=['同一 hybrid 状态下比较两个候选名额：只读已有标签，零新查询','',
           '仅当 scalar 前二候选已被任一方法评估过，才比较该步结果。缺失时不补查。',
           '覆盖受已查询候选影响；不能用以下子集估计所有动作的总体胜率或完整反事实路径。','']
    for rr in rows:
        value=f"hybrid 比该状态的 scalar 多改善 {rr['hybrid_vs_scalar_step_gain_pp']:+.5f} 个百分点" if rr['scalar_counterfactual_fully_observed'] else 'scalar 候选标签不完整，不下反事实结论'
        lines.append(f"{rr['context']} 第{rr['step']}轮: {value}")
        for t in rr['hybrid_actions']:
            lines.append(f"  {t['source']} {t['parameter']} {t['from']}->{t['to']}: scalar预测 {t['scalar_predicted_gain_pct']:+.4f}%, MSCS预测 {t['mscs_predicted_gain_pct']:+.4f}%, 实际 {t['actual_gain_pct']:+.4f}%, 可行 {t['feasible']}, 接受 {t['accepted']}")
    with (out/'MATCHED_STATE_AUDIT_ZH.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    inspect(sys.argv[1])
