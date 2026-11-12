"""Describe all saved even-query budgets without additional model evaluations."""
import json
from pathlib import Path
import sys
import numpy as np
from hybrid_support import ARMS


def read(p):
    return json.loads(Path(p).read_text())


def analyze(out):
    out=Path(out); summary=read(out/'summary.json'); rows=[]
    for pair in summary['paired']:
        target=pair['final_cpi']['scalar_excess']; curves={}; first={}
        for arm in ARMS:
            rr=[read(p) for p in sorted((out/'walks'/pair['context']/arm).glob('round_*.json'))]
            c0=rr[0]['root_cpi']
            curves[arm]=[{'queries':r['trials'][-1]['query'],'cpi':r['next_cpi'],
                          'gain_pct':100*(1-r['next_cpi']/c0)} for r in rr]
            hit=[p['queries'] for p in curves[arm] if p['cpi']<=target*(1+1e-6)]
            first[arm]=min(hit) if hit else None
        rows.append({'context':pair['context'],'group':pair['group'],'curves':curves,
                     'scalar_final_cpi_target':target,'first_even_budget_at_target':first,
                     'mean_gain_at_six_even_budgets_pct':{a:float(np.mean([p['gain_pct'] for p in curves[a]])) for a in ARMS}})
    with (out/'anytime.json').open('x') as f:
        json.dump({'scope':'descriptive saved-trajectory analysis; all budgets 2/4/6/8/10/12, not a new confirmation experiment',
                   'new_queries':0,'rows':rows},f,indent=2)
    lines=['已有轨迹的预算效率：零新增查询','',
           '对每个起点，统一以 scalar 在 12 次预算内的最终 CPI 为目标，列出各法首次达到目标的偶数预算。',
           '采用 1e-6 相对容差；未达到记为 None。另完整列出六个预算点，避免只展示有利预算。',
           '这属于事后描述性统计，小样本单 seed，不推断总体加速。','',
           '起点 | scalar | MSCS | hybrid （首次达到 scalar 最终 CPI 的候选验证数）']
    for row in rows:
        lines.append(row['context']+' | '+' | '.join(str(row['first_even_budget_at_target'][a]) for a in ARMS))
    lines += ['','全部预算点的 CPI 降幅%（scalar, MSCS, hybrid）：']
    for row in rows:
        lines.append(row['context'])
        for i in range(6):
            lines.append(f"  q={row['curves'][ARMS[0]][i]['queries']}: "+', '.join(f"{row['curves'][a][i]['gain_pct']:.5f}" for a in ARMS))
    with (out/'ANYTIME_ZH.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    analyze(sys.argv[1])
