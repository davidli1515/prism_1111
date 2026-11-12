"""Post-result illustration of one correction AND one failure; no new queries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matched_support import OUT, read, save, sha

summary = read(OUT/'summary.json')
cases = [('602.gcc_s_middle_r1_stride2', 'gcc: LQ 24 to 40'),
         ('605.mcf_s_middle_r1_stride2', 'mcf: LQ 79 to 95')]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), layout='constrained')
records = []
for ax, (cid, title) in zip(axes, cases):
    row = next(r for r in summary['cells'] if r['cell_id'] == cid)
    a = next(a for a in row['actions'] if a['parameter'] == 'LQEntries')
    values = []
    for key in ('predicted_pressure_cpi_change', 'actual_pressure_cpi_change'):
        component = 100*np.array(a[key]).mean(axis=0)/row['root_cpi']
        values.append(np.r_[component, component.sum()])
    np.testing.assert_allclose(values[0][-1], -a['mscs_gain_pct'], rtol=5e-4, atol=2e-5)
    np.testing.assert_allclose(values[1][-1], -a['actual_gain_pct'], rtol=5e-4, atol=2e-5)
    x = np.arange(4)
    ax.bar(x-.17, values[0], width=.32, color='#377eb8', label='MSCS prediction')
    ax.bar(x+.17, values[1], width=.32, color='#ff9c33', label='Full surrogate')
    ax.scatter([3], [-a['scalar_gain_pct']], marker='x', s=65, color='#202020',
               linewidth=2, label='Scalar prediction (total)', zorder=5)
    ax.axhline(0, linewidth=.8, color='gray')
    ax.set_xticks(x, ['FP', 'BP', 'SP', 'Total'])
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('CPI change (% of root; lower is better)')
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='best', fontsize=8)
    records.append(dict(cell_id=cid, selected_post_result=True,
        predicted_component_change_pct=values[0].tolist(),
        actual_component_change_pct=values[1].tolist(),
        scalar_predicted_total_change_pct=-a['scalar_gain_pct']))
fig.suptitle('Finite LQ response: one correction and one counterexample', fontsize=13)
fig.supxlabel('Stage-average CPI accounts; frozen-surrogate verification, not gem5', fontsize=9)
for extension in ('png','pdf'):
    p = OUT/f'component_cases.{extension}'
    assert not p.exists()
    fig.savefig(p, dpi=180)
plt.close(fig)
save(OUT/'component_cases.json', dict(cases=records,
    scope='Post-result paired illustration, not a new validation subset',
    source_sha256={str(OUT/'summary.json'):sha(OUT/'summary.json'), __file__:sha(__file__)}))
