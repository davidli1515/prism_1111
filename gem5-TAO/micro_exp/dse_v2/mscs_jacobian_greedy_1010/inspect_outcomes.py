"""Descriptive post-search diagnosis; consumes saved labels, launches no queries."""
import json
from pathlib import Path
import sys
import numpy as np

ARMS = ('scalar_linear', 'scalar_excess', 'mscs_logit')


def read(p):
    return json.loads(Path(p).read_text())


def inspect(out):
    out = Path(out); summary = read(out/'summary.json')
    findings = []
    for pair in summary['paired']:
        ctx = pair['context']; root = out/'walks'/ctx
        ds = {a: read(root/a/'decision_00.json') for a in ARMS}
        audit = read(out/'audits'/f'{ctx}.json')
        c0 = ds[ARMS[0]]['root_cpi']
        pred = {a: np.array(ds[a]['predicted_cpis']) for a in ARMS}
        truth = np.array(audit['actual_cpis'])
        predictions = np.load(root/'mscs_logit/initial_predictions.npz')
        q = predictions['root_m'][..., 1:]
        q = q/q.sum(-1, keepdims=True)
        d = predictions['logodds_slope']
        mean_d = (q[None]*d).sum(-1, keepdims=True)
        var_d = (q[None]*(d-mean_d)**2).sum(-1)
        states, trials, accepted = {}, {}, {}
        for a in ARMS:
            rounds = [read(p) for p in sorted((root/a).glob('round_*.json'))]
            states[a] = [r['root'] for r in rounds]
            trials[a] = [t['config'] for r in rounds for t in r['trials']]
            accepted[a] = [r['next_root'] for r in rounds if r['accepted_trial'] is not None]
        result = {'context': ctx, 'initial_candidate_count': len(truth),
                  'mscs_vs_excess_prediction_gap_pct_of_root_mean': float(np.mean(abs(pred['mscs_logit']-pred['scalar_excess']))/c0*100),
                  'mscs_vs_excess_prediction_gap_pct_of_root_max': float(np.max(abs(pred['mscs_logit']-pred['scalar_excess']))/c0*100),
                  'mscs_prediction_error_pct_of_root_mean': float(np.mean(abs(pred['mscs_logit']-truth))/c0*100),
                  'component_response_weighted_variance_mean': float(var_d.mean()),
                  'initial_top2': {a: ds[a]['chosen'] for a in ARMS},
                  'same_first_top2_set_as_excess': set(ds['mscs_logit']['chosen']) == set(ds['scalar_excess']['chosen']),
                  'same_ordered_trials_as_excess': trials['mscs_logit'] == trials['scalar_excess'],
                  'same_accepted_path_as_excess': accepted['mscs_logit'] == accepted['scalar_excess'],
                  'same_accepted_path_as_linear': accepted['mscs_logit'] == accepted['scalar_linear'],
                  'final_gain_pct': pair['gain_pct'],
                  'mscs_vs_best_scalar_gain_pp': pair['mscs_vs_best_scalar_gain_pp'],
                  'initial_root_clamped_slice_stages': int((predictions['root_m'][..., 0] <= .01).sum())}
        findings.append(result)
    with (out/'diagnosis.json').open('x') as f:
        json.dump({'scope': 'descriptive after the frozen search; zero new queries', 'contexts': findings}, f, indent=2)
    lines = ['离线诊断：MSCS 有限步长曲率是否改变选择（没有新增查询）', '']
    for r in findings:
        lines += [r['context'],
                  f"  MSCS 与 scalar_excess 初始 top2 集合相同: {r['same_first_top2_set_as_excess']}",
                  f"  整条被接受的路径相同: {r['same_accepted_path_as_excess']}",
                  f"  完整候选验证顺序相同: {r['same_ordered_trials_as_excess']}",
                  f"  两者 CPI 预测平均差: 起点 CPI 的 {r['mscs_vs_excess_prediction_gap_pct_of_root_mean']:.6f}%",
                  f"  MSCS CPI 预测平均误差: 起点 CPI 的 {r['mscs_prediction_error_pct_of_root_mean']:.6f}%", '']
    with (out/'DIAGNOSIS_ZH.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')
    plot_paths(out, summary)


def plot_paths(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    colors = ['#0072B2', '#E69F00', '#009E73']; styles = ['-', '--', ':']
    for ax, pair in zip(axes.flat, summary['paired']):
        for arm, color, style, marker in zip(ARMS, colors, styles, ['o', 's', 'x']):
            p = out/'walks'/pair['context']/arm
            rr = [read(f) for f in sorted(p.glob('round_*.json'))]
            c0 = rr[0]['root_cpi']; yy = [0.]+[100*(1-r['next_cpi']/c0) for r in rr]
            xx = [0]+[r['trials'][-1]['query'] for r in rr]
            ax.plot(xx, yy, color=color, linestyle=style, marker=marker, markersize=5, label=arm)
        ax.set_title(pair['context']); ax.set_xlabel('Verified candidates'); ax.set_ylabel('CPI reduction from root (%)')
        ax.grid(alpha=.2); ax.set_xticks(range(0, 13, 2))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.suptitle('Frozen surrogate, seed 42; fixed area cap = 1.10 x root area', y=.955, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(out/'greedy_paths.png', dpi=180)
    fig.savefig(out/'greedy_paths.pdf')
    plt.close(fig)


if __name__ == '__main__':
    inspect(sys.argv[1])
