"""Read-only audit of actually issued op classes; zero new model/simulator queries."""
import json
from pathlib import Path
from collections import defaultdict
from secant_support import HERE, ROOT, read, sha, save

GROUPS = {'FP_ALU': ('FloatAdd', 'FloatCmp', 'FloatCvt'),
          'FP_MultDiv': ('FloatMult', 'FloatMultAcc', 'FloatMisc', 'FloatDiv', 'FloatSqrt'),
          'IntMultDiv': ('IntMult', 'IntDiv')}


def scan():
    records = []; sources = {}
    for stats in sorted((ROOT / 'gem5_bounds/runs').glob('*/*/stats.txt')):
        cfg = stats.with_name('config.json')
        if not cfg.exists():
            continue
        # Last occurrence is the final stats block. Use issued counts so a
        # squashed instruction is not mistaken for an unused execution unit.
        issued = {}
        for line in stats.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith('system.switch_cpus_1.statIssuedInstType_0::'):
                issued[parts[0].split('::')[1]] = int(float(parts[1]))
        system = read(cfg)['system']; cpu = system.get('switch_cpus_1')
        if isinstance(cpu, list):
            assert len(cpu) == 1; cpu = cpu[0]
        assert cpu is not None
        units = cpu['fuPool']['FUList']; counts = {}; mapped = {}
        for name, classes in GROUPS.items():
            matching = [u for u in units if set(o['opClass'] for o in u['opList']) == set(classes)]
            mapped[name] = len(matching) == 1
            counts[name] = sum(issued[c] for c in classes) if all(c in issued for c in classes) else None
        records.append(dict(bench=stats.parent.parent.name,run=stats.parent.name,stats=str(stats),
                            issued_counts=counts,unit_mapping=mapped))
        sources[str(stats)] = sha(stats); sources[str(cfg)] = sha(cfg)
    groups = defaultdict(list)
    for r in records:
        groups[r['bench']].append(r)
    summary = {}
    for b, rows in groups.items():
        summary[b] = dict(runs=len(rows),groups={name:dict(
            all_counts_present=all(r['issued_counts'][name] is not None for r in rows),
            all_units_mapped=all(r['unit_mapping'][name] for r in rows),
            zero_issued_runs=sum(r['issued_counts'][name] == 0 for r in rows),
            max_issued=max((r['issued_counts'][name] for r in rows if r['issued_counts'][name] is not None),default=None))
            for name in GROUPS})
    return records, sources, summary


def main():
    out=HERE/'unused_fu_audit_01';out.mkdir(exist_ok=False)
    records,sources,summary=scan()
    save(out/'plan.json',dict(groups=GROUPS,source_sha256=sources,
        scope='Final ROI issued counts in archived gem5 bounds; not a global assertion about unseen workloads/configurations'))
    save(out/'records.json',records)
    save(out/'summary.json',dict(runs=len(records),by_workload=summary,new_model_queries=0,new_gem5_runs=0))
    assert len(records)==671
    assert all(sha(p)==h for p,h in sources.items())
    # Independent count of zero/nonzero stats lines, distinct from grouping.
    for name,classes in GROUPS.items():
        for r in records:
            values={}
            for line in Path(r['stats']).read_text().splitlines():
                for cls in classes:
                    key='system.switch_cpus_1.statIssuedInstType_0::'+cls
                    if line.startswith(key+' '):values[cls]=int(float(line.split()[1]))
            if len(values)==len(classes):assert sum(values.values())==r['issued_counts'][name]
            else:assert r['issued_counts'][name] is None
    save(out/'validation.json',dict(passed=True,all671_archived_stats_checked=True,
        issued_op_classes_and_native_FU_mapping_checked=True,sources_preserved=True,new_queries=0))
    print(json.dumps(read(out/'summary.json'),indent=2))


if __name__=='__main__':main()
