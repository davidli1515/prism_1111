"""Freeze eligibility and screen ordering; does not load the neural network."""
import json
from fresh_disagreement_support import *


def main():
    OUT.mkdir(exist_ok=False)
    preceding = read(PREVIOUS/'results_s42_01/plan.json')
    verify_sources(preceding)
    sources = dict(preceding['source_sha256'])
    sources[str(PREVIOUS/'results_s42_01/plan.json')] = sha(PREVIOUS/'results_s42_01/plan.json')
    directories = sorted(p for p in ROOT.glob('mscs_*_1010') if p.is_dir() and p != HERE)
    exposed, unknown, files = set(), set(), {}
    for directory in directories:
        for path in sorted(directory.rglob('*.json')):
            hint = next((bench_of(part) for part in path.parts if bench_of(part)), None)
            found, missing = configs_in(read(path), hint)
            exposed.update(found); unknown.update(missing); files[str(path)] = sha(path)
    for c in unknown:
        exposed.update((b, c) for b in BENCHES)
    sources.update(files)
    save(OUT/'exclusions.json', dict(directories=[str(p) for p in directories],
        source_sha256=files, unknown_context_configs=[dict(c) for c in sorted(unknown)],
        exposed=[dict(bench=b, config=dict(c)) for b, c in sorted(exposed)]))
    archives = []; schedules = []
    for bench in BENCHES:
        path = ROOT/'mscs_guard_v5/results_full_01'/bench/'s42/initial.json'
        archive = read(path); sources[str(path)] = sha(path)
        assert archive['checkpoint_sha256'] == sha(DEFAULT_CKPT)
        points = np.c_[archive['cpis'], archive['areas']]
        front = np.flatnonzero(pareto_mask(points)).tolist()
        records = []
        for index in front:
            c = archive['configs'][index]
            records.append(dict(archive_index=index, legal=bool(legal(c)),
                                exposed=(bench, config_key(c)) in exposed))
        order = [r['archive_index'] for r in records if r['legal'] and not r['exposed']][:MAX_SCREEN]
        roots = [dict(id=f'{bench}_freshidx{i}', bench=bench, archive_index=i,
                      root=archive['configs'][i], root_cpi=archive['cpis'][i],
                      root_area=archive['areas'][i], area_cap=1.10*archive['areas'][i],
                      area_sources=archive['area_sources'], source_directory=str(OUT)) for i in order]
        archives.append(dict(bench=bench, path=str(path), front_eligibility=records))
        schedules.append(dict(bench=bench, roots=roots))
        for l1 in sorted({r['root']['l1_size'] for r in roots}):
            paths = sorted((Path(PRECOMPUTE)/f'l1_{l1}'/bench).glob('slice_*/dse_precomputed.npy'))
            paths = [p for p in paths if int(p.parent.name.split('_')[1]) % 2 == 0][:50]
            assert len(paths) == 50
            sources.update({str(p): sha(p) for p in paths})
    for name in ('mean','std'):
        p = Path(NORM_CACHE)/f'18k_s42_{name}.npy'; sources[str(p)] = sha(p)
    for p in list(HERE.glob('*.py'))+[HERE/'PROTOCOL.txt']:
        sources[str(p)] = sha(p)
    save(OUT/'screen_plan.json', dict(seed=42, checkpoint=str(DEFAULT_CKPT),
        checkpoint_sha256=sha(DEFAULT_CKPT), archives=archives, schedules=schedules,
        max_screen_per_bench=MAX_SCREEN, source_sha256=sources,
        exclusions_sha256=sha(OUT/'exclusions.json'),
        scope='new roots within prior 1010 experiments; conditional disagreement audit; archive roots already evaluated'))
    print(json.dumps(dict(phase='screen_order_frozen', schedules=[dict(bench=s['bench'],
        indices=[r['archive_index'] for r in s['roots']]) for s in schedules],
        excluded=len(exposed), unknown_context_configs=len(unknown)), indent=2), flush=True)


if __name__ == '__main__':
    main()
