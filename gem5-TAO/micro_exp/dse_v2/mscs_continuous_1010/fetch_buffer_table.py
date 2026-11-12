"""Extend only the fetch-buffer feature lookup, preserving all original data."""
import json,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from secant_support import HERE,read,sha,save,array_save
from dse_infer import load_precomputed,encode_distribution

BENCHES=['605.mcf_s','619.lbm_s','654.roms_s','621.wrf_s']


def main():
    started=time.perf_counter();out=HERE/'fetch_buffer_table_01';out.mkdir(exist_ok=False)
    repo=HERE.parents[2];base=repo/'micro_exp/dse_exp';train=repo/'concord_analytical'
    assert read(HERE/'fetch_buffer_direct_audit_02/validation.json')['passed']
    code=[HERE/'fetch_buffer_table.cpp',HERE/'fetch_buffer_table.py',train/'analytical_models.cpp',
          train/'analytical_models.h',train/'trace_analysis.cpp',train/'trace_analysis.h',base/'dse_infer.py']
    sources={str(p):sha(p) for p in code};binary=out/'fetch_buffer_table'
    subprocess.run(['/usr/bin/g++','-O2','-std=c++17',str(code[0]),str(train/'trace_analysis.cpp'),'-o',str(binary)],check=True)
    jobs=[];ids={}
    for b in BENCHES:
        paths=sorted((base/'precompute/l1_64'/b).glob('slice_*/dse_precomputed.npy'))
        paths=[p for p in paths if int(p.parent.name.split('_')[1])%2==0][:50]
        assert len(paths)==50;ids[b]=[int(p.parent.name.split('_')[1]) for p in paths]
        for p in paths:
            sid=int(p.parent.name.split('_')[1]);trace=base/'traces'/b/f'slice_{sid}/o3trace.txt';assert trace.is_file()
            target=out/'direct'/b/f'slice_{sid}.json';target.parent.mkdir(parents=True,exist_ok=True)
            jobs.append((b,sid,p,trace,target))
    save(out/'plan.json',dict(benchmarks=BENCHES,slice_ids=ids,nb=[8,16,32,64],decode_widths=list(range(4,13)),
        icache_config=dict(l1_size=64,l2_size=1024,l1i_assoc=4,l2_assoc=16),source_sha256=sources,
        scope='Repair fetch-buffer truncation only. Other original analytical/cache approximations are preserved; no full physical-parity claim.'))
    def one(job):
        b,sid,p,trace,target=job;before={str(p):sha(p),str(trace):sha(trace)}
        result=subprocess.run([str(binary),str(trace),str(target)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError((b,sid,result.returncode,result.stderr))
        d=read(target);pre=load_precomputed(p);assert d['windows']==pre['W']
        table=np.zeros((4,9,101),dtype=np.float64);maxerr=0.;diffs=0
        for rec in d['runs']:
            ni=[8,16,32,64].index(rec['nb']);di=rec['dw']-4;encoded=encode_distribution(np.asarray(rec['throughput']))
            np.testing.assert_allclose(encoded,rec['features'],rtol=2e-6,atol=2e-6)
            if rec['nb']==8:
                old=pre['fb_thr'][7*9+di]
                np.testing.assert_allclose(old,rec['throughput'],rtol=2e-6,atol=2e-6)
                maxerr=max(maxerr,float(np.max(np.abs(old-np.asarray(rec['throughput'])))))
            table[ni,di]=rec['features']
        diffs=int(np.sum(np.max(np.abs(table-table[:1]),axis=-1)>1e-8))
        assert all(sha(path)==h for path,h in before.items())
        return b,sid,table,before,dict(bench=b,slice=sid,nb8_max_abs_throughput_error=maxerr,changed_config_distributions=diffs)
    tables={b:{} for b in BENCHES};raw_sources={};checks=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(one,j) for j in jobs]
        for n,f in enumerate(as_completed(futures),1):
            b,sid,table,src,check=f.result();tables[b][sid]=table;raw_sources.update(src);checks.append(check)
            if n%25==0:print(json.dumps(dict(phase='table',completed=n,total=len(jobs))),flush=True)
    artifacts={}
    for b in BENCHES:
        path=out/f'{b}.npz';array_save(path,features=np.stack([tables[b][sid] for sid in ids[b]]),slice_ids=np.array(ids[b]),nb=np.array([8,16,32,64]),decode_widths=np.arange(4,13))
        artifacts[str(path)]=sha(path)
    assert all(sha(p)==h for p,h in sources.items())
    save(out/'summary.json',dict(trace_slices=len(checks),checks=sorted(checks,key=lambda r:(r['bench'],r['slice'])),
        raw_sources_sha256=raw_sources,artifact_sha256=artifacts,new_model_queries=0,new_gem5_runs=0,wall_seconds=time.perf_counter()-started))
    save(out/'validation.json',dict(passed=True,all1800_nb8_series_match_existing_precompute=True,
        all_cpp_feature_distributions_reconstructed_in_python=True,exact_search_slice_order_preserved=True,sources_preserved=True))
    print(json.dumps(dict(phase='complete',trace_slices=len(checks),wall_seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
