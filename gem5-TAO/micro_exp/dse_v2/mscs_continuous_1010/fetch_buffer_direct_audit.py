"""Compare the capped DSE table with original direct training-code features."""
import json,subprocess,time
import numpy as np
from pathlib import Path
from secant_support import HERE,read,sha,save
from dse_infer import load_precomputed,encode_distribution


def main():
    started=time.perf_counter();out=HERE/'fetch_buffer_direct_audit_01';out.mkdir(exist_ok=False)
    repo=HERE.parents[2];base=repo/'micro_exp/dse_exp';training=repo/'concord_analytical'
    cpp=HERE/'fetch_buffer_direct_audit.cpp';binary=out/'fetch_buffer_direct_audit'
    code=[cpp,HERE/'fetch_buffer_direct_audit.py',base/'dse_infer.py',
          training/'analytical_models.cpp',training/'analytical_models.h',
          training/'trace_analysis.cpp',training/'trace_analysis.h']
    source={str(p):sha(p) for p in code}
    subprocess.run(['g++','-O2','-std=c++17',str(cpp),str(training/'trace_analysis.cpp'),'-o',str(binary)],check=True)
    benches=['605.mcf_s','619.lbm_s','654.roms_s','621.wrf_s'];rows=[]
    save(out/'plan.json',dict(benchmarks=benches,slices=[0],fetch_buffers=[8,16,32,64],decode_widths=[8,9,12],
        icache_config=dict(l1_size=64,l2_size=1024,l1i_assoc=4,l2_assoc=16),source_sha256=source,
        scope='Input feature numerical audit only; no model predictions, model edits, training or gem5 runs'))
    raw_sources={}
    for b in benches:
        trace=base/'traces'/b/'slice_0/o3trace.txt';prepath=base/'precompute/l1_64'/b/'slice_0/dse_precomputed.npy'
        raw_sources[str(trace)]=sha(trace);raw_sources[str(prepath)]=sha(prepath)
        output=out/f'{b}.json'
        subprocess.run([str(binary),str(trace),str(output)],check=True)
        direct=read(output);pre=load_precomputed(prepath);assert direct['windows']==pre['W']
        by={(r['nb'],r['dw']):r for r in direct['runs']}
        for dw in (8,9,12):
            stored=pre['fb_thr'][7*9+dw-4];encoded=encode_distribution(stored)
            # This equality anchors the comparison to the actual existing table
            # and trace. If it fails, stop rather than blame buffer truncation.
            np.testing.assert_allclose(stored,by[(8,dw)]['throughput'],rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(encoded,by[(8,dw)]['features'],rtol=2e-6,atol=2e-6)
            for nb in (8,16,32,64):
                actual=np.array(by[(nb,dw)]['features']);thr=np.array(by[(nb,dw)]['throughput'])
                np.testing.assert_allclose(encode_distribution(thr),actual,rtol=2e-6,atol=2e-6)
                rows.append(dict(bench=b,slice=0,decode_width=dw,fetch_buffer=nb,
                    feature_max_absolute_error=float(np.max(np.abs(encoded-actual))),
                    mean_throughput_capped=float(encoded[-1]),mean_throughput_direct=float(actual[-1]),
                    changed_distribution_coordinates=int(np.sum(np.abs(encoded-actual)>1e-8)),
                    changed_throughput_windows=int(np.sum(np.abs(stored-thr)>1e-8))))
    assert all(sha(p)==h for p,h in {**source,**raw_sources}.items())
    save(out/'summary.json',dict(rows=rows,differences_found=any(r['changed_distribution_coordinates'] for r in rows),
        raw_sources_sha256=raw_sources,new_model_queries=0,new_gem5_runs=0,wall_seconds=time.perf_counter()-started))
    save(out/'validation.json',dict(passed=True,all12_nb8_series_match_existing_precompute=True,
       _cpp_and_python_distribution_encoders_agree=True,original_sources_unchanged=True))
    print(json.dumps(read(out/'summary.json'),indent=2))


if __name__=='__main__':main()
