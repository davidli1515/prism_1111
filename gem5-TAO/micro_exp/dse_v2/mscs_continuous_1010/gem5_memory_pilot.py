"""Bounded three-config, private-directory gem5 validation; old sources reused."""
import json,os,sys,time,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'gem5_bounds'))
import run_gem5_bounds as bounds
import validate_joint_dse as original


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)


def verify_config(path,c):
    native=read(path);cpu=native['system']['switch_cpus_1']
    if isinstance(cpu,list):cpu=cpu[0]
    keys=('fetchWidth','decodeWidth','dispatchWidth','commitWidth','wbWidth','numROBEntries',
          'numIQEntries','LQEntries','SQEntries','fetchBufferSize')
    for k in keys:assert int(cpu[k])==c[k],(k,cpu[k],c[k])
    for k in ('renameWidth','issueWidth'):assert int(cpu[k])==c['dispatchWidth']
    fu=cpu['fuPool']['FUList']
    for k,i in [('IntALU',0),('IntMultDiv',1),('FP_ALU',2),('FP_MultDiv',3),('RdWrPort',8)]:assert int(fu[i]['count'])==c[k]
    original_cpu=native['system']['cpu']
    if isinstance(original_cpu,list):original_cpu=original_cpu[0]
    for name,param in [('icache','l1i_assoc'),('dcache','l1d_assoc')]:
        assert int(original_cpu[name]['size'])==1024*c['l1_size'] and int(original_cpu[name]['assoc'])==c[param]
    assert 'l2' not in native['system']
    return dict(cpu_functional_units_and_l1_match=True,l2_enabled=False,
                inactive_parameters=['l2_size','l2_assoc'])


def run_job(job):
    folder=Path(job['folder']);fifo=folder/'stack_fifo.txt';folder.mkdir(exist_ok=False)
    save(folder/'job.json',job);os.mkfifo(fifo)
    t=time.perf_counter();rc=None;error=None
    with (folder/'stack_last.txt').open('w') as last:
        reader=subprocess.Popen(['tail','-n','1',str(fifo)],stdout=last,stderr=subprocess.PIPE)
        try:
            with (folder/'run.log').open('w') as log:
                proc=subprocess.run(job['argv'],cwd=str(folder),stdout=log,stderr=subprocess.STDOUT,
                                    timeout=original.TIMEOUT,check=False)
                rc=proc.returncode
                if rc!=0:error='gem5_exit_'+str(rc)
        except subprocess.TimeoutExpired:error='timeout'
        finally:
            try:reader.communicate(timeout=30)
            except subprocess.TimeoutExpired:reader.kill();reader.communicate();error=error or 'trace_reader_timeout'
            if reader.returncode!=0:error=error or 'trace_reader_failure'
    fifo.unlink()
    cp=original.parse_cpi(str(folder/'stats.txt'));fields=bounds.parse_last(str(folder/'stack_last.txt'))
    verification=None
    if error is None:
        if not cp or fields is None:error='parse_failure'
        else:
            try:verification=verify_config(folder/'config.json',job['config'])
            except Exception as exc:error='configuration_mismatch: '+repr(exc)
    result=dict(role=job['role'],config=job['config'],proxy_cpi=job['proxy_cpi'],gem5_cpi=cp,
                fields28=fields,mscs16=bounds.stack16(fields) if fields is not None else None,
                returncode=rc,error=error,configuration_verification=verification,elapsed_seconds=time.perf_counter()-t)
    save(folder/'result.json',result)
    print(json.dumps(dict(phase='job_complete',role=job['role'],gem5_cpi=cp,error=error,elapsed_seconds=result['elapsed_seconds'])),flush=True)
    return result


def main():
    out=HERE/'gem5_memory_pilot_01';out.mkdir(exist_ok=False);bench='619.lbm_s'
    development=HERE/'paid_centers_walk_s42_01';raw=HERE/'pair_walk_s42_01'
    assert read(development/'validation.json')['passed'] and read(raw/'validation.json')['passed']
    pp=read(development/'plan.json');rr=read(development/'walks'/bench/'mscs_joint/summary.json')
    controls=[raw/'walks'/bench/'scalar_joint/summary.json',development/'walks'/bench/'scalar_joint/summary.json']
    control_records=[read(p) for p in controls];winner=min(range(2),key=lambda i:control_records[i]['final_cpi']);sc=control_records[winner]
    points=[('root',pp['root'],rr['initial_cpi']),('scalar_endpoint',sc['final_config'],sc['final_cpi']),
            ('mscs_endpoint',rr['final_config'],rr['final_cpi'])]
    insts=bounds.surrogate_insts(bench,64);jobs=[]
    for role,c,cp in points:
        folder=out/'runs'/role;argv=original.build_cmd(bench,c,str(folder),bounds.WARMUP,insts)
        argv+=['-P',f"{bounds.CPU}.dump_inst_trace_path='{folder/'stack_fifo'}'"]
        jobs.append(dict(role=role,bench=bench,config=c,proxy_cpi=cp,folder=str(folder),argv=argv))
    paths=[Path(__file__),HERE/'gem5_memory_pilot_protocol.txt',Path(bounds.__file__),Path(original.__file__),
           Path(original.GEM5_BIN),Path(original.GEM5_CONFIG),development/'plan.json',development/'validation.json',
           development/'walks'/bench/'mscs_joint/summary.json',*controls]
    sources={str(p):sha(p) for p in paths}
    save(out/'plan.json',dict(jobs=jobs,workers=2,warmup=bounds.WARMUP,roi_instructions=insts,
         selected_scalar_source=str(controls[winner]),source_sha256=sources,
         known_configuration_limit='Original generator and validation setup have no L2 enabled; L2 knobs do not change simulated hardware.',
         scope='post-hoc one-workload case; whole-ROI gem5 vs 50-slice surrogate, not matched-slice validation'))
    (out/'runs').mkdir()
    print(json.dumps(dict(phase='gem5_plan_frozen',jobs=3,workers=2,roi_instructions=insts)),flush=True)
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(run_job,j) for j in jobs]):rows.append(future.result())
    assert all(sha(p)==h for p,h in sources.items())
    all_ok=all(r['error'] is None for r in rows);by={r['role']:r for r in rows};comparison=None
    if all_ok:
        comparison={kind:{role:100*(1-by[role][kind]/by['root'][kind]) for role in ('scalar_endpoint','mscs_endpoint')}
                    for kind in ('proxy_cpi','gem5_cpi')}
    save(out/'summary.json',dict(results=by,all_successful=all_ok,gain_percent=comparison,source_hashes_preserved=True,
         scope='single selected development case; different aggregation windows, no generalization claim'))
    print(json.dumps(dict(phase='complete',all_successful=all_ok,gain_percent=comparison)),flush=True)


if __name__=='__main__':main()
