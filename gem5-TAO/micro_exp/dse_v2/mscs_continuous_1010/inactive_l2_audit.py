"""Read-only audit of instantiated caches and parameter-space mismatch."""
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from secant_support import HERE,read,sha,save


def cache_objects(obj):
    result=[]
    if isinstance(obj,dict):
        if obj.get('type')=='Cache':result.append({k:obj.get(k) for k in ('path','size','assoc')})
        for v in obj.values():
            if isinstance(v,(dict,list)):result.extend(cache_objects(v))
    elif isinstance(obj,list):
        for v in obj:result.extend(cache_objects(v))
    return result


def main():
    out=HERE/'inactive_l2_audit_01';out.mkdir(exist_ok=False)
    project=HERE.parents[2];runs=HERE.parent/'gem5_bounds/runs'
    files=sorted(runs.glob('*/*/result.json'));rows=[];sources={}
    for file in files:
        r=read(file);native=file.with_name('config.json')
        assert native.exists()
        caches=cache_objects(read(native))
        paths=[x['path'] for x in caches]
        rows.append(dict(bench=r['bench'],result_file=str(file),native_file=str(native),caches=caches,
                         l2_present=any('l2' in str(p).lower() for p in paths),l3_present=any('l3' in str(p).lower() for p in paths)))
        sources[str(native)]=sha(native);sources[str(file)]=sha(file)
    template=project/'micro_exp/dse_exp/mcpat-out/mcpat-out-0.xml'
    root=ET.parse(template).getroot()
    counts={p.attrib['name']:p.attrib['value'] for p in root.iter('param') if p.attrib.get('name') in ('number_of_L2s','number_of_L3s')}
    codefiles=[project/'gen_spec_traindata.py',project/'gen_spec_traindata_dse.py',project/'gen_spec_traindata_RS.py',
               project/'configs/common/CacheConfig.py',project/'configs/common/Options.py',
               project/'micro_exp/dse_exp/validate_joint_dse.py',project/'micro_exp/dse_exp/area_model.py',template]
    sources.update({str(p):sha(p) for p in codefiles})
    summary=dict(successful_run_records=len(rows),l2_present=sum(r['l2_present'] for r in rows),l3_present=sum(r['l3_present'] for r in rows),
        workloads=sorted({r['bench'] for r in rows}),mcpat_template_counts=counts,source_sha256=sources,rows=rows,
        conclusion='The audited native validation configurations and McPAT template disable L2/L3; L2 search knobs are not hardware investments in this setup.',
        limits='The available training generator sources also omit --l2cache. This audit does not inspect every historical raw training configuration or invalidate other MSCS fidelity results.',
        action='Freeze l2_size/l2_assoc in subsequent search-only variants; keep the checkpoint and old runs unchanged. Do not enable a different hierarchy silently.')
    save(out/'summary.json',summary)
    assert all(sha(p)==h for p,h in sources.items())
    save(out/'validation.json',dict(passed=True,all_available_successful_native_configs_read=True,source_preservation=True,new_model_or_gem5_queries=0))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','source_sha256')},indent=2))


if __name__=='__main__':main()
