"""Reproduce the first missing area basis, preserving backend output and XML."""
from pathlib import Path
import json
import subprocess
from secant_support import HERE,read,sha,save,moves,config_key
import area_model


def main():
    out=HERE/'reclaim_area_failure_s42_01';out.mkdir(exist_ok=False)
    old=HERE/'reclaim_allocation_s42_01'
    ctx=read(old/'plan.json')['contexts'][0];root=ctx['root']
    c=next(a['config'] for a in moves(root,4,{config_key(root)}) if a['parameter']=='fetchBufferSize' and a['config']['fetchBufferSize']==8)
    xml=out/'failed_basis.xml';area_model.modify_xml_for_config(area_model.MCPAT_TEMPLATE,c,xml)
    save(out/'plan.json',dict(config=c,root=root,old_plan_sha256=sha(old/'plan.json'),
        area_sources={str(p):sha(p) for p in (area_model.__file__,area_model.MCPAT_TEMPLATE,area_model.MCPAT_BIN)},
        scope='first missing area-basis reproduction; zero candidate CPI queries'))
    result=subprocess.run([area_model.MCPAT_BIN,'-infile',str(xml),'-print_level','5'],capture_output=True,text=True,timeout=60)
    (out/'stdout.txt').write_text(result.stdout);(out/'stderr.txt').write_text(result.stderr)
    try:value=area_model.parse_area(result.stdout) if result.returncode==0 else None
    except ValueError:value=None
    save(out/'result.json',dict(returncode=result.returncode,area=value,stdout_file=str(out/'stdout.txt'),stderr_file=str(out/'stderr.txt')))
    print(json.dumps(dict(returncode=result.returncode,area=value,stdout_tail=result.stdout[-1600:],stderr_tail=result.stderr[-1000:])),flush=True)


if __name__=='__main__':main()
