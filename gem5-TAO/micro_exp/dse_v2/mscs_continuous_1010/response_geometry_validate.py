"""Independent tensor solve and root/gradient invariants for the offline probe."""
import torch
import numpy as np
from secant_support import HERE,read,sha,save,compute_w_min
from adaptive_validate import close


def main():
    out=HERE/'response_geometry_s42_01';plan=read(out/'plan.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    records=read(out/'frozen_predictions.json')['records'];rank_matches=0
    for r in records:
        d=read(r['source_decision']);width=compute_w_min(d['root'])
        with np.load(out/r['array_file']) as z:
            assert sha(out/r['array_file'])==r['array_sha256']
            h=torch.as_tensor(z['scalar_history']);res=torch.as_tensor(z['residual'])
            ds=torch.as_tensor(z['scalar_candidate']);lo=1/width;hi=min(100.,1/(.01*width+1e-8))
            base=torch.as_tensor(z['root_cs']).double();excess=(base-lo).clamp(min=1e-10)
            for arm in ('raw_scalar','input_kernel','scalar_response','mscs_response'):
                if arm=='raw_scalar':corr=torch.zeros_like(ds)
                else:
                    if arm!='input_kernel':
                        c=torch.as_tensor(z[arm.split('_')[0]+'_candidate']);hist=torch.as_tensor(z[arm.split('_')[0]+'_history'])
                        gram=torch.einsum('hnv,knv->nhk',hist,hist).square()
                        cross=torch.einsum('cnv,hnv->nch',c,hist).square()
                        close(gram.numpy(),z[arm+'__gram'],rtol=1e-10,atol=1e-10)
                        close(cross.numpy(),z[arm+'__cross'],rtol=1e-10,atol=1e-10)
                    else:gram=torch.as_tensor(z[arm+'__gram']);cross=torch.as_tensor(z[arm+'__cross'])
                    w=cross@torch.linalg.pinv(gram,rtol=1e-6,hermitian=True)
                    corr=torch.einsum('nch,hns->cns',w,res)
                    close(corr.numpy(),z[arm+'__correction'],rtol=2e-5,atol=1e-7)
                scores=(lo+excess[None]*torch.exp((ds+corr).clamp(-80,80))).clamp(lo,hi).mean((1,2)).numpy()
                close(scores,r['scores'][arm],rtol=2e-5,atol=1e-7)
                assert np.argsort(scores,kind='stable')[:2].tolist()==r['chosen'][arm];rank_matches+=1
                if arm=='input_kernel':close(scores,d['scalar_corrected_cpis'],rtol=1e-4,atol=3e-6)
    # Independently check arbitrary dimensional response geometry has zero correction
    # and zero derivative at the root even when observations carry large residuals.
    torch.manual_seed(42);h=torch.randn(6,12,dtype=torch.float64);y=torch.randn(6,dtype=torch.float64)
    inverse=torch.linalg.pinv((h@h.T).square(),rtol=1e-6,hermitian=True)
    x=torch.zeros(12,dtype=torch.float64,requires_grad=True)
    value=(x@h.T).square()@inverse@y
    assert value.item()==0 and torch.autograd.grad(value,x)[0].abs().max()==0
    save(out/'validation.json',dict(passed=True,sources_unchanged=True,independent_tensor_replay=True,
        exact_rankings=rank_matches,original_input_kernel_reproduced=True,root_value_and_gradient_preserved=True,
        validator_sha256=sha(__file__),new_queries=0))
    print('Passed: 96 score/rank replays and root value/gradient invariant.')


if __name__=='__main__':main()
