"""Independent NumPy checks of recorded all-paid-center interpolation."""
import numpy as np


def validate_memory(out,bench,d):
    with np.load(d['prediction_file']) as all_arrays:
        z={k[len('memory__'):]:all_arrays[k] for k in all_arrays.files if k.startswith('memory__')}
    ds=z['scalar__distance_squared'];assert np.array_equal(ds,z['mscs__distance_squared']) and (ds>=0).all()
    zeros=ds==0;clamped=np.maximum(ds,np.finfo(ds.dtype).tiny)
    relative=clamped.min(0,keepdims=True)/clamped
    weights=np.where(zeros.any(0,keepdims=True),zeros/np.maximum(zeros.sum(0,keepdims=True),1),relative/relative.sum(0,keepdims=True))
    for kind in ('scalar','mscs'):
        np.testing.assert_allclose(z[kind+'__weights'],weights,rtol=1e-13,atol=1e-15)
        yy=z[kind+'__center_response'].astype(float)
        pp=(z[kind+'__center_response'][:,None]+z[kind+'__center_slope']).astype(float)
        expanded=weights.reshape(*weights.shape,*([1]*(pp.ndim-weights.ndim)))
        pred=(expanded*pp).sum(0).astype(np.float32)
        np.testing.assert_allclose(pred,z[kind+'__blended_response'],rtol=2e-6,atol=2e-6)
        assert len(yy)==len(d['history_ids'])
        for n,lid in enumerate(d['history_ids']):
            with np.load(out/'truth'/bench/f'{lid}.npz') as truth:mm=truth['m'].astype(float)
            target=np.log1p(-mm[...,0])-np.log(mm[...,0]) if kind=='scalar' else np.log(mm[...,1:])-np.log(mm[...,:1])
            np.testing.assert_allclose(yy[n],target,rtol=4e-6,atol=3e-6)
        if kind=='scalar':base=1/(1+np.exp(pred.astype(float)))
        else:
            full=np.concatenate([np.zeros_like(pred[...,:1]),pred],-1).astype(float)
            ex=np.exp(full-full.max(-1,keepdims=True));mm=ex/ex.sum(-1,keepdims=True);base=mm[...,0]
            np.testing.assert_allclose(mm,z['predicted_m'],rtol=4e-6,atol=3e-7)
        stages=np.minimum(100.,1/(np.maximum(base,.01)*np.array(d['widths'])[:,None,None]+1e-8))
        np.testing.assert_allclose(stages,z[kind+'__stage'],rtol=4e-6,atol=3e-6)
        np.testing.assert_allclose(stages.mean((1,2)),d[kind+'_scores'],rtol=4e-6,atol=3e-6)
