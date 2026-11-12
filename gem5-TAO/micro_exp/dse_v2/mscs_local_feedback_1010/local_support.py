"""Feedback is local to a resource and its increase/decrease direction."""
import copy
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ADAPTIVE=ROOT/'mscs_adaptive_greedy_1010'
HYBRID=ROOT/'mscs_hybrid_greedy_1010'
NEWROOTS=ROOT/'mscs_adaptive_newroots_1010'
sys.path.insert(0,str(ADAPTIVE))
from adaptive_support import (HybridLocalModel,rank_feedback,PARAMS,STEPS,TOPK,
    DEFAULT_CKPT,config_key,compute_w_min,legal,moves,Inputs,FeatureCache,
    load_model,exact,cpis,contributions,probabilities)

ARMS=('local',)
COMPARISON_ARMS=('scalar_excess','hybrid','adaptive','local')
REGRESSIONS=('602.gcc_s_middle','602.gcc_s_newrank9','605.mcf_s_middle')


class LocalModel(HybridLocalModel):
    def __init__(self,model,x,width,arm):
        assert arm=='local'
        super().__init__(model,x,width,'hybrid')


def bucket(action):
    assert action['parameter'] in PARAMS and action['sign'] in (-1,1)
    return f"{action['parameter']}:{action['sign']:+d}"


class LocalAllocator:
    def __init__(self):
        self.state=dict(completed_rounds=0,buckets={})

    def snapshot(self):
        return copy.deepcopy(self.state)

    def choose(self,actions,scalar,mscs):
        s,m=np.asarray(scalar,dtype=float),np.asarray(mscs,dtype=float)
        assert s.ndim==1 and s.shape==m.shape and len(s)==len(actions)
        assert np.isfinite(s).all() and np.isfinite(m).all()
        use=[]
        for action in actions:
            losses=self.state['buckets'].get(bucket(action),{}).get('loss2',{'scalar':0,'mscs':0})
            use.append(losses['mscs']<losses['scalar'])
        corrected=np.where(use,m,s)
        if not actions:
            return [],[],dict(anchor_scores=[],anchor_models=[])
        first=int(np.argmin(corrected))
        second=[j for j in np.argsort(m,kind='stable').tolist() if j!=first][:1]
        sources=['local_mscs_anchor' if use[first] else 'local_scalar_anchor']+['mscs_alternative']*len(second)
        return [first]+second,sources,dict(anchor_scores=corrected.tolist(),
            anchor_models=['mscs' if yes else 'scalar' for yes in use])

    def observe(self,root_cpi,actions,scalar,mscs,actual):
        assert len(actions)==len(scalar)==len(mscs)==len(actual)
        groups={}
        for i,action in enumerate(actions):
            groups.setdefault(bucket(action),[]).append(i)
        feedback={}
        for name,ids in groups.items():
            # No pair between different resources or opposite directions earns
            # credit in either bucket. Root-to-action evidence is always local.
            r=rank_feedback(root_cpi,*[[values[i] for i in ids] for values in (scalar,mscs,actual)])
            feedback[name]=dict(trial_indices=ids,feedback=r)
            state=self.state['buckets'].setdefault(name,dict(loss2={'scalar':0,'mscs':0},
                comparisons=0,discriminating_comparisons=0,observations=0))
            for model in ('scalar','mscs'):
                state['loss2'][model]+=r['loss2'][model]
            state['comparisons']+=len(r['comparisons'])
            state['discriminating_comparisons']+=r['discriminating_comparisons']
            state['observations']+=len(ids)
        self.state['completed_rounds']+=1
        return feedback
