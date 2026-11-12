"""Keep scalar ordering; demote only predicted-harmful resource withdrawals."""
import numpy as np

ARMS=('scalar_raw','scalar_memory','scalar_veto','mscs_veto')


def ranking(root,root_cpi,candidates,raw_scalar,memory_scalar,raw_mscs,arm):
    withdrawal=np.array([any(a['config'][k]<root[k] for k in root) for a in candidates],dtype=bool)
    blocked=np.zeros(len(candidates),dtype=bool)
    scores=np.asarray(memory_scalar if arm=='scalar_memory' else raw_scalar)
    if arm.endswith('_veto'):
        check=np.asarray(memory_scalar if arm=='scalar_veto' else raw_mscs)
        blocked=withdrawal & (check>=root_cpi*(1-1e-6))
    assert arm in ARMS
    order=sorted(range(len(candidates)),key=lambda i:(bool(blocked[i]),float(scores[i]),i))
    return order,blocked.tolist(),withdrawal.tolist()
