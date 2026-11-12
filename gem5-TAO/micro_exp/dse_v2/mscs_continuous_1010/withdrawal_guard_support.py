"""History can make a withdrawal less attractive, never more attractive."""
import numpy as np


def guard_scores(root,candidates,raw,memory):
    withdrawal=np.array([any(a['config'][k]<root[k] for k in root) for a in candidates],dtype=bool)
    raw=np.asarray(raw,dtype=float);memory=np.asarray(memory,dtype=float)
    return np.where(withdrawal,np.maximum(raw,memory),raw),withdrawal
