cd /scratch/gem5-TAO/micro_exp/exp2
python -c "
import os, sys, numpy as np
sys.path.insert(0, '/scratch/gem5-TAO/micro_exp/exp1')
from config import TRAIN_RATIO, D3_FEAT_END, CONCORDE_FEAT_END, SEED
from dataset import FeatureNormalizer

gem_dir = '/scratch2/wl730/micro_traindata_packed_all/gem'
found_dir = '/scratch/gem5-TAO/micro_exp/exp2/foundations'
files = sorted([f[:-4] for f in os.listdir(gem_dir) if f.endswith('.npy')])

for target in files:
    rng = np.random.RandomState(SEED)
    gem_feats, con_feats = [], []
    for name in files:
        if name == target: 
            rng.permutation(np.load(os.path.join(gem_dir, name+'.npy')).shape[0])
            continue
        raw = np.load(os.path.join(gem_dir, name+'.npy')).astype(np.float32)
        n = raw.shape[0]
        idx = rng.permutation(n)
        split = int(n * TRAIN_RATIO)
        gem_feats.append(raw[idx[:split], :D3_FEAT_END])
        con_feats.append(raw[idx[:split], :CONCORDE_FEAT_END])
    
    gn = FeatureNormalizer(); gn.fit(np.concatenate(gem_feats))
    cn = FeatureNormalizer(); cn.fit(np.concatenate(con_feats))
    
    bm_dir = os.path.join(found_dir, target)
    np.savez(os.path.join(bm_dir, 'gem_norm.npz'), mean=gn.mean, std=gn.std)
    np.savez(os.path.join(bm_dir, 'con_norm.npz'), mean=cn.mean, std=cn.std)
    print(f'  {target}: saved')
print('Done')
"
