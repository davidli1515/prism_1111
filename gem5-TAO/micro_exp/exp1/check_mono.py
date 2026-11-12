import numpy as np, os
gem_dir = "/scratch2/wl730/micro_traindata_packed/gem"
total, violations = 0, 0
for f in sorted(os.listdir(gem_dir)):
    gem = np.load(os.path.join(gem_dir, f))
    mscs = gem[:, 4681:4697].reshape(-1, 4, 4)  # (N, 4stages, 4states)
    fp = mscs[:, :, 1]  # FP across stages
    bp = mscs[:, :, 2]  # BP across stages
    sp = mscs[:, :, 3]  # SP across stages
    # Check violations
    fp_viol = (fp[:, 1:] > fp[:, :-1] + 1e-6).any(axis=1).sum()
    bp_viol = (bp[:, :-1] > bp[:, 1:] + 1e-6).any(axis=1).sum()
    sp_viol = (sp[:, 1:] > sp[:, :-1] + 1e-6).any(axis=1).sum()
    v = fp_viol + bp_viol + sp_viol
    total += gem.shape[0]
    violations += v
    if v > 0:
        print(f"{f[:-4]}: {v}/{gem.shape[0]} samples violate monotonicity")
print(f"\nTotal: {violations}/{total} violations ({violations/total*100:.1f}%)")
