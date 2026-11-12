import numpy as np
old_dir = "/scratch2/wl730/micro_traindata_packed/gem"
new_dir = "/scratch2/wl730/micro_traindata_packed_all/gem"

old = np.load(f"{old_dir}/508.namd_r.npy", allow_pickle=True)
new = np.load(f"{new_dir}/508.namd_r.npy", allow_pickle=True)

# Check arch param distribution (3850:3873)
old_arch = old[:, 3850:3873]
new_arch = new[:, 3850:3873]

print("Arch param ranges:")
for i, name in enumerate(["fetchW","decodeW","renameW","dispatchW","issueW","wbW","commitW",
                           "ROB","IQ","LQ","SQ","fetchBuf","IntALU","IntMD","FPALU","FPMD",
                           "RdWr","L1sz","L2sz","L1dA","L1iA","L2A","BP"]):
    o_min, o_max = old_arch[:,i].min(), old_arch[:,i].max()
    n_min, n_max = new_arch[:,i].min(), new_arch[:,i].max()
    o_uniq = len(np.unique(old_arch[:,i]))
    n_uniq = len(np.unique(new_arch[:,i]))
    if o_uniq != n_uniq or o_min != n_min or o_max != n_max:
        print(f"  {name:>10}: old=[{o_min:.0f},{o_max:.0f}] {o_uniq}vals  new=[{n_min:.0f},{n_max:.0f}] {n_uniq}vals  ← DIFFERENT")
    else:
        print(f"  {name:>10}: [{o_min:.0f},{o_max:.0f}] {o_uniq}vals  (same)")

# CPI distribution
old_mscs = old[:, 4681:4697].reshape(-1,4,4)
new_mscs = new[:, 4681:4697].reshape(-1,4,4)
old_wmin = np.minimum.reduce([old[:,3850],old[:,3851],old[:,3852],old[:,3853],old[:,3854],old[:,3856]])
new_wmin = np.minimum.reduce([new[:,3850],new[:,3851],new[:,3852],new[:,3853],new[:,3854],new[:,3856]])
old_cpi = 1.0/(old_mscs[:,3,0]*old_wmin+1e-8)
new_cpi = 1.0/(new_mscs[:,3,0]*new_wmin+1e-8)
print(f"\nCPI: old mean={old_cpi.mean():.2f} std={old_cpi.std():.2f} | new mean={new_cpi.mean():.2f} std={new_cpi.std():.2f}")
