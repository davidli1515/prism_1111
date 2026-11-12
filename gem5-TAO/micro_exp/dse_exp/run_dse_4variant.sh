cd /scratch/gem5-TAO/micro_exp/dse_exp

# 1. GEM Attribution (MSCS dominant + reclaim) — ours
python run_gem_dse.py --benchmark 605.mcf_s --methods gem_attribution --n-starts 50 --max-steps 30 --reclaim-threshold 0.05 --seed 42

# 2. GEM CPI gradient + reclaim
python run_gem_dse.py --benchmark 605.mcf_s --methods gem_scalar --n-starts 50 --max-steps 30 --seed 42

# 3. GEM CPI gradient (no reclaim)
python run_gem_dse.py --benchmark 605.mcf_s --methods gem_scalar_noreclaim --n-starts 50 --max-steps 30 --seed 42

# 4. GEM BO
python run_gem_dse.py --benchmark 605.mcf_s --methods gem_bo --bo-budget 500 --seed 42

# Plot
python plot_dse_results.py --benchmark 605.mcf_s