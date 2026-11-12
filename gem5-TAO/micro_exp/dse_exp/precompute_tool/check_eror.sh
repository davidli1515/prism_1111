# 对每个失败的bench，手动跑slice_0看gem5报什么错
for bench in 654.roms_s 657.xz_s 620.omnetpp_s; do
    echo "=== ${bench} ==="
    python -c "
import os, sys, shlex, subprocess
sys.path.insert(0, '/scratch/gem5-TAO/micro_exp/dse_exp')
from prepare_base_features import *

bench = '${bench}'
info = BENCHMARKS[bench]
ckpt_dir = os.path.join(CKPT_BASE, bench, info['ckpt_tick'])
slice_dir = '/tmp/test_${bench}'
os.makedirs(slice_dir, exist_ok=True)

argv = shlex.split(info['command'], posix=True)
i = 0
while i < len(argv):
    if argv[i] == '-I' and i+1 < len(argv):
        del argv[i:i+2]
    else:
        i += 1

argv.insert(1, '-d')
argv.insert(2, slice_dir)
p = DEFAULT_PARAMS
cpu = 'system.switch_cpus_1[0]'
argv += [
    f'--checkpoint-dir={ckpt_dir}', '-r', '1',
    '--standard-switch=1', '-W', '20000000', '-I', '100000',
    '-P', f\"{cpu}.fetchWidth={p['fetchWidth']}\",
]

print(' '.join(argv[:10]), '...')
result = subprocess.run(argv, cwd=info['cwd'], capture_output=True, text=True, timeout=300)
print(f'rc={result.returncode}')
print('STDERR:', result.stderr[-500:] if result.stderr else '(empty)')
print('STDOUT:', result.stdout[-500:] if result.stdout else '(empty)')
" 2>&1
    echo ""
done
