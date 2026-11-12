from collections import Counter

# 这里可以替换成从文件读取，比如: 
# with open('log.txt', 'r') as f:
#     lines = f.readlines()

text = """  [OK] 641.leela_s          ckpt=  1510100000000  gem5= 37.8s  conc= 11.8s
  [OK] 600.perlbench_s      ckpt=   240100000000  gem5= 39.6s  conc= 12.9s
  [OK] 631.deepsjeng_s      ckpt=  1384000000000  gem5= 49.6s  conc= 11.7s
  [OK] 631.deepsjeng_s      ckpt=   804200000000  gem5= 49.2s  conc= 13.2s
  [OK] 644.nab_s            ckpt=   217200000000  gem5= 44.3s  conc= 19.1s
  [OK] 641.leela_s          ckpt=  1120100000000  gem5= 51.4s  conc= 14.3s
  [OK] 641.leela_s          ckpt=  1429200000000  gem5= 50.4s  conc= 16.7s
  [OK] 623.xalancbmk_s      ckpt=   537100000000  gem5= 51.4s  conc= 21.5s
  [OK] 631.deepsjeng_s      ckpt=  1154700000000  gem5= 57.7s  conc= 17.5s
  [OK] 638.imagick_s        ckpt=    41800000000  gem5= 58.6s  conc= 17.2s
  [OK] 620.omnetpp_s        ckpt=    85300000000  gem5= 61.7s  conc= 14.8s
  [OK] 623.xalancbmk_s      ckpt=   885500000000  gem5= 61.0s  conc= 15.9s
  [OK] 619.lbm_s            ckpt=  2151900000000  gem5= 63.6s  conc= 14.3s
  [OK] 605.mcf_s            ckpt=   917300000000  gem5= 66.9s  conc= 11.1s
  [OK] 619.lbm_s            ckpt=  1900300000000  gem5= 65.1s  conc= 13.0s
  [OK] 623.xalancbmk_s      ckpt=   377600000000  gem5= 61.2s  conc= 17.4s
  [OK] 638.imagick_s        ckpt=   561400000000  gem5= 72.4s  conc= 13.7s
  [OK] 602.gcc_s            ckpt=   430300000000  gem5= 67.7s  conc= 18.6s
  [OK] 638.imagick_s        ckpt=   653700000000  gem5= 68.7s  conc= 18.5s
  [OK] 602.gcc_s            ckpt=   532900000000  gem5= 82.2s  conc= 12.1s
  [OK] 648.exchange2_s      ckpt=   444700000000  gem5= 39.5s  conc= 13.1s
  [OK] 508.namd_r           ckpt=   696800000000  gem5= 40.1s  conc= 11.9s
  [OK] 607.cactuBSSN_s      ckpt=    30600000000  gem5= 46.9s  conc= 12.3s
  [OK] 508.namd_r           ckpt=   420400000000  gem5= 43.9s  conc= 11.8s
  [OK] 621.wrf_s            ckpt=   162800000000  gem5= 48.1s  conc= 11.7s
  [OK] 526.blender_r        ckpt=   580900000000  gem5= 40.0s  conc= 11.9s
  [OK] 508.namd_r           ckpt=  1200100000000  gem5= 43.7s  conc= 11.6s
  [OK] 526.blender_r        ckpt=    43600000000  gem5= 42.7s  conc= 14.1s
  [OK] 526.blender_r        ckpt=   958700000000  gem5= 41.6s  conc= 14.2s
  [OK] 607.cactuBSSN_s      ckpt=   295000000000  gem5= 72.5s  conc= 11.6s
  [OK] 654.roms_s           ckpt=   306300000000  gem5= 70.6s  conc= 11.8s
  [OK] 657.xz_s             ckpt=   108200000000  gem5=130.5s  conc= 10.4s"""

benchmarks = []
for line in text.strip().split('\n'):
    if "[OK]" in line:
        parts = line.split()
        if len(parts) >= 2:
            benchmarks.append(parts[1])

# 统计各个 benchmark 出现的次数
counts = Counter(benchmarks)

# 打印结果
for bench, count in counts.most_common():
    print(f"{bench}: {count} 个")
