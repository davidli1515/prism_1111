# 详细检查每个L1 × benchmark的完成情况
echo "L1_KB | Benchmark | Slices | Status"
echo "------|-----------|--------|-------"
for l1 in 32 64 128 256; do
    for bench in $(ls /scratch/gem5-TAO/micro_exp/dse_exp/precompute/l1_${l1}/ 2>/dev/null); do
        dir="/scratch/gem5-TAO/micro_exp/dse_exp/precompute/l1_${l1}/${bench}"
        n=$(find "$dir" -name "dse_precomputed.npy" 2>/dev/null | wc -l)
        status="OK"
        [ "$n" -lt 100 ] && status="LOW"
        [ "$n" -eq 0 ] && status="MISSING"
        echo "${l1} | ${bench} | ${n} | ${status}"
    done
    # 检查哪些benchmark完全不存在
    for bench in $(ls /scratch/gem5-TAO/micro_exp/dse_exp/traces/ 2>/dev/null); do
        dir="/scratch/gem5-TAO/micro_exp/dse_exp/precompute/l1_${l1}/${bench}"
        [ -d "$dir" ] || echo "${l1} | ${bench} | 0 | NOT_FOUND"
    done
done

# 也检查run_multi_l1.log里有没有错误
echo ""
echo "=== Errors in log ==="
grep -i "error\|fail\|skip\|WARNING" /scratch/gem5-TAO/micro_exp/dse_exp/run_multi_l1.log 2>/dev/null | tail -20
