#!/bin/bash
# DSE Precompute — parallel run for both workloads
# Usage: bash run_precompute.sh [N_PARALLEL] [N_VERIFY]

set -e
TRACE_BASE="/scratch/gem5-TAO/micro_exp/dse_exp/traces"
OUT_BASE="/scratch/gem5-TAO/micro_exp/dse_exp/precompute"
TOOL="./dse_precompute"

N_PAR=${1:-16}; N_VER=${2:-5}; N_SLICES=400
WLS=("605.mcf_s" "623.xalancbmk_s")

[ -f "$TOOL" ] || { echo "Error: run 'make' first"; exit 1; }

for WL in "${WLS[@]}"; do
    TD="${TRACE_BASE}/${WL}"; OD="${OUT_BASE}/${WL}"
    [ -d "$TD" ] || { echo "SKIP: $TD"; continue; }
    echo "=== ${WL} ==="; mkdir -p "$OD"

    for ((s=0; s<N_SLICES; s+=N_PAR)); do
        e=$((s+N_PAR-1)); [ $e -ge $N_SLICES ] && e=$((N_SLICES-1))
        pids=()
        for ((i=s; i<=e; i++)); do
            [ -d "${TD}/slice_${i}" ] || continue
            $TOOL "$TD" "$OD" --verify $N_VER --slice-range $i $i \
                > "${OD}/slice_${i}.log" 2>&1 &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait $p; done
        echo "  slices ${s}-${e} done"
    done

    echo "  Results:"
    grep -h "verify:" ${OD}/slice_*.log 2>/dev/null | tail -3
    du -sh "$OD" 2>/dev/null
    echo ""
done
echo "=== All done ==="
