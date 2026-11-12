#!/bin/bash

echo "🚀 Starting parallel Checkpoint generation for mcf_s with precise timing..."
START_TIME=$(date +%s)

# 任务 1: Interval 3975
mkdir -p /scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/ckpt_3975
nohup build/ARM/gem5.opt -d /scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/m5out_3975 /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=AtomicSimpleCPU -c "/home/wl730/spec17/benchspec/CPU/605.mcf_s/exe/mcf_s_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in" --checkpoint-dir=/scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/ckpt_3975 --take-checkpoints=397500000000 --at-instruction --max-checkpoints=1 &> mcf_ckp_3975.txt &

# 任务 2: Interval 7785
mkdir -p /scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/ckpt_7785
nohup build/ARM/gem5.opt -d /scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/m5out_7785 /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=AtomicSimpleCPU -c "/home/wl730/spec17/benchspec/CPU/605.mcf_s/exe/mcf_s_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in" --checkpoint-dir=/scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/ckpt_7785 --take-checkpoints=778500000000 --at-instruction --max-checkpoints=1 &> mcf_ckp_7785.txt &

# 任务 3: Interval 9173
mkdir -p /scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/ckpt_9173
nohup build/ARM/gem5.opt -d /scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/m5out_9173 /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=AtomicSimpleCPU -c "/home/wl730/spec17/benchspec/CPU/605.mcf_s/exe/mcf_s_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in" --checkpoint-dir=/scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/ckpt_9173 --take-checkpoints=917300000000 --at-instruction --max-checkpoints=1 &> mcf_ckp_9173.txt &

# 任务 4: Interval 14749
mkdir -p /scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/ckpt_14749
nohup build/ARM/gem5.opt -d /scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/m5out_14749 /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=AtomicSimpleCPU -c "/home/wl730/spec17/benchspec/CPU/605.mcf_s/exe/mcf_s_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in" --checkpoint-dir=/scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/ckpt_14749 --take-checkpoints=1474900000000 --at-instruction --max-checkpoints=1 &> mcf_ckp_14749.txt &

# 任务 5: Interval 15224
mkdir -p /scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/ckpt_15224
nohup build/ARM/gem5.opt -d /scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/m5out_15224 /scratch/gem5-TAO/configs/example/se_new.py --cpu-type=AtomicSimpleCPU -c "/home/wl730/spec17/benchspec/CPU/605.mcf_s/exe/mcf_s_base.mytest-64" -o "/home/wl730/spec17/benchspec/CPU/605.mcf_s/run/run_base_refspeed_mytest-64.0000/inp.in" --checkpoint-dir=/scratch2/wl730/checkpoint_spec17_train/mcf_checkpoints/ckpt_15224 --take-checkpoints=1522400000000 --at-instruction --max-checkpoints=1 &> mcf_ckp_15224.txt &

echo "✅ All 5 Checkpoint jobs submitted! Waiting for them to finish..."

# wait 命令会阻塞脚本，直到上面所有挂载在后台 (&) 的任务全部运行结束
wait

END_TIME=$(date +%s)
ELAPSED=$(($END_TIME - $START_TIME))

HOURS=$(($ELAPSED / 3600))
MINS=$((($ELAPSED % 3600) / 60))
SECS=$(($ELAPSED % 60))

echo "🎉 All Checkpoints generated successfully!"
echo "⏱️ Total Time Elapsed: ${HOURS} hours, ${MINS} minutes, ${SECS} seconds."