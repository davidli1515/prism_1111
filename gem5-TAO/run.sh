#!/bin/bash

#/scratch/santosh/gem5/build/X86/gem5.opt /scratch/santosh/gem5/configs/example/se.py --caches --bp-type=TournamentBP -c ../../build/build_base_mytest-m64.0000/lbm_s -o "2000 reference.dat 0 0 200_200_260_ldc.of"


#/scratch/santosh/gem5/build/X86/gem5.opt --debug-flags=Fetch,Exec,Decode --debug-file=time_no_trace.out /scratch/santosh/gem5/configs/example/se.py --cpu-type=TimingSimpleCPU -c ../../build/build_base_mytest-m64.0000/lbm_s -o "2000 reference.dat 0 0 200_200_260_ldc.of" > timing_no_b_c_debug.log


/scratch/santosh/gem5/build/X86/gem5.opt --debug-flags=O3PipeView --debug-file=x86_timetrace.out /scratch/santosh/gem5/configs/example/se.py --cpu-type=TimingSimpleCPU -c ../../build/build_base_mytest-m64.0000/lbm_s -o "2000 reference.out 0 0 200_200_260_ldc.of"


/scratch/santosh/gem5/build/X86/gem5.opt --debug-flags=O3PipeView --debug-file=x86_o3trace.out /scratch/santosh/gem5/configs/example/se.py --cpu-type=O3CPU --caches -c ../../build/build_base_mytest-m64.0000/lbm_s -o '2000 reference.out 0 0 200_200_260_ldc.of'

/scratch/santosh/gem5/build/ARM/gem5.opt --debug-flags=O3PipeView --debug-file=arm_timetrace.out /scratch/santosh/gem5/configs/example/se.py --cpu-type=TimingSimpleCPU -c ../../build/build_base_mytest-64.0000/lbm_s -o "2000 reference.out 0 0 200_200_260_ldc.of" -I 100000


/scratch/santosh/gem5/build/ARM/gem5.opt --debug-flags=O3PipeView --debug-file=arm_o3trace.out /scratch/santosh/gem5/configs/example/se.py --cpu-type=O3CPU --caches -c ../../build/build_base_mytest-64.0000/lbm_s -o '2000 reference.out 0 0 200_200_260_ldc.of' -I 100000

#/scratch/santosh/gem5/build/X86/gem5.opt --debug-flags=Fetch,Exec,Decode --debug-file=time_b_c_trace.out /scratch/santosh/gem5/configs/example/se.py --cpu-type=TimingSimpleCPU --caches --bp-type=TournamentBP -c ../../build/build_base_mytest-m64.0000/lbm_s -o "2000 reference.dat 0 0 200_200_260_ldc.of" > timing_b_c_debug.log

#/scratch/santosh/gem5/build/X86/gem5.opt --debug-flags=Fetch,Exec,Decode --debug-file=o3_b_c_trace.out /scratch/santosh/gem5/configs/example/se.py --cpu-type=O3CPU --caches --bp-type=TournamentBP -c ../../build/build_base_mytest-m64.0000/lbm_s -o "2000 reference.dat 0 0 200_200_260_ldc.of" > o3_b_c_debug.log

#/scratch/santosh/gem5/build/X86/gem5.opt --debug-flags=Fetch,Exec,Decode --debug-file=o3_no_.out /scratch/santosh/gem5/configs/example/se.py --cpu-type=O3CPU --caches -c ../../build/build_base_mytest-m64.0000/lbm_s -o "2000 reference.dat 0 0 200_200_260_ldc.of" > o3_no_b_debug.log


