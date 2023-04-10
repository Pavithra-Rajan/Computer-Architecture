#!/usr/bin/bash 
export GEM5_DIR=/home/pavithra/gem5
export BENCHMARK=429.mcf/src/benchmark
export ARGUMENT=429.mcf/data/inp.in
for L1D_SIZE in 8kB 16kB 32kB
do
  for L2_SIZE in 2MB 4MB 8MB
  do
    time $GEM5_DIR/build/X86/gem5.opt -d ~/gem5/m5out/${L1D_SIZE}_${L2_SIZE} $GEM5_DIR/configs/example/se.py -c $BENCHMARK -o $ARGUMENT -I 21000000000 --cpu-type=TimingSimpleCPU --caches --l2cache --l1d_size=$L1D_SIZE --l1i_size=$L1D_SIZE --l2_size=$L2_SIZE --l1d_assoc=2 --l1i_assoc=2 --l2_assoc=1 --cacheline_size=64
  done
done 