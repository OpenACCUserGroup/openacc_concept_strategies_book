#!/bin/sh
echo "building CPU"
pgc++ --c++11 ThermoTables_OpenACC.C -o out_CPU
echo "building CPU+GPU"
pgc++ --c++11 -acc ThermoTables_OpenACC.C -o out_GPU
echo "building CPU multicore"
pgc++ --c++11 -acc -ta=multicore ThermoTables_OpenACC.C -o out_CPU_Multicore
echo "building CPU+GPU, optimized data locality"
pgc++ --c++11 -acc ThermoTables_OpenACC_Optimized.C -o out_GPU_Optimized

export ACC_NUM_CORES=6
nbIter=1000
for nbData in 100 1000 10000 100000 1000000
do
  echo ""
  echo "tests with nbData, nbIter = " $nbData $nbIter
  echo ""
  echo "running CPU, 1 Core"
  time ./out_CPU $nbData $nbIter
  echo ""
  echo "running CPU, multicore"
  time ./out_CPU_Multicore $nbData $nbIter
  echo ""
  echo "running CPU+GPU"
  time ./out_GPU $nbData $nbIter
  echo ""
  echo "running CPU+GPU, optimized data locality"
  time ./out_GPU_Optimized $nbData $nbIter
done
