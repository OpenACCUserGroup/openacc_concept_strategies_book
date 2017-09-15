#!/bin/sh
pgc++ --c++11 -pg ThermoTables_CPU.C -o out_CPU
./out_CPU 100000 1000 
gprof out_CPU gmon.out > profile.txt 
