#!/bin/bash
rm *.so *.o *.exe
#hipcc -c spmv_gpu.cpp -o spmv_gpu.o
#hipcc -shared spmv_gpu.o -o libtest.so
#g++ -c testspmv.cpp -L/public/home/ictapp/hx_group/wsj/test/spmv -ltest -o testspmv.exe

nvcc testspmv.cu -g -G -O3 -lcusparse -o main.exe
