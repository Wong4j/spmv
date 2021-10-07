#!/bin/bash
rm *.so *.o *.exe
nvcc testspmv.cu -O3 -lcusparse -o main.exe
