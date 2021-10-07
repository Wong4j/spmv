# spmv
Test the SpMV performance of matrices in CSR sparse formats on GPU

## Download sparse matrix
Download "Matrix Market (.mtx)" file from https://sparse.tamu.edu/
See the example in directory "matrix_collection/download_matrices.sh"

## Compile test code
```shell
./compile.sh
```

## Run SpMV
./main.exe xxx.mtx

## Profile SpMV
vim ncu_prof.sh
./ncu_prof.sh
