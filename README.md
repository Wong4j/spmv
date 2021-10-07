# SpMV
Test the SpMV performance of matrices in CSR sparse formats on GPU.

## Download sparse matrix
Download "Matrix Market (.mtx)" file from https://sparse.tamu.edu/. 
Then extract the .tar.gz file.

```shell
wget https://suitesparse-collection-website.herokuapp.com/MM/xxx.tar.gz

tar xvf xxx.tar.gz
```

See examples in directory "matrix_collection/download_matrices.sh"

## Compile test code
```shell
./compile.sh
```

## Run SpMV
```shell
./main.exe /path/to/xxx.mtx
```
output example:
```shell
shijie@Shijie-workstation:~/workspace/spmv$ ./main.exe matrix_collection/atmosmodd/atmosmodd.mtx 
nnz = 7
CSR scalr average time: 0.000330 second
CSR vector average time: 0.002919 second
CSR cusparse average time: 0.000277 second
correct result
```

## Profile SpMV
Change the variable "const int repeat = 1" in file "spmv_gpu.cuh" and recompile.
Then run:
```shell
./ncu_prof.sh xxx.mtx
```
Then open the ".ncu-rep" file using Nsight-compute

## Reference
Bell, Nathan, and Michael Garland. Efficient sparse matrix-vector multiplication on CUDA. Vol. 2. No. 5. Nvidia Technical Report NVR-2008-004, Nvidia Corporation, 2008.

CSR-formatter from https://github.com/notini/csr-formatter
