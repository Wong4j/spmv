#include <stdio.h>
#include <stdlib.h>
//CUDA RunTime API
//#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <cusparse.h>

static const int repeat = 1;
static const int thread_per_block = 256;
static const int ncol_per_row = 16;

#define CUDA_CALL(func) do{\
    cudaError_t cuda_ret = (func);\
    if(cuda_ret != cudaSuccess){\
        printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
        printf("Error Message: %s\n", cudaGetErrorString(cuda_ret));\
        printf("Error Function: %s\n", #func);\
        exit(1);\
    }\
}while(0)

#define CUDA_KERNEL_CALL(...) do{\
    if(cudaPeekAtLastError() != cudaSuccess){\
        printf("CUDA Error occured before the kernel call %s at line %d\n", #__VA_ARGS__, __LINE__);\
        exit(1);\
    }\
    __VA_ARGS__;\
    cudaError_t cuda_ret = cudaPeekAtLastError();\
    if(cuda_ret != cudaSuccess){\
        printf("CUDA Kernel Error at line %d in file %s\n", __LINE__, __FILE__);\
        printf("Error Message: %s\n", cudaGetErrorString(cuda_ret));\
        printf("Error Kernel: %s\n", #__VA_ARGS__);\
        exit(1);\
    }\
}while(0)

inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

__global__ void matMultCSR_scalar(const int nrow,
								  const int *ptr,
								  const int *indices,
								  const double *data,
								  const double *x,
								  		double *y)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;

	if(row < nrow)
	{
		double dot = 0;
		int row_start = ptr[row];
		int row_end = ptr[row+1];

		for(int jj = row_start; jj < row_end; jj++)
		{
			dot += data[jj] * x[indices[jj]];
		}
		y[row] = dot;
	}
}

__global__ void matMultCSR_vector(const int nrow,
								  const int *ptr,
								  const int *indices,
								  const double *data,
								  const double *x,
								  		double *y)
{
	__shared__ double vals[thread_per_block*2];
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int warp_id = thread_id / 32;
	int lane = thread_id & 31;

	int row = warp_id;

	if(row < nrow)
	{
		int row_start = ptr[row];
		int row_end = ptr[row+1];

		vals[threadIdx.x] = 0;
		for(int jj = row_start + lane; jj < row_end; jj += 32)
		{
			vals[threadIdx.x] += data[jj] * x[indices[jj]];
		}
		if(lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16];
		if(lane < 8)  vals[threadIdx.x] += vals[threadIdx.x + 8];
		if(lane < 4)  vals[threadIdx.x] += vals[threadIdx.x + 4];
		if(lane < 2)  vals[threadIdx.x] += vals[threadIdx.x + 2];
		if(lane < 1)  vals[threadIdx.x] += vals[threadIdx.x + 1];

		if(lane == 0)
			y[row] = vals[threadIdx.x];
	}
}


__global__ void matMultELL(const int nrow, 
						   const int ncol_per_row, 
						   const int *indices, 			
						   const double *data, 
						   const double *x, 
						   		 double *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < nrow)
    {
        double dot = 0;
        for(int i=0; i < ncol_per_row; i++)
        {
			int tmp = nrow * i + row;
            int col = indices[tmp];
			double val = data[tmp];
            dot += val * x[col];
        }
        y[row] = dot;
    }
}

void spmv_csr_scalar(const int nrow,
			  		 const int *ptr,
			  		 const int *indices,
			  		 const double *data,
			  		 const double *x,
			  		  	   double *y
			  		)
{
	int nz = ptr[nrow];
	int* d_ptr;
	int* d_indices;
	double* d_data;
	double* d_x;
	double* d_y;

//	struct timeval begin, end;

	CUDA_CALL(cudaMalloc((void**) &d_ptr, 	  sizeof(int) * (nrow+1)));
	CUDA_CALL(cudaMalloc((void**) &d_indices, sizeof(int) * nz));
	CUDA_CALL(cudaMalloc((void**) &d_data, 	  sizeof(double) * nz));
	CUDA_CALL(cudaMalloc((void**) &d_x, 	  sizeof(double) * nrow));
	CUDA_CALL(cudaMalloc((void**) &d_y, 	  sizeof(double) * nrow));

	CUDA_CALL(cudaMemcpy(d_ptr, 	ptr, 	 sizeof(int) * (nrow+1), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_indices, indices, sizeof(int) * nz, 	     cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_data, 	data, 	 sizeof(double) * nz,    cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_x, 		x, 		 sizeof(double) * nrow,  cudaMemcpyHostToDevice));

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

	dim3 blockDim(thread_per_block);
	dim3 gridDim(ceil_div(nrow, thread_per_block));
	//warm up
	for(int i=0; i < repeat; i++)
		CUDA_KERNEL_CALL(matMultCSR_scalar<<<gridDim, blockDim>>>(nrow, d_ptr, d_indices, d_data, d_x, d_y));
    CUDA_CALL(cudaDeviceSynchronize());
	
    cudaEventRecord(beg);
	for(int i=0; i < repeat; i++)
		CUDA_KERNEL_CALL(matMultCSR_scalar<<<gridDim, blockDim>>>(nrow, d_ptr, d_indices, d_data, d_x, d_y));
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEventRecord(end);

    CUDA_CALL(cudaEventSynchronize(beg));
    CUDA_CALL(cudaEventSynchronize(end));
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.0;
    printf("CSR scalr average time: %f second\n", elapsed_time / repeat);
	CUDA_CALL(cudaMemcpy(y, d_y, sizeof(double) * nrow, cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaEventDestroy(beg));
	CUDA_CALL(cudaEventDestroy(end));

	CUDA_CALL(cudaFree(d_ptr));
	CUDA_CALL(cudaFree(d_indices));
	CUDA_CALL(cudaFree(d_data));
	CUDA_CALL(cudaFree(d_x));
	CUDA_CALL(cudaFree(d_y));

}

void spmv_csr_vector(const int nrow,
			  		  const int *ptr,
			  		  const int *indices,
			  		  const double *data,
			  		  const double *x,
			  		  double *y
			  		  )
{
	int nz = ptr[nrow];
	int* d_ptr;
	int* d_indices;
	double* d_data;
	double* d_x;
	double* d_y;

//	struct timeval begin, end;

	CUDA_CALL(cudaMalloc((void**) &d_ptr, 	  sizeof(int) * (nrow+1)));
	CUDA_CALL(cudaMalloc((void**) &d_indices, sizeof(int) * nz));
	CUDA_CALL(cudaMalloc((void**) &d_data, 	  sizeof(double) * nz));
	CUDA_CALL(cudaMalloc((void**) &d_x, 	  sizeof(double) * nrow));
	CUDA_CALL(cudaMalloc((void**) &d_y, 	  sizeof(double) * nrow));

	CUDA_CALL(cudaMemcpy(d_ptr, 	ptr, 	 sizeof(int) * (nrow+1), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_indices, indices, sizeof(int) * nz, 	     cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_data, 	data, 	 sizeof(double) * nz,    cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_x, 		x, 		 sizeof(double) * nrow,  cudaMemcpyHostToDevice));

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

	dim3 blockDim2(thread_per_block);
	dim3 gridDim2(ceil_div(nrow * 32, thread_per_block));

	//warm up
	for(int i=0; i < repeat; i++)
		CUDA_KERNEL_CALL(matMultCSR_vector<<<gridDim2, blockDim2>>>(nrow, d_ptr, d_indices, d_data, d_x, d_y));
    CUDA_CALL(cudaDeviceSynchronize());
	
    cudaEventRecord(beg);
	for(int i=0; i < repeat; i++)
		CUDA_KERNEL_CALL(matMultCSR_vector<<<gridDim2, blockDim2>>>(nrow, d_ptr, d_indices, d_data, d_x, d_y));
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEventRecord(end);

    CUDA_CALL(cudaEventSynchronize(beg));
    CUDA_CALL(cudaEventSynchronize(end));
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.0;
    printf("CSR vector average time: %f second\n", elapsed_time / repeat);

	CUDA_CALL(cudaEventDestroy(beg));
	CUDA_CALL(cudaEventDestroy(end));
	CUDA_CALL(cudaMemcpy(y, d_y, sizeof(double) * nrow, cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(d_ptr));
	CUDA_CALL(cudaFree(d_indices));
	CUDA_CALL(cudaFree(d_data));
	CUDA_CALL(cudaFree(d_x));
	CUDA_CALL(cudaFree(d_y));
}

void spmv_csr_cusparse(const int nrow,
			  		   const int *ptr,
			  		   const int *indices,
			  		   const double *data,
			  		   const double *x,
			  		   double *y
			  		  )
{
	int nz = ptr[nrow];
	int* d_ptr;
	int* d_indices;
	double* d_data;
	double* d_x;
	double* d_y;

	CUDA_CALL(cudaMalloc((void**) &d_ptr, 	  sizeof(int) * (nrow+1)));
	CUDA_CALL(cudaMalloc((void**) &d_indices, sizeof(int) * nz));
	CUDA_CALL(cudaMalloc((void**) &d_data, 	  sizeof(double) * nz));
	CUDA_CALL(cudaMalloc((void**) &d_x, 	  sizeof(double) * nrow));
	CUDA_CALL(cudaMalloc((void**) &d_y, 	  sizeof(double) * nrow));

	CUDA_CALL(cudaMemcpy(d_ptr, 	ptr, 	 sizeof(int) * (nrow+1), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_indices, indices, sizeof(int) * nz, 	     cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_data, 	data, 	 sizeof(double) * nz,    cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_x, 		x, 		 sizeof(double) * nrow,  cudaMemcpyHostToDevice));

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

	double alpha = 1.0;
	double beta = 0.0;

	
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, nrow, nrow, nz,
                                      d_ptr, d_indices, d_data,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    // Create dense vector X
    cusparseCreateDnVec(&vecX, nrow, d_x, CUDA_R_64F);
    // Create dense vector y
    cusparseCreateDnVec(&vecY, nrow, d_y, CUDA_R_64F);
    // allocate an external buffer if needed
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_CSR_ALG1, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute SpMV
    cudaEventRecord(beg);
	for(int i=0; i < repeat; i++)
	    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
	                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
	                 CUSPARSE_SPMV_CSR_ALG1, dBuffer);
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEventRecord(end);

    CUDA_CALL(cudaEventSynchronize(beg));
    CUDA_CALL(cudaEventSynchronize(end));

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.0;
    printf("CSR cusparse average time: %f second\n", elapsed_time / repeat);

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
	CUDA_CALL(cudaEventDestroy(beg));
	CUDA_CALL(cudaEventDestroy(end));
    //--------------------------------------------------------------------------
    // device result check
    cudaMemcpy(y, d_y, nrow * sizeof(double), cudaMemcpyDeviceToHost);

	CUDA_CALL(cudaFree(d_ptr));
	CUDA_CALL(cudaFree(d_indices));
	CUDA_CALL(cudaFree(d_data));
	CUDA_CALL(cudaFree(d_x));
	CUDA_CALL(cudaFree(d_y));

}

void spmv_ell(const int nrow,
			  const int ncol_per_row,
			  const int *indices,
			  const double *data,
	  		  const double *x,
		    		double *y
			  )
{
	int size = nrow * ncol_per_row;
	int* d_indices;
	double* d_data;
	double* d_x;
	double* d_y;

//	struct timeval begin, end;

	CUDA_CALL(cudaMalloc((void**) &d_indices, sizeof(int) * size));
	CUDA_CALL(cudaMalloc((void**) &d_data, 	  sizeof(double) * size));
	CUDA_CALL(cudaMalloc((void**) &d_x, 	  sizeof(double) * nrow));
	CUDA_CALL(cudaMalloc((void**) &d_y, 	  sizeof(double) * nrow));

	CUDA_CALL(cudaMemcpy(d_indices, indices, sizeof(int) * size,    cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_data, 	data, 	 sizeof(double) * size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_x, 		x, 		 sizeof(double) * nrow, cudaMemcpyHostToDevice));

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

	dim3 blockDim(thread_per_block);
	dim3 gridDim(ceil_div(nrow, thread_per_block));

	//warm up
	for(int i=0; i < repeat; i++)
		CUDA_KERNEL_CALL(matMultELL<<<gridDim, blockDim>>>(nrow, ncol_per_row, d_indices, d_data, d_x, d_y));
    CUDA_CALL(cudaDeviceSynchronize());
	
    cudaEventRecord(beg);
	for(int i=0; i < repeat; i++)
		CUDA_KERNEL_CALL(matMultELL<<<gridDim, blockDim>>>(nrow, ncol_per_row, d_indices, d_data, d_x, d_y));
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEventRecord(end);

    CUDA_CALL(cudaEventSynchronize(beg));
    CUDA_CALL(cudaEventSynchronize(end));
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.0;
    printf("CSR vector average time: %f second\n", elapsed_time / repeat);

	CUDA_CALL(cudaEventDestroy(beg));
	CUDA_CALL(cudaEventDestroy(end));
	CUDA_CALL(cudaMemcpy(y, d_y, sizeof(double) * nrow, cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(d_indices));
	CUDA_CALL(cudaFree(d_data));
	CUDA_CALL(cudaFree(d_x));
	CUDA_CALL(cudaFree(d_y));
}
