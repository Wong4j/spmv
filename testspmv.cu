#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include "spmv_gpu.cuh"
#include <stdint.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include "csr_formatter.h"

using namespace std;

template <typename T>
void init_mat(T* mat, int n, T val)
{
	for(int i=0; i<n; i++)
	{
		mat[i] = val;
	}
}

template <typename T>
void print_mat(T* mat, int n)
{
	for(int i=0; i<n; i++)
	{
		cout << mat[i] << ' ';
	}
	cout << endl;
}


int main(int argc, char* argv[])
{
	assert(argc == 2);
    
	CSR csr_mat = assemble_csr_matrix_0base(argv[1]);
//	cout << csr_mat.val.size() << ' ' << csr_mat.col_ind.size() << ' ' << csr_mat.row_ptr.size() << endl;
//	print_mat(csr_mat.val.data(), 10);
//	print_mat(csr_mat.col_ind.data(), 10);
//	print_mat(csr_mat.row_ptr.data(), 10);

    int nrow = csr_mat.row_ptr.size() - 1;
    int* csr_row_ptr = csr_mat.row_ptr.data();
    int* csr_col_ind = csr_mat.col_ind.data();
	double* csr_val = csr_mat.val.data();
    double *x = new double[nrow]; 
    double *y = new double[nrow];
    double *y_ref = new double[nrow];
    double *y_ref2 = new double[nrow];
	init_mat(x, nrow, 1.0);
	init_mat(y, nrow, 0.0);
	init_mat(y_ref, nrow, 0.0);
	init_mat(y_ref2, nrow, 0.0);

	spmv_csr_scalar(nrow, csr_row_ptr, csr_col_ind, csr_val, x, y);
	spmv_csr_vector(nrow, csr_row_ptr, csr_col_ind, csr_val, x, y_ref);
	spmv_csr_cusparse(nrow, csr_row_ptr, csr_col_ind, csr_val, x, y_ref2);

	//check result
	double error = 1e-6;
	for(int i=0; i<nrow; i++)
	{
		if( fabs(y[i] - y_ref[i]) > error || fabs(y[i] - y_ref2[i]) > error )
			cout << "wrong result" << endl;
	}
	cout << "correct result" << endl;

//	spmv_ell(nrow, ncol_per_row, indices, data, x, y);

//	std::cout << y[10] << ' ' << y[11] << std::endl;
//	std::cout << y_ref[10] << ' ' << y_ref[11] << std::endl;
//	std::cout << y_ref2[10] << ' ' << y_ref2[11] << std::endl;

    delete[] x;
    delete[] y;
    delete[] y_ref;



}
