#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <math.h> 

using namespace std;

struct CSR {
	vector<double> val;
	vector<int>	col_ind;
	vector<int> row_ptr;
};

void printArray(vector<int> v){
	for(int i = 0; i < v.size(); i++)
		cout << v[i] << ' ';
	cout << '\n';
}

void printArray(vector<double> v){
	for(int i = 0; i < v.size(); i++)
		cout << v[i] << ' ';
	cout << '\n';
}

void printMatrix(CSR csr){
	
	int cont = 0;
	for(int i = 1; i < csr.row_ptr.size(); i++){
		int row_start = csr.row_ptr[i-1] - 1;
		int row_end = csr.row_ptr[i] - 1;
		vector<int>::const_iterator first = csr.col_ind.begin() + row_start;
		vector<int>::const_iterator last = csr.col_ind.begin() + row_end;	
		vector<int> row(first, last);		
		for(int j = 1; j < csr.row_ptr.size(); j++){
			if(std::count(row.begin(), row.end(), j) == 0)
				cout << '0' << ' ';
			else{
				cout << csr.val[cont] << ' ';
				cont++;
			}
		}
		std::cout << std::endl;
	}
}

int getDegree(vector<int> row_ptr, int vertex){
	return row_ptr[vertex] - row_ptr[vertex - 1];
}

vector<int> getAdjVertices(vector<int> col_ind, vector<int> row_ptr, int vertex){
	int row_start = row_ptr[vertex - 1];
	int row_end = row_ptr[vertex];
	vector<int>::const_iterator first = col_ind.begin() + row_start;
	vector<int>::const_iterator last = col_ind.begin() + row_end;	
	vector<int> adjVertices(first, last);		
	return adjVertices;
}

int getBandwidth(CSR csr){
	int bandwidth = std::numeric_limits<int>::min();
	for(int i = 1; i < csr.row_ptr.size() - 1; i++){ // i = current row id
		int row_start = csr.row_ptr[i-1];
		int row_end = csr.row_ptr[i];
		if (row_end - row_start == 1)
			continue;
		for (int j = row_start; j < row_end;j++){
			if (abs(csr.col_ind[j] - i) > bandwidth){
				bandwidth = abs(csr.col_ind[j] - i);
			}
				
		}
	}
	return bandwidth;
}

CSR assemble_csr_matrix(std::string filePath){
	int M, N, L;
	CSR matrix;
	std::ifstream fin(filePath);
	// Ignore headers and comments:
	while (fin.peek() == '%') fin.ignore(2048, '\n');
	// Read defining parameters:
	fin >> M >> N >> L;
	
	int last_row = 1;
	matrix.row_ptr.push_back(1);
	for (int l = 0; l < L; l++){
		int row, col;
		double data;
		fin >> row >> col >> data;
		matrix.col_ind.push_back(col);
		matrix.val.push_back(data);
		if (row > last_row){
			last_row = row;
			matrix.row_ptr.push_back(matrix.col_ind.size());
		}	
	}
	matrix.row_ptr.push_back(matrix.col_ind.size() + 1);
	fin.close();
	return matrix;
}

struct elem{
	int x;
	int y;
	double val;
};

bool comp(elem& a, elem& b)
{
	if(a.x < b.x) return true;
	if(a.x > b.x) return false;
	if(a.y < b.y) return true;
	return false;
}

CSR assemble_csr_matrix_0base(std::string filePath){
	int M, N, L;
	CSR matrix;
	vector<elem> all_elem;

	std::ifstream fin(filePath);
	// Ignore headers and comments:
	while (fin.peek() == '%') fin.ignore(2048, '\n');
	// Read defining parameters:
	fin >> M >> N >> L;


	elem tmp;
	double data;
	for(int i=0; i<L; i++)
	{
		fin >> M >> N >> data;
		tmp.x = M;
		tmp.y = N;
		tmp.val = data;
		all_elem.push_back(tmp);
	}

	sort(all_elem.begin(), all_elem.end(), comp);
	
	int tmp1 = 0;
	int nnz = 0;
	int last_row = 0;
	matrix.row_ptr.push_back(0);
	for (int l = 0; l < L; l++){
		int row, col;
		row = all_elem[l].x;
		col = all_elem[l].y;
		data = all_elem[l].val;
		if (row-1 > last_row){
			last_row = row-1;
			matrix.row_ptr.push_back(matrix.col_ind.size());
			int tmp2 = matrix.row_ptr.back();
			nnz = max(nnz, tmp2-tmp1);
			tmp1 = tmp2;
		}	
		matrix.col_ind.push_back(col-1);
		matrix.val.push_back(data);
	}
	matrix.row_ptr.push_back(matrix.col_ind.size());
	fin.close();
	cout << "nnz = " << nnz << endl;
	return matrix;
}

CSR assemble_simetric_csr_matrix(std::string filePath){
	int M, N, L;
	vector<int> rows, cols;
	vector<double> data;
	CSR matrix;
	std::ifstream fin(filePath);
	// Ignore headers and comments:
	while (fin.peek() == '%') fin.ignore(2048, '\n');
	// Read defining parameters:
	fin >> M >> N >> L;	
	matrix.row_ptr.push_back(0);
	for (int l = 0; l < L; l++){
		int row, col;
		double d;
		fin >> row >> col >> d;
		rows.push_back(row);
		cols.push_back(col);
		data.push_back(d);
	}
	fin.close();
	for (int l = 1; l <= M; l++){
		for (int k = 0; k < L; k++){
			if (cols[k] == l){
				matrix.col_ind.push_back(rows[k]);
				matrix.val.push_back(data[k]);					
			}	
			else if (rows[k] == l){
				matrix.col_ind.push_back(cols[k]);
				matrix.val.push_back(data[k]);				
			}
		}
		matrix.row_ptr.push_back(matrix.col_ind.size());
	}
	
	matrix.row_ptr.push_back(matrix.col_ind.size() + 1);
	
	return matrix;
}


