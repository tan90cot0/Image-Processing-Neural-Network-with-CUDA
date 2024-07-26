#include <bits/stdc++.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cstdio>
using namespace std;

// M is m x m, K is k x k and R is (m - k + 1) x (m - k + 1)
void conv (int m, int k, float* M, float* K, float* R) {
	int cnt = 0;
	for (int x = 0; x < m - k + 1; x++) {
		for (int y = 0; y < m - k + 1; y++) {
			float sum = 0;
			for (int u = 0; u < k; u++) {
				for (int v = 0; v < k; v++) {
					sum += M[m*(x + u) + y + v] * K[k*u + v];
				}
			}
			R[cnt++] = sum;
		}
	}
}

// M is m x m, K is k x k and R is m x m
void padded_conv (int m, int k, int p, float* M, float* K, float* R) {
	// make a matrix N that is (m + k - 1) x (m + k - 1)
	float N[(m + p)*(m + p)];
	// for now last p rows and columns are 0
	for (int i = 0; i < (m + p); i++) {
		for (int j = 0; j < (m + p); j++) {
			if (i < m && j < m)
				N[(m + p)*i + j] = M[m*i + j];
			else
				N[(m + p)*i + j] = 0;
		}
	}
	conv(m + p, k, N, K, R);
}

// apply relu activation on matrix A of size m x n
// result stored in B
void relu (int m, int n, float* A, float* B) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			B[n*i + j] = max((float) 0.0, A[n*i + j]);
		}
	}
}

// apply tanh activation on matrix A of size m x n
// result stored in B
void mat_tanh (int m, int n, float* A, float* B) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			B[n*i + j] = tanh(A[n*i + j]);
		}
	}
}

// pick max in each k x k sub matrix of A
// result stored in B, dim B = (m - k + 1) x (m - k + 1)
void max_pooling (int stride, int m, int k, float* A, float* B) {
	int cnt = 0;
	for (int i = 0; i + k <= m; i += stride) {
		for (int j = 0; j + k <= m; j += stride) {
			// A[i][j] is the top left corner
			float mx = A[m*i + j];
			for (int x = 0; x < k; x++) {
				for (int y = 0; y < k; y++) {
					mx = max(mx, A[m*(i + x) + (j + y)]);
				}
			}
			B[cnt++] = mx;
		}
	}
}

// pick mean in each k x k sub matrix of A
// result stored in B, dim B = (m - k + 1) x (m - k + 1)
void mean_pooling (int stride, int m, int k, float*A, float* B) {
	int cnt = 0;
	for (int i = 0; i < m; i += stride) {
		for (int j = 0; j < m; j += stride) {
			// A[i][j] is the top left corner
			float sum = 0;
			for (int x = 0; x < k; x++) {
				for (int y = 0; y < k; y++) {
					sum += A[m*(i + x) + (j + y)];
				}
			}
			B[cnt++] = sum/(k*k);
		}
	}
}

// v = sigmoid(u)
void sigmoid (int m, float* u, float* v) {
	for (int i = 0; i < m; i++) {
		v[i] = 1.0/(1.0 + exp(-u[i]));
	}
}

// v = softmax(u)
void softmax (int m, float* u, float* v) {
	float sum = 0;
	for (int i = 0; i < m; i++) {
		sum += exp(u[i]);
	}
	for (int i = 0; i < m; i++) {
		v[i] = exp(u[i])/sum;
	}
}

int main (int argc, char* argv[]) {

	std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL); std::cout.tie(NULL);

    cout << setprecision(12);

	if (string(argv[1]) == "1") { // 1=convolution
		int n = stoi(argv[2]), m = stoi(argv[3]), p = stoi(argv[4]);
		float M[n*n], K[m*m], R[(n - m + 1 + p)*(n - m + 1 + p)];
		for (int i = 5; i < 5 + n*n; i++) {
			M[i - 5] = stof(argv[i]);
		}
		for (int i = 5 + n*n; i < 5 + n*n + m*m; i++) {
			K[i - 5 - n*n] = stof(argv[i]);
		}
		padded_conv(n, m, p, M, K, R);
		for (int i = 0; i < (n - m + 1 + p) * (n - m + 1 + p); i++) {
			cout << R[i] << " ";
		}
	}
	else if (string(argv[1]) == "2") { // 2=non-linear-activations
		int n = stoi(argv[3]), m = stoi(argv[4]);
		float M[n*m], R[n*m];
		for (int i = 0; i < n*m; i++)
			M[i] = stof(argv[5 + i]);
		if (string(argv[2]) == "0") { // relu
			relu(n, m, M, R);
		}
		else if (string(argv[2]) == "1") { // tanh
			mat_tanh(n, m, M, R);
		}
		for (int i = 0; i < n*m; i++) {
			cout << R[i] << " ";
		}
	}
	else if (string(argv[1]) == "3") {	// 3=subsampling
		int m = stoi(argv[3]), n = stoi(argv[4]);
		float A[n*n], B[(n-m+1)*(n-m+1)];
		for (int i = 0; i < n*n; i++) {
			A[i] = stof(argv[i + 5]);
		}
		if (string(argv[2]) == "0") {	// 0 = max pool
			max_pooling(1, n, m , A, B);
		}
		else if (string(argv[2]) == "1") {	// 1 = mean pool
			mean_pooling(1, n, m , A, B);
		}
		for (int i = 0; i < (n - m + 1) * (n - m + 1); i++) {
			cout << B[i] << " ";
		}
	}
	else if (string(argv[1]) == "4") {	// 4 = converting a vector
		int m = argc - 3;
		float u[m], v[m];
		for (int i = 0; i < m; i++) {
			u[i] = stof(argv[3 + i]);
		}
		if (string(argv[2]) == "0") { // 0 = sigmoid
			sigmoid(m, u, v);
		}
		else if (string(argv[2]) == "1") {	// 1 = softmax
			softmax(m, u, v);
		}
		for (int i = 0; i < m; i++) {
			cout << v[i] << " " ;
		}
	}
	else {
		cerr << "Invalid option\n";
	}

	return 0;
}