#include <bits/stdc++.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cstdio>
#include <cuda_runtime.h>
using namespace std;

__global__ void convKernel(int m, int k, float* M, float* K, float* R) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < m - k + 1 && y < m - k + 1) {
        float sum = 0;
        for (int u = 0; u < k; u++) {
            for (int v = 0; v < k; v++) {
                sum += M[m * (x + u) + y + v] * K[k * u + v];
            }
        }
        R[x * (m - k + 1) + y] = sum;
    }
}

// M is m x m, K is k x k and R is (m - k + 1) x (m - k + 1)
void conv (int m, int k, float* M, float* K, float* R) {
	 float *d_M, *d_K, *d_R;
    int size_M = m * m * sizeof(float);
    int size_K = k * k * sizeof(float);
    int size_R = (m - k + 1) * (m - k + 1) * sizeof(float);

    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_K, size_K);
    cudaMalloc((void**)&d_R, size_R);

    cudaMemcpy(d_M, M, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, size_K, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m - k + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (m - k + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convKernel<<<numBlocks, threadsPerBlock>>>(m, k, d_M, d_K, d_R);

    cudaMemcpy(R, d_R, size_R, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_K);
    cudaFree(d_R);
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

__global__ void reluKernel(int m, int n, float* A, float* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        float val = A[n * i + j];
        B[n * i + j] = (val > 0) ? val : 0;
    }
}

// apply relu activation on matrix A of size m x n
// result stored in B
void relu (int m, int n, float* A, float* B) {
	float *d_A, *d_B;
    int size_A = m * n * sizeof(float);
    int size_B = m * n * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    reluKernel<<<numBlocks, threadsPerBlock>>>(m, n, d_A, d_B);

    cudaMemcpy(B, d_B, size_B, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}

__global__ void tanhKernel(int m, int n, float* A, float* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        float val = A[n * i + j];
        B[n * i + j] = tanh(val);
    }
}

// apply tanh activation on matrix A of size m x n
// result stored in B
void mat_tanh (int m, int n, float* A, float* B) {
	float *d_A, *d_B;
    int size_A = m * n * sizeof(float);
    int size_B = m * n * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    tanhKernel<<<numBlocks, threadsPerBlock>>>(m, n, d_A, d_B);

    cudaMemcpy(B, d_B, size_B, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}

__global__ void maxPoolingKernel(int stride, int m, int k, float* A, float* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < (m - k + 1) / stride && j < (m - k + 1) / stride) {
        int cnt = i * (m - k + 1) / stride + j;
        int start_i = i * stride;
        int start_j = j * stride;

        float mx = A[m * start_i + start_j];

        for (int x = 0; x < k; x++) {
            for (int y = 0; y < k; y++) {
                mx = fmaxf(mx, A[m * (start_i + x) + (start_j + y)]);
            }
        }

        B[cnt] = mx;
    }
}

// pick max in each k x k sub matrix of A
// result stored in B, dim B = (m - k + 1) x (m - k + 1)
void max_pooling (int stride, int m, int k, float* A, float* B) {
	float *d_A, *d_B;
    int size_A = m * m * sizeof(float);
    int size_B = ((m - k) / stride + 1) * ((m - k) / stride + 1) * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m - k + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (m - k + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    maxPoolingKernel<<<numBlocks, threadsPerBlock>>>(stride, m, k, d_A, d_B);

    cudaMemcpy(B, d_B, size_B, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}

__global__ void meanPoolingKernel(int stride, int m, int k, float* A, float* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < (m - k + 1) / stride && j < (m - k + 1) / stride) {
        int cnt = i * (m - k + 1) / stride + j;
        int start_i = i * stride;
        int start_j = j * stride;

        float sum = 0;

        for (int x = 0; x < k; x++) {
            for (int y = 0; y < k; y++) {
                sum += A[m * (start_i + x) + (start_j + y)];
            }
        }

        B[cnt] = sum / (k * k);
    }
}

// pick mean in each k x k sub matrix of A
// result stored in B, dim B = (m - k + 1) x (m - k + 1)
void mean_pooling (int stride, int m, int k, float*A, float* B) {
	float *d_A, *d_B;
    int size_A = m * m * sizeof(float);
    int size_B = ((m - k) / stride + 1) * ((m - k) / stride + 1) * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m - k + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (m - k + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    meanPoolingKernel<<<numBlocks, threadsPerBlock>>>(stride, m, k, d_A, d_B);

    cudaMemcpy(B, d_B, size_B, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}

__global__ void sigmoidKernel(int m, float* u, float* v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m) {
        v[i] = 1.0 / (1.0 + expf(-u[i]));
    }
}

// v = sigmoid(u)
void sigmoid (int m, float* u, float* v) {
	float *d_u, *d_v;
    int size = m * sizeof(float);

    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_v, size);

    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;

    sigmoidKernel<<<numBlocks, threadsPerBlock>>>(m, d_u, d_v);

    cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_v);
}

__global__ void softmaxKernel(int m, float* u, float* v, float* sumExp) {
    extern __shared__ float sdata[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m) {
        sdata[threadIdx.x] = expf(u[i]);
    } else {
        sdata[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Reduction to calculate the sum of exp(u[i])
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sumExp[blockIdx.x] = sdata[0];
    }

    __syncthreads();

    // Normalize the values
    if (i < m) {
        v[i] = expf(u[i]) / sumExp[0];
    }
}

// v = softmax(u)
void softmax (int m, float* u, float* v) {
	float *d_u, *d_v, *d_sumExp;
    int size = m * sizeof(float);

    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_v, size);
    cudaMalloc((void**)&d_sumExp, sizeof(float));

    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;

    softmaxKernel<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(m, d_u, d_v, d_sumExp);

    cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_sumExp);
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