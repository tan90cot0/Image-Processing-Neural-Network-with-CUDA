#include<bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void convKernel(int num_threads, int channels, int m, int k, float* M, float* K, float* R, float *B) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<num_threads){
		int filter = tid / (channels*(m-k+1)*(m-k+1));
		int channel = (tid / ((m-k+1)*(m-k+1))) % channels;
		int x = (tid / (m-k+1)) % (m-k+1);
		int y = tid % (m-k+1);
		float *M2 = &M[channel*m*m];
		float *K2 = &K[filter*channels*k*k + channel*k*k];
		float *R2 = &R[filter*(m-k+1)*(m-k+1)];
		float sum = 0;
		if(channel==0)
			sum = B[filter];

		#pragma unroll 2
		for (int u = 0; u < k; u++) {
			for (int v = 0; v < k; v++) {
				sum += M2[(x + u) * m + y + v] * K2[u * k + v];
			}
		}
		atomicAdd(&R2[x * (m - k + 1) + y], sum);
	}
}

__global__ void convKernel2(int num_threads, int num_filters, int channels, int in_sample_size, int out_sample_size, int m, int k, float* M, float* K, float* R, float *B) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<num_threads){
		int sample = tid / (channels*(m-k+1)*(m-k+1)*num_filters);
		int filter = (tid / (channels*(m-k+1)*(m-k+1)))%num_filters;
		int channel = (tid / ((m-k+1)*(m-k+1))) % channels;
		int x = (tid / (m-k+1)) % (m-k+1);
		int y = tid % (m-k+1);
		float *M2 = M + sample * in_sample_size + channel*m*m;
		float *K2 = K + filter*channels*k*k + channel*k*k;
		float *R2 = R + sample * out_sample_size + filter*(m-k+1)*(m-k+1);
		float sum = 0;
		if(channel==0)
			sum = B[filter];

		#pragma unroll 2
		for (int u = 0; u < k; u++) {
			for (int v = 0; v < k; v++) {
				sum += M2[(x + u) * m + y + v] * K2[u * k + v];
			}
		}
		atomicAdd(&R2[x * (m - k + 1) + y], sum);
	}
}

__global__ void relu_kernel(int num_threads, int m, int n, float* A) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<num_threads){
		int channel = tid / (m*n);
		int row = (tid / n) % m;
		int col = tid %n;
		int index = channel + row * n + col;

		A[index] = fmaxf(0.0f, A[index]);
	}
}

//generalise this
__global__ void max_pooling_kernel(int num_threads, int stride, int m, int k, float* A, float* B) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<num_threads){
		int new_size = (m-k+stride)/stride;
		int channel = tid / (new_size*new_size);
		int row = (tid / new_size) % new_size;
		int col = tid % new_size;
		float* A2 = A + channel*m*m;

		int base_row = row*2;
		int base_col = col*2;

		float mx = A2[m*base_row + base_col];
		for (int x = 0; x < k; x++) {
			for (int y = 0; y < k; y++) {
				mx = max(mx, A2[m*(base_row + x) + (base_col + y)]);
			}
		}

		B[channel*new_size*new_size + row*new_size + col] = mx;
	}
}

__global__ void softmax_kernel(int m, float* u, float* v) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < m) {
        float sum_exp = 0;
        for (int i = 0; i < m; i++) {
            sum_exp += expf(u[i]);
        }
        v[index] = expf(u[index]) / sum_exp;
    }
}

int main () {
	// Fast IO since we are reading a lot of weights
	// hmm does this work for .txt files tho?
	std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL); std::cout.tie(NULL);

	int num_samples = 10000, sample_size = 28 * 28; 
	float X[num_samples*sample_size];

	double io_time = 0, conv1_time = 0, pool1_time = 0, conv2_time = 0, pool2_time = 0, fc1_time = 0, relu_time = 0, fc2_time = 0, softmax_time = 0;

	auto start = chrono::high_resolution_clock::now();
	// load the input matrices
	// nothing to be parallelized here (?)
	ifstream is1("pre-proc-img/input.dat");
	is1.seekg(0, ios_base::end);
	is1.seekg(0, ios_base::beg);
	is1.read((char*) X, num_samples * sample_size * sizeof(float));
	is1.close();

	// load the trained weights
	// nothing to be parallelized here (?)
	
	// CONV1 Layer
	float conv1_filters[20][1][5 * 5], conv1_bias[20];
	ifstream is2("weights/conv1.dat");
	is2.seekg(0, ios_base::end);
	is2.seekg(0, ios_base::beg);
	is2.read((char*) conv1_filters, 20 * 1 * 5 * 5 * sizeof(float));
	is2.read((char*) conv1_bias, 20 * sizeof(float));
	is2.close();

	// CONV2 Layer
	float conv2_filters[50][20][5 * 5], conv2_bias[50];

	ifstream is3("weights/conv2.dat");
	is3.seekg(0, ios_base::end);
	is3.seekg(0, ios_base::beg);
	is3.read((char*) conv2_filters, 50 * 20 * 5 * 5 * sizeof(float));
	is3.read((char*) conv2_bias, 50 * sizeof(float));
	is3.close();

	// FC1 Layer
	float fc1_filters[500][50][4 * 4], fc1_bias[500];

	ifstream is4("weights/fc1.dat");
	is4.seekg(0, ios_base::end);
	is4.seekg(0, ios_base::beg);
	is4.read((char*) fc1_filters, 500 * 50 * 4 * 4 * sizeof(float));
	is4.read((char*) fc1_bias, 500 * sizeof(float));
	is4.close();

	// FC2 Layer
	float fc2_filters[10][500][1 * 1], fc2_bias[10];
	ifstream is5("weights/fc2.dat");
	is5.seekg(0, ios_base::end);
	is5.seekg(0, ios_base::beg);
	is5.read((char*) fc2_filters, 10 * 500 * 1 * 1 * sizeof(float));
	is5.read((char*) fc2_bias, 10 * sizeof(float));
	is5.close();
	
	float* X_cuda;
    cudaMalloc((void **)&X_cuda, num_samples * sample_size * sizeof(float));

	float* conv1_filters_cuda;
    cudaMalloc((void **)&conv1_filters_cuda, 1 * 20 * 5 * 5 * sizeof(float));
    cudaMemcpy(conv1_filters_cuda, conv1_filters, 1 * 20 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);

	float* conv2_filters_cuda;
    cudaMalloc((void **)&conv2_filters_cuda, 20 * 50 * 5 * 5 * sizeof(float));
    cudaMemcpy(conv2_filters_cuda, conv2_filters, 20 * 50 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);

	float* fc1_filters_cuda;
    cudaMalloc((void **)&fc1_filters_cuda, 50 * 500 * 4 * 4 * sizeof(float));
    cudaMemcpy(fc1_filters_cuda, fc1_filters, 50 * 500 * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);

	float* fc2_filters_cuda;
    cudaMalloc((void **)&fc2_filters_cuda, 500 * 10 * 1 * 1 * sizeof(float));
    cudaMemcpy(fc2_filters_cuda, fc2_filters, 500 * 10 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice);

	float* conv1_bias_cuda;
    cudaMalloc((void **)&conv1_bias_cuda, 20 * sizeof(float));
    cudaMemcpy(conv1_bias_cuda, conv1_bias, 20 * sizeof(float), cudaMemcpyHostToDevice);

	float* conv2_bias_cuda;
    cudaMalloc((void **)&conv2_bias_cuda, 50 * sizeof(float));
    cudaMemcpy(conv2_bias_cuda, conv2_bias, 50 * sizeof(float), cudaMemcpyHostToDevice);

	float* fc1_bias_cuda;
    cudaMalloc((void **)&fc1_bias_cuda, 500 * sizeof(float));
    cudaMemcpy(fc1_bias_cuda, fc1_bias, 500 * sizeof(float), cudaMemcpyHostToDevice);

	float* fc2_bias_cuda;
    cudaMalloc((void **)&fc2_bias_cuda, 10 * sizeof(float));
    cudaMemcpy(fc2_bias_cuda, fc2_bias, 10 * sizeof(float), cudaMemcpyHostToDevice);

	float* conv1_out_cuda;
    cudaMalloc((void **)&conv1_out_cuda, num_samples * 20 * 24 * 24 * sizeof(float));
	cudaMemset(conv1_out_cuda, 0, num_samples * 20 * 24 * 24 * sizeof(float));

	float* pool1_out_cuda;
    cudaMalloc((void **)&pool1_out_cuda, num_samples * 20 * 12 * 12 * sizeof(float));

	float* conv2_out_cuda;
    cudaMalloc((void **)&conv2_out_cuda, num_samples * 50 * 8 * 8 * sizeof(float));
	cudaMemset(conv2_out_cuda, 0, num_samples * 50 * 8 * 8 * sizeof(float));

	float* pool2_out_cuda;
    cudaMalloc((void **)&pool2_out_cuda, num_samples * 50 * 4 * 4 * sizeof(float));

	float* fc1_out_cuda;
    cudaMalloc((void **)&fc1_out_cuda, num_samples * 500 * 1 * 1 * sizeof(float));
	cudaMemset(fc1_out_cuda, 0, num_samples * 500 * 1 * 1 * sizeof(float));

	float* fc2_out_cuda;
    cudaMalloc((void **)&fc2_out_cuda, num_samples * 10 * 1 * 1 * sizeof(float));
	cudaMemset(fc2_out_cuda, 0, num_samples * 10 * 1 * 1 * sizeof(float));

	float results[num_samples][10];
	float* results_cuda;
	cudaMalloc((void **)&results_cuda, 10 * num_samples * sizeof(float));

	auto end = std::chrono::high_resolution_clock::now();
	chrono::duration<double> duration = end - start;
	io_time += duration.count();

	// iterate over the inputs
	// this has to be made batch wise in CUDA
	for (int sample = 0; sample < num_samples; sample++) {
		cudaMemcpy(X_cuda + sample * sample_size, X + sample * sample_size, sample_size * sizeof(float), cudaMemcpyHostToDevice);

		// compute forward pass across CONV1
		start = chrono::high_resolution_clock::now();
		convKernel<<<360, 32>>>(11520, 1, 28, 5, &X_cuda[sample*sample_size], conv1_filters_cuda, conv1_out_cuda + sample * 20 * 24 * 24, conv1_bias_cuda);
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		conv1_time += duration.count();

		// compute forward pass across POOL1
		start = chrono::high_resolution_clock::now();
		max_pooling_kernel<<<90, 32>>>(2880, 2, 24, 2, conv1_out_cuda + sample * 20 * 24 * 24, pool1_out_cuda + sample * 20 * 12 * 12);
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		pool1_time += duration.count();

		// compute forward pass across CONV2
		start = chrono::high_resolution_clock::now();
		convKernel<<<2000, 32>>>(64000, 20, 12, 5, pool1_out_cuda + sample * 20 * 12 * 12, conv2_filters_cuda, conv2_out_cuda + sample * 50 * 8 * 8, conv2_bias_cuda);
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		conv2_time += duration.count();

		// compute forward pass across POOL2
		start = chrono::high_resolution_clock::now();
		max_pooling_kernel<<<25, 32>>>(800, 2, 8, 2, conv2_out_cuda + sample * 50 * 8 * 8, pool2_out_cuda + sample * 50 * 4 * 4);
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		pool2_time += duration.count();

		// compute forward pass across FC1
		start = chrono::high_resolution_clock::now();
		convKernel<<<782, 32>>>(25000, 50, 4, 4, pool2_out_cuda + sample * 50 * 4 * 4, fc1_filters_cuda, fc1_out_cuda + sample * 500 * 1 * 1, fc1_bias_cuda);
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		fc1_time += duration.count();

		// compute forward pass across relu
		start = chrono::high_resolution_clock::now();
		relu_kernel<<<16, 32>>>(500, 1, 1, fc1_out_cuda + sample * 500 * 1 * 1);
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		relu_time += duration.count();

		// compute forward pass across FC2
		start = chrono::high_resolution_clock::now();
		convKernel<<<157, 32>>>(5000, 500, 1, 1, fc1_out_cuda + sample * 500 * 1 * 1, fc2_filters_cuda, fc2_out_cuda + sample * 10 * 1 * 1, fc2_bias_cuda);
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		fc2_time += duration.count();

		// compute forward pass across softmax
		start = chrono::high_resolution_clock::now();
		softmax_kernel<<<1, 10>>>(10, fc2_out_cuda + sample * 10 * 1 * 1, results_cuda+sample*10);
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		softmax_time += duration.count();
		cudaMemcpy(results, results_cuda, 10*num_samples*sizeof(float), cudaMemcpyDeviceToHost);
	}

	// uncomment below for timings
	/*cout << "Time spent in IO = " << io_time << "\n";
	cout << "Time spent in CONV1 = " << conv1_time << "\n";
	cout << "Time spent in POOL1 = " << pool1_time << "\n";
	cout << "Time spent in CONV2 = " << conv2_time << "\n";
	cout << "Time spent in POOL2 = " << pool2_time << "\n";
	cout << "Time spent in FC1 = " << fc1_time << "\n";
	cout << "Time spent in RELU = " << relu_time << "\n";
	cout << "Time spent in FC2 = " << fc2_time << "\n";
	cout << "Time spent in SOFTMAX = " << softmax_time << "\n";*/
	
	ofstream MyFile("output/dump_3.txt");
	for (int sample = 0; sample < num_samples; sample++) {
		MyFile << "Sample #" << sample << "\n";
		vector<pair<float, int>> prob;
		for (int i = 0; i < 10; i++) {
			prob.push_back({results[sample][i], i});
		}
		sort(prob.begin(), prob.end());
		for (int i = 0; i < 5; i++) {
			MyFile << 100*prob[9 - i].first << " class " << prob[9 - i].second << "\n";
		}
		MyFile << "\n";
	}
	MyFile.close();

	// uncomment below for accuracy
	/*FILE* pipe = popen("diff output.txt out.txt | grep \"^>\" | wc -l", "r");
    if (!pipe) {
        std::cerr << "popen() failed!" << std::endl;
        return 1;
    }

    // Read the command output
    char buffer[128];
    fgets(buffer, 128, pipe); // Assuming the output is less than 128 characters
    int diffCount = atoi(buffer); // Convert the output to integer
    pclose(pipe);

	float acc = (num_samples-diffCount)*100.0/num_samples;
    std::cout << "Accuracy: " << acc << "%"<<std::endl;*/

	return 0;
}