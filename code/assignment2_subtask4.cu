#include<bits/stdc++.h>
#include <fstream>
#include <cstdio>
using namespace std;

__global__ void conv2_Kernel(float* M, float* K, float* R, float *B) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<64000){
		int filter = tid / (1280);
		int channel = (tid / 64) % 20;
		int x = (tid / 8) % 8;
		int y = tid % 8;
		float *M2 = &M[channel*144];
		float *K2 = &K[filter*500 + channel*25];
		float *R2 = &R[filter*64];
		float sum = 0;
		if(channel==0)
			sum = B[filter];

		#pragma unroll
		for (int u = 0; u < 5; u++) {
			#pragma unroll
			for (int v = 0; v < 5; v++) {
				sum += M2[(x + u) * 12 + y + v] * K2[u * 5 + v];
			}
		}
		atomicAdd(&R2[x * 8 + y], sum);
	}
}

__global__ void conv1_Kernel(float* M, float* K, float* R, float *B) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<11520){
		int filter = tid / 576;
		int channel = 0;
		int x = (tid / 24) % 24;
		int y = tid % 24;
		float *M2 = &M[channel*784];
		float *K2 = &K[filter*25 + channel*25];
		float *R2 = &R[filter*576];
		float sum = 0;
		if(channel==0)
			sum = B[filter];

		#pragma unroll
		for (int u = 0; u < 5; u++) {
			#pragma unroll
			for (int v = 0; v < 5; v++) {
				sum += M2[(x + u) * 28 + y + v] * K2[u * 5 + v];
			}
		}
		atomicAdd(&R2[x * 24 + y], sum);
	}
}

__global__ void fc1_Kernel(float* M, float* K, float* R, float *B) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<25000){
		int filter = tid / 50;
		int channel = tid % 50;
		float *M2 = &M[channel*16];
		float *K2 = &K[filter*800 + channel*16];
		float sum = 0;
		if(channel==0)
			sum = B[filter];

		#pragma unroll
		for (int u = 0; u < 4; u++) {
			#pragma unroll
			for (int v = 0; v < 4; v++) {
				sum += M2[(u) * 4 + v] * K2[u * 4 + v];
			}
		}
		atomicAdd(R + filter, sum);
	}
}

__global__ void fc2_Kernel(float* M, float* K, float* R, float *B) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<5000){
		int filter = tid / 500;
		int channel = tid % 500;
		float sum = 0;
		if(channel==0)
			sum = B[filter];
		atomicAdd(R + filter, sum + M[channel] * K[filter*500 + channel]);
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

__global__ void relu_kernel2(int num_threads, int num_channels, int dim, int m, int n, float* A) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<num_threads){
		int sample = tid / (m*n*num_channels);
		int channel = (tid / (m*n))%num_channels;
		int row = (tid / n) % m;
		int col = tid %n;
		int index = sample*dim + channel + row * n + col;

		A[index] = fmaxf(0.0f, A[index]);
	}
}

__global__ void max_pool1_kernel(float* A, float* B) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<2880){
		int channel = tid / 144;
		int row = (tid / 12) % 12;
		int col = tid % 12;
		float* A2 = A + channel*576;

		int base_row = row*2;
		int base_col = col*2;

		float mx = A2[24*base_row + base_col];
		mx = max(mx, A2[24*(base_row + 1) + (base_col)]);
		mx = max(mx, A2[24*(base_row) + (base_col + 1)]);
		mx = max(mx, A2[24*(base_row + 1) + (base_col + 1)]);

		B[channel*144 + row*12 + col] = mx;
	}
}

__global__ void max_pool2_kernel(float* A, float* B) {
    int tid = ((blockIdx.x*blockDim.x)+threadIdx.x);
	if(tid<800){
		int channel = tid / 16;
		int row = (tid / 4) % 4;
		int col = tid % 4;
		float* A2 = A + channel*64;

		int base_row = row*2;
		int base_col = col*2;

		float mx = A2[8*base_row + base_col];
		mx = max(mx, A2[8*(base_row + 1) + (base_col)]);
		mx = max(mx, A2[8*(base_row) + (base_col + 1)]);
		mx = max(mx, A2[8*(base_row + 1) + (base_col + 1)]);

		B[channel*16 + row*4 + col] = mx;
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
	auto prog_start = chrono::high_resolution_clock::now();
	std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL); std::cout.tie(NULL);

	const int num_samples = 10000, sample_size = 28 * 28; 
	float X[num_samples][sample_size];

	// allocate all memory on device
	float* X_cuda;
	cudaMalloc((void **)&X_cuda, num_samples * sample_size * sizeof(float));
	float* conv1_filters_cuda;
	cudaMalloc((void **)&conv1_filters_cuda, 1 * 20 * 5 * 5 * sizeof(float));
	float* conv1_bias_cuda;
	cudaMalloc((void **)&conv1_bias_cuda, 20 * sizeof(float));
	float* conv1_out_cuda;
    cudaMalloc((void **)&conv1_out_cuda, num_samples * 20 * 24 * 24 * sizeof(float));
	cudaMemset(conv1_out_cuda, 0, num_samples * 20 * 24 * 24 * sizeof(float));
	float* pool1_out_cuda;
    cudaMalloc((void **)&pool1_out_cuda, num_samples * 20 * 12 * 12 * sizeof(float));
	float* conv2_filters_cuda;
    cudaMalloc((void **)&conv2_filters_cuda, 20 * 50 * 5 * 5 * sizeof(float));
    float* conv2_bias_cuda;
    cudaMalloc((void **)&conv2_bias_cuda, 50 * sizeof(float));
    float* conv2_out_cuda;
    cudaMalloc((void **)&conv2_out_cuda, num_samples * 50 * 8 * 8 * sizeof(float));
	cudaMemset(conv2_out_cuda, 0, num_samples * 50 * 8 * 8 * sizeof(float));
	float* pool2_out_cuda;
    cudaMalloc((void **)&pool2_out_cuda, num_samples * 50 * 4 * 4 * sizeof(float));
    float* fc1_filters_cuda;
    cudaMalloc((void **)&fc1_filters_cuda, 50 * 500 * 4 * 4 * sizeof(float));
    float* fc1_bias_cuda;
    cudaMalloc((void **)&fc1_bias_cuda, 500 * sizeof(float));
    float* fc1_out_cuda;
    cudaMalloc((void **)&fc1_out_cuda, num_samples * 500 * 1 * 1 * sizeof(float));
	cudaMemset(fc1_out_cuda, 0, num_samples * 500 * 1 * 1 * sizeof(float));
	float* fc2_filters_cuda;
    cudaMalloc((void **)&fc2_filters_cuda, 500 * 10 * 1 * 1 * sizeof(float));
    float* fc2_bias_cuda;
    cudaMalloc((void **)&fc2_bias_cuda, 10 * sizeof(float));
    float* fc2_out_cuda;
    cudaMalloc((void **)&fc2_out_cuda, num_samples * 10 * 1 * 1 * sizeof(float));
	cudaMemset(fc2_out_cuda, 0, num_samples * 10 * 1 * 1 * sizeof(float));
	float results[num_samples][10];
	float* results_cuda;
	cudaMalloc((void **)&results_cuda, 10 * num_samples * sizeof(float));

	// create asynchronous streams
	// each non-blocking
	const int num_streams = 2;
	cudaStream_t stream[num_streams];
	for (int i = 0; i < num_streams; i++)
		cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);

	auto io_start = chrono::high_resolution_clock::now();
	// load the input matrices
	// nothing to be parallelized here (?)

	ifstream is1("pre-proc-img/input.dat");
	is1.seekg(0, ios_base::end);
	is1.seekg(0, ios_base::beg);
	is1.read((char*) X, num_samples * sample_size * sizeof(float));
	is1.close();

	cudaMemcpyAsync(X_cuda, X, num_samples * sample_size * sizeof(float), cudaMemcpyHostToDevice, stream[0]);

	// load the trained weights
	// nothing to be parallelized here (?)
	
	// CONV1 Layer
	float conv1_filters[20][1][5 * 5], conv1_bias[20];

	ifstream is2("weights/conv1.dat");
	is2.seekg(0, ios_base::end);
	is2.seekg(0, ios_base::beg);
	is2.read((char*) conv1_filters, 20 * 1 * 5 * 5 * sizeof(float));

	cudaMemcpyAsync(conv1_filters_cuda, conv1_filters, 1 * 20 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
	
	is2.read((char*) conv1_bias, 20 * sizeof(float));
	is2.close();

    cudaMemcpyAsync(conv1_bias_cuda, conv1_bias, 20 * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    // compute forward pass across CONV1
	auto conv1_start = chrono::high_resolution_clock::now();
	for (int sample = 0; sample < num_samples; sample++)
		conv1_Kernel<<<360, 32, 0, stream[0]>>>(&X_cuda[sample*sample_size], conv1_filters_cuda, conv1_out_cuda + sample * 11520, conv1_bias_cuda);

	// compute forward pass across POOL1
	auto pool1_start = chrono::high_resolution_clock::now();
	for (int sample = 0; sample < num_samples; sample++)
		max_pool1_kernel<<<90, 32, 0, stream[0]>>>(conv1_out_cuda + sample * 11520, pool1_out_cuda + sample * 2880);

	// CONV2 Layer
	float conv2_filters[50][20][5 * 5], conv2_bias[50];

	ifstream is3("weights/conv2.dat");
	is3.seekg(0, ios_base::end);
	is3.seekg(0, ios_base::beg);
	is3.read((char*) conv2_filters, 50 * 20 * 5 * 5 * sizeof(float));

	cudaMemcpyAsync(conv2_filters_cuda, conv2_filters, 20 * 50 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice, stream[1]);

	is3.read((char*) conv2_bias, 50 * sizeof(float));
	is3.close();

    cudaMemcpyAsync(conv2_bias_cuda, conv2_bias, 50 * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    // compute forward pass across CONV2
	auto conv2_start = chrono::high_resolution_clock::now();
	for (int sample = 0; sample < num_samples; sample++)
		conv2_Kernel<<<2000, 32, 0, stream[0]>>>(pool1_out_cuda + sample * 2880, conv2_filters_cuda, conv2_out_cuda + sample * 3200, conv2_bias_cuda);

	// compute forward pass across POOL2
	auto pool2_start = chrono::high_resolution_clock::now();
	for (int sample = 0; sample < num_samples; sample++)
		max_pool2_kernel<<<25, 32, 0, stream[0]>>>(conv2_out_cuda + sample * 3200, pool2_out_cuda + sample * 800);

	// FC1 Layer
	float fc1_filters[500][50][4 * 4], fc1_bias[500];

	ifstream is4("weights/fc1.dat");
	is4.seekg(0, ios_base::end);
	is4.seekg(0, ios_base::beg);
	is4.read((char*) fc1_filters, 500 * 50 * 4 * 4 * sizeof(float));

	cudaMemcpyAsync(fc1_filters_cuda, fc1_filters, 50 * 500 * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
	
	is4.read((char*) fc1_bias, 500 * sizeof(float));
	is4.close();

    cudaMemcpyAsync(fc1_bias_cuda, fc1_bias, 500 * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    // compute forward pass across FC1
	auto fc1_start = chrono::high_resolution_clock::now();
	for (int sample = 0; sample < num_samples; sample++)
		fc1_Kernel<<<782, 32, 0, stream[0]>>>(pool2_out_cuda + sample * 800, fc1_filters_cuda, fc1_out_cuda + sample * 500, fc1_bias_cuda);
	// compute forward pass across relu
	auto relu_start = chrono::high_resolution_clock::now();
	for (int sample = 0; sample < num_samples; sample++)
		relu_kernel<<<16, 32, 0, stream[0]>>>(500, 1, 1, fc1_out_cuda + sample * 500);
	// relu_kernel2<<<16*num_samples, 32>>>(500*num_samples, 500, 500, 1, 1, fc1_out_cuda);

	// FC2 Layer
	float fc2_filters[10][500][1 * 1], fc2_bias[10];
	ifstream is5("weights/fc2.dat");
	is5.seekg(0, ios_base::end);
	is5.seekg(0, ios_base::beg);
	is5.read((char*) fc2_filters, 10 * 500 * 1 * 1 * sizeof(float));

	cudaMemcpyAsync(fc2_filters_cuda, fc2_filters, 500 * 10 * 1 * 1 * sizeof(float), cudaMemcpyHostToDevice, stream[1]);

	is5.read((char*) fc2_bias, 10 * sizeof(float));
	is5.close();
	
    cudaMemcpyAsync(fc2_bias_cuda, fc2_bias, 10 * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
	// compute forward pass across FC2
	auto fc2_start = chrono::high_resolution_clock::now();
	for (int sample = 0; sample < num_samples; sample++)
		fc2_Kernel<<<157, 32, 0, stream[0]>>>(fc1_out_cuda + sample * 500, fc2_filters_cuda, fc2_out_cuda + sample * 10, fc2_bias_cuda);	

	// compute forward pass across softmax
	auto softmax_start = chrono::high_resolution_clock::now();
	for (int sample = 0; sample < num_samples; sample++)
		softmax_kernel<<<1, 10, 0, stream[0]>>>(10, fc2_out_cuda + sample * 10, results_cuda + sample * 10);
	auto softmax_end = std::chrono::high_resolution_clock::now();

	cudaDeviceSynchronize();
	for (int i = 0; i < num_streams; i++)
        cudaStreamDestroy(stream[i]);

	cudaMemcpy(results, results_cuda, 10*num_samples*sizeof(float), cudaMemcpyDeviceToHost);

	ofstream MyFile("output/dump_4.txt");
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

	/*double nano = 1000000000;
	cout << "Time spent in IO = " << ((conv1_start-io_start).count())/nano << "\n";
	cout << "Time spent in CONV1 = " << ((pool1_start-conv1_start).count())/nano << "\n";
	cout << "Time spent in POOL1 = " << ((conv2_start-pool1_start).count())/nano << "\n";
	cout << "Time spent in CONV2 = " << ((pool2_start-conv2_start).count())/nano << "\n";
	cout << "Time spent in POOL2 = " << ((fc1_start-pool2_start).count())/nano << "\n";
	cout << "Time spent in FC1 = " << ((relu_start-fc1_start).count())/nano << "\n";
	cout << "Time spent in RELU = " << ((fc2_start-relu_start).count())/nano << "\n";
	cout << "Time spent in FC2 = " << ((softmax_start-fc2_start).count())/nano << "\n";
	cout << "Time spent in SOFTMAX = " << ((softmax_end-softmax_start).count())/nano << "\n";*/
	
	//auto prog_end = chrono::high_resolution_clock::now();
	//cout << "Total time = " << ((prog_end-prog_start).count())/nano << "\n\n\n";

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