#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "fp16_conversion.h"
#include "nvmlPower.hpp"
#include <vector>

using namespace std;

// #define FP16MM

inline const char* cublasGetErrorString(cublasStatus_t status)
{
	switch (status)
	{
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
	if (result != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
		assert(result == CUBLAS_STATUS_SUCCESS);
	}
	return result;
}

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
template<typename T>
inline void CPU_fill_rand(T *A, int nr_rows_A, int nr_cols_A) {
	int a = 1;

	for (int i = 0; i < nr_rows_A * nr_cols_A; i++) {
		A[i] = (T)rand() / (T)(RAND_MAX / a);
	}
}

int single_stream_hgemm(int use_tensorcore, int matrix_size, int repeats = 10) {
	int min_m_k_n = 2;
	int max_m_k_n = 4096 * 4;
	min_m_k_n = max_m_k_n = matrix_size;
	int verbose = 1;

	cout << "\ncublasHgemm test result:\n" << endl;

	if (verbose)
		cout << "running with"
		<< " min_m_k_n: " << min_m_k_n
		<< " max_m_k_n: " << max_m_k_n
		<< " use_tensor: " << use_tensorcore
		<< " dType bits:" <<sizeof(__half)*8
		<< " repeats: " << repeats
		<< endl;

	cublasStatus_t stat;
	cublasHandle_t handle;

	checkCublas(cublasCreate(&handle));
	if (use_tensorcore == 1) {
		checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
		//checkCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
		//checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
	}

	if (verbose) cout << "allocating device variables" << endl;

	// Allocate 3 arrays on CPU

	float *h_A = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
	float *h_B = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
	float *h_C = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));

	float *h_hA = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(__half));
	float *h_hB = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(__half));
	float *h_hC = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(__half));

	CPU_fill_rand(h_A, max_m_k_n, max_m_k_n);
	CPU_fill_rand(h_B, max_m_k_n, max_m_k_n);
	CPU_fill_rand(h_C, max_m_k_n, max_m_k_n);

	__half *d_A, *d_B, *d_C;
	checkCuda(cudaMalloc(&d_A, max_m_k_n * max_m_k_n * sizeof(__half)));
	checkCuda(cudaMalloc(&d_B, max_m_k_n * max_m_k_n * sizeof(__half)));
	checkCuda(cudaMalloc(&d_C, max_m_k_n * max_m_k_n * sizeof(__half)));

	for (int i = 0; i < max_m_k_n * max_m_k_n; i++) {
		h_hA[i] = approx_float_to_half(h_A[i]);
		h_hA[i] = approx_float_to_half(h_B[i]);
		h_hA[i] = approx_float_to_half(h_C[i]);
	}

	checkCuda(cudaMemcpy(d_A, h_hA, max_m_k_n * max_m_k_n * sizeof(__half), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B, h_hB, max_m_k_n * max_m_k_n * sizeof(__half), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_C, h_hC, max_m_k_n * max_m_k_n * sizeof(__half), cudaMemcpyHostToDevice));


	int lda, ldb, ldc, m, n, k;
	const __half alf = approx_float_to_half(1.0);
	const __half bet = approx_float_to_half(0.0);
	const __half *alpha = &alf;
	const __half *beta = &bet;


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int size = min_m_k_n;
	{
		double sum = 0.0;

		for (int rep = 0; rep < repeats; rep++) {

			m = n = k = size;
			lda = m;
			ldb = k;
			ldc = m;
			if (rep == repeats - 1) {
				nvmlAPIRun();
				cudaEventRecord(start, 0);
			}

			stat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);	

			cudaDeviceSynchronize();
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);

			if (rep == repeats - 1) {
				float elapsed;
				cudaEventElapsedTime(&elapsed, start, stop);
				elapsed /= 1000.0f;
				sum = elapsed;
				nvmlAPIEnd();
			}
			if (stat != CUBLAS_STATUS_SUCCESS) {
				cerr << "cublasSgemmBatched failed" << endl;
				exit(1);
			}
			assert(!cudaGetLastError());

			std::swap(d_A, d_C);

		}

		cout << "float16: size " << size << " average: " << sum << " s " << endl;

	}

	checkCuda(cudaMemcpy(h_hA, d_A, max_m_k_n * max_m_k_n * sizeof(__half), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_hB, d_B, max_m_k_n * max_m_k_n * sizeof(__half), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_hC, d_C, max_m_k_n * max_m_k_n * sizeof(__half), cudaMemcpyDeviceToHost));

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

int single_stream_sgemm(int use_tensorcore, int matrix_size, int repeats = 10) {
	int min_m_k_n = 2;
	int max_m_k_n = 4096 * 4;
	min_m_k_n = max_m_k_n = matrix_size;
	int verbose = 1;

	cout << "\ncublasSgemm test result:\n" << endl;

	if (verbose)
		cout << "running with"
		<< " min_m_k_n: " << min_m_k_n
		<< " max_m_k_n: " << max_m_k_n
		<< " use_tensor: "<< use_tensorcore
		<< " repeats: " << repeats
		<< " dType bits:" << sizeof(float) * 8
		<< endl;

	cublasStatus_t stat;
	cublasHandle_t handle;

	checkCublas(cublasCreate(&handle));	
	if (use_tensorcore == 1) {
		checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
		//checkCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
		//checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
	}

	if (verbose) cout << "allocating device variables" << endl;

	// Allocate 3 arrays on CPU

	float *h_A = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
	float *h_B = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
	float *h_C = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));

	CPU_fill_rand(h_A, max_m_k_n, max_m_k_n);
	CPU_fill_rand(h_B, max_m_k_n, max_m_k_n);
	CPU_fill_rand(h_C, max_m_k_n, max_m_k_n);

	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	checkCuda(cudaMalloc(&d_A, max_m_k_n * max_m_k_n * sizeof(float)));
	checkCuda(cudaMalloc(&d_B, max_m_k_n * max_m_k_n * sizeof(float)));
	checkCuda(cudaMalloc(&d_C, max_m_k_n * max_m_k_n * sizeof(float)));

	checkCuda(cudaMemcpy(d_A, h_A, max_m_k_n * max_m_k_n * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B, h_B, max_m_k_n * max_m_k_n * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_C, h_C, max_m_k_n * max_m_k_n * sizeof(float), cudaMemcpyHostToDevice));

	int lda, ldb, ldc, m, n, k;
	const float alf = 1.0f;
	const float bet = 0.0f;
	const float *alpha = &alf;
	const float *beta = &bet;


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int size = min_m_k_n;
	{
		double sum = 0.0;
		
		for (int rep = 0; rep < repeats; rep++) {
			
			m = n = k = size;
			lda = m;
			ldb = k;
			ldc = m;
			if (rep == repeats - 1) {
				nvmlAPIRun();
				cudaEventRecord(start, 0);
			}

			stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);

			cudaDeviceSynchronize();
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);

			if (rep == repeats - 1) {
				float elapsed;
				cudaEventElapsedTime(&elapsed, start, stop);
				elapsed /= 1000.0f;
				sum = elapsed;
				nvmlAPIEnd();
			}
			if (stat != CUBLAS_STATUS_SUCCESS) {
				cerr << "cublasSgemmBatched failed" << endl;
				exit(1);
			}
			assert(!cudaGetLastError());
			
			std::swap(d_A, d_C);
	
		}

		cout << "float32: size " << size << " average: " << sum  << " s " << endl;

	}

	checkCuda(cudaMemcpy(h_A, d_A, max_m_k_n * max_m_k_n * sizeof(float), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_B, d_B, max_m_k_n * max_m_k_n * sizeof(float), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_C, d_C, max_m_k_n * max_m_k_n * sizeof(float), cudaMemcpyDeviceToHost));

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

int single_stream_dgemm(int use_tensorcore, int matrix_size, int repeats = 10) {
	int min_m_k_n = 2;
	int max_m_k_n = 4096 * 4;
	min_m_k_n = max_m_k_n = matrix_size;
	int verbose = 1;

#ifndef FP16MM
	cout << "\ncublasSgemm test result:\n" << endl;
#else
	cout << "\ncublasHgemm test result:\n" << endl;
#endif

	if (verbose)
		cout << "running with"
		<< " min_m_k_n: " << min_m_k_n
		<< " max_m_k_n: " << max_m_k_n
		<< " use_tensor: " << use_tensorcore
		<< " repeats: " << repeats
		<< " dType bits:" << sizeof(double) * 8
		<< endl;

	cublasStatus_t stat;
	cublasHandle_t handle;

	checkCublas(cublasCreate(&handle));
	if (use_tensorcore == 1) {
		checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
		//checkCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
		//checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
	}

	if (verbose) cout << "allocating device variables" << endl;

	// Allocate 3 arrays on CPU

	double *h_A = (double *)malloc(max_m_k_n * max_m_k_n * sizeof(double));
	double *h_B = (double *)malloc(max_m_k_n * max_m_k_n * sizeof(double));
	double *h_C = (double *)malloc(max_m_k_n * max_m_k_n * sizeof(double));

	CPU_fill_rand(h_A, max_m_k_n, max_m_k_n);
	CPU_fill_rand(h_B, max_m_k_n, max_m_k_n);
	CPU_fill_rand(h_C, max_m_k_n, max_m_k_n);

	// Allocate 3 arrays on GPU
	double *d_A, *d_B, *d_C;
	checkCuda(cudaMalloc(&d_A, max_m_k_n * max_m_k_n * sizeof(double)));
	checkCuda(cudaMalloc(&d_B, max_m_k_n * max_m_k_n * sizeof(double)));
	checkCuda(cudaMalloc(&d_C, max_m_k_n * max_m_k_n * sizeof(double)));

	checkCuda(cudaMemcpy(d_A, h_A, max_m_k_n * max_m_k_n * sizeof(double), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B, h_B, max_m_k_n * max_m_k_n * sizeof(double), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_C, h_C, max_m_k_n * max_m_k_n * sizeof(double), cudaMemcpyHostToDevice));

	int lda, ldb, ldc, m, n, k;
	const double alf = 1.0f;
	const double bet = 0.0f;
	const double *alpha = &alf;
	const double *beta = &bet;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int size = min_m_k_n;
	{
		double sum = 0.0;

		for (int rep = 0; rep < repeats; rep++) {

			m = n = k = size;
			lda = m;
			ldb = k;
			ldc = m;
			if (rep == repeats - 1) {
				nvmlAPIRun();
				cudaEventRecord(start, 0);
			}

			stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
		
			cudaDeviceSynchronize();
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);

			if (rep == repeats - 1) {
				float elapsed;
				cudaEventElapsedTime(&elapsed, start, stop);
				elapsed /= 1000.0f;
				sum = elapsed;
				nvmlAPIEnd();
			}
			if (stat != CUBLAS_STATUS_SUCCESS) {
				cerr << "cublasSgemmBatched failed" << endl;
				exit(1);
			}
			assert(!cudaGetLastError());

			std::swap(d_A, d_C);

		}

		cout << "float64: size " << size << " average: " << sum << " s " << endl;

	}

	checkCuda(cudaMemcpy(h_A, d_A, max_m_k_n * max_m_k_n * sizeof(double), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_B, d_B, max_m_k_n * max_m_k_n * sizeof(double), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_C, d_C, max_m_k_n * max_m_k_n * sizeof(double), cudaMemcpyDeviceToHost));

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

int multi_stream(int num_streams, int use_tensorcore, int matrix_size, int repeats = 10) {
	int min_m_k_n = 2;
	int max_m_k_n = 4096 * 4;
	min_m_k_n = max_m_k_n = matrix_size;
	int verbose = 1;

#ifndef FP16MM
	cout << "\ncublasSgemm test result:\n" << endl;
#else
	cout << "\ncublasHgemm test result:\n" << endl;
#endif

	if (verbose)
		cout << "running with"
		<< " streams    : "<< num_streams
		<< " min_m_k_n: " << min_m_k_n
		<< " max_m_k_n: " << max_m_k_n
		<< " use_tensor: " << use_tensorcore
		<< " repeats: " << repeats
		<< endl;

	vector<cublasStatus_t> vecstat(num_streams);
	cublasHandle_t handle;

	vector<cudaStream_t> vecStreams(num_streams);
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&vecStreams[i]);
	}
	vector<cublasHandle_t> vecHandle(num_streams);

	for (int i=0; i< num_streams; i++)
		checkCublas(cublasCreate(&vecHandle[i]));

	if (use_tensorcore == 1) {
#if 1
		if (num_streams == 2) {
			checkCublas(cublasSetMathMode(vecHandle[1], CUBLAS_TENSOR_OP_MATH));
		}
#else
		for (int i = 0; i < num_streams; i++) {
			checkCublas(cublasSetMathMode(vecHandle[i], CUBLAS_TENSOR_OP_MATH));
		}
#endif
	}

	for (int i = 0; i < num_streams; i++) {
		checkCublas(cublasSetStream(vecHandle[i], vecStreams[i]));
	}

	if (verbose) cout << "allocating device variables" << endl;

	// Allocate 3 arrays on CPU

	float *h_A = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float)*num_streams);
	float *h_B = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float)*num_streams);
	float *h_C = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float)*num_streams);

	CPU_fill_rand(h_A, max_m_k_n, max_m_k_n*num_streams);
	CPU_fill_rand(h_B, max_m_k_n, max_m_k_n*num_streams);
	CPU_fill_rand(h_C, max_m_k_n, max_m_k_n*num_streams);

#ifndef FP16MM
	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	vector<float*> vec_d_A(num_streams), vec_d_B(num_streams), vec_d_C(num_streams);
	checkCuda(cudaMalloc(&d_A, max_m_k_n * max_m_k_n * sizeof(float)*num_streams));
	checkCuda(cudaMalloc(&d_B, max_m_k_n * max_m_k_n * sizeof(float)*num_streams));
	checkCuda(cudaMalloc(&d_C, max_m_k_n * max_m_k_n * sizeof(float)*num_streams));

	checkCuda(cudaMemcpy(d_A, h_A, max_m_k_n * max_m_k_n * sizeof(float)*num_streams, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B, h_B, max_m_k_n * max_m_k_n * sizeof(float)*num_streams, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_C, h_C, max_m_k_n * max_m_k_n * sizeof(float)*num_streams, cudaMemcpyHostToDevice));

	for (int i = 0; i < num_streams; i++) {
		vec_d_A[i] = d_A + i * max_m_k_n * max_m_k_n;
		vec_d_B[i] = d_B + i * max_m_k_n * max_m_k_n;
		vec_d_C[i] = d_C + i * max_m_k_n * max_m_k_n;
	}

	int lda, ldb, ldc, m, n, k;
	const float alf = 1.0f;
	const float bet = 0.0f;
	const float *alpha = &alf;
	const float *beta = &bet;

#else

	__half *d_A, *d_B, *d_C;
	checkCuda(cudaMalloc(&d_A, max_m_k_n * max_m_k_n * sizeof(__half)));
	checkCuda(cudaMalloc(&d_B, max_m_k_n * max_m_k_n * sizeof(__half)));
	checkCuda(cudaMalloc(&d_C, max_m_k_n * max_m_k_n * sizeof(__half)));

	for (int i = 0; i < max_m_k_n * max_m_k_n; i++) {
		d_A[i] = approx_float_to_half(h_A[i]);
		d_B[i] = approx_float_to_half(h_B[i]);
		d_C[i] = approx_float_to_half(h_C[i]);
	}

	int lda, ldb, ldc, m, n, k;
	const __half alf = approx_float_to_half(1.0);
	const __half bet = approx_float_to_half(0.0);
	const __half *alpha = &alf;
	const __half *beta = &bet;

#endif

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//for (int size = min_m_k_n; size <= max_m_k_n; size = size * 2) 
	{
		const int size = matrix_size;
		double sum = 0.0;
		
		for (int rep = 0; rep < repeats; rep++) {
			
			m = n = k = matrix_size;
			lda = m;
			ldb = k;
			ldc = m;

			if (repeats - 1 == rep) {
				nvmlAPIRun();
				cudaEventRecord(start, 0);
			}

			for (int i = 0; i < num_streams; i++) {
				vecstat[i] = cublasSgemm(vecHandle[i], CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, vec_d_A[i], lda, vec_d_B[i], ldb, beta, vec_d_C[i], ldc);
			}

//#ifndef FP16MM
//			stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
//#else
//			stat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
//#endif
			cudaDeviceSynchronize();
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);

			if (repeats - 1 == rep) {
				float elapsed;
				cudaEventElapsedTime(&elapsed, start, stop);
				elapsed /= 1000.0f;
				sum = elapsed;
				nvmlAPIEnd();
			}

			for (int i = 0; i < num_streams; i++) {
				auto& stat = vecstat[i];
				if (stat != CUBLAS_STATUS_SUCCESS) {
					cerr << "cublasSgemmBatched failed" << endl;
					exit(1);
				}
				assert(!cudaGetLastError());
			}
			for (int i = 0; i < num_streams; i++) {
				std::swap(vec_d_A[i], vec_d_C[i]);
			}
		}

#ifndef FP16MM	
		cout << "float32: size "
#else
		cout << "float16: size "
#endif
			<< size << " average: " << sum  << " s " << endl;

	}

	checkCuda(cudaMemcpy(h_A, d_A, max_m_k_n * max_m_k_n * sizeof(float)*num_streams, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_B, d_B, max_m_k_n * max_m_k_n * sizeof(float)*num_streams, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_C, d_C, max_m_k_n * max_m_k_n * sizeof(float)*num_streams, cudaMemcpyDeviceToHost));

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

int main(int argc, char ** argv) {
	int num_streams = 1;
	int use_tensor_core = 0;
	int repeats = 1;
	int dType = 0;
	int matrix_size = 1024 * 1;
	if (argc == 6) {
		num_streams = atoi(argv[1]);
		use_tensor_core = atoi(argv[2]);
		repeats = atoi(argv[3]);
		dType = atoi(argv[4]);
		matrix_size = atoi(argv[5]);
	}

	if (num_streams == 1) {
		if (dType == 0)
			single_stream_sgemm(use_tensor_core, matrix_size, repeats);
		else if (dType == 1)
			single_stream_dgemm(use_tensor_core, matrix_size, repeats);
		else if (dType == 2)
			single_stream_hgemm(use_tensor_core, matrix_size, repeats);
	}else
		multi_stream(num_streams, use_tensor_core, matrix_size, repeats);

	return 0;
}