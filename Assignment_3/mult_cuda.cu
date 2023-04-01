#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrix_multiply(float *A, float *B, float *C, int M, int N, int K)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < K) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[i*N+k] * B[k*K+j];
        }
        C[i*K+j] = sum;
    }
}

int main()
{
    int M = 1000; // number of rows in A
    int N = 2000; // number of columns in A and rows in B
    int K = 3000; // number of columns in B
    float *A, *B, *C; // matrices
    float *d_A, *d_B, *d_C; // device matrices
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    // allocate memory for host matrices
    A = (float*)malloc(sizeA);
    B = (float*)malloc(sizeB);
    C = (float*)malloc(sizeC);

    // initialize host matrices with random data
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i*N+j] = rand() / (float)RAND_MAX;
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            B[i*K+j] = rand() / (float)RAND_MAX;
        }
    }

    // allocate memory for device matrices
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // copy host matrices to device
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // set grid and block sizes for kernel launch
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // launch kernel to perform matrix multiplication on device
    matrix_multiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // copy result matrix from device to host
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // print result matrix
    printf("Result matrix:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            printf("%f ", C[i*K+j]);
        }
        printf("\n");
    }

    // free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
