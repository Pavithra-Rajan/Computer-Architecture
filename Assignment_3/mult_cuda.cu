#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrix_multiply(float *A, float *B, float *C, int ROWS, int COLS)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ROWS && j < COLS)
    {
        float sum = 0.0f;
        for (int k = 0; k < COLS; ++k)
        {
            sum += A[i * COLS + k] * B[k * COLS + j];
        }
        C[i * COLS + j] = sum;
    }
}

int main()
{
    int ROWS = 2000; // number of rows
    int COLS = 2000; // number of columns 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *A, *B, *C;       // matrices
    float *d_A, *d_B, *d_C; // device matrices
    size_t sizeA = ROWS * COLS * sizeof(float);
    size_t sizeB = COLS * COLS * sizeof(float);
    size_t sizeC = ROWS * COLS * sizeof(float);

    // allocate memory for host matrices
    A = (float *)malloc(sizeA);
    B = (float *)malloc(sizeB);
    C = (float *)malloc(sizeC);

    // initialize host matrices with random data
    for (int i = 0; i < ROWS; ++i)
    {
        for (int j = 0; j < COLS; ++j)
        {
            A[i * COLS + j] = rand() / (float)RAND_MAX;
        }
    }
    for (int i = 0; i < COLS; ++i)
    {
        for (int j = 0; j < COLS; ++j)
        {
            B[i * COLS + j] = rand() / (float)RAND_MAX;
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
    dim3 numBlocks((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // launch kernel to perform matrix multiplication on device
    cudaEventRecord(start);
    matrix_multiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time with CUDA: %f seconds\n", milliseconds / 1000.0f);

    // copy result matrix from device to host
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    /*
    // print result matrix
    printf("Result matrix:\n");
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < K; ++j) {
            printf("%f ", C[i*K+j]);
        }
        printf("\n");
    }*/

    // free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
