#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 15000

__global__ void matrix_add(int *a, int *b, int *c)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = i * N + j;
    if (i < N && j < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    size_t size = N * N * sizeof(int);

    // Allocate memory on the host
    h_a = (int*) malloc(size);
    h_b = (int*) malloc(size);
    h_c = (int*) malloc(size);

    // Initialize matrices a and b
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_a[i * N + j] = i + j;
            h_b[i * N + j] = i - j;
        }
    }

    // Allocate memory on the device
    cudaMalloc((void**) &d_a, size);
    cudaMalloc((void**) &d_b, size);
    cudaMalloc((void**) &d_c, size);

    // Copy matrices a and b from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Start the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the kernel to perform matrix addition
    matrix_add<<<gridDim, blockDim>>>(d_a, d_b, d_c);

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time with CUDA: %f seconds\n", milliseconds/ 1000.0f);

    // Copy matrix c from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the results
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (h_c[i * N + j] != h_a[i * N + j] + h_b[i * N + j])
            {
                printf("Error: Incorrect result at (%d,%d)\n", i, j);
                return 1;
            }
        }
    }

    // Free memory on the host and device
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("Matrix addition completed successfully!\n");

    return 0;
}
