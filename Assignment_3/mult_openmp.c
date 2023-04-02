#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/resource.h>

#define N 2000
int A[N][N], B[N][N], C[N][N];

int main() {
    int i, j, k;
    
    // Initialize matrices with random values
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
            C[i][j] = 0;
        }
    }

    double start_time = omp_get_wtime();

    // Multiply matrices A and B
    #pragma omp parallel for private(j, k) shared(A, B, C)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end_time = omp_get_wtime();
    double thread_time = end_time - start_time;

    printf("Execution time with OpenMP: %f seconds\n", thread_time);
    /*
    // Print result
    printf("Result matrix C:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", C[i][j]);
        }
        printf("\n");
    }
    */
    return 0;
}
