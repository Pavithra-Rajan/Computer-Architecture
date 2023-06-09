#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/resource.h>

#define N 15000

int matrix1[N][N];
int matrix2[N][N];
int result[N][N];

int main()
{

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            matrix1[i][j] = i + j;
            matrix2[i][j] = i * j;
        }
    }

    double start_time = omp_get_wtime();

    int i, j;
#pragma omp parallel for shared(matrix1, matrix2, result) private(i, j)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    double end_time = omp_get_wtime();
    double thread_time = end_time - start_time;

    printf("Execution time with threads: %f seconds\n", thread_time);
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("Memory usage: %f MB\n", (float) usage.ru_maxrss/(1024*1024));

    start_time = omp_get_wtime();

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    end_time = omp_get_wtime();
    double no_thread_time = end_time - start_time;

    printf("Execution time without threads: %f seconds\n", no_thread_time);

    double speedup = no_thread_time / thread_time;
    printf("Speedup: %f\n", speedup);

    return 0;
}
