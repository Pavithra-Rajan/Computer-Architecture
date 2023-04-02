#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define N 5
#define THREADS 4
int a[N][N], b[N][N], c[N][N];

struct thread_data
{
    int row_start;
    int row_end;
};

void *matrix_addition(void *arg)
{
    struct thread_data *data = (struct thread_data *)arg;
    int i, j;
    for (i = data->row_start; i < data->row_end; i++)
    {
        for (j = 0; j < N; j++)
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    pthread_exit(NULL);
}

int main()
{
    int i, j;
    pthread_t threads[THREADS];
    struct thread_data thread_args[THREADS];
    clock_t start_time, end_time;
    double no_thread_time, thread_time;

    // Initialize matrices
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a[i][j] = rand() % 100;
            b[i][j] = rand() % 100;
        }
    }

    start_time = clock();
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    end_time = clock();
    no_thread_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Execution time without threads: %f seconds\n", no_thread_time);

    // Create threads and distribute work
    start_time = clock();
    for (i = 0; i < THREADS; i++)
    {
        thread_args[i].row_start = i * N / THREADS;
        thread_args[i].row_end = (i + 1) * N / THREADS;
        pthread_create(&threads[i], NULL, matrix_addition, (void *)&thread_args[i]);
    }

    // Wait for threads to finish
    for (i = 0; i < THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    end_time = clock();
    thread_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Execution time with threads: %f seconds\n", thread_time);
    /*
    for (i = 0; i < N; i++) 
    {
        for (j = 0; j < N; j++) 
        {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }
    */
    double speedup = no_thread_time / thread_time;
    printf("Speedup: %f\n", speedup);

    return 0;
}
