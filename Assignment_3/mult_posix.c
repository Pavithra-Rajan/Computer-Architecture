#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define ROWS 2000
#define COLS 2000
#define THREADS 4

int mat1[ROWS][COLS];
int mat2[ROWS][COLS];
int result[ROWS][COLS];

struct thread_arg {
    int start_row;
    int end_row;
};

void *multiply(void *arg) {
    struct thread_arg *t_arg = (struct thread_arg *) arg;

    for (int i = t_arg->start_row; i < t_arg->end_row; i++) {
        for (int j = 0; j < COLS; j++) {
            int sum = 0;
            for (int k = 0; k < ROWS; k++) {
                sum += mat1[i][k] * mat2[k][j];
            }
            result[i][j] = sum;
        }
    }

    pthread_exit(NULL);
}

int main() {
    pthread_t threads[THREADS];
    struct thread_arg t_args[THREADS];

    clock_t start_time, end_time;
    double thread_time;

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            mat1[i][j] = rand() % 10;
            mat2[i][j] = rand() % 10;
        }
    }

    start_time = clock();

    int block_size = ROWS / THREADS;
    for (int i = 0; i < THREADS; i++) {
        t_args[i].start_row = i * block_size;
        t_args[i].end_row = (i + 1) * block_size;
        if (i == THREADS - 1) {
            t_args[i].end_row = ROWS;
        }
        pthread_create(&threads[i], NULL, multiply, (void *) &t_args[i]);
    }

    for (int i = 0; i < THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    end_time = clock();
    thread_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Execution time with POSIX threads: %f seconds\n", thread_time);

    return 0;
}
