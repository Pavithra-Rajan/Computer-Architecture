# Computer-Architecture
This repository contains all the source files and resources for CS4032D Computer Architecture, NITC

There are 3 assignments that were organised for this course which are as follows:
## Assignment 1
1. Using the program results after running the cache analysis program. determine
- What is the overall size and block size of the second-level cache?
- What is the miss penalty of the second-level cache?
- What is the associativity of the second-level cache?
- What is the size of the main memory?
- What is the paging time if the page size is 4 KB?

2. If necessary, modify the code to measure the following system characteristics. Plot the experimental results with elapsed time on the y-axis and the memory stride on the x-axis. Use logarithmic scales for both axes, and draw a line for each cache size.

- What is the system page size?
- How many entries are there in the TLB?
- What is the miss penalty for the TLB?
- What is the associativity of the TLB?

These are Q 2.4 and Q 2.5 from ```Computer Architecture: A Quantitative Approach
Book by David A Patterson and John L. Hennessy Edition 3```

## Assignment 2 (Group)
The assignment entails comparing the performance of cache based on hit time, miss rate and average memory access time by varying the cache size of L1 and L2.
L1 = 16 KB, 32 KB, 64 KB
L2 = 2 MB, 4 MB, 8 MB

## Assignment 3 (Group)
1. Performance comparison of pthread and openMP for a parallelizable problem
2. Compare the performance of pthread, OpenMP and CUDA for two different problems out of which one is purely vectorizable (DLP) and the other is parallelizable (TLP)

## Directory Structure
```console
.
├── Assignment_1
│   ├── plot2.png
│   ├── plot2.py
│   ├── plot.png
│   ├── plot.py
│   ├── README.md
│   ├── README.pdf
│   ├── Report.pdf
│   ├── test2.c
│   └── test.c
├── Assignment_2
│   ├── FIGURES
│   │   ├── l1_vs_l1hit.png
│   │   ├── l2_vs_AMAT_l2.png
│   │   ├── l2_vs_l2hit.png
│   │   ├── l2_vs_l2miss.png
│   │   ├── table1.png
│   │   └── table2.png
│   ├── logs.txt
│   ├── plot.py
│   ├── README.md
│   ├── README.pdf
│   ├── RESULTS
│   │   ├── 16KB_2MB.txt
│   │   ├── 16KB_4MB.txt
│   │   ├── 16KB_8MB.txt
│   │   ├── 32KB_2MB.txt
│   │   ├── 32KB_4MB.txt
│   │   ├── 32KB_8MB.txt
│   │   ├── 64KB_2MB.txt
│   │   ├── 64KB_4MB.txt
│   │   └── 64KB_8MB.txt
│   └── SPEC
│       ├── 401.bzip2
│       ├── 429.mcf
│       ├── 456.hmmer
│       ├── 458.sjeng
│       ├── 470.lbm
│       ├── m5out
│       ├── mcf.out
│       └── runGem5.sh
├── Assignment_3
│   ├── matrix_add_cuda.cu
│   ├── matrix_add_openmp.c
│   ├── matrix_add_posix.c
│   ├── mult_cuda.cu
│   ├── mult_openmp.c
│   ├── mult_posix.c
│   ├── README.md
│   └── README.pdf
└── README.md
```