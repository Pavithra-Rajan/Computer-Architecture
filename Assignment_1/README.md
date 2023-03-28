# Assignment 1 CS4032D Computer Architecture
<strong>Name:</strong> Pavithra Rajan

<strong>Roll Number:</strong> B190632CS

This folder contains all the necessary files for the submission.
1. ```test.c```: The C program that computes the access time for various values of strides across different cache sizes. The cache size, the stride value and the read access time in ns is stored as ```output.txt```.
2. ```test2.c```: The same C program as above with minor changes to solve the Ex 2.5. he cache size, the stride value and the read access time in ns is stored as ```output2.txt```.
3. ```plot.py```: A python program to plot the stride versus read graph from the file ```output.txt```. The plotted graph will be saved as  ```plot.py```.
4. ```plot2.py```: A python program to plot the log base 2 stride versus log base 2 read graph from the file ```output2.txt```. The plotted graph will be saved as  ```plot2.py```.

To run the C programs:
```console
gcc test.c
```

To run the Python plotting programs:
```console
python3 plot.py
```

