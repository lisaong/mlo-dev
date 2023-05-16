# Banded Matrix Multiplication in CUDA
Create a function that implements matrix multiplication between a banded matrix and a dense matrix.

Matrix multiplication is often shown as `T0 += T1 * T2`, where T0, T1 and T2 are dense matrices. This is almost identical, except that the A matrix in this case, is a banded matrix.

This can be described as follows:

![banded_matmul](img/banded_matmul.png)

* n0 is the number of rows in T0 and T1
* n1 is the number of columns in T0 and T2
* n2 is the inner or shared dimension, i.e. number of columns in T1 and number of rows in T2

The matrix T1 is a banded matrix, a type of sparse matrix. It’s often given the form T1 on the left to save space in RAM, but logically represents Ť1 on the right.

(See [Band matrix - Wikipedia]() if you want more info. Banded matrices have interesting mathematical properties that make them very useful in special situations. But we won’t get into that here).

## Requirements

### Create a program that implements the following:

Matrix multiplication between a banded matrix and a dense matrix. The only size to consider is when T0 is 1024 rows by 1024 columns (i.e. n0 == n1 == 1024.).

### Measure your performance:

Run your function until 10 sec has passed, then count the number of iterations.

Print your performance in FLOPS, calculated by:

FLOPS = NumberOfIterations x NumberOfOps / ElapsedTimeInSec

Let’s assume the dense theorical number of ops, which would make NumberOfOps = 1024 x 1024 x 1024 x 2.
Use 32-bit floating point data types for the input and output matrices.

In mathematics, particularly matrix theory, a band matrix or banded matrix is a sparse matrix whose non-zero entries are confined to a diagonal band, comprising the main diagonal and zero or more diagonals.

# Results

GPU: NVIDIA A6000

## Naive

```shell
Using device 0
Values match
Blocksize: 16, Iterations: 56, FLOPS: 1.19722e+10, GFLOPS: 11.9722
Blocksize: 24, Iterations: 47, FLOPS: 9.88634e+09, GFLOPS: 9.88634
Blocksize: 32, Iterations: 36, FLOPS: 7.54052e+09, GFLOPS: 7.54052
Skipping Blocksize: 40, invalid configuration argument
Skipping Blocksize: 48, invalid configuration argument
Skipping Blocksize: 56, invalid configuration argument
Skipping Blocksize: 64, invalid configuration argument
Skipping Blocksize: 72, invalid configuration argument
Skipping Blocksize: 80, invalid configuration argument
Skipping Blocksize: 88, invalid configuration argument
Skipping Blocksize: 96, invalid configuration argument
Skipping Blocksize: 104, invalid configuration argument
Skipping Blocksize: 112, invalid configuration argument
Skipping Blocksize: 120, invalid configuration argument
Skipping Blocksize: 128, invalid configuration argument
```

## Loading T1 in shared memory

```shell
Using device 0
Values match
Blocksize: 16, Iterations: 66, FLOPS: 1.40425e+10, GFLOPS: 14.0425
Blocksize: 24, Iterations: 53, FLOPS: 1.11449e+10, GFLOPS: 11.1449
Blocksize: 32, Iterations: 37, FLOPS: 7.58615e+09, GFLOPS: 7.58615
Skipping Blocksize: 40, invalid configuration argument
Skipping Blocksize: 48, invalid configuration argument
Skipping Blocksize: 56, invalid configuration argument
Skipping Blocksize: 64, invalid configuration argument
Skipping Blocksize: 72, invalid configuration argument
Skipping Blocksize: 80, invalid configuration argument
Skipping Blocksize: 88, invalid configuration argument
Skipping Blocksize: 96, invalid configuration argument
Skipping Blocksize: 104, invalid configuration argument
Skipping Blocksize: 112, invalid configuration argument
Skipping Blocksize: 120, invalid configuration argument
Skipping Blocksize: 128, invalid configuration argument
```

## Loading both T0 and T1 in shared memory

```shell
Using device 0
Values match
Blocksize: 16, Iterations: 72, FLOPS: 1.52138e+10, GFLOPS: 15.2138
Blocksize: 24, Iterations: 74, FLOPS: 1.57493e+10, GFLOPS: 15.7493
Blocksize: 32, Iterations: 43, FLOPS: 8.97539e+09, GFLOPS: 8.97539
Skipping Blocksize: 40, invalid configuration argument
Skipping Blocksize: 48, invalid configuration argument
Skipping Blocksize: 56, invalid configuration argument
Skipping Blocksize: 64, invalid configuration argument
Skipping Blocksize: 72, invalid configuration argument
Skipping Blocksize: 80, invalid configuration argument
Skipping Blocksize: 88, invalid configuration argument
Skipping Blocksize: 96, invalid configuration argument
Skipping Blocksize: 104, invalid configuration argument
Skipping Blocksize: 112, invalid configuration argument
Skipping Blocksize: 120, invalid configuration argument
Skipping Blocksize: 128, invalid configuration argument
```


# Profiling

## Install nsys
Download link (login required): https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2023-2

```shell
sudo apt install libglib2.0-0
sudo dpkg -i NsightSystems-linux-cli-public-2023.2.1.122-3259852.deb
```

## Run nsys

```shell
nsys profile --stats=true --force-overwrite true -o bmm ./bmm 3
nsys analyze bmm.sqlite
```