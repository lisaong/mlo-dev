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
Blocksize: 1, Iterations: 21, FLOPS: 4.50047e+09, GFLOPS: 4.50047
Blocksize: 2, Iterations: 38, FLOPS: 8.06564e+09, GFLOPS: 8.06564
Blocksize: 3, Iterations: 54, FLOPS: 1.11806e+10, GFLOPS: 11.1806
Blocksize: 4, Iterations: 66, FLOPS: 1.39674e+10, GFLOPS: 13.9674
Blocksize: 5, Iterations: 67, FLOPS: 1.42881e+10, GFLOPS: 14.2881
Blocksize: 6, Iterations: 66, FLOPS: 1.41225e+10, GFLOPS: 14.1225
Blocksize: 7, Iterations: 65, FLOPS: 1.3676e+10, GFLOPS: 13.676
Blocksize: 8, Iterations: 84, FLOPS: 1.76639e+10, GFLOPS: 17.6639
Blocksize: 9, Iterations: 57, FLOPS: 1.21648e+10, GFLOPS: 12.1648
Blocksize: 10, Iterations: 63, FLOPS: 1.34411e+10, GFLOPS: 13.4411
Blocksize: 11, Iterations: 64, FLOPS: 1.36352e+10, GFLOPS: 13.6352
Blocksize: 12, Iterations: 57, FLOPS: 1.21009e+10, GFLOPS: 12.1009
Blocksize: 13, Iterations: 57, FLOPS: 1.2082e+10, GFLOPS: 12.082
Blocksize: 14, Iterations: 56, FLOPS: 1.16946e+10, GFLOPS: 11.6946
Blocksize: 15, Iterations: 58, FLOPS: 1.23224e+10, GFLOPS: 12.3224
Blocksize: 16, Iterations: 58, FLOPS: 1.23393e+10, GFLOPS: 12.3393
Blocksize: 17, Iterations: 53, FLOPS: 1.09683e+10, GFLOPS: 10.9683
Blocksize: 18, Iterations: 52, FLOPS: 1.08051e+10, GFLOPS: 10.8051
Blocksize: 19, Iterations: 51, FLOPS: 1.0918e+10, GFLOPS: 10.918
Blocksize: 20, Iterations: 49, FLOPS: 1.04837e+10, GFLOPS: 10.4837
Blocksize: 21, Iterations: 48, FLOPS: 1.02256e+10, GFLOPS: 10.2256
Blocksize: 22, Iterations: 48, FLOPS: 1.01988e+10, GFLOPS: 10.1988
Blocksize: 23, Iterations: 47, FLOPS: 9.76345e+09, GFLOPS: 9.76345
Blocksize: 24, Iterations: 47, FLOPS: 9.89697e+09, GFLOPS: 9.89697
Blocksize: 25, Iterations: 46, FLOPS: 9.51334e+09, GFLOPS: 9.51334
Blocksize: 26, Iterations: 44, FLOPS: 9.44853e+09, GFLOPS: 9.44853
Blocksize: 27, Iterations: 42, FLOPS: 9.00172e+09, GFLOPS: 9.00172
Blocksize: 28, Iterations: 44, FLOPS: 9.23293e+09, GFLOPS: 9.23293
Blocksize: 29, Iterations: 41, FLOPS: 8.74093e+09, GFLOPS: 8.74093
Blocksize: 30, Iterations: 40, FLOPS: 8.56835e+09, GFLOPS: 8.56835
Blocksize: 31, Iterations: 39, FLOPS: 7.98696e+09, GFLOPS: 7.98696
Blocksize: 32, Iterations: 36, FLOPS: 7.53703e+09, GFLOPS: 7.53703
Skipping Blocksize: 33, invalid configuration argument
Skipping Blocksize: 34, invalid configuration argument
Skipping Blocksize: 35, invalid configuration argument
Skipping Blocksize: 36, invalid configuration argument
Skipping Blocksize: 37, invalid configuration argument
Skipping Blocksize: 38, invalid configuration argument
Skipping Blocksize: 39, invalid configuration argument
Skipping Blocksize: 40, invalid configuration argument
```

## Loading both T0 and T1 in shared memory

```shell
Using device 0
Values match
Blocksize: 1, Iterations: 31, FLOPS: 6.37505e+09, GFLOPS: 6.37505
Blocksize: 2, Iterations: 62, FLOPS: 1.32159e+10, GFLOPS: 13.2159
Blocksize: 3, Iterations: 92, FLOPS: 1.95945e+10, GFLOPS: 19.5945
Blocksize: 4, Iterations: 119, FLOPS: 2.53178e+10, GFLOPS: 25.3178
Blocksize: 5, Iterations: 139, FLOPS: 2.94961e+10, GFLOPS: 29.4961
Blocksize: 6, Iterations: 112, FLOPS: 2.38121e+10, GFLOPS: 23.8121
Blocksize: 7, Iterations: 117, FLOPS: 2.48224e+10, GFLOPS: 24.8224
Blocksize: 8, Iterations: 123, FLOPS: 2.60096e+10, GFLOPS: 26.0096
Blocksize: 9, Iterations: 107, FLOPS: 2.28657e+10, GFLOPS: 22.8657
Blocksize: 10, Iterations: 108, FLOPS: 2.27974e+10, GFLOPS: 22.7974
Blocksize: 11, Iterations: 97, FLOPS: 2.0563e+10, GFLOPS: 20.563
Blocksize: 12, Iterations: 100, FLOPS: 2.11867e+10, GFLOPS: 21.1867
Blocksize: 13, Iterations: 94, FLOPS: 1.99625e+10, GFLOPS: 19.9625
Blocksize: 14, Iterations: 97, FLOPS: 2.0792e+10, GFLOPS: 20.792
Blocksize: 15, Iterations: 100, FLOPS: 2.13562e+10, GFLOPS: 21.3562
Blocksize: 16, Iterations: 74, FLOPS: 1.58194e+10, GFLOPS: 15.8194
Blocksize: 17, Iterations: 95, FLOPS: 2.02418e+10, GFLOPS: 20.2418
Blocksize: 18, Iterations: 89, FLOPS: 1.90742e+10, GFLOPS: 19.0742
Blocksize: 19, Iterations: 87, FLOPS: 1.8631e+10, GFLOPS: 18.631
Blocksize: 20, Iterations: 83, FLOPS: 1.75005e+10, GFLOPS: 17.5005
Blocksize: 21, Iterations: 85, FLOPS: 1.80881e+10, GFLOPS: 18.0881
Blocksize: 22, Iterations: 84, FLOPS: 1.78631e+10, GFLOPS: 17.8631
Blocksize: 23, Iterations: 83, FLOPS: 1.76517e+10, GFLOPS: 17.6517
Blocksize: 24, Iterations: 74, FLOPS: 1.55824e+10, GFLOPS: 15.5824
Blocksize: 25, Iterations: 81, FLOPS: 1.73278e+10, GFLOPS: 17.3278
Blocksize: 26, Iterations: 81, FLOPS: 1.73323e+10, GFLOPS: 17.3323
Blocksize: 27, Iterations: 80, FLOPS: 1.68716e+10, GFLOPS: 16.8716
Blocksize: 28, Iterations: 77, FLOPS: 1.636e+10, GFLOPS: 16.36
Blocksize: 29, Iterations: 80, FLOPS: 1.71538e+10, GFLOPS: 17.1538
Blocksize: 30, Iterations: 80, FLOPS: 1.69657e+10, GFLOPS: 16.9657
Blocksize: 31, Iterations: 82, FLOPS: 1.72482e+10, GFLOPS: 17.2482
Blocksize: 32, Iterations: 43, FLOPS: 8.9052e+09, GFLOPS: 8.9052
Skipping Blocksize: 33, invalid configuration argument
Skipping Blocksize: 34, invalid configuration argument
Skipping Blocksize: 35, invalid configuration argument
Skipping Blocksize: 36, invalid configuration argument
Skipping Blocksize: 37, invalid configuration argument
Skipping Blocksize: 38, invalid configuration argument
Skipping Blocksize: 39, invalid configuration argument
Skipping Blocksize: 40, invalid configuration argument
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