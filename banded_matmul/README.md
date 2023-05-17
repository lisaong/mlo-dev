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
GridDim: 1024, BlockDim: 1, Iterations: 21, FLOPS: 4.46812e+09, GFLOPS: 4.46812
GridDim: 512, BlockDim: 2, Iterations: 38, FLOPS: 8.13318e+09, GFLOPS: 8.13318
GridDim: 341, BlockDim: 3, Iterations: 54, FLOPS: 1.12896e+10, GFLOPS: 11.2896
GridDim: 256, BlockDim: 4, Iterations: 66, FLOPS: 1.40688e+10, GFLOPS: 14.0688
GridDim: 204, BlockDim: 5, Iterations: 67, FLOPS: 1.43721e+10, GFLOPS: 14.3721
GridDim: 170, BlockDim: 6, Iterations: 66, FLOPS: 1.40414e+10, GFLOPS: 14.0414
GridDim: 146, BlockDim: 7, Iterations: 65, FLOPS: 1.36381e+10, GFLOPS: 13.6381
GridDim: 128, BlockDim: 8, Iterations: 84, FLOPS: 1.77256e+10, GFLOPS: 17.7256
GridDim: 113, BlockDim: 9, Iterations: 57, FLOPS: 1.2093e+10, GFLOPS: 12.093
GridDim: 102, BlockDim: 10, Iterations: 63, FLOPS: 1.32877e+10, GFLOPS: 13.2877
GridDim: 93, BlockDim: 11, Iterations: 64, FLOPS: 1.34467e+10, GFLOPS: 13.4467
GridDim: 85, BlockDim: 12, Iterations: 57, FLOPS: 1.18814e+10, GFLOPS: 11.8814
GridDim: 78, BlockDim: 13, Iterations: 57, FLOPS: 1.19221e+10, GFLOPS: 11.9221
GridDim: 73, BlockDim: 14, Iterations: 55, FLOPS: 1.17952e+10, GFLOPS: 11.7952
GridDim: 68, BlockDim: 15, Iterations: 58, FLOPS: 1.22193e+10, GFLOPS: 12.2193
GridDim: 64, BlockDim: 16, Iterations: 58, FLOPS: 1.22419e+10, GFLOPS: 12.2419
GridDim: 60, BlockDim: 17, Iterations: 52, FLOPS: 1.11224e+10, GFLOPS: 11.1224
GridDim: 56, BlockDim: 18, Iterations: 51, FLOPS: 1.09447e+10, GFLOPS: 10.9447
GridDim: 53, BlockDim: 19, Iterations: 51, FLOPS: 1.08172e+10, GFLOPS: 10.8172
GridDim: 51, BlockDim: 20, Iterations: 49, FLOPS: 1.04148e+10, GFLOPS: 10.4148
GridDim: 48, BlockDim: 21, Iterations: 48, FLOPS: 1.01491e+10, GFLOPS: 10.1491
GridDim: 46, BlockDim: 22, Iterations: 48, FLOPS: 1.01476e+10, GFLOPS: 10.1476
GridDim: 44, BlockDim: 23, Iterations: 47, FLOPS: 9.72043e+09, GFLOPS: 9.72043
GridDim: 42, BlockDim: 24, Iterations: 47, FLOPS: 9.8438e+09, GFLOPS: 9.8438
GridDim: 40, BlockDim: 25, Iterations: 46, FLOPS: 9.45908e+09, GFLOPS: 9.45908
GridDim: 39, BlockDim: 26, Iterations: 44, FLOPS: 9.41163e+09, GFLOPS: 9.41163
GridDim: 37, BlockDim: 27, Iterations: 42, FLOPS: 8.96827e+09, GFLOPS: 8.96827
GridDim: 36, BlockDim: 28, Iterations: 44, FLOPS: 9.18653e+09, GFLOPS: 9.18653
GridDim: 35, BlockDim: 29, Iterations: 41, FLOPS: 8.67209e+09, GFLOPS: 8.67209
GridDim: 34, BlockDim: 30, Iterations: 40, FLOPS: 8.51254e+09, GFLOPS: 8.51254
GridDim: 33, BlockDim: 31, Iterations: 39, FLOPS: 7.96174e+09, GFLOPS: 7.96174
GridDim: 32, BlockDim: 32, Iterations: 36, FLOPS: 7.51567e+09, GFLOPS: 7.51567
```

## Loading both T0 and T1 in shared memory

```shell
Using device 0
Values match
GridDim: 1024, BlockDim: 1, FLOPS: 6.27819e+09, GFLOPS: 6.27819
GridDim: 512, BlockDim: 2, FLOPS: 1.31284e+10, GFLOPS: 13.1284
GridDim: 341, BlockDim: 3, FLOPS: 1.94658e+10, GFLOPS: 19.4658
GridDim: 256, BlockDim: 4, FLOPS: 2.51048e+10, GFLOPS: 25.1048
GridDim: 204, BlockDim: 5, FLOPS: 2.93987e+10, GFLOPS: 29.3987
GridDim: 170, BlockDim: 6, FLOPS: 2.36977e+10, GFLOPS: 23.6977
GridDim: 146, BlockDim: 7, FLOPS: 2.48095e+10, GFLOPS: 24.8095
GridDim: 128, BlockDim: 8, FLOPS: 2.60028e+10, GFLOPS: 26.0028
GridDim: 113, BlockDim: 9, FLOPS: 2.2888e+10, GFLOPS: 22.888
GridDim: 102, BlockDim: 10, FLOPS: 2.28008e+10, GFLOPS: 22.8008
GridDim: 93, BlockDim: 11, FLOPS: 2.05235e+10, GFLOPS: 20.5235
GridDim: 85, BlockDim: 12, FLOPS: 2.11752e+10, GFLOPS: 21.1752
GridDim: 78, BlockDim: 13, FLOPS: 1.99365e+10, GFLOPS: 19.9365
GridDim: 73, BlockDim: 14, FLOPS: 2.07937e+10, GFLOPS: 20.7937
GridDim: 68, BlockDim: 15, FLOPS: 2.13597e+10, GFLOPS: 21.3597
GridDim: 64, BlockDim: 16, FLOPS: 1.58893e+10, GFLOPS: 15.8893
GridDim: 60, BlockDim: 17, FLOPS: 2.02486e+10, GFLOPS: 20.2486
GridDim: 56, BlockDim: 18, FLOPS: 1.91084e+10, GFLOPS: 19.1084
GridDim: 53, BlockDim: 19, FLOPS: 1.86555e+10, GFLOPS: 18.6555
GridDim: 51, BlockDim: 20, FLOPS: 1.75671e+10, GFLOPS: 17.5671
GridDim: 48, BlockDim: 21, FLOPS: 1.8088e+10, GFLOPS: 18.088
GridDim: 46, BlockDim: 22, FLOPS: 1.78564e+10, GFLOPS: 17.8564
GridDim: 44, BlockDim: 23, FLOPS: 1.76299e+10, GFLOPS: 17.6299
GridDim: 42, BlockDim: 24, FLOPS: 1.56316e+10, GFLOPS: 15.6316
GridDim: 40, BlockDim: 25, FLOPS: 1.73372e+10, GFLOPS: 17.3372
GridDim: 39, BlockDim: 26, FLOPS: 1.7299e+10, GFLOPS: 17.299
GridDim: 37, BlockDim: 27, FLOPS: 1.6872e+10, GFLOPS: 16.872
GridDim: 36, BlockDim: 28, FLOPS: 1.63622e+10, GFLOPS: 16.3622
GridDim: 35, BlockDim: 29, FLOPS: 1.7155e+10, GFLOPS: 17.155
GridDim: 34, BlockDim: 30, FLOPS: 1.69653e+10, GFLOPS: 16.9653
GridDim: 33, BlockDim: 31, FLOPS: 1.72524e+10, GFLOPS: 17.2524
GridDim: 32, BlockDim: 32, FLOPS: 8.90529e+09, GFLOPS: 8.90529
```


# Profiling

## nsys
Download link (login required): https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2023-2

```shell
sudo apt install libglib2.0-0
sudo dpkg -i NsightSystems-linux-cli-public-2023.2.1.122-3259852.deb

nsys profile --stats=true --force-overwrite true -o bmm_smem ./bmm_smem
nsys analyze bmm_smem.sqlite
```

## NSight Compute CLI

ncu supercedes nvprof

Download link (login required): https://developer.nvidia.com/tools-overview/nsight-compute/get-started
User guide: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvprof-guide

```shell
sudo ./nsight-compute-linux-2023.1.1.4-32678585.run

sudo /usr/local/NVIDIA-Nsight-Compute-2023.1/ncu -f -o profile bmm_smem
```