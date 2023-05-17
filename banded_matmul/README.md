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
GridDim: 1024, BlockDim: 1, Iterations: 21, FLOPS: 4.4446e+09, GFLOPS: 4.4446
GridDim: 512, BlockDim: 2, Iterations: 38, FLOPS: 8.02375e+09, GFLOPS: 8.02375
GridDim: 341, BlockDim: 3, Iterations: 53, FLOPS: 1.13308e+10, GFLOPS: 11.3308
GridDim: 256, BlockDim: 4, Iterations: 66, FLOPS: 1.38938e+10, GFLOPS: 13.8938
GridDim: 204, BlockDim: 5, Iterations: 67, FLOPS: 1.41319e+10, GFLOPS: 14.1319
GridDim: 170, BlockDim: 6, Iterations: 66, FLOPS: 1.38136e+10, GFLOPS: 13.8136
GridDim: 146, BlockDim: 7, Iterations: 64, FLOPS: 1.3607e+10, GFLOPS: 13.607
GridDim: 128, BlockDim: 8, Iterations: 83, FLOPS: 1.75569e+10, GFLOPS: 17.5569
GridDim: 113, BlockDim: 9, Iterations: 57, FLOPS: 1.19644e+10, GFLOPS: 11.9644
GridDim: 102, BlockDim: 10, Iterations: 62, FLOPS: 1.32819e+10, GFLOPS: 13.2819
GridDim: 93, BlockDim: 11, Iterations: 63, FLOPS: 1.34197e+10, GFLOPS: 13.4197
GridDim: 85, BlockDim: 12, Iterations: 56, FLOPS: 1.20007e+10, GFLOPS: 12.0007
GridDim: 78, BlockDim: 13, Iterations: 56, FLOPS: 1.20121e+10, GFLOPS: 12.0121
GridDim: 73, BlockDim: 14, Iterations: 55, FLOPS: 1.16347e+10, GFLOPS: 11.6347
GridDim: 68, BlockDim: 15, Iterations: 58, FLOPS: 1.21145e+10, GFLOPS: 12.1145
GridDim: 64, BlockDim: 16, Iterations: 58, FLOPS: 1.21388e+10, GFLOPS: 12.1388
GridDim: 60, BlockDim: 17, Iterations: 52, FLOPS: 1.1003e+10, GFLOPS: 11.003
GridDim: 56, BlockDim: 18, Iterations: 51, FLOPS: 1.08321e+10, GFLOPS: 10.8321
GridDim: 53, BlockDim: 19, Iterations: 51, FLOPS: 1.07423e+10, GFLOPS: 10.7423
GridDim: 51, BlockDim: 20, Iterations: 49, FLOPS: 1.02662e+10, GFLOPS: 10.2662
GridDim: 48, BlockDim: 21, Iterations: 48, FLOPS: 9.99048e+09, GFLOPS: 9.99048
GridDim: 46, BlockDim: 22, Iterations: 48, FLOPS: 9.96348e+09, GFLOPS: 9.96348
GridDim: 44, BlockDim: 23, Iterations: 46, FLOPS: 9.77553e+09, GFLOPS: 9.77553
GridDim: 42, BlockDim: 24, Iterations: 46, FLOPS: 9.87436e+09, GFLOPS: 9.87436
GridDim: 40, BlockDim: 25, Iterations: 45, FLOPS: 9.5015e+09, GFLOPS: 9.5015
GridDim: 39, BlockDim: 26, Iterations: 44, FLOPS: 9.25867e+09, GFLOPS: 9.25867
GridDim: 37, BlockDim: 27, Iterations: 42, FLOPS: 8.79311e+09, GFLOPS: 8.79311
GridDim: 36, BlockDim: 28, Iterations: 44, FLOPS: 9.03963e+09, GFLOPS: 9.03963
GridDim: 35, BlockDim: 29, Iterations: 41, FLOPS: 8.54128e+09, GFLOPS: 8.54128
GridDim: 34, BlockDim: 30, Iterations: 40, FLOPS: 8.37832e+09, GFLOPS: 8.37832
GridDim: 33, BlockDim: 31, Iterations: 38, FLOPS: 8.04553e+09, GFLOPS: 8.04553
GridDim: 32, BlockDim: 32, Iterations: 36, FLOPS: 7.39213e+09, GFLOPS: 7.39213
```

## Loading both T0 and T1 in shared memory

```shell
Using device 0
Values match
GridDim: 1024, BlockDim: 1, FLOPS: 6.28527e+09, GFLOPS: 6.28527
GridDim: 512, BlockDim: 2, FLOPS: 1.30269e+10, GFLOPS: 13.0269
GridDim: 341, BlockDim: 3, FLOPS: 1.95321e+10, GFLOPS: 19.5321
GridDim: 256, BlockDim: 4, FLOPS: 2.50464e+10, GFLOPS: 25.0464
GridDim: 204, BlockDim: 5, FLOPS: 2.90732e+10, GFLOPS: 29.0732
GridDim: 170, BlockDim: 6, FLOPS: 2.35608e+10, GFLOPS: 23.5608
GridDim: 146, BlockDim: 7, FLOPS: 2.46638e+10, GFLOPS: 24.6638
GridDim: 128, BlockDim: 8, FLOPS: 2.58231e+10, GFLOPS: 25.8231
GridDim: 113, BlockDim: 9, FLOPS: 2.27405e+10, GFLOPS: 22.7405
GridDim: 102, BlockDim: 10, FLOPS: 2.26456e+10, GFLOPS: 22.6456
GridDim: 93, BlockDim: 11, FLOPS: 2.0411e+10, GFLOPS: 20.411
GridDim: 85, BlockDim: 12, FLOPS: 2.1061e+10, GFLOPS: 21.061
GridDim: 78, BlockDim: 13, FLOPS: 1.98267e+10, GFLOPS: 19.8267
GridDim: 73, BlockDim: 14, FLOPS: 2.04912e+10, GFLOPS: 20.4912
GridDim: 68, BlockDim: 15, FLOPS: 2.12457e+10, GFLOPS: 21.2457
GridDim: 64, BlockDim: 16, FLOPS: 1.56349e+10, GFLOPS: 15.6349
GridDim: 60, BlockDim: 17, FLOPS: 2.01745e+10, GFLOPS: 20.1745
GridDim: 56, BlockDim: 18, FLOPS: 1.87657e+10, GFLOPS: 18.7657
GridDim: 53, BlockDim: 19, FLOPS: 1.83403e+10, GFLOPS: 18.3403
GridDim: 51, BlockDim: 20, FLOPS: 1.74651e+10, GFLOPS: 17.4651
GridDim: 48, BlockDim: 21, FLOPS: 1.80357e+10, GFLOPS: 18.0357
GridDim: 46, BlockDim: 22, FLOPS: 1.774e+10, GFLOPS: 17.74
GridDim: 44, BlockDim: 23, FLOPS: 1.75419e+10, GFLOPS: 17.5419
GridDim: 42, BlockDim: 24, FLOPS: 1.56089e+10, GFLOPS: 15.6089
GridDim: 40, BlockDim: 25, FLOPS: 1.70866e+10, GFLOPS: 17.0866
GridDim: 39, BlockDim: 26, FLOPS: 1.70595e+10, GFLOPS: 17.0595
GridDim: 37, BlockDim: 27, FLOPS: 1.6799e+10, GFLOPS: 16.799
GridDim: 36, BlockDim: 28, FLOPS: 1.63182e+10, GFLOPS: 16.3182
GridDim: 35, BlockDim: 29, FLOPS: 1.68879e+10, GFLOPS: 16.8879
GridDim: 34, BlockDim: 30, FLOPS: 1.69144e+10, GFLOPS: 16.9144
GridDim: 33, BlockDim: 31, FLOPS: 1.71906e+10, GFLOPS: 17.1906
GridDim: 32, BlockDim: 32, FLOPS: 8.96925e+09, GFLOPS: 8.96925
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

sudo /usr/local/NVIDIA-Nsight-Compute-2023.1/ncu -o profile bmm_smem
```