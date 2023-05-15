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
sh run.sh 

Using device 3
Blocksize: 16, Iterations: 516, FLOPS: 1.10554e+11, GFLOPS: 110.554
Blocksize: 32, Iterations: 413, FLOPS: 8.86346e+10, GFLOPS: 88.6346
Blocksize: 48, Iterations: 1415, FLOPS: 3.03762e+11, GFLOPS: 303.762
Blocksize: 64, Iterations: 1418, FLOPS: 3.04093e+11, GFLOPS: 304.093
Blocksize: 80, Iterations: 1419, FLOPS: 3.04315e+11, GFLOPS: 304.315
Blocksize: 96, Iterations: 1421, FLOPS: 3.04827e+11, GFLOPS: 304.827
Blocksize: 112, Iterations: 1419, FLOPS: 3.04482e+11, GFLOPS: 304.482
Blocksize: 128, Iterations: 1417, FLOPS: 3.04223e+11, GFLOPS: 304.223
Blocksize: 144, Iterations: 1420, FLOPS: 3.04768e+11, GFLOPS: 304.768
Blocksize: 160, Iterations: 1418, FLOPS: 3.04192e+11, GFLOPS: 304.192
Blocksize: 176, Iterations: 1419, FLOPS: 3.0446e+11, GFLOPS: 304.46
Blocksize: 192, Iterations: 1417, FLOPS: 3.04202e+11, GFLOPS: 304.202
Blocksize: 208, Iterations: 1419, FLOPS: 3.04672e+11, GFLOPS: 304.672
Blocksize: 224, Iterations: 1419, FLOPS: 3.04457e+11, GFLOPS: 304.457
Blocksize: 240, Iterations: 1418, FLOPS: 3.04125e+11, GFLOPS: 304.125
Blocksize: 256, Iterations: 1417, FLOPS: 3.04167e+11, GFLOPS: 304.167
Blocksize: 272, Iterations: 1419, FLOPS: 3.04413e+11, GFLOPS: 304.413
Blocksize: 288, Iterations: 1418, FLOPS: 3.04479e+11, GFLOPS: 304.479
Blocksize: 304, Iterations: 1418, FLOPS: 3.04312e+11, GFLOPS: 304.312
Blocksize: 320, Iterations: 1420, FLOPS: 3.04585e+11, GFLOPS: 304.585
Blocksize: 336, Iterations: 1421, FLOPS: 3.04841e+11, GFLOPS: 304.841
Blocksize: 352, Iterations: 1420, FLOPS: 3.04919e+11, GFLOPS: 304.919
Blocksize: 368, Iterations: 1419, FLOPS: 3.04547e+11, GFLOPS: 304.547
Blocksize: 384, Iterations: 1421, FLOPS: 3.0505e+11, GFLOPS: 305.05
Blocksize: 400, Iterations: 1421, FLOPS: 3.04842e+11, GFLOPS: 304.842
Blocksize: 416, Iterations: 1421, FLOPS: 3.05064e+11, GFLOPS: 305.064
Blocksize: 432, Iterations: 1421, FLOPS: 3.05114e+11, GFLOPS: 305.114
Blocksize: 448, Iterations: 1421, FLOPS: 3.0502e+11, GFLOPS: 305.02
Blocksize: 464, Iterations: 1421, FLOPS: 3.04965e+11, GFLOPS: 304.965
Blocksize: 480, Iterations: 1420, FLOPS: 3.04773e+11, GFLOPS: 304.773
Blocksize: 496, Iterations: 1420, FLOPS: 3.04561e+11, GFLOPS: 304.561
Blocksize: 512, Iterations: 1426, FLOPS: 3.0609e+11, GFLOPS: 306.09
Blocksize: 528, Iterations: 1424, FLOPS: 3.05432e+11, GFLOPS: 305.432
Blocksize: 544, Iterations: 1425, FLOPS: 3.05658e+11, GFLOPS: 305.658
Blocksize: 560, Iterations: 1422, FLOPS: 3.05264e+11, GFLOPS: 305.264
Blocksize: 576, Iterations: 1423, FLOPS: 3.05304e+11, GFLOPS: 305.304
Blocksize: 592, Iterations: 1422, FLOPS: 3.05337e+11, GFLOPS: 305.337
Blocksize: 608, Iterations: 1424, FLOPS: 3.05479e+11, GFLOPS: 305.479
Blocksize: 624, Iterations: 1423, FLOPS: 3.05552e+11, GFLOPS: 305.552
Blocksize: 640, Iterations: 1423, FLOPS: 3.0555e+11, GFLOPS: 305.55
Blocksize: 656, Iterations: 1423, FLOPS: 3.05424e+11, GFLOPS: 305.424
Blocksize: 672, Iterations: 1421, FLOPS: 3.05039e+11, GFLOPS: 305.039
Blocksize: 688, Iterations: 1422, FLOPS: 3.05108e+11, GFLOPS: 305.108
Blocksize: 704, Iterations: 1422, FLOPS: 3.05089e+11, GFLOPS: 305.089
Blocksize: 720, Iterations: 1420, FLOPS: 3.04849e+11, GFLOPS: 304.849
Blocksize: 736, Iterations: 1423, FLOPS: 3.05294e+11, GFLOPS: 305.294
Blocksize: 752, Iterations: 1423, FLOPS: 3.05368e+11, GFLOPS: 305.368
Blocksize: 768, Iterations: 1422, FLOPS: 3.05109e+11, GFLOPS: 305.109
Blocksize: 784, Iterations: 1421, FLOPS: 3.05082e+11, GFLOPS: 305.082
Blocksize: 800, Iterations: 1421, FLOPS: 3.05037e+11, GFLOPS: 305.037
Blocksize: 816, Iterations: 1421, FLOPS: 3.04747e+11, GFLOPS: 304.747
Blocksize: 832, Iterations: 1421, FLOPS: 3.04998e+11, GFLOPS: 304.998
Blocksize: 848, Iterations: 1418, FLOPS: 3.04238e+11, GFLOPS: 304.238
Blocksize: 864, Iterations: 1421, FLOPS: 3.04779e+11, GFLOPS: 304.779
Blocksize: 880, Iterations: 1420, FLOPS: 3.04582e+11, GFLOPS: 304.582
Blocksize: 896, Iterations: 1418, FLOPS: 3.04403e+11, GFLOPS: 304.403
Blocksize: 912, Iterations: 1420, FLOPS: 3.04605e+11, GFLOPS: 304.605
Blocksize: 928, Iterations: 1419, FLOPS: 3.04657e+11, GFLOPS: 304.657
Blocksize: 944, Iterations: 1419, FLOPS: 3.04535e+11, GFLOPS: 304.535
Blocksize: 960, Iterations: 1420, FLOPS: 3.04878e+11, GFLOPS: 304.878
Blocksize: 976, Iterations: 1418, FLOPS: 3.04364e+11, GFLOPS: 304.364
Blocksize: 992, Iterations: 1421, FLOPS: 3.05141e+11, GFLOPS: 305.141
Blocksize: 1008, Iterations: 1421, FLOPS: 3.05136e+11, GFLOPS: 305.136
Blocksize: 1024, Iterations: 1420, FLOPS: 3.04889e+11, GFLOPS: 304.889
```

## PackedB

```shell
Blocksize: 16, Iterations: 639, FLOPS: 1.37127e+11, GFLOPS: 137.127
Blocksize: 32, Iterations: 425, FLOPS: 9.08455e+10, GFLOPS: 90.8455
Blocksize: 48, Iterations: 1416, FLOPS: 3.03892e+11, GFLOPS: 303.892
Blocksize: 64, Iterations: 1417, FLOPS: 3.04165e+11, GFLOPS: 304.165
Blocksize: 80, Iterations: 1418, FLOPS: 3.04101e+11, GFLOPS: 304.101
Blocksize: 96, Iterations: 1419, FLOPS: 3.04469e+11, GFLOPS: 304.469
Blocksize: 112, Iterations: 1418, FLOPS: 3.04146e+11, GFLOPS: 304.146
Blocksize: 128, Iterations: 1417, FLOPS: 3.03931e+11, GFLOPS: 303.931
Blocksize: 144, Iterations: 1419, FLOPS: 3.04453e+11, GFLOPS: 304.453
Blocksize: 160, Iterations: 1420, FLOPS: 3.04601e+11, GFLOPS: 304.601
Blocksize: 176, Iterations: 1418, FLOPS: 3.04182e+11, GFLOPS: 304.182
Blocksize: 192, Iterations: 1419, FLOPS: 3.04413e+11, GFLOPS: 304.413
Blocksize: 208, Iterations: 1420, FLOPS: 3.0471e+11, GFLOPS: 304.71
Blocksize: 224, Iterations: 1418, FLOPS: 3.04245e+11, GFLOPS: 304.245
Blocksize: 240, Iterations: 1415, FLOPS: 3.0379e+11, GFLOPS: 303.79
Blocksize: 256, Iterations: 1421, FLOPS: 3.05081e+11, GFLOPS: 305.081
Blocksize: 272, Iterations: 1420, FLOPS: 3.04893e+11, GFLOPS: 304.893
Blocksize: 288, Iterations: 1420, FLOPS: 3.04916e+11, GFLOPS: 304.916
Blocksize: 304, Iterations: 1421, FLOPS: 3.0489e+11, GFLOPS: 304.89
Blocksize: 320, Iterations: 1420, FLOPS: 3.04589e+11, GFLOPS: 304.589
Blocksize: 336, Iterations: 1421, FLOPS: 3.04822e+11, GFLOPS: 304.822
Blocksize: 352, Iterations: 1422, FLOPS: 3.05016e+11, GFLOPS: 305.016
Blocksize: 368, Iterations: 1420, FLOPS: 3.04543e+11, GFLOPS: 304.543
Blocksize: 384, Iterations: 1421, FLOPS: 3.05114e+11, GFLOPS: 305.114
Blocksize: 400, Iterations: 1420, FLOPS: 3.04536e+11, GFLOPS: 304.536
Blocksize: 416, Iterations: 1421, FLOPS: 3.04876e+11, GFLOPS: 304.876
Blocksize: 432, Iterations: 1419, FLOPS: 3.04685e+11, GFLOPS: 304.685
Blocksize: 448, Iterations: 1421, FLOPS: 3.04851e+11, GFLOPS: 304.851
Blocksize: 464, Iterations: 1419, FLOPS: 3.04637e+11, GFLOPS: 304.637
Blocksize: 480, Iterations: 1421, FLOPS: 3.04741e+11, GFLOPS: 304.741
Blocksize: 496, Iterations: 1419, FLOPS: 3.04667e+11, GFLOPS: 304.667
Blocksize: 512, Iterations: 1421, FLOPS: 3.04746e+11, GFLOPS: 304.746
Blocksize: 528, Iterations: 1422, FLOPS: 3.05044e+11, GFLOPS: 305.044
Blocksize: 544, Iterations: 1418, FLOPS: 3.04156e+11, GFLOPS: 304.156
Blocksize: 560, Iterations: 1420, FLOPS: 3.04756e+11, GFLOPS: 304.756
Blocksize: 576, Iterations: 1420, FLOPS: 3.0489e+11, GFLOPS: 304.89
Blocksize: 592, Iterations: 1421, FLOPS: 3.05029e+11, GFLOPS: 305.029
Blocksize: 608, Iterations: 1418, FLOPS: 3.0444e+11, GFLOPS: 304.44
Blocksize: 624, Iterations: 1417, FLOPS: 3.04199e+11, GFLOPS: 304.199
Blocksize: 640, Iterations: 1420, FLOPS: 3.04559e+11, GFLOPS: 304.559
Blocksize: 656, Iterations: 1420, FLOPS: 3.04787e+11, GFLOPS: 304.787
Blocksize: 672, Iterations: 1420, FLOPS: 3.04676e+11, GFLOPS: 304.676
Blocksize: 688, Iterations: 1419, FLOPS: 3.04421e+11, GFLOPS: 304.421
Blocksize: 704, Iterations: 1420, FLOPS: 3.0472e+11, GFLOPS: 304.72
Blocksize: 720, Iterations: 1420, FLOPS: 3.04859e+11, GFLOPS: 304.859
Blocksize: 736, Iterations: 1420, FLOPS: 3.04775e+11, GFLOPS: 304.775
Blocksize: 752, Iterations: 1420, FLOPS: 3.04791e+11, GFLOPS: 304.791
Blocksize: 768, Iterations: 1420, FLOPS: 3.04711e+11, GFLOPS: 304.711
Blocksize: 784, Iterations: 1420, FLOPS: 3.04803e+11, GFLOPS: 304.803
Blocksize: 800, Iterations: 1418, FLOPS: 3.04479e+11, GFLOPS: 304.479
Blocksize: 816, Iterations: 1419, FLOPS: 3.04443e+11, GFLOPS: 304.443
Blocksize: 832, Iterations: 1419, FLOPS: 3.04383e+11, GFLOPS: 304.383
Blocksize: 848, Iterations: 1421, FLOPS: 3.0497e+11, GFLOPS: 304.97
Blocksize: 864, Iterations: 1418, FLOPS: 3.04327e+11, GFLOPS: 304.327
Blocksize: 880, Iterations: 1419, FLOPS: 3.0449e+11, GFLOPS: 304.49
Blocksize: 896, Iterations: 1419, FLOPS: 3.04574e+11, GFLOPS: 304.574
Blocksize: 912, Iterations: 1421, FLOPS: 3.04795e+11, GFLOPS: 304.795
Blocksize: 928, Iterations: 1421, FLOPS: 3.04939e+11, GFLOPS: 304.939
Blocksize: 944, Iterations: 1420, FLOPS: 3.04515e+11, GFLOPS: 304.515
Blocksize: 960, Iterations: 1416, FLOPS: 3.03803e+11, GFLOPS: 303.803
Blocksize: 976, Iterations: 1418, FLOPS: 3.04484e+11, GFLOPS: 304.484
Blocksize: 992, Iterations: 1417, FLOPS: 3.04268e+11, GFLOPS: 304.268
Blocksize: 1008, Iterations: 1420, FLOPS: 3.04688e+11, GFLOPS: 304.688
Blocksize: 1024, Iterations: 1419, FLOPS: 3.04413e+11, GFLOPS: 304.413
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