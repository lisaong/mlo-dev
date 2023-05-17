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
GridDim,BlockDim,FLOPS,GFLOPS
1024,1,4.46582e+09,4.46582
512,2,8.11437e+09,8.11437
342,3,1.12676e+10,11.2676
256,4,1.39888e+10,13.9888
205,5,1.43354e+10,14.3354
171,6,1.39758e+10,13.9758
147,7,1.39525e+10,13.9525
128,8,1.78088e+10,17.8088
114,9,1.20373e+10,12.0373
103,10,1.40226e+10,14.0226
94,11,1.38998e+10,13.8998
86,12,1.21356e+10,12.1356
79,13,1.24467e+10,12.4467
74,14,1.16926e+10,11.6926
69,15,1.22876e+10,12.2876
64,16,1.21886e+10,12.1886
61,17,1.09938e+10,10.9938
57,18,1.0791e+10,10.791
54,19,1.09303e+10,10.9303
52,20,1.04108e+10,10.4108
49,21,1.02745e+10,10.2745
47,22,1.02665e+10,10.2665
45,23,9.74759e+09,9.74759
43,24,9.85403e+09,9.85403
41,25,9.51803e+09,9.51803
40,26,9.37249e+09,9.37249
38,27,8.91651e+09,8.91651
37,28,9.28119e+09,9.28119
36,29,8.61685e+09,8.61685
35,30,8.40325e+09,8.40325
34,31,8.03466e+09,8.03466
32,32,7.57236e+09,7.57236
```

## Loading both T0 and T1 in shared memory

```shell
Using device 0
Values match
GridDim,BlockDim,FLOPS,GFLOPS
1024,1,6.26655e+09,6.26655
512,2,1.32315e+10,13.2315
342,3,1.96143e+10,19.6143
256,4,2.52358e+10,25.2358
205,5,2.95197e+10,29.5197
171,6,2.39349e+10,23.9349
147,7,2.50248e+10,25.0248
128,8,2.6216e+10,26.216
114,9,2.29079e+10,22.9079
103,10,2.30398e+10,23.0398
94,11,2.07379e+10,20.7379
86,12,2.13665e+10,21.3665
79,13,1.99787e+10,19.9787
74,14,2.07439e+10,20.7439
69,15,2.13635e+10,21.3635
64,16,1.57438e+10,15.7438
61,17,2.02642e+10,20.2642
57,18,1.90755e+10,19.0755
54,19,1.86163e+10,18.6163
52,20,1.77472e+10,17.7472
49,21,1.81736e+10,18.1736
47,22,1.79018e+10,17.9018
45,23,1.77121e+10,17.7121
43,24,1.58222e+10,15.8222
41,25,1.73057e+10,17.3057
40,26,1.73905e+10,17.3905
38,27,1.71164e+10,17.1164
37,28,1.65226e+10,16.5226
36,29,1.72156e+10,17.2156
35,30,1.69715e+10,16.9715
34,31,1.74204e+10,17.4204
32,32,8.97544e+09,8.97544
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