# Implementation of banded matrix multiplication using Triton

#%%
import torch
import triton
import triton.language as tl

#%%

@triton.jit
def naive_bmm_kernel(t0_ptr, t1_ptr, t2_ptr, N, stride_t0m, stride_t0n, stride_t1m, stride_t1k, stride_t2k, stride_t2n, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)

# %%
