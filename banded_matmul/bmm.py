# Implementation of banded matrix multiplication using Triton

#%%
import torch
import triton
import triton.language as tl

#%%
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N0': 32, 'BLOCK_SIZE_N1': 32, 'BLOCK_SIZE_N2': 32})
    ],
    key=['n0', 'n1', 'n2']
)
@triton.jit
def naive_bmm_kernel(t0_ptr, t1_ptr, t2_ptr, n0, n1, n2, stride_t0n0, stride_t0n1, stride_t1n0, stride_t1n2, stride_t2n2, stride_t2n1, BLOCK_SIZE_N0: tl.constexpr, BLOCK_SIZE_N1: tl.constexpr, BLOCK_SIZE_N2: tl.constexpr):
    pid = tl.program_id(axis=0)

    grid_n0 = (n0 + BLOCK_SIZE_N0 - 1) // BLOCK_SIZE_N0
    grid_n1 = (n1 + BLOCK_SIZE_N1 - 1) // BLOCK_SIZE_N1
    pid_n0 = pid // grid_n0
    pid_n1 = pid % grid_n1

    # Create pointers for the first blocks of T1 and T2
    # t1_ptrs is a block of [BLOCK_SIZE_N0, BLOCK_SIZE_N2] pointers
    # t2_ptrs is a block of [BLOCK_SIZE_N2, BLOCK_SIZE_N1] pointers
    offs_t1n0 = pid_n0 * BLOCK_SIZE_N0 + tl.arange(0, BLOCK_SIZE_N0)
    offs_t2n1 = pid_n1 * BLOCK_SIZE_N1 + tl.arange(0, BLOCK_SIZE_N1)
    offs_n2 = tl.arange(0, BLOCK_SIZE_N2)
    t1_ptrs = t1_ptr + ((offs_t1n0[:, None] * stride_t1n0) + (offs_n2[None, :] * stride_t1n2))


def naive_bmm(t0, t1, t2, BLOCK_SIZE_K=32):
    n0, n2 = t1.shape
    n2, n1 = t2.shape
    assert n2 % BLOCK_SIZE_K == 0, "n2 must be divisible by BLOCK_SIZE_K"

    grid = lambda META: (
        triton.cdiv(n0, META['BLOCK_SIZE_N0']) * triton.cdiv(n1, META['BLOCK_SIZE_N1']),
    )
    naive_bmm_kernel[grid](t0, t1, t2, n0, n1, n2, t0.stride(0), t0.stride(1), t1.stride(0), t1.stride(1), t2.stride(0), t2.stride(1))
    return t0


# %%
def bmm_reference(t0: torch.tensor, t1: torch.tensor, t2: torch.tensor) -> torch.tensor:
    # use numpy to compute the reference
    t0_cpu = t0.cpu().numpy()
    t1_cpu = t1.cpu().numpy()
    t2_cpu = t2.cpu().numpy()
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
            for k in range(t1.shape[0]):
                if (i+k) < t2.shape[0]:
                    t0_cpu[i, j] += t1_cpu[i, k] * t2_cpu[i + k, j]
    return t0_cpu

# %%
torch.manual_seed(0)
n0 = n1 = n2 = 256
t0 = torch.randn(n0, n1, dtype=torch.float32, device='cuda')
t1 = torch.randn(n1, n2, dtype=torch.float32, device='cuda')
t2 = torch.randn(n2, n1, dtype=torch.float32, device='cuda')

# %%
# ref_output = bmm_reference(t0, t1, t2)
# print(ref_output)

# %%
triton_output = naive_bmm(t0, t1, t2)

# %%
if triton.testing.allclose(triton_output, ref_output):
    print("✅ Triton and Reference match")
else:
    print("❌ Triton and Reference differ")