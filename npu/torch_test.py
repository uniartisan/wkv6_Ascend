from typing import Optional
from typing import Optional, Tuple

import torch
import torch_npu
import rwkv6_vector
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
torch.npu.config.allow_internal_format = False


def naive_recurrent_rwkv6(
    B, T, C, H,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    dtype = torch.float32
):
    orig_dtype = q.dtype
    B, H, T, K, V = *q.shape, v.shape[-1]
    q, k, v, w, u, h = map(lambda x: x.to(dtype), (q, k, v, w, u, h))
    o = torch.zeros_like(v)

    for i in range(T):
        q_i = q[:, :, i, :] 
        k_i = k[:, :, i] * (K ** -0.5)
        v_i = v[:, :, i, :] * (V ** -0.5)
        w_i = w[:, :, i]
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * (-w_i[..., None].exp()).exp() + kv_i
   
    return o.to(orig_dtype), h.to(orig_dtype)



def generate_random_tensor(shape, dtype, low=-1, high=1):
    """
    生成指定形状和数据类型的随机张量。
    """
    return torch.rand(shape, dtype=dtype, device=torch.device("npu:0")).uniform_(low, high)


def compare_outputs(B, T, C, H, data_type):
    """
    从文件中读取输入数据，并比较 naive_recurrent_rwkv6 和 rwkv_time_mix_torch 的输出。
    """
    # 定义形状参数

    N = C // H  # 头的维度
    param_shape = (B, H, T, N)
    u_shape = (H, N)
    h_shape = (B, H, N, N)


    k = generate_random_tensor(param_shape, data_type, low=-1, high=1)
    v = generate_random_tensor(param_shape, data_type, low=-1, high=1)
    w = generate_random_tensor(param_shape, data_type, low=-1, high=1)
    q = generate_random_tensor(param_shape, data_type, low=-1, high=1)
    u = generate_random_tensor(u_shape, data_type, low=-1, high=1)
    h = torch.randn(h_shape, dtype=data_type, device=torch.device("npu:0"))
    h2 = h.clone()


    # Call naive_recurrent_rwkv6
    with torch.no_grad():
        o_base, state = naive_recurrent_rwkv6(B, T, C, H, q, k, v, w, u, h.transpose(-1, -2))
        # 检查 base 没有 nan
        if torch.isnan(o_base).any():
            print("o_base has NaN.")
            # return
        # 调用 rwkv6_vector
        o2, state2 = rwkv6_vector.run_rwkv6_vector(B, T, C, H, q, k, v, w, u, h2)
        if torch.isnan(o2).any():
            print("o_2 has NaN.")
            # return
        o_base = o_base.cpu()
        o2 = o2.cpu()

        # 比较 o_base 和 o2
        diff = (o_base - o2).abs()
        if torch.allclose(o_base, o2, rtol=1e-3, atol=1e-3):
            print("o_base and o2 are the same.")
            print("Ascend NPU wkv6 kernel is correct.")
        else:
            print("o_base and o2 are different.")
            
        print(f"Max difference: {diff.max()}")
        print(f"Mean difference: {diff.mean()}")
        # print(f"Difference details: {diff}")
        
        print((state.transpose(-1, -2).to(torch.float16) - state2).abs().max())


# Example usage
if __name__ == "__main__":
    B, T, C, H = 1, 4096, 4096, 64
    dtype = torch.float16
    torch.manual_seed(42)
    compare_outputs(B, T, C, H, dtype)
