from typing import Optional, Tuple
import torch
import torch_npu
import numpy as np
import sys
import os

sys.path.append("/root/wkv6/wkv6_Ascend/npu/build")
import rwkv6_vector

torch.npu.config.allow_internal_format = False

# 全局变量，用于记录所有测试中的最大差异
max_diff = 0.0
max_state_diff = 0.0

def naive_recurrent_rwkv6(
    B, T, C, H,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    h0: torch.Tensor,
    dtype=torch.float32
):
    orig_dtype = q.dtype
    B, H, T, K, V = *q.shape, v.shape[-1]
    orig_device = q.device
    o = torch.zeros_like(v)
    h = h0.clone()

    w = -torch.exp(w)
    w = w.exp()


    for i in range(T):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i]
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * (w_i[..., None]) + kv_i

    return o.to(dtype).to(orig_device), h.to(dtype).to(orig_device)

def generate_random_tensor(shape, dtype, low=-30, high=30):
    """
    生成指定形状和数据类型的随机张量，范围更广。
    """
    return torch.rand(shape, dtype=dtype, device=torch.device("npu:0")).uniform_(low, high)

def compare_outputs(B, T, HEADS, HEADS_DIM, data_type):
    """
    从文件中读取输入数据，并比较 naive_recurrent_rwkv6 和 rwkv_time_mix_torch 的输出。
    """
    global max_diff, max_state_diff

    # 定义形状参数
    C = HEADS * HEADS_DIM
    N = HEADS_DIM
    param_shape = (B, HEADS, T, N)
    u_shape = (HEADS, N)
    h_shape = (B, HEADS, N, N)

    # k = generate_random_tensor(param_shape, data_type)
    # v = generate_random_tensor(param_shape, data_type)
    # w = generate_random_tensor(param_shape, data_type)
    # q = generate_random_tensor(param_shape, data_type)
    # u = generate_random_tensor(u_shape, data_type)
    # h = generate_random_tensor(h_shape, data_type)
    
    q = generate_random_tensor(param_shape, data_type, low=-30, high=15.163)  # receptance
    k = generate_random_tensor(param_shape, data_type, low=-27.221, high=22.123)  # key
    v = generate_random_tensor(param_shape, data_type, low=-255.707, high=136.692)  # value
    w = generate_random_tensor(param_shape, data_type, low=-16.727, high=7.503)  # time_decay
    u = generate_random_tensor(u_shape, data_type, low=-4.375, high=8.688)  # time_first
    h = generate_random_tensor(h_shape, data_type, low=-3302.162, high=4094.870)  # initial_state
    # h = h.zero_()


    h2 = h.clone()

    # Call naive_recurrent_rwkv6
    with torch.no_grad():
        o_base, state = naive_recurrent_rwkv6(B, T, HEADS, HEADS_DIM, q, k, v, w, u, h.transpose(-1, -2))
        if torch.isnan(o_base).any():
            print("o_base has NaN.")

        # 调用 rwkv6_vector
        o2, state2 = rwkv6_vector.run_rwkv6_vector(B, T, HEADS, HEADS_DIM, q, k, v, w, u, h2)
        if torch.isnan(o2).any():
            print("o_2 has NaN.")

        if torch.isnan(state2).any():
            print("state2 has NaN.")


        o_base = o_base.cpu().float()
        o2 = o2.cpu().float()
        state = state.cpu().float()
        state2 = state2.cpu().float()

        # 比较 o_base 和 o2
        diff = (o_base - o2).abs()
        state_diff = (state.transpose(-1, -2).float() - state2.float()).abs()

        # 更新全局最大差异
        max_diff = max(max_diff, diff.max().item())
        max_state_diff = max(max_state_diff, state_diff.max().item())

        if torch.allclose(o_base, o2, rtol=1e-3, atol=1e-3) and torch.allclose(state, state2, rtol=1e-3, atol=1e-3):
            pass
        else:
            print(B, T, HEADS, HEADS_DIM, data_type, "Mean difference: ", diff.mean(), "Max difference: ", diff.max(),
                  "Mean state difference: ", state_diff.mean(), "Max state difference: ", state_diff.max())

# Example usage
if __name__ == "__main__":
    dtype = torch.float32
    torch.manual_seed(42)

    # 定义不同的 B, T, HEAD, HEADDIM 组合
    configs = [
        (1, 4096, 128, 64),
        (1, 1, 64, 64),
        (1, 32, 64, 64),
        (1, 40, 64, 64),
        (1, 56, 32, 64),
        (1, 1, 32, 64),
        (1, 56, 40, 64)
    ]


    # 遍历每个配置并调用 compare_outputs 函数
    for config in configs:
        B, T, HEAD, HEADDIM = config
        print("========================================")
        compare_outputs(B, T, HEAD, HEADDIM, dtype)

    # 输出所有测试中的最大差异
    print("\n========================================")
    print(f"All tests completed. Max output difference: {max_diff}")
    print(f"Max state difference: {max_state_diff}")