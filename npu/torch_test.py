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
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
):
    orig_dtype = q.dtype
    B, H, T, K, V = *q.shape, v.shape[-1]
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)

    if initial_state is not None:
        h += initial_state

    for i in range(T):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i]
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * (-w_i[:, None].exp()).exp() + kv_i
    ht = h if output_final_state else None
    return o.to(orig_dtype), ht


def naive_recurrent_rwkv6_expanded(
        B, T, C, H,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
):
    orig_dtype = q.dtype
    D = C // H  # 头的维度
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    h = torch.zeros(B, H, D, D, dtype=torch.float32, device=q.device)
    h = torch.zeros(B, H, D, D, dtype=orig_dtype, device=q.device)  # 隐藏状态
    o = torch.zeros_like(v)

    # 如果有初始状态，加到隐藏状态中
    if initial_state is not None:
        h += initial_state.to(dtype=orig_dtype)

    # 处理 u 的形状

    u_expand = u[:, :, None]  # (1, H, D, 1)

    # 逐元素计算
    for t in range(T):      # time step
        for b in range(B):  # batch
            for h_idx in range(H):  # head

                # 获取当前时间步的 q, k, v, w
                q_i = q[b, h_idx, t, :]  # (D,)
                k_i = k[b, h_idx, t, :]  # (D,)
                v_i = v[b, h_idx, t, :]  # (D,)
                w_i = w[b, h_idx, t, :]  # (D,)
                # u_i =

                # 计算 kv_i = k_i * v_i^T
                kv_i = k_i[:, None] * v_i[None, :]  # (D, D)

                # 计算输出 o_i = (h + u_expand * kv_i) * q_i
                o_i = (h[b, h_idx] + u_expand[h_idx] * kv_i) * \
                    q_i[:, None]  # (D, D)
                o[b, h_idx, t, :] = o_i.sum(dim=0)  # (D,)

                # 更新隐藏状态 h = h * exp(w_i) + kv_i
                h[b, h_idx] = h[b, h_idx] * (-w_i[:, None].exp()).exp() + kv_i

    # 如果需要返回最终状态
    ht = h if output_final_state else None
    return o.to(orig_dtype), ht


def rwkv_numpy_forward(q, k, v, w, u):
    """
    RWKV forward pass using NumPy.
    
    Args:
        q (torch.Tensor): Query tensor of shape (B, H, T, N).
        k (torch.Tensor): Key tensor of shape (B, H, T, N).
        v (torch.Tensor): Value tensor of shape (B, H, T, N).
        w (torch.Tensor): Weight tensor of shape (B, H, T, N).
        u (torch.Tensor): Parameter tensor of shape (H, N).
    
    Returns:
        torch.Tensor: Output tensor of shape (B, H, T, N).
    """
    # Convert PyTorch tensors to NumPy arrays
    q_np = q.cpu().numpy()
    k_np = k.cpu().numpy()
    v_np = v.cpu().numpy()
    w_np = w.cpu().numpy()
    u_np = u.cpu().numpy()

    # Get dimensions
    B, H, T, N = q_np.shape

    # Initialize output array
    o_np = np.zeros_like(q_np)

    # Perform the RWKV computation
    for b in range(B):
        for h in range(H):
            for i in range(N):
                state = np.zeros((N), dtype=q_np.dtype)
                for t in range(T):
                    for j in range(N):
                        x = k_np[b, h, t, j] * v_np[b, h, t, i]
                        s = state[j]
                        o_np[b, h, t, i] += q_np[b, h, t, j] * \
                            (u_np[h, j] * x + s)
                        state[j] = s * np.exp(w_np[b, h, t, j]) + x

    # Convert the result back to a PyTorch tensor
    o_torch = torch.from_numpy(o_np)

    return o_torch


def load_bin_file(file_path, shape, dtype=torch.float32):
    """
    从二进制文件中加载数据并转换为指定形状和类型的张量。
    """
    data = np.fromfile(file_path, dtype=np.float32)  # 假设文件是 float32 格式
    return torch.from_numpy(data.reshape(shape)).to(dtype)


def compare_outputs(B, T, C, H, data_type=torch.float32):
    """
    从文件中读取输入数据，并比较 naive_recurrent_rwkv6 和 rwkv_time_mix_torch 的输出。
    """
    # 定义形状参数

    N = C // H  # 头的维度
    param_shape = (B, H, T, N)
    u_shape = (H, N)

    # 从文件中加载输入数据并还原形状
    base_path = "/root/wkv6/wkv6_Ascend/npu/input_npu/"
    k = load_bin_file(base_path + "input_k.bin", param_shape,
                      data_type).to(torch.device("npu:0"))
    v = load_bin_file(base_path + "input_v.bin", param_shape,
                      data_type).to(torch.device("npu:0"))
    w = load_bin_file(base_path + "input_w.bin", param_shape,
                      data_type).to(torch.device("npu:0"))
    q = load_bin_file(base_path + "input_r.bin", param_shape,
                      data_type).to(torch.device("npu:0"))
    u = load_bin_file(base_path + "input_u.bin", u_shape,
                      data_type).to(torch.device("npu:0"))

    output_path = "/root/wkv6/wkv6_Ascend/npu/output_npu/"
    o_base = load_bin_file(output_path + "output_o_golden.bin",
                           param_shape, data_type).to(torch.device("npu:0"))

    # Call naive_recurrent_rwkv6
    with torch.no_grad():

        # 调用 rwkv6_vector
        # o2 = torch.empty_like(q).to(torch.device("npu:0"))
        o2 = rwkv6_vector.run_rwkv6_vector(B, T, C, H, q, k, v, w, u)
        o_base = o_base.cpu()
        o2 = o2.cpu()

        # 比较 o_base 和 o2
        if torch.allclose(o_base, o2, rtol=1e-5, atol=1e-5):
            print("o_base and o2 are the same.")
            print("Ascend NPU wkv6 kernel is correct.")
        else:
            print("o_base and o2 are different.")
            diff = (o_base - o2).abs()
            print(f"Max difference: {diff.max()}")
            print(f"Mean difference: {diff.mean()}")
            # print(f"Difference details: {diff}")


# Example usage
if __name__ == "__main__":
    B, T, C, H = 1, 64, 4096, 64
    compare_outputs(B, T, C, H)
