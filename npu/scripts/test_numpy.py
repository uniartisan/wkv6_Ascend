import torch
import torch_npu

import numpy as np
from typing import Optional, Tuple
def naive_recurrent_rwkv6(
    B, T, C, H,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
):
    orig_dtype = q.dtype
    B, H, T, K, V = *q.shape, v.shape[-1]
    q, k, v, w, u = map(lambda x: x.float().npu(), (q, k, v, w, u))
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
        h = h * (-w_i[..., None].exp()).exp() + kv_i
    ht = h if output_final_state else None
    return o.to(orig_dtype).cpu(), ht

from typing import Optional
import torch

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

                # 计算 kv_i = k_i * v_i^T
                kv_i = k_i[:, None] * v_i[None, :]  # (D, D)

                # 计算输出 o_i = (h + u_expand * kv_i) * q_i
                o_i = (h[b, h_idx] + u_expand[h_idx] * kv_i) * q_i[:, None]  # (D, D)
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
    q_np = q.numpy()
    k_np = k.numpy()
    v_np = v.numpy()
    w_np = w.numpy()
    u_np = u.numpy()
    
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
                        o_np[b, h, t, i] += q_np[b, h, t, j] * (u_np[h, j] * x + s)
                        state[j] = s * np.exp(-np.exp(w[b, h, t, j]))  + x
    
    # Convert the result back to a PyTorch tensor
    o_torch = torch.from_numpy(o_np)
    
    return o_torch

def compare_outputs(B, T, C, H, data_type=torch.float32):
    """
    Compare outputs of naive_recurrent_rwkv6 and rwkv_time_mix_torch.
    """
    # Generate random input data
    N = C // H
    param_shape = (B, H, T, N)
    u_shape = (H, N)

    k = torch.rand(param_shape, dtype=data_type).uniform_(-1,1).to(torch.device("cpu"))
    v = torch.rand(param_shape, dtype=data_type).uniform_(-1,1).to(torch.device("cpu"))
    w = torch.rand(param_shape, dtype=data_type).uniform_(-8,-6).to(torch.device("cpu"))
    q = torch.rand(param_shape, dtype=data_type).uniform_(-1,1).to(torch.device("cpu"))
    u = torch.rand(u_shape, dtype=data_type).uniform_(-1,1).to(torch.device("cpu"))

    # Call naive_recurrent_rwkv6
    o_naive, _ = naive_recurrent_rwkv6(B, T, C, H, q, k, v, w, u)

    # Call rwkv_time_mix_torch
    o_time_mix = rwkv_numpy_forward(q, k, v, w, u)


    print(o_time_mix.shape, o_naive.shape)

    # Compare the two outputs
    if torch.allclose(o_naive, o_time_mix, rtol=1e-5, atol=1e-5):
        print("Outputs are the same.")
    else:
        print("Outputs are different.")
        print((o_naive - o_time_mix).max())


# Example usage
if __name__ == "__main__":
    B, T, C, H = 1, 16, 1024, 32
    compare_outputs(B, T, C, H)