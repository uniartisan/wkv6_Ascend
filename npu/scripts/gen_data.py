import numpy as np
import os
import torch
import torch_npu
from typing import Optional, Tuple


def rwkv_time_mix(B, T, C, H, data_type, input_dir, output_dir):
    N = C // H  # 头的维度
    param_shape = (B, H, T, N)
    u_shape = (H, N)
    k = np.random.uniform(-1, 1, param_shape).astype(data_type)
    v = np.random.uniform(-1, 1, param_shape).astype(data_type)
    w = np.random.uniform(-8, -6, param_shape).astype(data_type)
    q = np.random.uniform(-1, 1, param_shape).astype(data_type)
    u = np.random.uniform(-1, 1, u_shape).astype(data_type)
    o = np.zeros(param_shape).astype(data_type)

    # save k, v, w, r, u, o original values

    k.tofile(os.path.join(input_dir, "input_k.bin"))
    v.tofile(os.path.join(input_dir, "input_v.bin"))
    w.tofile(os.path.join(input_dir, "input_w.bin"))
    q.tofile(os.path.join(input_dir, "input_r.bin"))
    u.tofile(os.path.join(input_dir, "input_u.bin"))
    o.tofile(os.path.join(input_dir, "input_o.bin"))

    np.save(os.path.join(input_dir, "input_k.bin.npy"), k)
    np.save(os.path.join(input_dir, "input_v.bin.npy"), v)
    np.save(os.path.join(input_dir, "input_w.bin.npy"), w)
    np.save(os.path.join(input_dir, "input_r.bin.npy"), q)
    np.save(os.path.join(input_dir, "input_u.bin.npy"), u)
    np.save(os.path.join(input_dir, "input_o.bin.npy"), o)

    for b in range(B):
        for h in range(H):
            print("Generating data: h=", h)
            for i in range(N):
                state = np.zeros((N), dtype=data_type)
                for t in range(T):
                    for j in range(N):
                        x = k[b, h, t, j] * v[b, h, t, i]
                        s = state[j]
                        o[b, h, t, i] += q[b, h, t, j] * (u[h, j] * x + s)
                        state[j] = s * np.exp(-np.exp(w[b, h, t, j])) + x

    # output o_golden bin
    o.tofile(os.path.join(output_dir, "output_o_golden.bin"))
    np.save(os.path.join(output_dir, "output_o_golden.bin.npy"), o)
    return

def naive_recurrent_rwkv6(
    B, T, C, H,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
):
    orig_dtype = q.dtype
    B, H, T, K, V = *q.shape, v.shape[-1]
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)



    for i in range(T):
        q_i = q[:, :, i, :] 
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i]
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * (-w_i[..., None].exp()).exp() + kv_i
   
    return o.to(orig_dtype)

def rwkv_time_mix_torch(B, T, C, H, data_type, input_dir, output_dir):
    N = C // H  # 头的维度
    param_shape = (B, H, T, N)
    u_shape = (H, N)

    # 生成随机数据
    k = torch.rand(param_shape, dtype=data_type).uniform_(-1, 1)
    v = torch.rand(param_shape, dtype=data_type).uniform_(-1, 1)
    w = torch.rand(param_shape, dtype=data_type).uniform_(-8, -6)
    q = torch.rand(param_shape, dtype=data_type).uniform_(-1, 1)
    u = torch.rand(u_shape, dtype=data_type).uniform_(-1, 1)

    # 保存输入数据
    k.numpy().tofile(os.path.join(input_dir, "input_k.bin"))
    v.numpy().tofile(os.path.join(input_dir, "input_v.bin"))
    w.numpy().tofile(os.path.join(input_dir, "input_w.bin"))
    q.numpy().tofile(os.path.join(input_dir, "input_r.bin"))
    u.numpy().tofile(os.path.join(input_dir, "input_u.bin"))

    np.save(os.path.join(input_dir, "input_k.bin.npy"), k.numpy())
    np.save(os.path.join(input_dir, "input_v.bin.npy"), v.numpy())
    np.save(os.path.join(input_dir, "input_w.bin.npy"), w.numpy())
    np.save(os.path.join(input_dir, "input_r.bin.npy"), q.numpy())
    np.save(os.path.join(input_dir, "input_u.bin.npy"), u.numpy())

    # 使用 PyTorch 计算输出
    with torch.no_grad():
        o = naive_recurrent_rwkv6(B, T, C, H, q.npu(), k.npu(), v.npu(), w.npu(), u.npu())
        o = o.cpu()

    # 保存输出数据
    o.numpy().tofile(os.path.join(output_dir, "output_o_golden.bin"))
    np.save(os.path.join(output_dir, "output_o_golden.bin.npy"), o.numpy())

if __name__ == "__main__":
    B, T, C, H = 1, 64, 4096, 64
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.system("rm -rf ../input_npu")
    os.system("rm -rf ../output_npu")
    os.system("mkdir ../input_npu")
    os.system("mkdir ../output_npu")
    input_dir = "../input_npu"
    output_dir = "../output_npu"
    data_type = torch.float32
    rwkv_time_mix_torch(B, T, C, H, data_type, input_dir, output_dir)
