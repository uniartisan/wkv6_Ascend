import torch
import torch_npu
import sys, os
sys.path.append(os.getcwd())
import rwkv6_vector
torch.npu.config.allow_internal_format = False

from typing import Optional, Tuple

def naive_recurrent_rwkv6(
        B, T, C, H,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float = -1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    u_2d: bool = False
):
    orig_dtype = q.dtype
    B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]

    h = torch.zeros(B, H, K, V, dtype=orig_dtype, device=q.device)
    o = torch.zeros_like(v)



    if initial_state is not None:
        h += initial_state.to(dtype=orig_dtype)

    w = w.exp()

    if u_2d:
        u_expand = u[None, ..., None]
    else:
        u_expand = u[..., None]

    for i in range(T):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i]
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u_expand * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i

    ht = h if output_final_state else None
    return o.to(orig_dtype), ht


def test_wkv_custom_ops():
    # 分配Host侧输入内存，并进行数据初始化
    B = 1
    L = T = 56
    C = 4096
    H = 64
    D = C // H
    dtype = torch.float16
    require_grad = True
    device = 'npu'
    torch.manual_seed(42)
    q = (torch.randn(B, H, L, D).to(device).to(dtype)).requires_grad_(require_grad)
    k = (torch.randn(B, H, L, D).to(device).to(dtype)).requires_grad_(require_grad)
    v = torch.randn(B, H, L, D).to(device).to(dtype).requires_grad_(require_grad)
    w = torch.nn.functional.logsigmoid(torch.randn(B, H, L, D)).to(device).to(dtype).requires_grad_(require_grad)
    u = (torch.randn(H, D).to(device).to(dtype)).requires_grad_(require_grad)

    # 分配Device侧输入内存，并将数据从Host上拷贝到Device上
    with torch.no_grad():
        o1, _ = naive_recurrent_rwkv6(B, T, C, H, q, k, v, w, u, scale=1.0, u_2d=True)
    o2 = rwkv6_vector.run_rwkv6_vector(B, T, C, H, k,v,w,q,u)

    # torch.testing.assert_close(o1, o2, rtol=1e-3, atol=1e-3)
    # print(o2.to('cpu'))
    print(o1-o2)



if __name__ == "__main__":
    test_wkv_custom_ops()