from typing import Optional, Tuple
import torch
import torch_npu
import rwkv6_vector
import numpy as np
import sys
import os
import time  # 导入 time 模块

sys.path.append(os.getcwd())
torch.npu.config.allow_internal_format = False


def benchmark(B, T, C, H, q, k, v, w, u, num_runs=10):
    """
    多次运行 rwkv6_vector.run_rwkv6_vector 并计算平均运行时间。
    
    Args:
        B, T, C, H: 输入张量的形状参数。
        q, k, v, w, u: 输入张量。
        num_runs: 运行次数，默认为 10 次。
    """
    # 预热（避免第一次运行时间不准确）
    for _ in range(3):
        with torch.no_grad():
            _ = rwkv6_vector.run_rwkv6_vector(B, T, C, H, q, k, v, w, u)
        torch.npu.synchronize() 

    # 记录运行时间
    total_time = 0.0
    for _ in range(num_runs):
        
        start_time = time.time()  # 记录开始时间
        with torch.no_grad():
            _ = rwkv6_vector.run_rwkv6_vector(B, T, C, H, q, k, v, w, u)
        torch.npu.synchronize() 
        end_time = time.time()  # 记录结束时间
        total_time += (end_time - start_time)  # 累加运行时间

    # 计算平均运行时间
    avg_time = total_time / num_runs
    print(f"Kernel Average running time over {num_runs} runs: {avg_time:.6f} seconds")


    

# 示例用法
if __name__ == "__main__":
    B, T, C, H = 8, 4096, 4096, 64
    L = T
    device = torch.device("npu:0")
    dtype = torch.float32
    D = C // H

    # 生成随机输入张量
    q = torch.randn(B, H, L, D).to(device).to(dtype)
    k = torch.randn(B, H, L, D).to(device).to(dtype)
    v = torch.randn(B, H, L, D).to(device).to(dtype)
    w = torch.randn(B, H, L, D).uniform_(-8, -6).to(device).to(dtype)
    u = torch.randn(H, D).to(device).to(dtype)

    # 运行 benchmark
    benchmark(B, T, C, H, q, k, v, w, u, num_runs=10)  # 默认运行 10 次