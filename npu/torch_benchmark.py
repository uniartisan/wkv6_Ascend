from typing import Optional, Tuple
import torch
import torch_npu
import rwkv6_vector
import numpy as np
import sys
import os
import time
import math
sys.path.append(os.getcwd())
torch.npu.config.allow_internal_format = False


# 配置 experimental_config 参数
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,  # 采集 AI Core 性能指标
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,  # 采集 pipeline 气泡等指标
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)


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
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()  # 记录开始时间
            _ = rwkv6_vector.run_rwkv6_vector(B, T, C, H, q, k, v, w, u)
            torch.npu.synchronize() 
            end_time = time.time()  # 记录结束时间
            total_time += (end_time - start_time)  # 累加运行时间

    # 计算平均运行时间
    avg_time = total_time / num_runs

    # 启动性能数据采集
    # with torch_npu.profiler.profile(
    #     activities=[
    #         torch_npu.profiler.ProfilerActivity.CPU,
    #         torch_npu.profiler.ProfilerActivity.NPU
    #     ],
    #     schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
    #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("/root/wkv6/wkv6_Ascend/npu/result"),
    #     record_shapes=False,
    #     profile_memory=False,
    #     with_stack=False,
    #     with_modules=False,
    #     with_flops=True,  # 开启 FLOPs 采集
    #     experimental_config=experimental_config) as prof:
        
    #     for step in range(num_runs):
    #         _ = rwkv6_vector.run_rwkv6_vector(B, T, C, H, q, k, v, w, u)
    #         prof.step()
    print(f"WKV6 Vector Kernel Average running time over {num_runs} runs: {avg_time:.6f} seconds, token length: {T}")


def benchmark_flash_attention(B, T, C, H, q, k, v, attn_mask=None, num_runs=10):
    """
    多次运行 FlashAttentionScore 并计算平均运行时间。
    """
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = torch_npu.npu_fusion_attention(
                q, k, v, H, "BNSD", 
                atten_mask=attn_mask,
                scale=1.0 / math.sqrt(C // H),  # 缩放系数
                keep_prob=1.0,  # 无 dropout
                sparse_mode=0  # 默认模式
            )
            torch.npu.synchronize()

    # 记录运行时间
    total_time = 0.0
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = torch_npu.npu_fusion_attention(
                q, k, v, H, "BNSD", 
                atten_mask=attn_mask,
                scale=1.0 / math.sqrt(C // H),  # 缩放系数
                keep_prob=1.0,  # 无 dropout
                sparse_mode=0  # 默认模式
            )
            torch.npu.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)

    # 计算平均运行时间
    avg_time = total_time / num_runs

    
    print(f"FlashAttentionScore Average running time over {num_runs} runs: {avg_time:.6f} seconds, token length: {T}")

# 示例用法
if __name__ == "__main__":
    B, T, C, H = 8, 4096*4, 4096, 64  # Batch size, Sequence length, Embedding dimension, Head number
    L = T
    device = torch.device("npu:0")
    dtype = torch.bfloat16  # FlashAttentionScore 支持 float16 和 bfloat16

    # 生成随机输入张量
    q = torch.randn(B, H, L, C // H).to(device).to(dtype)  # [B, H, L, C//H]
    k = torch.randn(B, H, L, C // H).to(device).to(dtype)  # [B, H, L, C//H]
    v = torch.randn(B, H, L, C // H).to(device).to(dtype)  # [B, H, L, C//H]

    # 生成注意力掩码（可选）
    attn_mask = torch.randint(0, 2, (B, 1, L, L), dtype=torch.bool).to(device)  # [B, 1, L, L]

    # 运行 benchmark
    benchmark_flash_attention(B, T, C, H, q, k, v, attn_mask=attn_mask, num_runs=10)
    del q, k ,v, attn_mask

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
