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
    print(
        f"WKV6 Vector Kernel Average running time over {num_runs} runs: {avg_time:.6f} seconds, token length: {T}")


def benchmark_flash_attention(B, T, C, H, q, k, v, num_runs=10):
    """
    多次运行 FlashAttentionScore 并计算平均运行时间。
    """
    # 预热
    # 生成下三角掩码
    attn_mask = torch.triu(torch.ones(
        (B, 1, T, T), dtype=torch.bool), diagonal=1).to(q.device)
    attn_mask = torch.logical_not(attn_mask)  # 取反，True 表示保留，False 表示遮蔽

    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = torch_npu.npu_fusion_attention(
                q, k, v, H, "BNSD",
                scale=1.0 / math.sqrt(D),  # 缩放系数
                keep_prob=1.0,  # 无 dropout
            )
            torch.npu.synchronize()

    # 记录运行时间
    total_time = 0.0
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = torch_npu.npu_fusion_attention(
                q, k, v, H, "BNSD",
                scale=1.0 / math.sqrt(D),  # 缩放系数
                keep_prob=1.0,  # 无 dropout
            )
            torch.npu.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)

    # 计算平均运行时间
    avg_time = total_time / num_runs

    print(
        f"FlashAttentionScore Average running time over {num_runs} runs: {avg_time:.6f} seconds, token length: {T}")


# 示例用法
if __name__ == "__main__":
    test_list = [(8, 4096, 64, 64), (8, 4096*2, 64, 64), (8, 4096*3, 64, 64),
                 (8, 4096*4, 64, 64), (8, 4096, 128, 64), (8, 4096, 32, 64), (8, 4096, 8, 64), ]
    for B, L, H, D in test_list:
        C = H * D
        print(
            f"Running benchmark with B={B}, T={L}, H={H}, D={D}... BF16 vs FP32")
        device = torch.device("npu:0")
        dtype = torch.bfloat16  # FlashAttentionScore 支持 float16 和 bfloat16

        # 生成随机输入张量
        q = torch.randn(B, H, L, D).to(device).to(dtype)  # [B, H, L, C//H]
        k = torch.randn(B, H, L, D).to(device).to(dtype)  # [B, H, L, C//H]
        v = torch.randn(B, H, L, D).to(device).to(dtype)  # [B, H, L, C//H]

        # 运行 benchmark
        benchmark_flash_attention(B, L, C, H, q, k, v, num_runs=10)
        del q, k, v

        device = torch.device("npu:0")
        dtype = torch.float32

        # 生成随机输入张量
        q = torch.randn(B, H, L, D).to(device).to(dtype)
        k = torch.randn(B, H, L, D).to(device).to(dtype)
        v = torch.randn(B, H, L, D).to(device).to(dtype)
        w = torch.randn(B, H, L, D).uniform_(-8, -6).to(device).to(dtype)
        u = torch.randn(H, D).to(device).to(dtype)

        # 运行 benchmark
        benchmark(B, L, C, H, q, k, v, w, u, num_runs=10)  # 默认运行 10 次
        del q, k, v, w, u
