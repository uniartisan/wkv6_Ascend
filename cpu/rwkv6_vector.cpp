/**
 * @file rwkv6_vector.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 * Author: Yuqing Sun (s30040711)
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;

class KernelRWKV6Vector
{
public:
    __aicore__ inline KernelRWKV6Vector() {}
    __aicore__ inline void Init(uint32_t B, uint32_t T, uint32_t C, uint32_t H, 
                                GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, GM_ADDR u, GM_ADDR o, uint32_t tileLength)
    {
        // k:[B, H, T, N]
        // v:[B, H, T, N]
        // w:[B, H, T, N]
        // r:[B, H, T, N]
        // u:[H, N]
        // o:[B, H, T, N]
        this->B = B;
        this->T = T;
        this->C = C;
        this->H = H;
        this->N = C / H;
        // 在T维度上切tiling
        this->tileLength = tileLength;
        this->tileNum = this->T / tileLength;
        // B * H 能被核数整除的情况，不能整除时将remainder按照每个core一个head进行处理
        uint32_t totalHeads = this->B * this->H;
        uint32_t blockNum = GetBlockNum();
        uint32_t currentBlock = GetBlockIdx();
        uint32_t baseHeadsPerCore = totalHeads / blockNum; // 基础分配
        uint32_t remainderHeads = totalHeads % blockNum;   // 余数
        // 计算当前核心实际处理的head数量
        this->headPerCore = baseHeadsPerCore;
        if (currentBlock < remainderHeads)
        {
            this->headPerCore += 1; // 前面几个核多处理一个head
        }
        // 计算当前核心的数据偏移
        uint32_t headOffset = baseHeadsPerCore * currentBlock;
        if (currentBlock < remainderHeads)
        {
            headOffset += currentBlock;
        }
        else
        {
            headOffset += remainderHeads;
        }
        uint32_t uh_offset = headOffset % H;
        this->sizePerCore = this->headPerCore * T * N;
        kGm.SetGlobalBuffer((__gm__ half *)k + headOffset * T * N, this->sizePerCore);
        vGm.SetGlobalBuffer((__gm__ half *)v + headOffset * T * N, this->sizePerCore);
        wGm.SetGlobalBuffer((__gm__ half *)w + headOffset * T * N, this->sizePerCore);
        rGm.SetGlobalBuffer((__gm__ half *)r + headOffset * T * N, this->sizePerCore);
        oGm.SetGlobalBuffer((__gm__ half *)o + headOffset * T * N, this->sizePerCore);
        uGm.SetGlobalBuffer((__gm__ half *)u + uh_offset * this->N, this->headPerCore * this->N);
        // k,v,w,r,u,o每次搬运[tileLength, N]大小的tensor
        pipe.InitBuffer(inQueueK, BUFFER_NUM, this->tileLength * this->N * sizeof(half));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->tileLength * this->N * sizeof(half));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * this->N * sizeof(half));
        pipe.InitBuffer(inQueueR, BUFFER_NUM, this->tileLength * this->N * sizeof(half));
        pipe.InitBuffer(inQueueU, BUFFER_NUM, this->N * sizeof(half));
        // 其中 o 既是输入也是输出，所以既需要vecin的buffer也需要vecout的buffer
        pipe.InitBuffer(inQueueO, BUFFER_NUM, this->tileLength * this->N * sizeof(half));
        pipe.InitBuffer(outQueueO, BUFFER_NUM, this->tileLength * this->N * sizeof(half));
        // state及中间变量，每个中间变量大小为[N, N]
        pipe.InitBuffer(stateBuf, 3 * this->N * this->N * sizeof(half));
        // FIXME: state 应该是可以传入的参数
        // 用于储存broadcast结果
        pipe.InitBuffer(broadBuf0, this->N * this->N * sizeof(half));
        pipe.InitBuffer(broadBuf1, this->N * this->N * sizeof(half));
        pipe.InitBuffer(broadBuf2, this->N * this->N * sizeof(half));
        // 设置broadcast shape参数
        SetBroadShapes();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> stateLocal = stateBuf.Get<half>();
        LocalTensor<half> broadLocal0 = broadBuf0.Get<half>();
        LocalTensor<half> broadLocal1 = broadBuf1.Get<half>();
        LocalTensor<half> broadLocal2 = broadBuf2.Get<half>();

        for (uint32_t h = 0; h < this->headPerCore; h++)
        {
            // copy tensor u[h,:]
            CopyInU(h);
            LocalTensor<half> uLocal = inQueueU.DeQue<half>();
            // broadcast u and store in broadLocal0:[1, N] to [N, N]
            BroadCast<half, 2, 0>(broadLocal0, uLocal, broadDstShape, broadSrcShape);

            for (uint32_t tile = 0; tile < this->tileNum; tile++)
            {
                // copy tensor k,v,w,r,o[b, h, tile * tileLength:(tile+1)*tileLength, :]
                CopyInKVWRO(h, tile);
                LocalTensor<half> kLocal = inQueueK.DeQue<half>();
                LocalTensor<half> vLocal = inQueueV.DeQue<half>();
                LocalTensor<half> wLocal = inQueueW.DeQue<half>();
                LocalTensor<half> rLocal = inQueueR.DeQue<half>();
                LocalTensor<half> oLocal = inQueueO.DeQue<half>();
                Compute(kLocal, vLocal, wLocal, rLocal, oLocal, stateLocal, broadLocal0, broadLocal1, broadLocal2, h, tile);
                CopyOutO(h, tile);
            }
            inQueueU.FreeTensor(uLocal);
        }
    }

private:
    __aicore__ inline void SetBroadShapes()
    {
        // 除了v以外所有的broadcast shape: [1, N] -> [N, N]
        broadDstShape[0] = this->N;
        broadDstShape[1] = this->N;
        broadSrcShape[0] = 1;
        broadSrcShape[1] = this->N;
        // v的broadcast shape: [N, 1] -> [N, N]
        vDstShape[0] = this->N;
        vDstShape[1] = this->N;
        vSrcShape[0] = this->N;
        vSrcShape[1] = 1;
    }

    __aicore__ inline void CopyInU(uint32_t progress_h)
    {
        // copy in u[h,:]
        uint32_t offset = progress_h * this->N;
        LocalTensor<half> uLocal = inQueueU.AllocTensor<half>();
        DataCopy(uLocal, uGm[offset], this->N);
        inQueueU.EnQue<half>(uLocal);
    }

    __aicore__ inline void CopyInKVWRO(uint32_t progress_h, uint32_t progress_tile)
    {
        // copy k,v,w,r,o[b, h, tile*tileLength:(tile+1)*tileLength, :]
        uint32_t offset = progress_h * this->T * this->N + progress_tile * this->tileLength * this->N;
        LocalTensor<half> kLocal = inQueueK.AllocTensor<half>();
        LocalTensor<half> vLocal = inQueueV.AllocTensor<half>();
        LocalTensor<half> wLocal = inQueueW.AllocTensor<half>();
        LocalTensor<half> rLocal = inQueueR.AllocTensor<half>();
        LocalTensor<half> oLocal = inQueueO.AllocTensor<half>();
        DataCopy(kLocal, kGm[offset], this->tileLength * this->N);
        DataCopy(vLocal, vGm[offset], this->tileLength * this->N);
        DataCopy(wLocal, wGm[offset], this->tileLength * this->N);
        DataCopy(rLocal, rGm[offset], this->tileLength * this->N);
        DataCopy(oLocal, oGm[offset], this->tileLength * this->N);
        inQueueK.EnQue<half>(kLocal);
        inQueueV.EnQue<half>(vLocal);
        inQueueW.EnQue<half>(wLocal);
        inQueueR.EnQue<half>(rLocal);
        inQueueO.EnQue<half>(oLocal);
    }

    __aicore__ inline void CopyOutO(uint32_t progress_h, uint32_t progress_tile)
    {
        // copy out o[b, h, tile*tileLength:(tile+1)*tileLength,:]
        uint32_t offset = progress_h * this->T * this->N + progress_tile * this->tileLength * N;
        LocalTensor<half> oOutLocal = outQueueO.DeQue<half>();
        DataCopy(oGm[offset], oOutLocal, this->tileLength * this->N);
        outQueueO.FreeTensor(oOutLocal);
    }

    __aicore__ inline void Compute(LocalTensor<half> kLocal, LocalTensor<half> vLocal, LocalTensor<half> wLocal,
                                   LocalTensor<half> rLocal, LocalTensor<half> oLocal, LocalTensor<half> stateLocal,
                                   LocalTensor<half> broadLocal0, LocalTensor<half> broadLocal1, LocalTensor<half> broadLocal2,
                                   uint32_t progress_h, uint32_t progress_tile)
    {
        uint32_t offset0 = 0; // reserved for state vectors
        uint32_t offset1 = this->N * this->N;
        uint32_t offset2 = this->N * this->N * 2;

        if (progress_tile == 0)
        {
            Muls(stateLocal[offset0], stateLocal[offset0], (half)0, this->N * this->N);
        }

        for (uint32_t t = 0; t < this->tileLength; t++)
        {
            // compute kv = k.mT@v, offset1
            // broadcast v from [N,1] to [N, N]
            BroadCast<half, 2, 1>(broadLocal2, vLocal[t * this->N], vDstShape, vSrcShape);
            // broadcast k from [1,N] to [N, N]
            BroadCast<half, 2, 0>(broadLocal1, kLocal[t * this->N], broadDstShape, broadSrcShape);
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset1], broadLocal1, broadLocal2, this->N * this->N);

            PipeBarrier<PIPE_V>();

            // compute ukv = u * kv, shape: N * N, offset2, u was stored in broadLocal0
            Mul(stateLocal[offset2], broadLocal0, stateLocal[offset1], this->N * this->N);

            PipeBarrier<PIPE_V>();

            // compute sukv = state + ukv, shape:N * N, offset2
            Add(stateLocal[offset2], stateLocal[offset2], stateLocal[offset0], this->N * this->N);

            PipeBarrier<PIPE_V>();

            // compute state = w * state, shape:N * N, state
            // broadcast w from [1, N] to [N, N]
            BroadCast<half, 2, 0>(broadLocal1, wLocal[t * this->N], broadDstShape, broadSrcShape);
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset0], broadLocal1, stateLocal[offset0], this->N * this->N);

            PipeBarrier<PIPE_V>();

            // compute state = state + kv, shape:N*N, state
            Add(stateLocal[offset0], stateLocal[offset0], stateLocal[offset1], this->N * this->N);

            // compute out = r * sukv, shape:N * N, offset2
            // broadcast r from [1, N] to [N, N]
            BroadCast<half, 2, 0>(broadLocal1, rLocal[t * this->N], broadDstShape, broadSrcShape);
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset2], broadLocal1, stateLocal[offset2], this->N * this->N);

            PipeBarrier<PIPE_V>();

            // compute reduceSum(out), shape: N
            // mask=N, repeatTimes=N, dstRepStride=1, srcBlkStride=1, srcRepStride=N*sizeof(half)/32=4
            WholeReduceSum(oLocal[t * this->N], stateLocal[offset2], this->N, this->N, 1, 1, this->N * sizeof(half) / 32);
        }

        // move o from vecin to vecout then free vecin o
        LocalTensor<half> oOutLocal = outQueueO.AllocTensor<half>();
        DataCopy(oOutLocal, oLocal, this->tileLength * this->N);
        outQueueO.EnQue<half>(oOutLocal);
        inQueueO.FreeTensor(oLocal);

        // free k,v,w,r vecin for reuse
        inQueueK.FreeTensor(kLocal);
        inQueueV.FreeTensor(vLocal);
        inQueueW.FreeTensor(wLocal);
        inQueueR.FreeTensor(rLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueK, inQueueV, inQueueW, inQueueR, inQueueU, inQueueO;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueO;
    GlobalTensor<half> kGm, vGm, wGm, rGm, uGm, oGm;
    TBuf<QuePosition::VECCALC> stateBuf, broadBuf0, broadBuf1, broadBuf2;
    uint32_t B, T, C, H, N;
    uint32_t tileLength, tileNum;
    uint32_t batchPerCore, sizePerCore, headPerCore, uSizePerCore;
    uint32_t broadDstShape[2], broadSrcShape[2];
    uint32_t vDstShape[2], vSrcShape[2];
};

// implementation of kernel function
extern "C" __global__ __aicore__ void rwkv6_vector(uint32_t B, uint32_t T, uint32_t C, uint32_t H,
                                                   GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, GM_ADDR u, GM_ADDR o, uint32_t tileLength)
{
    KernelRWKV6Vector op;
    op.Init(B, T, C, H, k, v, w, r, u, o, tileLength);
    op.Process();
}
