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
        this->tileNum = this->T / tileLength; // 余数还需要考虑。
        this->tileNumremainer = this->T % tileLength; // 余数
        if (this->tileNumremainer > 0)
        {
            this->hasRemainer = true;
        } else {
            this->hasRemainer = false;
        }
        // B * H 能被核数整除的情况，不能整除时将remainer按照每个core一个head进行处理
        uint32_t totalHeads = this->B * this->H;
        uint32_t blockNum = GetBlockNum();
        uint32_t currentBlock = GetBlockIdx();
        uint32_t baseHeadsPerCore = totalHeads / blockNum; // 基础分配
        uint32_t remainerHeads = totalHeads % blockNum;   // 余数
        // 计算当前核心实际处理的head数量
        this->headPerCore = baseHeadsPerCore;
        if (currentBlock < remainerHeads)
        {
            this->headPerCore += 1; // 前面几个核多处理一个head
        }
        // 计算当前核心的数据偏移
        uint32_t headOffset = baseHeadsPerCore * currentBlock;
        if (currentBlock < remainerHeads)
        {
            headOffset += currentBlock;
        }
        else
        {
            headOffset += remainerHeads;
        }
        uint32_t uh_offset = headOffset % H;
        this->sizePerCore = this->headPerCore * T * N;
        kGm.SetGlobalBuffer((__gm__ float *)k + headOffset * T * N, this->sizePerCore);
        vGm.SetGlobalBuffer((__gm__ float *)v + headOffset * T * N, this->sizePerCore);
        wGm.SetGlobalBuffer((__gm__ float *)w + headOffset * T * N, this->sizePerCore);
        rGm.SetGlobalBuffer((__gm__ float *)r + headOffset * T * N, this->sizePerCore);
        oGm.SetGlobalBuffer((__gm__ float *)o + headOffset * T * N, this->sizePerCore);
        uGm.SetGlobalBuffer((__gm__ float *)u + uh_offset * this->N, this->headPerCore * this->N);
        // k,v,w,r,u,o每次搬运[tileLength, N]大小的tensor
        pipe.InitBuffer(inQueueK, BUFFER_NUM, this->tileLength * this->N * sizeof(float));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->tileLength * this->N * sizeof(float));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * this->N * sizeof(float));
        pipe.InitBuffer(inQueueR, BUFFER_NUM, this->tileLength * this->N * sizeof(float));
        pipe.InitBuffer(inQueueU, BUFFER_NUM, this->N * sizeof(float));
        // 其中 o 既是输入也是输出，所以既需要vecin的buffer也需要vecout的buffer
        pipe.InitBuffer(inQueueO, BUFFER_NUM, this->tileLength * this->N * sizeof(float));
        pipe.InitBuffer(outQueueO, BUFFER_NUM, this->tileLength * this->N * sizeof(float));
        // state及中间变量，每个中间变量大小为[N, N]
        pipe.InitBuffer(stateBuf, 3 * this->N * this->N * sizeof(float));
        // FIXME: state 应该是可以传入的参数
        // 用于储存broadcast结果
        pipe.InitBuffer(broadBuf0, this->N * this->N * sizeof(float));
        pipe.InitBuffer(broadBuf1, this->N * this->N * sizeof(float));
        pipe.InitBuffer(broadBuf2, this->N * this->N * sizeof(float));
        // 设置broadcast shape参数
        SetBroadShapes();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> stateLocal = stateBuf.Get<float>();
        LocalTensor<float> broadLocal0 = broadBuf0.Get<float>();
        LocalTensor<float> broadLocal1 = broadBuf1.Get<float>();
        LocalTensor<float> broadLocal2 = broadBuf2.Get<float>();

        for (uint32_t h = 0; h < this->headPerCore; h++)
        {
            // copy tensor u[h,:]
            CopyInU(h);
            LocalTensor<float> uLocal = inQueueU.DeQue<float>();
            // broadcast u and store in broadLocal0:[1, N] to [N, N]
            BroadCast<float, 2, 0>(broadLocal0, uLocal, broadDstShape, broadSrcShape);

            for (uint32_t tile = 0; tile < this->tileNum; tile++)
            {
                // copy tensor k,v,w,r,o[b, h, tile * tileLength:(tile+1)*tileLength, :]
                CopyInKVWRO(h, tile, false);
                LocalTensor<float> kLocal = inQueueK.DeQue<float>();
                LocalTensor<float> vLocal = inQueueV.DeQue<float>();
                LocalTensor<float> wLocal = inQueueW.DeQue<float>();
                LocalTensor<float> rLocal = inQueueR.DeQue<float>();
                LocalTensor<float> oLocal = inQueueO.DeQue<float>();
                Compute(kLocal, vLocal, wLocal, rLocal, oLocal, stateLocal, broadLocal0, broadLocal1, broadLocal2, h, tile);
                CopyOutO(h, tile, false);
            }

            // 处理余数
            if (this->hasRemainer)
            {
                CopyInKVWRO(h, this->tileNum, true);
                LocalTensor<float> kLocal = inQueueK.DeQue<float>();
                LocalTensor<float> vLocal = inQueueV.DeQue<float>();
                LocalTensor<float> wLocal = inQueueW.DeQue<float>();
                LocalTensor<float> rLocal = inQueueR.DeQue<float>();
                LocalTensor<float> oLocal = inQueueO.DeQue<float>();
                Compute(kLocal, vLocal, wLocal, rLocal, oLocal, stateLocal, broadLocal0, broadLocal1, broadLocal2, h, this->tileNum);
                CopyOutO(h, this->tileNum, true);
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
        LocalTensor<float> uLocal = inQueueU.AllocTensor<float>();
        DataCopy(uLocal, uGm[offset], this->N);
        inQueueU.EnQue<float>(uLocal);
    }

    __aicore__ inline void CopyInKVWRO(uint32_t progress_h, uint32_t progress_tile, bool remainer)
    {
        // copy k,v,w,r,o[b, h, tile*tileLength:(tile+1)*tileLength, :]
        uint32_t currentTileLength = this->tileLength;
        if (remainer)
        {
            currentTileLength = this->tileNumremainer;
        }
        
        uint32_t offset = progress_h * this->T * this->N + progress_tile * this->tileLength * this->N;
        LocalTensor<float> kLocal = inQueueK.AllocTensor<float>();
        LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
        LocalTensor<float> wLocal = inQueueW.AllocTensor<float>();
        LocalTensor<float> rLocal = inQueueR.AllocTensor<float>();
        LocalTensor<float> oLocal = inQueueO.AllocTensor<float>();
        DataCopy(kLocal, kGm[offset], currentTileLength * this->N);
        DataCopy(vLocal, vGm[offset], currentTileLength * this->N);
        DataCopy(wLocal, wGm[offset], currentTileLength * this->N);
        DataCopy(rLocal, rGm[offset], currentTileLength * this->N);
        DataCopy(oLocal, oGm[offset], currentTileLength * this->N);
        inQueueK.EnQue<float>(kLocal);
        inQueueV.EnQue<float>(vLocal);
        inQueueW.EnQue<float>(wLocal);
        inQueueR.EnQue<float>(rLocal);
        inQueueO.EnQue<float>(oLocal);
    }

    __aicore__ inline void CopyOutO(uint32_t progress_h, uint32_t progress_tile, bool remainer)
    {
        // copy out o[b, h, tile*tileLength:(tile+1)*tileLength,:]
        uint32_t currentTileLength = this->tileLength;
        if (remainer)
        {
            currentTileLength = this->tileNumremainer;
        }
        uint32_t offset = progress_h * this->T * this->N + progress_tile * this->tileLength * this->N;
        LocalTensor<float> oOutLocal = outQueueO.DeQue<float>();
        DataCopy(oGm[offset], oOutLocal, currentTileLength * this->N);
        outQueueO.FreeTensor(oOutLocal);
    }

    __aicore__ inline void Compute(LocalTensor<float> kLocal, LocalTensor<float> vLocal, LocalTensor<float> wLocal,
                                   LocalTensor<float> rLocal, LocalTensor<float> oLocal, LocalTensor<float> stateLocal,
                                   LocalTensor<float> broadLocal0, LocalTensor<float> broadLocal1, LocalTensor<float> broadLocal2,
                                   uint32_t progress_h, uint32_t progress_tile)
    {
        uint32_t offset0 = 0; // reserved for state vectors
        uint32_t offset1 = this->N * this->N;
        uint32_t offset2 = this->N * this->N * 2;

        if (progress_tile == 0)
        {
            Muls(stateLocal[offset0], stateLocal[offset0], (float)0, this->N * this->N);
        }

        for (uint32_t t = 0; t < this->tileLength; t++)
        {
            // compute kv = k.mT@v, offset1
            // broadcast v from [N,1] to [N, N]
            BroadCast<float, 2, 1>(broadLocal2, vLocal[t * this->N], vDstShape, vSrcShape);
            // broadcast k from [1,N] to [N, N]
            BroadCast<float, 2, 0>(broadLocal1, kLocal[t * this->N], broadDstShape, broadSrcShape);
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset1], broadLocal1, broadLocal2, this->N * this->N);

            PipeBarrier<PIPE_V>();

            // compute ukv = u * kv, shape: N * N, offset2, u was stored in broadLocal0
            Mul(stateLocal[offset2], broadLocal0, stateLocal[offset1], this->N * this->N);

            PipeBarrier<PIPE_V>();

            // compute sukv = state + ukv, shape:N * N, offset2
            Add(stateLocal[offset2], stateLocal[offset2], stateLocal[offset0], this->N * this->N);

            PipeBarrier<PIPE_V>();

            // compute state = exp(-exp(w)) * state, shape:N * N, state
            // broadcast w from [1, N] to [N, N]
            Exp(wLocal[t * this->N], wLocal[t * this->N], this->N);
            float negOne = -1;
            Muls(wLocal[t * this->N], wLocal[t * this->N], negOne, this->N);
            Exp(wLocal[t * this->N], wLocal[t * this->N], this->N);
            BroadCast<float, 2, 0>(broadLocal1, wLocal[t * this->N], broadDstShape, broadSrcShape);
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset0], broadLocal1, stateLocal[offset0], this->N * this->N);

            PipeBarrier<PIPE_V>();

            // compute state = state + kv, shape:N*N, state
            Add(stateLocal[offset0], stateLocal[offset0], stateLocal[offset1], this->N * this->N);

            // compute out = r * sukv, shape:N * N, offset2
            // broadcast r from [1, N] to [N, N]
            BroadCast<float, 2, 0>(broadLocal1, rLocal[t * this->N], broadDstShape, broadSrcShape);
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset2], broadLocal1, stateLocal[offset2], this->N * this->N);

            PipeBarrier<PIPE_V>();

            // compute reduceSum(out), shape: N
            // mask=N, repeatTimes=N, dstRepStride=1, srcBlkStride=1, srcRepStride=N*sizeof(float)/32=4
            WholeReduceSum(oLocal[t * this->N], stateLocal[offset2], this->N, this->N, 1, 1, this->N * sizeof(float) / 32);
        }

        // move o from vecin to vecout then free vecin o
        LocalTensor<float> oOutLocal = outQueueO.AllocTensor<float>();
        DataCopy(oOutLocal, oLocal, this->tileLength * this->N);
        outQueueO.EnQue<float>(oOutLocal);
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
    GlobalTensor<float> kGm, vGm, wGm, rGm, uGm, oGm;
    TBuf<QuePosition::VECCALC> stateBuf, broadBuf0, broadBuf1, broadBuf2;
    uint32_t B, T, C, H, N;
    uint32_t tileLength, tileNum, tileNumremainer;
    bool hasRemainer;
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
