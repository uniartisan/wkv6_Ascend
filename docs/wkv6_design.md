# RWKV6 Vector算子设计文档

作者: Sunyuqing 30040711
日期: 2024年10月

## 1. 设计目标

### 1.1 场景与基本说明

实现RWKV6的time mix单元recurrent模式下的计算逻辑:

```python
for b in range(B):
    for h in range(H):
        for i in range(N):
            state = np.zeros((N), dtype=data_type)
            for t in range(T):
                for j in range(N):
                    x = k[b, t, h, j] * v[b, t, h, i]
                    s = state[j]
                    o[b, t, h, i]+=r[b, t, h, j] * (u[h, j] * x + s)
                    state[j] = s * w[b, t, h, j] + x
```

接口:

```cpp
extern "C" __global__ __aicore__ void rwkv6_vector(uint32_t B, uint32_t T, uint32_t C, uint32_t H, GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, GM_ADDR u, GM_ADDR o, uint32_t tileLength)
```

详细入参说明:

- `B`: 输入的Batch数 (shape: [], type: uint32)
- `T`: 输入的序列长度 (shape: [], type: uint32) 
- `C`: 输入的维度 (shape: [], type: uint32)
- `H`: 输入的attention数, H能被C整除, N = C // H (shape: [], type: uint32)
- `k`: 输入矩阵 k (shape: [B, H, T, N], datatype: half)
- `v`: 输入矩阵 v (shape: [B, H, T, N], datatype: half)
- `w`: 输入矩阵 w (shape: [B, H, T, N], datatype: half)
- `r`: 输入矩阵 r (shape: [B, H, T, N], datatype: half, data format: ND)
- `u`: 输入矩阵 u (shape: [H, N], datatype: half)
- `o`: 输入、输出矩阵 o (shape: [B, H, T, N], datatype: half)
- `tileLength`: 输入的T维度tiling参数 (shape: [], type: uint32_t)

### 1.2 数据类型

目前的实现版本中只涉及half类型, 后续根据需要将wkv计算部分转换为fp32精度。

## 2. 算子实现方案

### 2.1 分核, Tiling, 数据搬运

输入矩阵为 k, v, w, r, u, o, 其中 k, v, w, r, o 矩阵的 shape 为 [B, H, T, N], u 矩阵的 shape 为 [H, N]; 输出矩阵为 o。
因此选择在 B 和 H 维度进行分核, 对 k, v, w, r, o 在 T 维度进行 tiling 切分后分 tile 搬运。

#### 2.1.1 分核

当前代码版本中实现 B * H 可以被核数整除的情况, 每个核上处理 B * H/coreNum 份数据。矩阵 u 只有 [H, N] 两个维度, 需要计算当前的核对应的 h。

#### 2.1.2 Tiling

对数据排布为 [B, H, T, N] 顺序的 k, v, w, r, o 矩阵, 在 T 维度切分后, 每次搬运的数据块大小为 tileLength * N 个元素。 具体 tileLength 可以根据 N 的大小、UB 内存 192KB、DataCopy 指令单次搬运量越大连续 datablock 越长性能越好, 等综合考虑。

#### 2.1.3 其他 Tiling 方案

矩阵 k, v, w, r, o 如果按照 [B, T, H, N] 的数据排布顺序, 则另一种方案是在对 H 维度进行 Tiling 切分, 每次搬运的数据块大小为 h_tileLength * N 个元素。但此种情况下, 由于计算过程需要同一个 state 矩阵 (N * N) 从 t=1 迭代到 t=T, 则在 UB 上需要存放 k * h_tileLength * N * N 份内存作为计算过程的中间变量 (k 为中间变量的个数)。如果 h_tileLength 过小, 则会导致搬运指令较过多影响性能。

因此综合考虑后选择将 H 维度与 T 维度对换, 并且选择在 T 维度进行切分, 此种情况下在 UB 上需要存放的中间变量为 k * N * N, tileLength 可以选择较大的值, 提高搬运指令性能和 mte2 mte3 带宽利用率。

### 2.2 算法流程

算子输入: k, v, w, r, o [tileLength, N] 由 GM 搬运至 UB;
输出: o[tileLength, N] 由 UB 搬运至 GM;
tileLength 为 T 维度 tiling 参数;
state 中间变量: [N, N] 存放于 UB 上

对下述步骤重复 tileLength 遍:

1. 首先获得当前 t 对应的 kt, vt, wt, rt, 获得当前 h 对应的 uh
2. 对 kt 进行 broadcast 操作: [1, N] -> [N, N], 对 vt 进行 broadcast 操作: [N, 1] -> [N, N], 进行 vector 乘操作, kv = kt * vt 存放于 UB 上
3. 对 uh 进行 broadcast 操作, [1,N] -> [N, N], 并进行 vector 乘操作, ukv = uh * kv 并存放于 UB 上
4. 计算 sukv = state + ukv, 结果存放于 UB 上
5. 对 wt 进行 broadcast 操作: [1, N] -> [N, N], 并更新 state = state * wt, 结果存放于 UB 上
6. 更新 state = state + kv, 结果存放于 UB 上
7. 对 rt 进行 broadcast 操作: [1, N] -> [N, N], 并计算 out = rt * sukv, 结果存放于 UB 上
8. 对 out 进行 ReduceSum 操作, ot = reducesum(out, dim=1), 将 ot 赋值给 o 矩阵对应位置

### 2.3 相关代码

#### 2.3.1 Pytorch 伪代码

```python
for b in range(B):
    for h in range(H):
        for i in range(N):
            state = np.zeros((N), dtype=data_type)
            for t in range(T):
                for j in range(N):
                    x = k[b, t, h, j] * v[b, t, h, i]
                    s = state[j]
                    o[b, t, h, i]+=r[b, t, h, j] * (u[h, j] * x + s)
                    state[j] = s * w[b, t, h, j] + x
```

#### 2.3.2 Pytorch 算子实现等价代码

```python
def rwkv6_vector_torch(B, T, C, H, k, v, w, r, u, o):
    """
    input: k,v,w,r,u,o
    output: o
    k, v, w, r, o: shape [B, H, T, N]
    u: shape [H, N]
    """
    N = C // H
    for b in range(B):
        for h in range(H):
            uh = u[h].unsqueeze(0).broadcast_to((N, N))
            state = torch.zeros((N, N), dtype=torch.float16)
            for t in range(T):
                kt = k[b, h, t].unsqueeze(0).broadcast_to((N, N))
                vt = v[b, h, t].unsqueeze(1).broadcast_to((N, N))
                rt = r[b, h, t].unsqueeze(0).broadcast_to((N, N))
                wt = w[b, h, t].unsqueeze(0).broadcast_to((N, N))
                kv = torch.mul(kt, vt)
                ws = torch.mul(wt, state)
                state = torch.add(ws, kv)
                ukv = torch.mul(uh, kv)
                out_r = torch.mul(rt, ukv)
                out = torch.sum(out_r, dim=1)
                o[b, h, t] = out
    return o
```

#### 2.3.3 算子实现: Process 函数

#### 2.3.4 算子实现: Compute 函数

```cpp
__aicore__ inline void Compute(LocalTensor<half> kLocal, LocalTensor<half> vLocal, LocalTensor<half> wLocal, LocalTensor<half> rLocal, LocalTensor<half> oLocal, LocalTensor<half> stateLocal, LocalTensor<half> broadLocal0, LocalTensor<half> broadLocal1, LocalTensor<half> broadLocal2, uint32_t progress_h, uint32_t progress_tile)
{
    uint32_t offset0 = 0; //reserved for state vectors
    uint32_t offset1 = this->N * this->N;
    uint32_t offset2 = this->N * this->N * 2;

    if (progress_tile == 0) {
        Muls(stateLocal[offset0], stateLocal[offset0], (half)0, this->N * this->N);
    }

    for (uint32_t t=0; t < this->tileLength; t++) {
        //compute kv = k.mT@v, offset1
        //broadcast v from [N,1] to [N, N]
        BroadCast<half, 2, 1>(broadLocal2, vLocal[t*this->N], vDstShape, vSrcShape);
        //broadcast k from [1,N] to [N, N]
        BroadCast<half, 2, 0>(broadLocal1, kLocal[t*this->N], broadDstShape, broadSrcShape);
        Mul(stateLocal[offset1], broadLocal1, broadLocal2, this->N * this->N);
        PipeBarrier<PIPE_V>();

        //compute ukv = u * kv, shape: N * N, offset2, u was stored in broadLocal0
        Mul(stateLocal[offset2], broadLocal0, stateLocal[offset1], this->N * this->N);
        PipeBarrier<PIPE_V>();

        //compute sukv = state + ukv, shape:N * N, offset2
        Add(stateLocal[offset2], stateLocal[offset2], stateLocal[offset0], this->N * this->N);
        PipeBarrier<PIPE_V>();

        //compute state = w * state, shape:N * N, state
        //broadcast w from [1, N] to [N, N]
        BroadCast<half, 2, 0>(broadLocal1, wLocal[t*this->N], broadDstShape, broadSrcShape);
        Mul(stateLocal[offset0], broadLocal1, stateLocal[offset0], this->N * this->N);
        PipeBarrier<PIPE_V>();

        //compute state = state + kv, shape:N*N, state
        Add(stateLocal[offset0], stateLocal[offset0], stateLocal[offset1], this->N * this->N);

        //compute out = r * sukv, shape:N * N, offset2
        //broadcast r from [1, N] to [N, N]
        BroadCast<half, 2, 0>(broadLocal1, rLocal[t*this->N], broadDstShape, broadSrcShape);
        Mul(stateLocal[offset2], broadLocal1, stateLocal[offset2], this->N * this->N);
        PipeBarrier<PIPE_V>();

        //compute reduceSum(out), shape: N
        //mask=N, repeatTimes=N, dstRepStride=1, srcBlkStride=1, srcRepStride=N*sizeof(half)/32=4
        WholeReduceSum(oLocal[t*this->N], stateLocal[offset2], this->N, this->N, 1, 1, this->N*sizeof(half)/32);
    }

    //move o from vecin to vecout then free vecin o
    LocalTensor<half> oOutLocal = outQueueO.AllocTensor<half>();
    DataCopy(oOutLocal, oLocal, this->tileLength * this->N);
    outQueueO.EnQue<half>(oOutLocal);
    inQueueO.FreeTensor(oLocal);

    //free k,v,w,r vecin for reuse
    inQueueK.FreeTensor(kLocal);
    inQueueV.FreeTensor(vLocal);
    inQueueW.FreeTensor(wLocal);	
    inQueueR.FreeTensor(rLocal);
}
```

### 2.4 AIV UB Buffer 内存

输入输出:
- k,v,w,r,o 各搬入大小为 tileLength * N 的部分; o 同时也是输出矩阵因此还需要一个大小为 tileLength * N 的 buffer 存放输出结果
- u 矩阵搬入大小为 N 的部分

中间变量:
- 计算过程中需要存放 3 份大小为 N * N 的矩阵;
- 另外需要 3 个大小为 N * N 的 buffer 用于存放各个 broadcast 的结果。