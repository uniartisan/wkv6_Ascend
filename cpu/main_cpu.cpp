#include "data_utils.h"
#include "acl/acl.h"
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void rwkv6_vector(uint32_t B, uint32_t T, uint32_t C, uint32_t H, 
                                GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, GM_ADDR u, GM_ADDR o, uint32_t tileLength);
constexpr uint32_t B = 1;
constexpr uint32_t T = 32;
constexpr uint32_t C = 256;
constexpr uint32_t H = 8;
constexpr uint32_t tileLength = 4;
constexpr uint32_t CORE_NUM = 1;


int64_t CompareResult(void* outputData, int64_t outSize)
{
    void* goldenData;
    goldenData = (uint8_t*)AscendC::GmAlloc(outSize);
    size_t goldenSize = outSize;
    bool ret = ReadFile("./output/output_o_golden.bin", goldenSize, goldenData, goldenSize);
    if (ret) {
        printf("ReadFile output golden sucess!\n");
    } else {
        return -1;
    }

    constexpr float EPS = 1e-3;
    int64_t wrongNum = 0;
    for (int i = 0;i< outSize / (sizeof(float)/2);i++) {
        
        float real = static_cast<float>(((__fp16*)outputData)[i]);
        float golden = static_cast<float>(((__fp16*)goldenData)[i]);
        float ae = std::abs(real-golden);
        float re = ae / abs(golden);
        if (ae > EPS && re > EPS) {
            std::cout << "i=" << i << " test failed: output real o: " << real << ", output golden o: " << golden << std::endl;
            wrongNum++;
        }
        else {
            std::cout << "i=" << i << " test passed: output real o: " << real << ", output golden o: " << golden << std::endl;
        }
        
    }
    AscendC::GmFree((void*)goldenData);
    return wrongNum;
}

int32_t main(int32_t argc, char* argv[])
{
    //ascendc::gmalloc
    uint32_t paramShape = B * T * C;
    uint32_t uShape = C;
    std::cout << "paramSize:" << paramShape * sizeof(float) /2 << ", while paramShape:" << paramShape <<std::endl;
    size_t paramSize = paramShape * sizeof(float) /2;
    size_t uSize = uShape * sizeof(float) /2;
    printf("paramSize %d\n", paramSize);
    printf("uSize %d\n", uSize);

    uint8_t* k = (uint8_t*)AscendC::GmAlloc(paramSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(paramSize);
    uint8_t* w = (uint8_t*)AscendC::GmAlloc(paramSize);
    uint8_t* r = (uint8_t*)AscendC::GmAlloc(paramSize);
    uint8_t* o = (uint8_t*)AscendC::GmAlloc(paramSize);
    uint8_t* u = (uint8_t*)AscendC::GmAlloc(uSize);
    //readfile
    ReadFile("./input/input_k.bin", paramSize, k, paramSize);
    ReadFile("./input/input_v.bin", paramSize, v, paramSize);
    ReadFile("./input/input_w.bin", paramSize, w, paramSize);
    ReadFile("./input/input_r.bin", paramSize, r, paramSize);
    ReadFile("./input/input_o.bin", paramSize, o, paramSize);
    ReadFile("./input/input_u.bin", uSize, u, uSize);
    //setkernelmode::aiv_mode
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    //icpu_run_kf
    ICPU_RUN_KF(rwkv6_vector, CORE_NUM, B, T, C, H, k, v, w, r, u, o, tileLength);
    //compare result
    int64_t wrongNum = CompareResult(o, paramSize);
    //ascendc::gmfree
    AscendC::GmFree((void*)k);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)w);
    AscendC::GmFree((void*)r);
    AscendC::GmFree((void*)u);
    AscendC::GmFree((void*)o);

    if (wrongNum != 0) {
        printf("test failed\n");
        printf("wrongNum = %d \n", wrongNum);
    }
    else {
        printf("test pass!\n");
    }
    return 0;
}
