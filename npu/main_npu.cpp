
#include "data_utils.h"
#include "acl/acl.h"
#include "aclrtlaunch_rwkv6_vector.h"

constexpr uint32_t B = 1;
constexpr uint32_t T = 128+8;
constexpr uint32_t C = 4096;
constexpr uint32_t H = 64;
constexpr uint32_t tileLength = 32;
constexpr uint32_t CORE_NUM = 48;

int64_t CompareResult(void *outputData, int64_t outSize)
{
    uint8_t *goldenData;
    CHECK_ACL(aclrtMallocHost((void **)(&goldenData), outSize));
    size_t goldenSize = outSize;
    bool ret = ReadFile("./output_npu/output_o_golden.bin", goldenSize, goldenData, goldenSize);
    if (ret)
    {
        printf("ReadFile output golden sucess!\n");
    }
    else
    {
        return -1;
    }

    constexpr float EPS = 1e-3;
    int64_t wrongNum = 0;
    float maxError = 0;
    for (int i = 0; i < outSize / (sizeof(float)); i++)
    {
        float real = static_cast<float>(((float *)outputData)[i]);
        float golden = static_cast<float>(((float *)goldenData)[i]);
        float ae = std::abs(real - golden);
        float re = ae / abs(golden);
        maxError = std::max(maxError, static_cast<float>(std::abs(real - golden)));
        if (ae > EPS && re > EPS)
        {
            std::cout << "i=" << i << " test failed: output real o: " << real << ", output golden o: " << golden << std::endl;
            wrongNum++;
        }
        // else
        // {
        //     std::cout << "i=" << i << " test passed: output real o: " << real << ", output golden o: " << golden << std::endl;
        // }
    }
    CHECK_ACL(aclrtFreeHost(goldenData));
    std::cout << "maxError: " << maxError << std::endl;
    return wrongNum;
}

int32_t main(int32_t argc, char *argv[])
{
    uint32_t paramShape = B * T * C;
    uint32_t uShape = C;
    std::cout << "paramSize:" << paramShape * sizeof(float) << ", while paramShape:" << paramShape << std::endl;
    size_t paramSize = paramShape * sizeof(float);
    size_t uSize = uShape * sizeof(float);
    printf("paramSize %d\n", paramSize);
    printf("uSize %d\n", uSize);

    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *kHost, *vHost, *wHost, *rHost, *uHost, *oHost;
    uint8_t *kDevice, *vDevice, *wDevice, *rDevice, *uDevice, *oDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&kHost), paramSize));
    CHECK_ACL(aclrtMallocHost((void **)(&vHost), paramSize));
    CHECK_ACL(aclrtMallocHost((void **)(&wHost), paramSize));
    CHECK_ACL(aclrtMallocHost((void **)(&rHost), paramSize));
    CHECK_ACL(aclrtMallocHost((void **)(&uHost), uSize));
    CHECK_ACL(aclrtMallocHost((void **)(&oHost), paramSize));
    CHECK_ACL(aclrtMalloc((void **)&kDevice, paramSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&vDevice, paramSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wDevice, paramSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&rDevice, paramSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&uDevice, uSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&oDevice, paramSize, ACL_MEM_MALLOC_HUGE_FIRST));

    bool ret = false;
    ret = ReadFile("./input_npu/input_k.bin", paramSize, kHost, paramSize);
    if (ret)
    {
        printf("ReadFile k sucess!\n");
    }
    ret = ReadFile("./input_npu/input_v.bin", paramSize, vHost, paramSize);
    if (ret)
    {
        printf("ReadFile v sucess!\n");
    }
    ret = ReadFile("./input_npu/input_w.bin", paramSize, wHost, paramSize);
    if (ret)
    {
        printf("ReadFile w sucess!\n");
    }
    ret = ReadFile("./input_npu/input_r.bin", paramSize, rHost, paramSize);
    if (ret)
    {
        printf("ReadFile r sucess!\n");
    }
    ret = ReadFile("./input_npu/input_o.bin", paramSize, oHost, paramSize);
    if (ret)
    {
        printf("ReadFile o sucess!\n");
    }
    ret = ReadFile("./input_npu/input_u.bin", uSize, uHost, uSize);
    if (ret)
    {
        printf("ReadFile u sucess!\n");
    }

    CHECK_ACL(aclrtMemcpy(kDevice, paramSize, kHost, paramSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(vDevice, paramSize, vHost, paramSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(wDevice, paramSize, wHost, paramSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(rDevice, paramSize, rHost, paramSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(uDevice, uSize, uHost, uSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(oDevice, paramSize, oHost, paramSize, ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtlaunch_rwkv6_vector(CORE_NUM, stream, B, T, C, H, kDevice, vDevice, wDevice, rDevice, uDevice, oDevice, tileLength);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(oHost, paramSize, oDevice, paramSize, ACL_MEMCPY_DEVICE_TO_HOST));

    int64_t wrongNum = CompareResult(oHost, paramSize);
    if (wrongNum != 0)
    {
        printf("test failed\n");
        printf("wrongNum = %d \n", wrongNum);
        printf("totalNum = %d \n", paramSize / (sizeof(float)));
    }
    else
    {
        printf("test pass!\n");
    }
    return 0;

    CHECK_ACL(aclrtFree(kDevice));
    CHECK_ACL(aclrtFree(vDevice));
    CHECK_ACL(aclrtFree(wDevice));
    CHECK_ACL(aclrtFree(rDevice));
    CHECK_ACL(aclrtFree(uDevice));
    CHECK_ACL(aclrtFree(oDevice));
    CHECK_ACL(aclrtFreeHost(kHost));
    CHECK_ACL(aclrtFreeHost(vHost));
    CHECK_ACL(aclrtFreeHost(wHost));
    CHECK_ACL(aclrtFreeHost(rHost));
    CHECK_ACL(aclrtFreeHost(uHost));
    CHECK_ACL(aclrtFreeHost(oHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}
