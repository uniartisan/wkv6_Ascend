#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "aclrtlaunch_rwkv6_vector.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"


namespace my_wkv {
at::Tensor run_wkv6(uint32_t B, uint32_t T, uint32_t C, uint32_t H,
                        const at::Tensor &k, const at::Tensor &v, const at::Tensor &w,
                        const at::Tensor &r, const at::Tensor &u, const at::Tensor &output)
{
    // const at::Tensor &o, uint32_t tileLength
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    // at::Tensor output = at::empty_like(r); // output tensor has the same shape as r(querry tensor)
    uint32_t tileLength = 32;
    uint32_t blockDim = 8;

    ACLRT_LAUNCH_KERNEL(rwkv6_vector)
    (blockDim, acl_stream, B, T, C, H, const_cast<void *>(k.storage().data()), const_cast<void *>(v.storage().data()),
    const_cast<void *>(w.storage().data()), const_cast<void *>(r.storage().data()), const_cast<void *>(u.storage().data()),
     const_cast<void *>(output.storage().data()), tileLength);
    return output;
}
} // namespace my_wkv

PYBIND11_MODULE(rwkv6_vector, m)
{
    m.doc() = "rwkv6_vector pybind11 interfaces"; // optional module docstring
    m.def("run_rwkv6_vector", &my_wkv::run_wkv6, "");
}
