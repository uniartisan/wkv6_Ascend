#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "aclrtlaunch_rwkv6_vector.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace my_wkv
{
    std::tuple<at::Tensor, at::Tensor> run_wkv6(uint32_t B, uint32_t T, uint32_t HEAD_NUMS, uint32_t HEAD_SIZE, 
                        const at::Tensor &r, const at::Tensor &k, const at::Tensor &v, const at::Tensor &w,
                        const at::Tensor &u, const at::Tensor &h_input)
    {
        // const at::Tensor &o, uint32_t tileLength
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        at::Tensor output = at::empty_like(r); // output tensor has the same shape as r(querry tensor)
        at::Tensor h_output = at::empty_like(h_input); // output state has the same shape as h_input
        uint32_t tileLength = 32;
        uint32_t blockDim = 48;
        float scale = 1.0 / sqrtf(HEAD_SIZE);

        ACLRT_LAUNCH_KERNEL(rwkv6_vector)
        (blockDim, acl_stream, B, T, HEAD_NUMS, HEAD_SIZE, scale, const_cast<void *>(k.storage().data()), const_cast<void *>(v.storage().data()),
         const_cast<void *>(w.storage().data()), const_cast<void *>(r.storage().data()), const_cast<void *>(u.storage().data()),
         const_cast<void *>(output.storage().data()), const_cast<void *>(h_input.storage().data()), const_cast<void *>(h_output.storage().data()),
         tileLength);
        return std::make_tuple(output, h_output);
    }
} // namespace my_wkv

PYBIND11_MODULE(rwkv6_vector, m)
{
    m.doc() = "rwkv6_vector pybind11 interfaces"; // optional module docstring
    m.def("run_rwkv6_vector", &my_wkv::run_wkv6, "");
}
