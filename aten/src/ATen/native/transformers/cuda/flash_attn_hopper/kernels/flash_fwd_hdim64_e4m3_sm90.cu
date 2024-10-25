// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include <ATen/native/transformers/cuda/flash_attn_hopper/flash_fwd_launch_template.h>
namespace pytorch_flash_hopper{

template<>
void run_mha_fwd_<cutlass::float_e4m3_t, 64>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_fp8_hdim64<cutlass::float_e4m3_t>(params, stream);
}
} // namespace pytorch_flash_hopper
