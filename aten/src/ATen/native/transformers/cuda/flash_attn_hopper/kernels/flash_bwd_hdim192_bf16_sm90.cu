// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include <ATen/native/transformers/cuda/flash_attn_hopper/flash_bwd_launch_template.h>
namespace pytorch_flash_hopper{

template<>
void run_mha_bwd_<cutlass::bfloat16_t, 192>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim192<cutlass::bfloat16_t>(params, stream);
}
} // namespace pytorch_flash_hopper
