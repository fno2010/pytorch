#pragma once
#include <cstddef>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

namespace pytorch_flash_hopper {

TORCH_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        const float softmax_scale,
        bool is_causal,
        c10::optional<at::Tensor> &q_scale_,  // 1
        c10::optional<at::Tensor> &k_scale_,  // 1
        c10::optional<at::Tensor> &v_scale_,  // 1
        int window_size_left,
        int window_size_right,
        const float softcap);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
               c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
               c10::optional<at::Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
               int const max_seqlen_q,
               int const max_seqlen_k,
               const float softmax_scale,
               bool is_causal,
               c10::optional<at::Tensor> &q_scale_,  // 1
               c10::optional<at::Tensor> &k_scale_,  // 1
               c10::optional<at::Tensor> &v_scale_,  // 1
               int window_size_left,
               int window_size_right,
               const float softcap);


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
               const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &softmax_lse,     // b x h x seqlen_q
               c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
               c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
               c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
               c10::optional<at::Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float softmax_scale,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic);

} // namespace pytorch_flash_hopper
