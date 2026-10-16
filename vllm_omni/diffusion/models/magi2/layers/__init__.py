# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MAGI-2 native defaults with explicit hardware-optimized layer overrides."""

from vllm_omni.platforms import current_omni_platform

from .native import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
    MHCHandler,
    MultiModalityRMSNorm,
    apply_rotary_emb,
    compute_topk_probs_and_indices,
    correct_out_lse_with_sink,
    global_sort_routes,
    rotate_half,
    torch_mh_moe_forward,
    torch_varlen_attention_with_sink,
    varlen_attention_with_sink,
)

if current_omni_platform.is_cuda():
    from .cuda import (
        Magi2MultiHeadMoE,
        compute_topk_probs_and_indices,
        global_sort_routes,
        varlen_attention_with_sink,
    )
elif current_omni_platform.is_musa():
    from .musa import Magi2MultiHeadMoE, MHCHandler, varlen_attention_with_sink
elif current_omni_platform.is_npu():
    from .npu import Magi2MultiHeadMoE, MultiModalityRMSNorm, apply_rotary_emb, varlen_attention_with_sink

__all__ = [
    "MHCHandler",
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "MultiModalityRMSNorm",
    "apply_rotary_emb",
    "compute_topk_probs_and_indices",
    "correct_out_lse_with_sink",
    "global_sort_routes",
    "rotate_half",
    "torch_mh_moe_forward",
    "torch_varlen_attention_with_sink",
    "varlen_attention_with_sink",
]
