# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch

from vllm_omni.diffusion.layers.swiglu7 import SwiGLU7

from .attention_ops import correct_out_lse_with_sink, torch_varlen_attention_with_sink, varlen_attention_with_sink
from .dispatcher import ModalityDispatcher
from .embedding import ElementWiseFourierEmbed
from .linear import Magi2GroupedLinear, make_grouped_linear
from .mh_moe import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
    compute_topk_probs_and_indices,
    global_sort_routes,
    torch_mh_moe_forward,
)
from .mhc import MHCHandler, MHCTensorTuple, sinkhorn_knopp
from .normalization import MultiModalityRMSNorm
from .rope import apply_rotary_emb, rotate_half

_swiglu7_op = SwiGLU7()


def swiglu7(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Released GPT-OSS-style clamped SwiGLU activation."""

    # Keep the native expression visible to torch.compile while dispatching
    # the fused CustomOp for eager inference.
    if not torch.compiler.is_compiling():
        return _swiglu7_op(x, alpha, limit, out_dtype)
    return _swiglu7_op.forward_native(x, alpha, limit, out_dtype)


__all__ = [
    "ElementWiseFourierEmbed",
    "MHCHandler",
    "MHCTensorTuple",
    "Magi2GroupedLinear",
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "ModalityDispatcher",
    "MultiModalityRMSNorm",
    "apply_rotary_emb",
    "compute_topk_probs_and_indices",
    "correct_out_lse_with_sink",
    "global_sort_routes",
    "make_grouped_linear",
    "rotate_half",
    "sinkhorn_knopp",
    "swiglu7",
    "torch_mh_moe_forward",
    "torch_varlen_attention_with_sink",
    "varlen_attention_with_sink",
]
