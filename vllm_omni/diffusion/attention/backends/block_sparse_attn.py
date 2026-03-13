# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)


class BlockSparseAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "BLOCK_SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["BlockSparseAttentionImpl"]:
        return BlockSparseAttentionImpl


class BlockSparseAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.sparsity = float(os.environ.get("BLOCK_SPARSE_ATTN_SPARSITY", 0.0))

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """NPU sparse block attention implementation using mindiesd."""
        try:
            from mindiesd import sparse_attention
        except ImportError:
            raise ImportError(
                "BlockSparseAttentionBackend NPU implementation requires MindIE-SD. "
                "Please install MindIE-SD to enable NPU attention support. "
                "For installation details, see https://gitcode.com/Ascend/MindIE-SD"
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata else None
        txt_len = attn_metadata.txt_len if attn_metadata else 0
        latent_shape_q = attn_metadata.latent_shape_q if attn_metadata else None
        latent_shape_k = attn_metadata.latent_shape_k if attn_metadata else None

        output = sparse_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=self.softmax_scale,
            is_causal=self.causal,
            head_num=self.num_heads,
            input_layout="BSND",
            sparse_type="rf_v2",
            txt_len=txt_len,
            block_size=128,
            latent_shape_q=latent_shape_q,
            latent_shape_k=latent_shape_k,
            sparsity=self.sparsity,
        )

        return output
