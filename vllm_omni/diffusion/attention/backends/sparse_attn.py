# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)


class SparseAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SparseAttentionImpl"]:
        return SparseAttentionImpl


class SparseAttentionImpl(AttentionImpl):
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
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.num_kv_heads = num_kv_heads

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
        sparsity: float = 0.0,
        **extra_sparse_args,
    ) -> torch.Tensor:
        """NPU sparse attention implementation using mindiesd."""
        try:
            from mindiesd import sparse_attention
        except ImportError:
            raise ImportError(
                "SparseAttentionBackend NPU implementation requires MindIE-SD. "
                "Please install MindIE-SD to enable NPU attention support. "
                "For installation details, see https://gitcode.com/Ascend/MindIE-SD"
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata else None
        input_layout = extra_sparse_args.get("input_layout", "BSND")

        if input_layout == "BNSD":
            # Default layout of q, k, v is BSND. 
            # If input_layout of sparse_attention is BNSD, q, k, v need to transpose.
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()

        output = sparse_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=self.softmax_scale,
            is_causal=self.causal,
            head_num=self.num_heads,
            input_layout=input_layout,
            inner_precise=extra_sparse_args.get("inner_precise", 0),
            sparse_type=extra_sparse_args.get("sparse_type", "ada_bsa"),
            txt_len=extra_sparse_args.get("txt_len", 0),
            block_size=extra_sparse_args.get("block_size", 128),
            latent_shape_q=extra_sparse_args.get("latent_shape_q", None),
            latent_shape_k=extra_sparse_args.get("latent_shape_k", None),
            keep_sink=extra_sparse_args.get("keep_sink", True),
            keep_recent=extra_sparse_args.get("keep_recent", True),
            cdf_threshold=extra_sparse_args.get("cdf_threshold", 1.0),
            sparsity=sparsity,
        )

        if input_layout == "BNSD":
            output = output.transpose(1, 2).contiguous()

        return output
