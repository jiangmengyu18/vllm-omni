# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_fused_moe import (
    HunyuanFusedMoEDefault,
    _set_forward_context_num_tokens,
)


def _group_is_enabled(group: Any | None) -> bool:
    return group is not None and dist.get_world_size(group) > 1


class MindIESDHunyuanFusedMoE(HunyuanFusedMoEDefault):
    """NPU adapter that executes HunyuanImage3 MoE with MindIE-SD."""

    def __init__(self, *, prefix: str = "", **kwargs: Any) -> None:
        # Extract NPU-specific kwargs before calling super().__init__();
        # nn.Module.__setattr__ blocks attribute assignment before Module.__init__().
        tp_group = kwargs.pop("tp_group", None)
        ep_group = kwargs.pop("ep_group", None)
        tokens_full = kwargs.pop("tokens_full", None)
        if tokens_full is None:
            tokens_full = kwargs.pop("input_is_full", True)
        else:
            kwargs.pop("input_is_full", None)
        kwargs.pop("reduce_results", None)
        dispatcher_type = kwargs.pop("dispatcher_type", None)
        mindiesd_shared_experts = kwargs.get("shared_experts")

        # `--quantization ascend` passes quant_config (e.g. AscendModelSlimConfig).
        # We handle both quantized and unquantized paths in _prepare_mindiesd_weights
        # by checking the actual weight dtype.
        self._quant_type = "none"

        super().__init__(prefix=prefix, **kwargs)

        self.tp_group = tp_group
        self.ep_group = ep_group
        self.tokens_full = tokens_full
        self.dispatcher_type = dispatcher_type
        self._mindiesd_shared_experts = mindiesd_shared_experts
        self._mindiesd_weights_prepared = False
        process_weights_after_loading = self.quant_method.process_weights_after_loading

        def process_mindiesd_weights_after_loading(layer: Any) -> None:
            process_weights_after_loading(layer)
            layer._prepare_mindiesd_weights()

        self.quant_method.process_weights_after_loading = process_mindiesd_weights_after_loading

    def _prepare_mindiesd_weights(self) -> None:
        if self._mindiesd_weights_prepared:
            return

        if self.w13_weight.dtype == torch.int8:
            # --- Quantized path (e.g. --quantization ascend with W8A8_DYNAMIC) ---
            # After AscendW8A8DynamicFusedMoEMethod.process_weights_after_loading:
            #   w13_weight: [E, H, 2*I] int8, NZ format
            #   w13_weight_scale: [E, 2*I]
            #   w2_weight: [E, I, H] int8, NZ format
            #   w2_weight_scale: [E, H]
            self._quant_type = "int8"

            # Remove NZ format for MindIE-SD grouped_matmul
            self.w13_weight.data = torch_npu.npu_format_cast(self.w13_weight.data, 0)
            self.w2_weight.data = torch_npu.npu_format_cast(self.w2_weight.data, 0)
        else:
            # --- Unquantized path ---
            # Original layout: [E, 2*I, H] -> transpose to [E, H, 2*I] for MindIE-SD
            self._quant_type = "none"
            self.w13_weight.data = self.w13_weight.data.transpose(-1, -2).contiguous()
            self.w2_weight.data = self.w2_weight.data.transpose(-1, -2).contiguous()

        self._mindiesd_weights_prepared = True

    def _forward_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self._mindiesd_shared_experts is None:
            return None
        shared_out = self._mindiesd_shared_experts(hidden_states)
        if _group_is_enabled(self.tp_group):
            dist.all_reduce(shared_out, group=self.tp_group)
        return shared_out

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        self._prepare_mindiesd_weights()
        _set_forward_context_num_tokens(hidden_states.shape[0])

        from mindiesd.layers.fused_moe import fused_moe

        moe_kwargs = {
            "hidden_states": hidden_states,
            "router_logits": router_logits,
            "num_experts": self.global_num_experts,
            "top_k": self.top_k,
            "w13_weight": self.w13_weight,
            "w2_weight": self.w2_weight,
            "w13_bias": getattr(self, "w13_bias", None),
            "w2_bias": getattr(self, "w2_bias", None),
            "quant_type": self._quant_type,
            "tp_group": self.tp_group,
            "ep_group": self.ep_group,
            "dispatcher_type": self.dispatcher_type,
            "tokens_full": self.tokens_full,
            "renormalize": self.renormalize,
            "custom_routing_function": self.custom_routing_function,
            "reduce_results": True,
        }
        if self._quant_type == "int8":
            moe_kwargs["w13_weight_scale"] = self.w13_weight_scale
            moe_kwargs["w2_weight_scale"] = self.w2_weight_scale

        routed_out = fused_moe(**moe_kwargs)
        shared_out = self._forward_shared_experts(hidden_states)
        if shared_out is None:
            return routed_out
        return routed_out + shared_out
