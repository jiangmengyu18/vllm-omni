# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed as dist

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
        if kwargs.get("quant_config") is not None:
            raise NotImplementedError("Quantized MindIE-SD fused_moe is not implemented yet.")

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

        routed_out = fused_moe(
            hidden_states=hidden_states,
            w13_weight=self.w13_weight,
            w2_weight=self.w2_weight,
            router_logits=router_logits,
            num_experts=self.global_num_experts,
            top_k=self.top_k,
            w13_bias=getattr(self, "w13_bias", None),
            w2_bias=getattr(self, "w2_bias", None),
            tokens_full=self.tokens_full,
            reduce_results=True,
            dispatcher_type=self.dispatcher_type,
            tp_group=self.tp_group,
            ep_group=self.ep_group,
            renormalize=self.renormalize,
            custom_routing_function=self.custom_routing_function,
        )
        shared_out = self._forward_shared_experts(hidden_states)
        if shared_out is None:
            return routed_out
        return routed_out + shared_out
