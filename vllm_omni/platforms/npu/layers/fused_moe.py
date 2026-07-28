# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import vllm.distributed.parallel_state as vllm_parallel_state
import vllm.forward_context as _vllm_fc
from vllm.config import VllmConfig
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.parallel_state import (
    get_tp_group,
)
from vllm.distributed.parallel_state import (
    init_model_parallel_group as vllm_init_model_parallel_group,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
    get_data_parallel_world_size,
    get_expert_parallel_group_ranks,
    get_sequence_parallel_world_size,
    get_world_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context as omni_get_ctx


def _ensure_forward_context_attr(name: str, annotation: Any, default: Any) -> None:
    if name not in _vllm_fc.ForwardContext.__annotations__:
        _vllm_fc.ForwardContext.__annotations__[name] = annotation
    if not hasattr(_vllm_fc.ForwardContext, name):
        setattr(_vllm_fc.ForwardContext, name, default)


def _init_mc2_group_for_diffusion(
    backend: str,
    local_rank: int,
    group_ranks: list[list[int]],
) -> None:
    import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

    if getattr(vllm_ascend_parallel_state, "_MC2", None) is not None:
        return
    vllm_ascend_parallel_state._MC2 = vllm_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        group_name="mc2",
    )


def _select_moe_comm_method(vllm_config: VllmConfig) -> MoECommType | None:
    soc_version = get_ascend_device_type()
    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A3}:
        moe_comm_type = MoECommType.ALLTOALL
    elif soc_version in {AscendDeviceType._310P}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A5}:
        moe_comm_type = MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    return moe_comm_type


def prepare_fused_moe_runtime() -> None:
    vllm_config = omni_get_ctx().vllm_config

    backend = torch.distributed.get_backend(get_world_group().device_group)
    local_rank = get_world_group().local_rank
    _init_mc2_group_for_diffusion(
        backend=backend,
        local_rank=local_rank,
        group_ranks=get_expert_parallel_group_ranks(),
    )

    moe_comm_type = _select_moe_comm_method(vllm_config=vllm_config)
    _ensure_forward_context_attr("num_tokens", int | None, None)
    _ensure_forward_context_attr("in_profile_run", bool, False)
    _ensure_forward_context_attr("moe_comm_type", MoECommType | None, moe_comm_type)
    _ensure_forward_context_attr("moe_comm_method", Any, _MoECommMethods.get(moe_comm_type))
    _ensure_forward_context_attr("flash_comm_v1_enabled", bool, False)
    _ensure_forward_context_attr("max_tokens_across_dp", int | None, None)
    _ensure_forward_context_attr("max_tokens_across_pcp", int | None, None)


def reset_fused_moe_forward_context() -> None:
    try:
        forward_context = _vllm_fc.get_forward_context()
    except AssertionError:
        return

    # MoE input token counts can change between transformer forwards, for
    # example when text tokens are present only in some forwards. Clear cached
    # DP/PCP padding bounds so the first MoE layer recomputes them.
    forward_context.max_tokens_across_dp = None
    forward_context.max_tokens_across_pcp = None


def _set_max_tokens(forward_context: Any, hidden_states: torch.Tensor) -> None:
    if (
        getattr(forward_context, "max_tokens_across_dp", None) is not None
        or getattr(forward_context, "max_tokens_across_pcp", None) is not None
    ):
        return

    num_tokens_before_pcp = hidden_states.shape[0]
    dp_metadata = getattr(forward_context, "dp_metadata", None)
    if dp_metadata is not None:
        max_tokens_across_dp = int(dp_metadata.num_tokens_across_dp_cpu.max().item())
        forward_context.max_tokens_across_dp = max_tokens_across_dp
        # DP prepare pads each rank to max_tokens_across_dp, then gathers
        # all DP ranks before the PCP gather.
        num_tokens_before_pcp = max_tokens_across_dp * get_dp_group().world_size

    pcp_group = getattr(vllm_parallel_state, "_PCP", None)
    if pcp_group is None or getattr(pcp_group, "world_size", 1) <= 1:
        return

    gathered_num_tokens: list[int | None] = [None] * pcp_group.world_size
    torch.distributed.all_gather_object(
        gathered_num_tokens,
        int(num_tokens_before_pcp),
        group=pcp_group.cpu_group,
    )
    if any(num_tokens is None for num_tokens in gathered_num_tokens):
        raise RuntimeError(f"Failed to gather MoE PCP token counts: {gathered_num_tokens}")
    forward_context.max_tokens_across_pcp = max(gathered_num_tokens)


def fused_moe_forward_context_pre_hook(
    module: Any,
    args: Any,
    kwargs: Any,
) -> None:
    forward_context = _vllm_fc.get_forward_context()

    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None and args:
        hidden_states = args[0]
    if hidden_states is not None:
        _set_max_tokens(forward_context, hidden_states)

    forward_context.moe_comm_type = _select_moe_comm_method(vllm_config=omni_get_ctx().vllm_config)
    forward_context.moe_comm_method = _MoECommMethods.get(forward_context.moe_comm_type)


class MindIESDAscendMoERunner(AscendMoERunner):
    """Ascend MoE runner that delegates routed-expert execution to MindIE-SD."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        from mindiesd import fused_moe

        self._fused_moe = fused_moe
        self._quant_config = self._build_quant_config()
        self._tp_group = self._get_enabled_device_group(get_tp_group)
        if self._tp_group is None:
            self._tp_rank = 0
            self._tp_world_size = 1
        else:
            self._tp_rank = dist.get_rank(self._tp_group)
            self._tp_world_size = dist.get_world_size(self._tp_group)
        self._ep_group = self._get_enabled_device_group(get_ep_group)
        self._inputs_sharded = self._get_inputs_sharded()
        self._can_reduce_merged_output = self._tp_group is not None and not self._inputs_sharded

    @staticmethod
    def _get_enabled_device_group(get_group: Any) -> Any | None:
        try:
            group = get_group()
        except AssertionError:
            return None
        if group is None or getattr(group, "world_size", 1) <= 1:
            return None
        return group.device_group

    def _build_quant_config(self) -> Any | None:
        from mindiesd.quantization.config import QuantConfig
        from mindiesd.quantization.mode import QuantAlgorithm

        if self.quant_type == QuantType.W8A8:
            return QuantConfig(quant_algo=QuantAlgorithm.W8A8_DYNAMIC)
        if self.quant_type == QuantType.W8A8MXFP:
            return QuantConfig(quant_algo=QuantAlgorithm.W8A8_MXFP8)
        if self.quant_type != QuantType.NONE:
            raise NotImplementedError(f"MindIE-SD fused MoE does not support quant_type={self.quant_type}.")
        return None

    def _get_inputs_sharded(self) -> bool:
        has_token_sharding = (
            get_sequence_parallel_world_size() > 1
            or get_classifier_free_guidance_world_size() > 1
            or get_data_parallel_world_size() > 1
        )
        return self._ep_group is not None and has_token_sharding

    def _slice_inputs(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        if not self._inputs_sharded or self._tp_world_size == 1:
            return hidden_states, router_logits, None

        num_tokens = hidden_states.shape[0]
        pad_size = (-num_tokens) % self._tp_world_size
        if pad_size > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
            router_logits = F.pad(router_logits, (0, 0, 0, pad_size))

        hidden_states = hidden_states.chunk(self._tp_world_size, dim=0)[self._tp_rank]
        router_logits = router_logits.chunk(self._tp_world_size, dim=0)[self._tp_rank]
        return hidden_states.contiguous(), router_logits.contiguous(), num_tokens

    def _gather_routed_out(self, routed_out: torch.Tensor, original_num_tokens: int | None) -> torch.Tensor:
        if original_num_tokens is None:
            return routed_out

        gathered = [torch.empty_like(routed_out) for _ in range(self._tp_world_size)]
        dist.all_gather(gathered, routed_out.contiguous(), group=self._tp_group)
        return torch.cat(gathered, dim=0)[:original_num_tokens]

    def _finalize_output(
        self,
        hidden_states: torch.Tensor,
        routed_out: torch.Tensor,
        dispatcher_type: str,
    ) -> torch.Tensor:
        reduce_merged = dispatcher_type == "static" and self._can_reduce_merged_output
        if self._shared_experts is None:
            if reduce_merged:
                dist.all_reduce(routed_out, group=self._tp_group)
            return routed_out

        shared_out = self._shared_experts(hidden_states)
        if reduce_merged:
            output = routed_out + shared_out
            dist.all_reduce(output, group=self._tp_group)
            return output

        if self._tp_group is not None:
            dist.all_reduce(shared_out, group=self._tp_group)
        return routed_out + shared_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        moe_hidden_states, moe_router_logits, original_num_tokens = self._slice_inputs(hidden_states, router_logits)
        routed_experts = self.routed_experts
        moe_kwargs = {
            "hidden_states": moe_hidden_states,
            "router_logits": moe_router_logits,
            "num_experts": self.moe_config.num_experts,
            "top_k": self.top_k,
            "w13_weight": routed_experts.w13_weight,
            "w2_weight": routed_experts.w2_weight,
            "w13_bias": getattr(routed_experts, "w13_bias", None),
            "w2_bias": getattr(routed_experts, "w2_bias", None),
            "quant_config": self._quant_config,
            "tp_group": self._tp_group,
            "ep_group": self._ep_group,
            "dispatcher_type": None,
            "renormalize": self.renormalize,
            "custom_routing_function": self.custom_routing_function,
            "inputs_sharded": self._inputs_sharded,
            "reduce_routed_out": not self._can_reduce_merged_output,
            "return_dispatcher_type": True,
        }
        if self._quant_config is not None:
            moe_kwargs["w13_weight_scale"] = routed_experts.w13_weight_scale
            moe_kwargs["w2_weight_scale"] = routed_experts.w2_weight_scale

        routed_out, dispatcher_type = self._fused_moe(**moe_kwargs)
        routed_out = self._gather_routed_out(routed_out, original_num_tokens)
        return self._finalize_output(hidden_states, routed_out, dispatcher_type)
