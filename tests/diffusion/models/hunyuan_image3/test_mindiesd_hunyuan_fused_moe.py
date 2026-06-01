# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import torch


def _install_fake_mindiesd(mocker, fused_moe):
    mindiesd = types.ModuleType("mindiesd")
    layers = types.ModuleType("mindiesd.layers")
    fused_moe_module = types.ModuleType("mindiesd.layers.fused_moe")
    fused_moe_module.fused_moe = fused_moe
    mocker.patch.dict(
        sys.modules,
        {
            "mindiesd": mindiesd,
            "mindiesd.layers": layers,
            "mindiesd.layers.fused_moe": fused_moe_module,
        },
    )


def _make_layer(moe_module, shared_experts=None):
    layer = object.__new__(moe_module.MindIESDHunyuanFusedMoE)
    layer._mindiesd_weights_prepared = True
    layer._mindiesd_shared_experts = shared_experts
    layer.w13_weight = torch.randn(2, 4, 16)
    layer.w2_weight = torch.randn(2, 8, 4)
    layer.global_num_experts = 2
    layer.top_k = 1
    layer.tokens_full = True
    layer.dispatcher_type = None
    layer.tp_group = None
    layer.ep_group = None
    layer._quant_type = "none"
    layer.renormalize = False
    layer.custom_routing_function = None
    return layer


def test_forward_calls_mindiesd_fused_moe(mocker):
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    routed_out = torch.randn(3, 4)
    fused_moe = mocker.MagicMock(return_value=routed_out)
    _install_fake_mindiesd(mocker, fused_moe)
    mocker.patch.object(moe_module, "_set_forward_context_num_tokens")

    hidden_states = torch.randn(3, 4)
    router_logits = torch.randn(3, 2)
    layer = _make_layer(moe_module)

    output = layer.forward(hidden_states, router_logits)

    assert output is routed_out
    fused_moe.assert_called_once_with(
        hidden_states=hidden_states,
        router_logits=router_logits,
        num_experts=2,
        top_k=1,
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_bias=None,
        w2_bias=None,
        quant_type="none",
        tp_group=None,
        ep_group=None,
        dispatcher_type=None,
        tokens_full=True,
        renormalize=False,
        custom_routing_function=None,
        reduce_results=True,
    )


def test_forward_adds_shared_experts(mocker):
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    routed_out = torch.ones(3, 4)
    shared_out = torch.full((3, 4), 2.0)
    fused_moe = mocker.MagicMock(return_value=routed_out)
    shared_experts = mocker.MagicMock(return_value=shared_out)
    _install_fake_mindiesd(mocker, fused_moe)
    mocker.patch.object(moe_module, "_set_forward_context_num_tokens")
    all_reduce = mocker.patch.object(moe_module.dist, "all_reduce")
    mocker.patch.object(moe_module.dist, "get_world_size", return_value=2)
    tp_group = object()

    layer = _make_layer(moe_module, shared_experts=shared_experts)
    layer.tp_group = tp_group
    hidden_states = torch.randn(3, 4)
    router_logits = torch.randn(3, 2)

    output = layer.forward(hidden_states, router_logits)

    shared_experts.assert_called_once_with(hidden_states)
    all_reduce.assert_called_once_with(shared_out, group=tp_group)
    assert fused_moe.call_args.kwargs["reduce_results"] is True
    assert torch.equal(output, routed_out + shared_out)


def test_forward_passes_int8_weight_scales_to_mindiesd(mocker):
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    routed_out = torch.randn(3, 4)
    fused_moe = mocker.MagicMock(return_value=routed_out)
    _install_fake_mindiesd(mocker, fused_moe)
    mocker.patch.object(moe_module, "_set_forward_context_num_tokens")

    layer = _make_layer(moe_module)
    layer._quant_type = "int8"
    layer.w13_weight_scale = torch.randn(2, 16)
    layer.w2_weight_scale = torch.randn(2, 4)
    hidden_states = torch.randn(3, 4)
    router_logits = torch.randn(3, 2)

    output = layer.forward(hidden_states, router_logits)

    assert output is routed_out
    assert fused_moe.call_args.kwargs["quant_type"] == "int8"
    assert fused_moe.call_args.kwargs["w13_weight_scale"] is layer.w13_weight_scale
    assert fused_moe.call_args.kwargs["w2_weight_scale"] is layer.w2_weight_scale


def test_prepare_mindiesd_weights_transposes_once():
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    layer = _make_layer(moe_module)
    layer._mindiesd_weights_prepared = False
    w13_weight = layer.w13_weight
    w2_weight = layer.w2_weight

    layer._prepare_mindiesd_weights()
    first_w13 = layer.w13_weight
    first_w2 = layer.w2_weight
    layer._prepare_mindiesd_weights()

    assert torch.equal(first_w13, w13_weight.transpose(-1, -2).contiguous())
    assert torch.equal(first_w2, w2_weight.transpose(-1, -2).contiguous())
    assert layer.w13_weight is first_w13
    assert layer.w2_weight is first_w2


def test_prepare_mindiesd_int8_weights_keeps_vllm_ascend_scales(mocker):
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    layer = _make_layer(moe_module)
    layer._mindiesd_weights_prepared = False
    layer.w13_weight = torch.randint(-8, 8, (2, 4, 16), dtype=torch.int8)
    layer.w2_weight = torch.randint(-8, 8, (2, 8, 4), dtype=torch.int8)
    layer.w13_weight_scale = torch.randn(2, 16)
    layer.w2_weight_scale = torch.randn(2, 4)
    format_cast = mocker.patch.object(moe_module.torch_npu, "npu_format_cast", side_effect=lambda tensor, _: tensor)

    layer._prepare_mindiesd_weights()

    assert layer._quant_type == "int8"
    assert layer.w13_weight_scale.shape == (2, 16)
    assert layer.w2_weight_scale.shape == (2, 4)
    assert not hasattr(layer, "w13_scale")
    assert not hasattr(layer, "w2_scale")
    assert format_cast.call_count == 2
