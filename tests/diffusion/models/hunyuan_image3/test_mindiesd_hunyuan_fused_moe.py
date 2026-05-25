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
    layer.reduce_results = None
    layer.dispatcher_type = "static"
    layer.tp_group = None
    layer.ep_group = None
    layer.renormalize = False
    layer.custom_routing_function = None
    return layer


def _mock_group_world_size(group):
    return 2 if group is not None else 1


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
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        router_logits=router_logits,
        num_experts=2,
        top_k=1,
        w13_bias=None,
        w2_bias=None,
        tokens_full=True,
        reduce_results=None,
        dispatcher_type="static",
        tp_group=None,
        ep_group=None,
        renormalize=False,
        custom_routing_function=None,
    )


def test_forward_adds_shared_experts_and_reduces_tp(mocker):
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    routed_out = torch.ones(3, 4)
    shared_out = torch.full((3, 4), 2.0)
    fused_moe = mocker.MagicMock(return_value=routed_out)
    shared_experts = mocker.MagicMock(return_value=shared_out)
    _install_fake_mindiesd(mocker, fused_moe)
    mocker.patch.object(moe_module, "_set_forward_context_num_tokens")
    mocker.patch.object(moe_module, "_get_group_world_size", side_effect=_mock_group_world_size)
    all_reduce = mocker.patch.object(moe_module.dist, "all_reduce")
    tp_group = object()

    layer = _make_layer(moe_module, shared_experts=shared_experts)
    layer.tp_group = tp_group
    hidden_states = torch.randn(3, 4)
    router_logits = torch.randn(3, 2)

    output = layer.forward(hidden_states, router_logits)

    shared_experts.assert_called_once_with(hidden_states)
    all_reduce.assert_called_once()
    assert all_reduce.call_args.kwargs["group"] is tp_group
    assert all_reduce.call_args.args[0] is output
    assert fused_moe.call_args.kwargs["reduce_results"] is False
    assert torch.equal(output, routed_out + shared_out)


def test_ep_keeps_routed_reduce_results_for_mindiesd(mocker):
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    fused_moe = mocker.MagicMock(return_value=torch.ones(3, 4))
    shared_experts = mocker.MagicMock(return_value=torch.ones(3, 4))
    _install_fake_mindiesd(mocker, fused_moe)
    mocker.patch.object(moe_module, "_set_forward_context_num_tokens")
    mocker.patch.object(
        moe_module,
        "_get_group_world_size",
        side_effect=_mock_group_world_size,
    )
    all_reduce = mocker.patch.object(moe_module.dist, "all_reduce")

    layer = _make_layer(moe_module, shared_experts=shared_experts)
    layer.tp_group = object()
    layer.ep_group = object()
    layer.dispatcher_type = "dynamic"

    output = layer.forward(torch.randn(3, 4), torch.randn(3, 2))

    assert fused_moe.call_args.kwargs["reduce_results"] is None
    all_reduce.assert_called_once()
    assert all_reduce.call_args.args[0] is output


def test_static_ep_reduces_routed_output_inside_mindiesd(mocker):
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    routed_out = torch.ones(3, 4)
    shared_out = torch.full((3, 4), 2.0)
    fused_moe = mocker.MagicMock(return_value=routed_out)
    shared_experts = mocker.MagicMock(return_value=shared_out)
    _install_fake_mindiesd(mocker, fused_moe)
    mocker.patch.object(moe_module, "_set_forward_context_num_tokens")
    mocker.patch.object(moe_module, "_get_group_world_size", side_effect=_mock_group_world_size)
    ep_group = object()

    layer = _make_layer(moe_module, shared_experts=shared_experts)
    layer.ep_group = ep_group

    output = layer.forward(torch.randn(3, 4), torch.randn(3, 2))

    assert fused_moe.call_args.kwargs["reduce_results"] is True
    assert torch.equal(output, routed_out + shared_out)


def test_static_ep_tp_reduces_routed_on_ep_and_combined_output_on_tp(mocker):
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    routed_out = torch.ones(3, 4)
    shared_out = torch.full((3, 4), 2.0)
    fused_moe = mocker.MagicMock(return_value=routed_out)
    shared_experts = mocker.MagicMock(return_value=shared_out)
    _install_fake_mindiesd(mocker, fused_moe)
    mocker.patch.object(moe_module, "_set_forward_context_num_tokens")
    mocker.patch.object(moe_module, "_get_group_world_size", side_effect=_mock_group_world_size)
    all_reduce = mocker.patch.object(moe_module.dist, "all_reduce")
    ep_group = object()
    tp_group = object()

    layer = _make_layer(moe_module, shared_experts=shared_experts)
    layer.ep_group = ep_group
    layer.tp_group = tp_group

    output = layer.forward(torch.randn(3, 4), torch.randn(3, 2))

    assert fused_moe.call_args.kwargs["reduce_results"] is True
    all_reduce.assert_called_once()
    assert all_reduce.call_args.args[0] is output
    assert all_reduce.call_args.kwargs["group"] is tp_group


def test_ep_tp_without_shared_reduces_routed_output_on_tp(mocker):
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    routed_out = torch.ones(3, 4)
    fused_moe = mocker.MagicMock(return_value=routed_out)
    _install_fake_mindiesd(mocker, fused_moe)
    mocker.patch.object(moe_module, "_set_forward_context_num_tokens")
    mocker.patch.object(moe_module, "_get_group_world_size", side_effect=_mock_group_world_size)
    all_reduce = mocker.patch.object(moe_module.dist, "all_reduce")

    layer = _make_layer(moe_module)
    layer.tp_group = object()
    layer.ep_group = object()
    layer.dispatcher_type = "dynamic"

    output = layer.forward(torch.randn(3, 4), torch.randn(3, 2))

    assert output is routed_out
    all_reduce.assert_called_once_with(routed_out, group=layer.tp_group)


def test_explicit_reduce_results_is_preserved(mocker):
    import vllm_omni.platforms.npu.models.mindiesd_hunyuan_fused_moe as moe_module

    fused_moe = mocker.MagicMock(return_value=torch.ones(3, 4))
    shared_experts = mocker.MagicMock(return_value=torch.ones(3, 4))
    _install_fake_mindiesd(mocker, fused_moe)
    mocker.patch.object(moe_module, "_set_forward_context_num_tokens")
    mocker.patch.object(moe_module, "_get_group_world_size", side_effect=_mock_group_world_size)
    mocker.patch.object(moe_module.dist, "all_reduce")

    layer = _make_layer(moe_module, shared_experts=shared_experts)
    layer.tp_group = object()
    layer.reduce_results = True

    layer.forward(torch.randn(3, 4), torch.randn(3, 2))

    assert fused_moe.call_args.kwargs["reduce_results"] is True


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
