# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace


def _group(world_size, device_group):
    return SimpleNamespace(world_size=world_size, device_group=device_group)


def _resolve_kwargs(mocker, *, tp_world_size=1, ep_world_size=1):
    import vllm_omni.platforms.npu.platform as npu_platform

    tp_device_group = object()
    ep_device_group = object()
    tp_group = _group(tp_world_size, tp_device_group)
    ep_group = _group(ep_world_size, ep_device_group) if ep_world_size > 0 else None
    mocker.patch.object(npu_platform, "_has_mindiesd_fused_moe", return_value=True)
    mocker.patch.object(npu_platform, "get_tp_group", return_value=tp_group)
    mocker.patch.object(npu_platform, "_try_get_ep_group", return_value=ep_group)

    kwargs = npu_platform.NPUOmniPlatform.get_diffusion_model_impl_kwargs("hunyuan_fused_moe", {})
    return kwargs, tp_device_group, ep_device_group


def test_pure_tp_passes_tp_group_without_dispatcher_override(mocker):
    kwargs, tp_device_group, _ = _resolve_kwargs(mocker, tp_world_size=2, ep_world_size=1)

    assert kwargs["tp_group"] is tp_device_group
    assert kwargs["tokens_full"] is True
    assert "ep_group" not in kwargs
    assert kwargs["dispatcher_type"] is None


def test_ep_passes_ep_group_without_dispatcher_override(mocker):
    kwargs, _, ep_device_group = _resolve_kwargs(mocker, tp_world_size=1, ep_world_size=2)

    assert kwargs["ep_group"] is ep_device_group
    assert kwargs["tokens_full"] is True
    assert "tp_group" not in kwargs
    assert kwargs["dispatcher_type"] is None
