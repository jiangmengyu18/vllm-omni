# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace


def _group(world_size, device_group):
    return SimpleNamespace(world_size=world_size, device_group=device_group)


def _resolve_kwargs(mocker, *, device_type, tp_world_size=1, ep_world_size=1):
    import vllm_omni.platforms.npu.platform as npu_platform

    tp_device_group = object()
    ep_device_group = object()
    tp_group = _group(tp_world_size, tp_device_group)
    ep_group = _group(ep_world_size, ep_device_group) if ep_world_size > 0 else None
    mocker.patch.object(npu_platform, "_has_mindiesd_fused_moe", return_value=True)
    mocker.patch.object(npu_platform, "get_tp_group", return_value=tp_group)
    mocker.patch.object(npu_platform, "_try_get_ep_group", return_value=ep_group)
    mocker.patch.object(npu_platform, "get_ascend_device_type", return_value=device_type)

    kwargs = npu_platform.NPUOmniPlatform.get_diffusion_model_impl_kwargs("hunyuan_fused_moe", {})
    return kwargs, tp_device_group, ep_device_group


def test_pure_tp_uses_static_dispatcher_on_a3(mocker):
    import vllm_omni.platforms.npu.platform as npu_platform

    kwargs, tp_device_group, _ = _resolve_kwargs(
        mocker,
        device_type=npu_platform.AscendDeviceType.A3,
        tp_world_size=2,
        ep_world_size=1,
    )

    assert kwargs["dispatcher_type"] == "static"
    assert kwargs["tp_group"] is tp_device_group
    assert "ep_group" not in kwargs


def test_ep_uses_dynamic_dispatcher_on_a3(mocker):
    import vllm_omni.platforms.npu.platform as npu_platform

    kwargs, _, ep_device_group = _resolve_kwargs(
        mocker,
        device_type=npu_platform.AscendDeviceType.A3,
        tp_world_size=1,
        ep_world_size=2,
    )

    assert kwargs["dispatcher_type"] == "dynamic"
    assert kwargs["ep_group"] is ep_device_group
    assert "tp_group" not in kwargs


def test_ep_uses_static_dispatcher_on_a2(mocker):
    import vllm_omni.platforms.npu.platform as npu_platform

    kwargs, _, ep_device_group = _resolve_kwargs(
        mocker,
        device_type=npu_platform.AscendDeviceType.A2,
        tp_world_size=1,
        ep_world_size=2,
    )

    assert kwargs["dispatcher_type"] == "static"
    assert kwargs["ep_group"] is ep_device_group
