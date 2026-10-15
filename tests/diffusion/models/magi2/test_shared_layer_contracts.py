# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Semantic contracts for MAGI-2 layers deduplicated into shared layers."""

from __future__ import annotations

import importlib

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import norm as shared_norm
from vllm_omni.diffusion.layers import rope as shared_rope
from vllm_omni.diffusion.models.magi2 import layers as magi2_layers

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def _rms_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    *,
    dim: int | tuple[int, ...] = -1,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Match the legacy order: FP32 normalize and scale, then output cast."""

    original_dtype = x.dtype
    result = x.float()
    result = result * torch.rsqrt(result.square().mean(dim=dim, keepdim=True) + eps)
    if weight is not None:
        result = result * weight.float()
    return result.to(out_dtype or original_dtype)


def _legacy_rotate_half(
    x: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    if not interleaved:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)
    first, second = x[..., ::2], x[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _legacy_apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """The deleted MAGI-2 native implementation, retained as a test oracle."""

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"RoPE dimension {rotary_dim} exceeds head dimension {x.shape[-1]}")
    if interleaved:
        cos = cos.unsqueeze(-2).repeat_interleave(2, dim=-1)
        sin = sin.unsqueeze(-2).repeat_interleave(2, dim=-1)
    else:
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    prefix = x[..., :rotary_dim]
    rotated = prefix * cos + _legacy_rotate_half(prefix, interleaved) * sin
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def _require_npu() -> torch.device:
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        pytest.skip("requires torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an available NPU")
    return torch.device("npu")


@pytest.mark.cpu
def test_rms_norm_native_keeps_fp32_scale_before_output_cast() -> None:
    eps = 1e-6
    x = torch.tensor(
        [[0.125, -0.75, 1.5, 3.0], [-2.0, 0.375, 0.625, 4.0]],
        dtype=torch.bfloat16,
    )
    gamma = torch.tensor([1.03125, 0.8125, 1.1875, 0.96875])

    actual = shared_norm.rms_norm_native(
        x,
        gamma,
        eps,
        dim=-1,
        out_dtype=torch.float32,
    )
    expected = _rms_reference(
        x,
        gamma,
        eps,
        dim=-1,
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    unscaled = shared_norm.rms_norm_native(
        x,
        weight=None,
        eps=eps,
        dim=-1,
        out_dtype=torch.float32,
    )
    expected_unscaled = _rms_reference(
        x,
        None,
        eps,
        dim=-1,
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(unscaled, expected_unscaled, rtol=0, atol=0)


@pytest.mark.cpu
def test_multimodality_norm_state_dict_keeps_flat_fp32_zero_delta() -> None:
    dim, modalities, patterns = 8, 3, 2
    norm = magi2_layers.MultiModalityRMSNorm(
        dim,
        num_modality=modalities,
        num_patterns=patterns,
    )

    state = norm.state_dict()
    assert set(state) == {"weight"}
    assert state["weight"].shape == (modalities * patterns * dim,)
    assert state["weight"].dtype == torch.float32
    torch.testing.assert_close(
        state["weight"],
        torch.zeros_like(state["weight"]),
        rtol=0,
        atol=0,
    )


@pytest.mark.cpu
def test_multimodality_norm_applies_delta_plus_one_in_original_token_order() -> None:
    eps, dim, modalities = 1e-6, 4, 3
    norm = magi2_layers.MultiModalityRMSNorm(
        dim,
        eps=eps,
        num_modality=modalities,
        out_dtype=torch.float32,
    )
    delta = torch.tensor(
        [
            [0.0, 0.25, -0.125, 0.5],
            [-0.25, 0.0, 0.375, -0.5],
            [0.125, -0.375, 0.0, 0.75],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        norm.weight.copy_(delta.flatten())

    x = torch.tensor(
        [
            [0.5, -1.0, 2.0, 0.25],
            [1.5, 0.75, -0.5, 2.5],
            [-2.0, 0.5, 1.0, -0.25],
            [0.125, 3.0, -1.5, 0.75],
            [2.0, -0.75, 0.375, 1.25],
            [-0.5, 1.25, 2.5, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    modality_mapping = torch.tensor([2, 0, 1, 2, 0, 1], dtype=torch.int32)
    dispatcher = magi2_layers.ModalityDispatcher(
        modality_mapping,
        modalities,
    )

    # MAGI-2 packs by modality before calling the normalization layer.
    actual = dispatcher.inverse_permute(norm(dispatcher.permute(x), dispatcher))
    normalized = _rms_reference(
        x,
        None,
        eps,
        out_dtype=torch.float32,
    )
    gamma_by_token = delta[modality_mapping.long()] + 1.0
    expected = normalized * gamma_by_token

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_multimodality_norm_pattern_two_normalizes_each_pattern() -> None:
    eps, dim, patterns = 1e-6, 4, 2
    norm = magi2_layers.MultiModalityRMSNorm(
        dim,
        eps=eps,
        num_patterns=patterns,
        out_dtype=torch.float32,
    )
    delta = torch.tensor(
        [[0.0, 0.25, -0.125, 0.5], [-0.25, 0.0, 0.375, -0.5]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        norm.weight.copy_(delta.flatten())
    x = torch.tensor(
        [
            [[0.5, -1.0, 2.0, 0.25], [1.5, 0.75, -0.5, 2.5]],
            [[-2.0, 0.5, 1.0, -0.25], [0.125, 3.0, -1.5, 0.75]],
            [[2.0, -0.75, 0.375, 1.25], [-0.5, 1.25, 2.5, -1.0]],
        ],
        dtype=torch.bfloat16,
    )

    actual = norm(x)
    expected = _rms_reference(
        x,
        delta + 1.0,
        eps,
        dim=-1,
        out_dtype=torch.float32,
    )

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("batch_dependent", [False, True])
@pytest.mark.parametrize("rotary_dim", [96, 128])
def test_shared_native_rope_matches_legacy_batch_and_partial_tail(
    interleaved: bool,
    batch_dependent: bool,
    rotary_dim: int,
) -> None:
    generator = torch.Generator().manual_seed(17)
    batch, sequence, heads, head_dim = 2, 5, 3, 128
    x = torch.randn(
        batch,
        sequence,
        heads,
        head_dim,
        generator=generator,
    )
    phase_shape = (batch, sequence, rotary_dim // 2) if batch_dependent else (sequence, rotary_dim // 2)
    phase = torch.randn(*phase_shape, generator=generator)
    cos, sin = phase.cos(), phase.sin()

    actual = shared_rope.apply_rotary_emb_torch(
        x,
        cos,
        sin,
        interleaved=interleaved,
    )
    expected = _legacy_apply_rotary_emb(
        x,
        cos,
        sin,
        interleaved=interleaved,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual[..., rotary_dim:], x[..., rotary_dim:])


@pytest.mark.cpu
def test_shared_native_rope_preserves_legacy_dtype_promotion() -> None:
    generator = torch.Generator().manual_seed(18)
    x = torch.randn(2, 4, 3, 128, generator=generator).to(torch.bfloat16)
    phase = torch.randn(2, 4, 48, generator=generator)
    cos, sin = phase.cos(), phase.sin()

    actual = shared_rope.apply_rotary_emb_torch(x, cos, sin)
    expected = _legacy_apply_rotary_emb(x, cos, sin)

    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual[..., 96:], x[..., 96:].float())


@pytest.mark.cpu
def test_shared_native_rope_preserves_value_error_for_oversized_rotary_dim() -> None:
    x = torch.randn(1, 2, 3, 128)
    cos = torch.randn(2, 65)
    sin = torch.randn(2, 65)

    with pytest.raises(
        ValueError,
        match=r"RoPE dimension 130 exceeds head dimension 128",
    ):
        shared_rope.apply_rotary_emb_torch(x, cos, sin)


@pytest.mark.cpu
def test_shared_npu_rope_helper_falls_back_for_cpu_tensor() -> None:
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(2, 4, 3, 128, generator=generator)
    phase = torch.randn(2, 4, 48, generator=generator)
    cos, sin = phase.cos(), phase.sin()

    actual = shared_rope.apply_rotary_emb_npu(x, cos, sin)
    expected = _legacy_apply_rotary_emb(x, cos, sin)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual[..., 96:], x[..., 96:])


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_rms_norm_npu_matches_fp32_reference() -> None:
    device = _require_npu()
    eps = 1e-6
    generator = torch.Generator().manual_seed(20)
    x_cpu = torch.randn(17, 64, generator=generator)
    gamma_cpu = torch.randn(64, generator=generator) * 0.1 + 1.0
    x = x_cpu.to(device)
    gamma = gamma_cpu.to(device)

    actual = shared_norm.rms_norm_npu(x, gamma, eps)
    expected = _rms_reference(
        x_cpu,
        gamma_cpu,
        eps,
        out_dtype=torch.float32,
    )

    assert actual.device.type == "npu"
    assert actual.dtype == torch.float32
    assert actual.shape == x.shape
    # The NPU reduction order can differ from the CPU oracle; this is numerical
    # parity, not a bitwise CPU/NPU equivalence requirement.
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=2e-4,
        atol=2e-4,
    )


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_shared_npu_rope_keeps_batch_cos_dtype_and_partial_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _require_npu()
    generator = torch.Generator().manual_seed(21)
    batch, sequence, heads, head_dim, rotary_dim = 2, 5, 4, 128, 96
    x_cpu = torch.randn(
        batch,
        sequence,
        heads,
        head_dim,
        generator=generator,
    ).to(torch.bfloat16)
    phase = torch.randn(batch, sequence, rotary_dim // 2, generator=generator)
    cos_cpu, sin_cpu = phase.cos(), phase.sin()
    x = x_cpu.to(device)
    cos, sin = cos_cpu.to(device), sin_cpu.to(device)

    torch_npu = importlib.import_module("torch_npu")
    real_op = torch_npu.npu_rotary_mul
    calls = []

    def checked_rotary_mul(prefix, op_cos, op_sin, *, rotary_mode):
        assert prefix.shape == (batch, sequence, heads, rotary_dim)
        assert op_cos.shape == op_sin.shape == (batch, sequence, 1, rotary_dim)
        assert prefix.is_contiguous() and op_cos.is_contiguous() and op_sin.is_contiguous()
        assert prefix.dtype == op_cos.dtype == op_sin.dtype == x.dtype
        assert rotary_mode == "half"
        calls.append(True)
        return real_op(prefix, op_cos, op_sin, rotary_mode=rotary_mode)

    monkeypatch.setattr(torch_npu, "npu_rotary_mul", checked_rotary_mul)
    actual = shared_rope.apply_rotary_emb_npu(x, cos, sin)
    assert calls == [True]
    expected = _legacy_apply_rotary_emb(
        x_cpu,
        cos_cpu.to(x_cpu.dtype),
        sin_cpu.to(x_cpu.dtype),
    )

    assert actual.device.type == "npu"
    assert actual.dtype == x.dtype
    assert actual.shape == x.shape
    assert torch.equal(actual[..., rotary_dim:], x[..., rotary_dim:])
    # BF16 rotary_mul may use a different fused evaluation order from the CPU
    # expression, so compare within a BF16-scale numerical tolerance.
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )

    bad_cos = torch.randn(batch, sequence, 65, device=device)
    bad_sin = torch.randn(batch, sequence, 65, device=device)
    with pytest.raises(ValueError, match=r"RoPE dimension 130"):
        shared_rope.apply_rotary_emb_npu(x, bad_cos, bad_sin)


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.parametrize(
    ("shape", "interleaved"),
    [
        ((5, 4, 128), False),
        ((1, 5, 4, 128), True),
    ],
    ids=["rank-three", "interleaved"],
)
def test_shared_npu_rope_unsupported_layouts_use_native_semantics(
    shape: tuple[int, ...],
    interleaved: bool,
) -> None:
    device = _require_npu()
    generator = torch.Generator().manual_seed(22)
    x = torch.randn(*shape, generator=generator).to(torch.bfloat16).to(device)
    sequence = shape[-3]
    phase = torch.randn(sequence, 48, generator=generator).to(device)
    cos, sin = phase.cos(), phase.sin()

    actual = shared_rope.apply_rotary_emb_npu(
        x,
        cos,
        sin,
        interleaved=interleaved,
    )
    expected = shared_rope.apply_rotary_emb_torch(
        x,
        cos,
        sin,
        interleaved=interleaved,
    )

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.cpu(),
        expected.cpu(),
        rtol=2e-4,
        atol=2e-4,
    )


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_multimodality_npu_norm_handles_an_empty_modality_group() -> None:
    device = _require_npu()
    eps, dim, modalities = 1e-6, 8, 3
    norm = magi2_layers.MultiModalityRMSNorm(
        dim,
        eps=eps,
        num_modality=modalities,
        out_dtype=torch.float32,
    ).to(device)
    delta = torch.tensor(
        [
            [0.0, 0.25, -0.125, 0.5, 0.125, -0.25, 0.375, 0.0],
            [-0.25, 0.0, 0.375, -0.5, 0.25, 0.125, 0.0, -0.125],
            [0.125, -0.375, 0.0, 0.75, -0.25, 0.5, -0.125, 0.25],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        norm.weight.copy_(delta.flatten().to(device))

    generator = torch.Generator().manual_seed(23)
    x_cpu = torch.randn(5, dim, generator=generator).to(torch.bfloat16)
    modality_mapping_cpu = torch.tensor([2, 0, 2, 0, 2], dtype=torch.int32)
    x = x_cpu.to(device)
    modality_mapping = modality_mapping_cpu.to(device)
    dispatcher = magi2_layers.ModalityDispatcher(
        modality_mapping,
        modalities,
    )

    actual = dispatcher.inverse_permute(norm(dispatcher.permute(x), dispatcher))
    normalized = _rms_reference(
        x_cpu,
        None,
        eps,
        out_dtype=torch.float32,
    )
    expected = normalized * (delta[modality_mapping_cpu.long()] + 1.0)

    assert actual.dtype == torch.float32
    assert actual.shape == x.shape
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=2e-4,
        atol=2e-4,
    )


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_multimodality_npu_norm_pattern_two_uses_native_contract() -> None:
    device = _require_npu()
    eps, dim, patterns = 1e-6, 8, 2
    norm = magi2_layers.MultiModalityRMSNorm(
        dim,
        eps=eps,
        num_patterns=patterns,
        out_dtype=torch.float32,
    ).to(device)
    generator = torch.Generator().manual_seed(24)
    delta = torch.randn(patterns, dim, generator=generator) * 0.1
    with torch.no_grad():
        norm.weight.copy_(delta.flatten().to(device))
    x_cpu = torch.randn(7, patterns, dim, generator=generator).to(torch.bfloat16)
    x = x_cpu.to(device)

    actual = norm(x)
    expected = _rms_reference(
        x_cpu,
        delta + 1.0,
        eps,
        out_dtype=torch.float32,
    )

    assert actual.dtype == torch.float32
    assert actual.shape == x.shape
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=2e-4,
        atol=2e-4,
    )
