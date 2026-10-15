# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from ..cuda.mh_moe import Magi2MultiHeadMoE
from .attention_ops import varlen_attention_with_sink

__all__ = ["Magi2MultiHeadMoE", "varlen_attention_with_sink"]
