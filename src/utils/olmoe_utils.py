import torch
import torch.nn as nn
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.activations import ACT2FN

from ..quantization.qlinear import QLinear
from ..quantization.quantizer import Quantizer
from ..transforms.transforms import BaseTransform, IdentityTransform
from .llama_utils import QuantizedLlamaAttention


class QuantizedOlmoeExpert(nn.Module):
    """Feed-forward expert used inside OLMoE blocks."""

    def __init__(
        self,
        config: OlmoeConfig,
        weight_quantizer: Quantizer = None,
        act_quantizer: Quantizer = None,
        gate_up_in_transform: BaseTransform = IdentityTransform(),
        down_in_transform: BaseTransform = IdentityTransform(),
    ):
        super().__init__()
        self.up_proj = QLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            weight_quantizer=weight_quantizer,
            act_quantizer=act_quantizer,
        )
        self.gate_proj = QLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            weight_quantizer=weight_quantizer,
            act_quantizer=act_quantizer,
        )
        self.down_proj = QLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            weight_quantizer=weight_quantizer,
            act_quantizer=act_quantizer,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_up_in_transform = gate_up_in_transform
        self.down_in_transform = down_in_transform
        self._train_mode = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_in_transform(x)
        up = self.up_proj(x, self.gate_up_in_transform)
        gate = self.gate_proj(x, self.gate_up_in_transform)
        x = self.act_fn(gate) * up
        x = self.down_in_transform(x)
        return self.down_proj(x, self.down_in_transform)

    def fix_parametrization(self) -> None:
        self.up_proj.fix_parametrization(self.gate_up_in_transform)
        self.gate_proj.fix_parametrization(self.gate_up_in_transform)
        self.down_proj.fix_parametrization(self.down_in_transform)
        self._train_mode = False


class QuantizedOlmoeMLP(nn.Module):
    """Mixture-of-experts MLP for OLMoE models."""

    def __init__(
        self,
        config: OlmoeConfig,
        weight_quantizer: Quantizer = None,
        act_quantizer: Quantizer = None,
        gate_up_in_transform: BaseTransform = IdentityTransform(),
        down_in_transform: BaseTransform = IdentityTransform(),
    ):
        super().__init__()
        self.num_experts = getattr(config, "num_experts", 1)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 1)
        self.router = QLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            weight_quantizer=weight_quantizer,
            act_quantizer=act_quantizer,
        )
        self.experts = nn.ModuleList(
            [
                QuantizedOlmoeExpert(
                    config,
                    weight_quantizer=weight_quantizer,
                    act_quantizer=act_quantizer,
                    gate_up_in_transform=gate_up_in_transform,
                    down_in_transform=down_in_transform,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.gate_up_in_transform = gate_up_in_transform
        self._train_mode = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Router operates on rotated input
        router_inp = self.gate_up_in_transform(x)
        router_logits = self.router(router_inp, self.gate_up_in_transform)
        routing_weights = torch.softmax(router_logits, dim=-1)
        if self.num_experts_per_tok < self.num_experts:
            topk_weights, topk_indices = torch.topk(
                routing_weights, self.num_experts_per_tok, dim=-1
            )
            routing_weights = torch.zeros_like(routing_weights).scatter(
                -1, topk_indices, topk_weights
            )
        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # Combine experts
        output = (expert_outputs * routing_weights.unsqueeze(-2)).sum(dim=-1)
        return output

    def fix_parametrization(self) -> None:
        self.router.fix_parametrization(self.gate_up_in_transform)
        for expert in self.experts:
            expert.fix_parametrization()
        self._train_mode = False


class QuantizedOlmoeAttention(QuantizedLlamaAttention):
    """Attention layer for OLMoE models. Reuses Llama attention implementation."""

    def __init__(
        self,
        config: OlmoeConfig,
        layer_idx: int,
        weight_quantizer: Quantizer = None,
        act_quantizer: Quantizer = None,
        qkv_in_transform: BaseTransform = IdentityTransform(),
        o_in_transform: BaseTransform = IdentityTransform(),
    ) -> None:
        super().__init__(
            config,
            layer_idx,
            weight_quantizer=weight_quantizer,
            act_quantizer=act_quantizer,
            qkv_in_transform=qkv_in_transform,
            o_in_transform=o_in_transform,
        )
