import torch
from typing import Optional, Callable, Set
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack


class DecompressionAwareAttention(Qwen2Attention):
    """Qwen2 attention that detects when model attends to reference tokens."""

    def __init__(
        self, config, layer_idx: int, reference_token_ids: Optional[Set[int]] = None
    ):
        super().__init__(config, layer_idx)
        self.reference_token_ids = reference_token_ids or set()
        self._detection_enabled = True

    def set_reference_tokens(self, token_ids: Set[int]):
        self.reference_token_ids = token_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        if (
            self._detection_enabled
            and attn_weights is not None
            and input_ids is not None
        ):
            self._detect_reference_attention(attn_weights, input_ids)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _detect_reference_attention(
        self, attn_weights: torch.Tensor, input_ids: torch.LongTensor
    ):
        if not self.reference_token_ids:
            return

        reference_positions = []
        for pos, token_id in enumerate(input_ids[0].tolist()):
            if token_id in self.reference_token_ids:
                reference_positions.append(pos)

        if not reference_positions:
            return

        avg_weights = attn_weights.mean(dim=1)
        last_token_weights = avg_weights[0, -1, :]

        for pos in reference_positions:
            if pos < last_token_weights.shape[0]:
                weight = last_token_weights[pos].item()
                if weight > 0.01:
                    print(
                        f"[Layer {self.layer_idx}] Attention to ref @ pos {pos}: {weight:.4f}"
                    )


def replace_attention_layers(model, reference_token_ids: Set[int]):
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        new_attn = DecompressionAwareAttention(
            config=model.config,
            layer_idx=layer_idx,
            reference_token_ids=reference_token_ids,
        )
        new_attn.q_proj = old_attn.q_proj
        new_attn.k_proj = old_attn.k_proj
        new_attn.v_proj = old_attn.v_proj
        new_attn.o_proj = old_attn.o_proj
        layer.self_attn = new_attn

    return model
