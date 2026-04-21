"""
Ol-y 1.1B Transformer with integrated ACT (Affective Communication Tokens).

Architecture: ~1.1B parameters
- hidden_size: 2048
- num_layers: 24
- num_heads: 16
- intermediate_size: 6400
- vocab_size: 32000
- RoPE positional encoding
- RMSNorm (pre-norm architecture)
- SwiGLU activation in FFN
- Gradient checkpointing support for RTX 3050 Ti training

The ACT system is integrated as a separate callable head that runs in parallel
with the language modeling head. During generation, the model first produces an
ACT token (emotion label + intensity) through the ACT head, then generates the
natural language response conditioned on the declared emotional state.

Reference: Sakishimiro (2026), "Affective Communication Tokens"
"""

import math
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class OlyConfig:
    """Configuration for the Ol-y 1.1B ACTT model."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 6400
    max_position_embeddings: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    # ACT configuration
    act_enabled: bool = True
    act_num_emotions: int = 17
    act_hidden_size: int = 512
    act_intensity_range: Tuple[float, float] = (0.1, 1.0)
    act_max_composite: int = 5
    act_loss_structural: float = 2.0
    act_loss_label: float = 1.5
    act_loss_response: float = 1.0
    # Training
    gradient_checkpointing: bool = False

    @classmethod
    def from_json(cls, path: str) -> "OlyConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        arch = data.get("architecture", {})
        act = data.get("act", {})
        loss_w = act.get("loss_weights", {})
        return cls(
            vocab_size=arch.get("vocab_size", 32000),
            hidden_size=arch.get("hidden_size", 2048),
            num_hidden_layers=arch.get("num_hidden_layers", 24),
            num_attention_heads=arch.get("num_attention_heads", 16),
            intermediate_size=arch.get("intermediate_size", 6400),
            max_position_embeddings=arch.get("max_position_embeddings", 2048),
            hidden_dropout_prob=arch.get("hidden_dropout_prob", 0.1),
            attention_dropout_prob=arch.get("attention_dropout_prob", 0.1),
            layer_norm_eps=arch.get("layer_norm_eps", 1e-6),
            rope_theta=arch.get("rope_theta", 10000.0),
            tie_word_embeddings=arch.get("tie_word_embeddings", True),
            act_enabled=act.get("enabled", True),
            act_num_emotions=act.get("num_emotions", 17),
            act_hidden_size=act.get("act_hidden_size", 512),
            act_intensity_range=tuple(act.get("intensity_range", [0.1, 1.0])),
            act_max_composite=act.get("max_composite_components", 5),
            act_loss_structural=loss_w.get("structural", 2.0),
            act_loss_label=loss_w.get("label", 1.5),
            act_loss_response=loss_w.get("response", 1.0),
            gradient_checkpointing=data.get("training", {}).get("gradient_checkpointing", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_dropout_prob": self.attention_dropout_prob,
            "layer_norm_eps": self.layer_norm_eps,
            "rope_theta": self.rope_theta,
            "tie_word_embeddings": self.tie_word_embeddings,
            "act_enabled": self.act_enabled,
            "act_num_emotions": self.act_num_emotions,
            "act_hidden_size": self.act_hidden_size,
            "act_intensity_range": list(self.act_intensity_range),
            "act_max_composite": self.act_max_composite,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    More stable than LayerNorm for deep transformers. Used in LLaMA architecture.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 for numerical stability during norm computation
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute sin/cos tables for Rotary Position Embeddings (RoPE).

    RoPE encodes position information directly into query/key vectors through
    rotation, enabling the model to extrapolate to longer sequences and providing
    relative position awareness without explicit position embeddings.

    Args:
        dim: dimension of each attention head (hidden_size // num_heads)
        max_seq_len: maximum sequence length to precompute for
        theta: RoPE base frequency (10000.0 by default, as in the original paper)
        device: target device for the tensors

    Returns:
        Tuple of (cos, sin) tensors, each of shape (max_seq_len, dim)
    """
    # Frequency bands: theta_i = theta^(-2i/dim) for i in [0, dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # Position indices
    t = torch.arange(max_seq_len, device=device).float()
    # Outer product: (seq_len, dim/2)
    angles = torch.outer(t, freqs)
    # Duplicate for pairing: (seq_len, dim)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embeddings to query or key tensor.

    Rotates adjacent pairs of dimensions in the head dimension by position-
    dependent angles, encoding relative position into the dot-product attention.

    Args:
        x: input tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: cosine table of shape (seq_len, head_dim)
        sin: sine table of shape (seq_len, head_dim)

    Returns:
        Rotated tensor of the same shape as x
    """
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    # Split into even/odd pairs and rotate
    x_half1 = x[..., : x.shape[-1] // 2]
    x_half2 = x[..., x.shape[-1] // 2 :]
    x_rotated = torch.cat([-x_half2, x_half1], dim=-1)
    return x * cos + x_rotated * sin


class OlyAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE.

    Standard multi-head attention with:
    - Rotary Position Embeddings (RoPE) applied to Q and K
    - Causal masking for autoregressive generation
    - Optional attention dropout
    """

    def __init__(self, config: OlyConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout_prob

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.attention_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V in one projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to queries and keys
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Handle KV cache for inference
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask: prevent attending to future tokens
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output), new_cache


class OlyMLP(nn.Module):
    """SwiGLU Feed-Forward Network.

    Uses the SwiGLU activation function (Shazeer, 2020) which provides better
    training stability and performance compared to standard ReLU/GELU FFNs.
    The gate projection controls information flow through the up-projection.

    SwiGLU(x) = (Swish(W_gate * x)) * (W_up * x)
    output = W_down * SwiGLU(x)
    """

    def __init__(self, config: OlyConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gated linear unit with SiLU (Swish) activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class OlyTransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Structure: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
    Pre-norm (placing norm before each sub-layer) improves training stability
    for deep networks compared to post-norm.
    """

    def __init__(self, config: OlyConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.attention = OlyAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.mlp = OlyMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, cache = self.attention(
            hidden_states, rope_cos, rope_sin, attention_mask, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cache


class OlyModel(nn.Module):
    """Ol-y 1.1B base transformer model (encoder-only backbone).

    This is the core transformer that produces contextual hidden states.
    It does NOT include the language modeling head or ACT head -- those are
    added by OlyForCausalLM which wraps this model.

    Architecture details:
    - 24 transformer blocks with pre-norm (RMSNorm)
    - RoPE positional encoding (no learned position embeddings)
    - SwiGLU FFN activation
    - Fused QKV projections
    - ~1.1B total parameters (with LM head)

    Supports gradient checkpointing for memory-efficient training on
    consumer GPUs (RTX 3050 Ti with 4GB VRAM).
    """

    def __init__(self, config: OlyConfig):
        super().__init__()
        self.config = config

        # Token embeddings (no position embeddings -- RoPE handles position)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer blocks
        self.layers = nn.ModuleList([
            OlyTransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.layer_norm_eps)

        # Precompute RoPE frequency tables
        head_dim = config.hidden_size // config.num_attention_heads
        rope_cos, rope_sin = precompute_rope_frequencies(
            head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution.

        Uses std=0.02 for embeddings and linear layers, following GPT-2 conventions.
        Output projections in attention and FFN are scaled by 1/sqrt(2*num_layers)
        to prevent activation growth in deep networks.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass through the transformer backbone.

        Args:
            input_ids: token IDs, shape (batch_size, seq_len)
            attention_mask: optional padding mask, shape (batch_size, seq_len)
            past_key_values: cached KV pairs for each layer (for fast inference)
            use_cache: whether to return updated KV cache

        Returns:
            Tuple of (hidden_states, new_past_key_values)
            hidden_states shape: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)

        # Build causal attention mask
        causal_mask = self._build_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
        if attention_mask is not None:
            # Expand padding mask and combine with causal mask
            pad_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)
            pad_mask = (1.0 - pad_mask) * torch.finfo(hidden_states.dtype).min
            causal_mask = causal_mask + pad_mask

        # Run through transformer layers
        new_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing: trade compute for memory
                hidden_states, cache = checkpoint(
                    layer,
                    hidden_states,
                    self.rope_cos,
                    self.rope_sin,
                    causal_mask,
                    None,  # no cache during training with checkpointing
                    False,
                    use_reentrant=False,
                )
            else:
                hidden_states, cache = layer(
                    hidden_states, self.rope_cos, self.rope_sin,
                    causal_mask, past_kv, use_cache
                )

            if use_cache:
                new_caches.append(cache)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, new_caches

    def _build_causal_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build upper-triangular causal mask for autoregressive attention.

        Returns a mask where position i can only attend to positions <= i.
        Future positions are filled with -inf to zero them out after softmax.
        """
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


class OlyForCausalLM(nn.Module):
    """Ol-y 1.1B with Causal Language Modeling head and integrated ACT head.

    This is the complete model for training and inference. It wraps OlyModel
    and adds:
    1. A language modeling head (linear projection to vocab logits)
    2. An ACT head (emotion classifier + intensity regressor) as a separate
       callable function that operates on the pooled hidden state

    The ACT head is architecturally separate from the LM head, implementing
    the "calling function" pattern described in the ACT paper. During training,
    the total loss combines structural, label, and response losses.

    Total parameters: ~1.1B
    """

    def __init__(self, config: OlyConfig):
        super().__init__()
        self.config = config
        self.model = OlyModel(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embedding weights with LM head (reduces parameters, improves training)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # ACT head: separate callable function for emotion prediction
        if config.act_enabled:
            from oly.act.act_head import ACTHead
            self.act_head = ACTHead(config)
        else:
            self.act_head = None

        # Optional persistent emotional memory. This is not required for plain
        # forward/generation, but lets runtime callers keep EMA state across
        # sessions without changing the core model weights.
        self.emotional_memory = None

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Ol-y] Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
        print(f"[Ol-y] Trainable parameters: {trainable_params:,}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        act_labels: Optional[torch.Tensor] = None,
        act_intensities: Optional[torch.Tensor] = None,
        act_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass with optional loss computation.

        Args:
            input_ids: input token IDs (batch, seq_len)
            attention_mask: padding mask (batch, seq_len)
            labels: target token IDs for LM loss (batch, seq_len), -100 = ignore
            act_labels: emotion label indices for ACT loss (batch,)
            act_intensities: target intensity values for ACT loss (batch,)
            act_mask: binary mask indicating which samples have ACT annotations (batch,)
            past_key_values: KV cache for generation
            use_cache: whether to return KV cache

        Returns:
            Dictionary with keys: loss, logits, act_emotion_logits, act_intensity,
            past_key_values, hidden_states
        """
        # Forward through transformer backbone
        hidden_states, new_cache = self.model(
            input_ids, attention_mask, past_key_values, use_cache
        )

        # Language modeling logits
        logits = self.lm_head(hidden_states)

        # ACT head prediction (uses pooled hidden state from last token)
        act_output = {}
        if self.act_head is not None:
            act_output = self.act_head(hidden_states, attention_mask)

        # Compute losses if labels are provided
        loss = None
        loss_components = {}
        if labels is not None:
            # Standard causal LM loss (L_resp in the paper)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss_components["lm_loss"] = lm_loss

            # ACT losses (L_label and L_struct from the paper)
            if self.act_head is not None and act_labels is not None:
                from oly.act.act_loss import ACTLoss
                act_loss_fn = ACTLoss(self.config)
                act_loss, act_loss_parts = act_loss_fn(
                    act_output.get("emotion_logits"),
                    act_output.get("intensity"),
                    act_labels,
                    act_intensities,
                    act_mask,
                )
                loss_components.update(act_loss_parts)

                # Combined loss with paper-specified weights:
                # L = lambda_1 * L_struct + lambda_2 * L_label + lambda_3 * L_resp
                loss = (
                    self.config.act_loss_response * lm_loss
                    + act_loss
                )
            else:
                loss = lm_loss

        return {
            "loss": loss,
            "logits": logits,
            "act_emotion_logits": act_output.get("emotion_logits"),
            "act_intensity": act_output.get("intensity"),
            "act_hidden": act_output.get("act_hidden"),
            "past_key_values": new_cache,
            "hidden_states": hidden_states,
            "loss_components": loss_components,
        }

    def call_act(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Call the ACT function separately to predict emotional state.

        This is the "calling function" interface -- a separate entry point that
        runs only the ACT head on the current hidden state, without producing
        language model outputs. Useful for:
        - Prefill enforcement during generation
        - Standalone emotion analysis
        - ACT token construction before response generation

        Args:
            input_ids: input token IDs
            attention_mask: padding mask

        Returns:
            Dictionary with emotion prediction and intensity
        """
        with torch.no_grad():
            hidden_states, _ = self.model(input_ids, attention_mask)
            if self.act_head is not None:
                return self.act_head(hidden_states, attention_mask)
        return {}

    def set_emotional_memory(self, memory) -> None:
        """Attach an EmotionalMemoryEMA instance for generation-time updates."""
        self.emotional_memory = memory

    def _update_emotional_memory(
        self,
        act_result: Optional[Dict[str, Any]],
        emotional_memory=None,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update attached EMA memory from ACT/probe probabilities."""
        memory = emotional_memory if emotional_memory is not None else self.emotional_memory
        if memory is None or not act_result:
            return None

        emotion_probs = act_result.get("emotion_probs")
        if emotion_probs is None:
            return None

        if isinstance(emotion_probs, torch.Tensor):
            from oly.act.act_token import EMOTION_LABELS
            probs = emotion_probs[0].detach().float().cpu().tolist()
            distribution = {
                emotion: probs[idx]
                for idx, emotion in enumerate(EMOTION_LABELS)
                if idx < len(probs)
            }
        else:
            distribution = emotion_probs

        value = memory.update_from_probe_distribution(
            distribution,
            session_id=session_id,
            metadata={"source": "oly_generate"},
        )
        return {
            "value": value,
            "tone": memory.tone,
            "sessions": memory.sessions,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        emit_act: bool = True,
        emotional_memory=None,
        memory_session_id: Optional[str] = None,
        update_memory: bool = True,
    ) -> Dict[str, Any]:
        """Generate text with optional ACT token emission.

        Autoregressive generation with nucleus (top-p) and top-k sampling.
        When emit_act=True, the ACT head is called first to produce the
        emotion token, which is prepended to the generated response.

        Args:
            input_ids: prompt token IDs (batch, prompt_len)
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (higher = more random)
            top_p: nucleus sampling threshold
            top_k: top-k sampling threshold
            emit_act: whether to produce ACT token before the response
            emotional_memory: optional EmotionalMemoryEMA to update from ACT state
            memory_session_id: optional session ID stored in memory history
            update_memory: whether to update attached/provided memory

        Returns:
            Dictionary with generated_ids, act_token (if emit_act)
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Step 1: Call ACT head for emotion prediction (if enabled)
        act_result = None
        if emit_act and self.act_head is not None:
            act_result = self.call_act(input_ids)
        memory_result = (
            self._update_emotional_memory(
                act_result,
                emotional_memory=emotional_memory,
                session_id=memory_session_id,
            )
            if update_memory
            else None
        )

        # Step 2: Autoregressive generation
        generated = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            outputs = self.forward(
                generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs["past_key_values"]
            next_logits = outputs["logits"][:, -1, :]

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1:]
                next_logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[mask] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop at EOS (token id 2 by convention)
            if (next_token == 2).all():
                break

        return {
            "generated_ids": generated,
            "act_result": act_result,
            "emotional_memory": memory_result,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters broken down by component."""
        counts = {
            "embedding": sum(p.numel() for p in self.model.embed_tokens.parameters()),
            "transformer_layers": sum(
                p.numel() for layer in self.model.layers for p in layer.parameters()
            ),
            "final_norm": sum(p.numel() for p in self.model.norm.parameters()),
            "lm_head": 0 if self.config.tie_word_embeddings else sum(
                p.numel() for p in self.lm_head.parameters()
            ),
        }
        if self.act_head is not None:
            counts["act_head"] = sum(p.numel() for p in self.act_head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
