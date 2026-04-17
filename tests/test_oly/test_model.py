"""
Tests for the Ol-y 1.1B transformer model architecture.

Validates:
- Model initialization and parameter count
- Forward pass shapes and outputs
- Gradient flow through all components
- Gradient checkpointing functionality
- KV cache for inference
- ACT head integration
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oly.model.transformer import OlyConfig, OlyModel, OlyForCausalLM


@pytest.fixture
def small_config():
    """A tiny model config for fast testing (not 1.1B -- just a few million params)."""
    return OlyConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        act_enabled=True,
        act_num_emotions=17,
        act_hidden_size=64,
        gradient_checkpointing=False,
    )


@pytest.fixture
def model(small_config):
    """Create a small model for testing."""
    return OlyForCausalLM(small_config)


class TestOlyConfig:
    """Tests for model configuration."""

    def test_default_config(self):
        config = OlyConfig()
        assert config.vocab_size == 32000
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.intermediate_size == 6400
        assert config.act_enabled is True
        assert config.act_num_emotions == 17

    def test_config_to_dict(self):
        config = OlyConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["vocab_size"] == 32000
        assert d["act_enabled"] is True

    def test_config_from_json(self, tmp_path):
        """Test loading config from JSON file."""
        config_data = {
            "architecture": {
                "vocab_size": 500,
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 128,
            },
            "act": {
                "enabled": True,
                "num_emotions": 17,
                "act_hidden_size": 32,
            },
        }
        config_path = tmp_path / "test_config.json"
        import json
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = OlyConfig.from_json(str(config_path))
        assert config.vocab_size == 500
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 1


class TestOlyModel:
    """Tests for the base transformer model."""

    def test_forward_shape(self, small_config):
        model = OlyModel(small_config)
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        hidden_states, cache = model(input_ids)
        assert hidden_states.shape == (batch_size, seq_len, small_config.hidden_size)
        assert cache is None  # use_cache=False by default

    def test_with_attention_mask(self, small_config):
        model = OlyModel(small_config)
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -5:] = 0  # mask last 5 tokens

        hidden_states, _ = model(input_ids, attention_mask=attention_mask)
        assert hidden_states.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_kv_cache(self, small_config):
        model = OlyModel(small_config)
        model.eval()

        input_ids = torch.randint(0, small_config.vocab_size, (1, 10))
        hidden_states, cache = model(input_ids, use_cache=True)

        assert cache is not None
        assert len(cache) == small_config.num_hidden_layers
        # Each cache entry is (key, value) tuple
        for k, v in cache:
            assert k.shape[2] == 10  # seq_len cached
            assert v.shape[2] == 10

    def test_gradient_checkpointing(self, small_config):
        small_config.gradient_checkpointing = True
        model = OlyModel(small_config)
        model.train()

        input_ids = torch.randint(0, small_config.vocab_size, (1, 16))
        hidden_states, _ = model(input_ids)
        loss = hidden_states.sum()
        loss.backward()

        # Verify gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestOlyForCausalLM:
    """Tests for the complete model with LM and ACT heads."""

    def test_forward_with_labels(self, model, small_config):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        act_labels = torch.randint(0, small_config.act_num_emotions, (batch_size,))
        act_intensities = torch.rand(batch_size) * 0.9 + 0.1

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            act_labels=act_labels,
            act_intensities=act_intensities,
        )

        assert outputs["loss"] is not None
        assert outputs["logits"].shape == (batch_size, seq_len, small_config.vocab_size)
        assert outputs["act_emotion_logits"].shape == (batch_size, small_config.act_num_emotions)
        assert outputs["act_intensity"].shape == (batch_size,)

    def test_forward_without_labels(self, model, small_config):
        input_ids = torch.randint(0, small_config.vocab_size, (1, 16))
        outputs = model(input_ids=input_ids)
        assert outputs["loss"] is None
        assert outputs["logits"] is not None

    def test_act_head_callable(self, model, small_config):
        """Test that ACT head can be called separately as a function."""
        input_ids = torch.randint(0, small_config.vocab_size, (1, 16))
        act_result = model.call_act(input_ids)

        assert "emotion_logits" in act_result
        assert "intensity" in act_result
        assert "emotion_pred" in act_result
        # Intensity should be in [0.1, 1.0]
        assert act_result["intensity"].item() >= 0.1
        assert act_result["intensity"].item() <= 1.0

    def test_loss_backward(self, model, small_config):
        """Test that loss backpropagation works correctly."""
        input_ids = torch.randint(0, small_config.vocab_size, (1, 16))
        labels = torch.randint(0, small_config.vocab_size, (1, 16))
        act_labels = torch.randint(0, small_config.act_num_emotions, (1,))
        act_intensities = torch.tensor([0.7])

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            act_labels=act_labels,
            act_intensities=act_intensities,
        )

        outputs["loss"].backward()
        # Check gradients exist for trainable parameters
        trainable_with_grad = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                trainable_with_grad += 1
        assert trainable_with_grad > 0

    def test_parameter_count(self, model):
        """Test that parameter counting works."""
        counts = model.count_parameters()
        assert "total" in counts
        assert "embedding" in counts
        assert "transformer_layers" in counts
        assert counts["total"] > 0

    def test_generate(self, model, small_config):
        """Test text generation with ACT token emission."""
        input_ids = torch.randint(0, small_config.vocab_size, (1, 8))
        result = model.generate(input_ids, max_new_tokens=10, emit_act=True)

        assert "generated_ids" in result
        assert "act_result" in result
        assert result["generated_ids"].shape[1] >= 8  # at least prompt length

    def test_tied_embeddings(self, small_config):
        """Verify embedding weight tying between input and LM head."""
        small_config.tie_word_embeddings = True
        model = OlyForCausalLM(small_config)
        assert model.lm_head.weight is model.model.embed_tokens.weight


class TestFullScaleConfig:
    """Test that the full 1.1B config produces the expected parameter count."""

    @pytest.mark.slow
    def test_1b_parameter_count(self):
        """Verify ~1.1B parameters with the production config.

        This test is marked slow because it allocates the full model.
        Skip in CI with: pytest -m 'not slow'
        """
        config = OlyConfig()  # default = production config
        model = OlyForCausalLM(config)
        total = sum(p.numel() for p in model.parameters())
        # Should be approximately 1.1B (between 1.0B and 1.2B)
        assert 1_000_000_000 < total < 1_200_000_000, f"Got {total:,} parameters"
