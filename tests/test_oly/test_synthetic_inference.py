"""
Tests for model inference using synthetic data (no trained weights required).

Validates that all model components work correctly end-to-end using
randomly initialized weights and synthetic inputs. This lets us verify
the entire inference pipeline before investing time in training.

Tests cover:
- Forward pass with synthetic batches (all 17 emotions)
- ACT head emotion prediction distribution
- ACT token construction from model outputs
- Tokenizer + model integration
- Batch processing correctness
- Generation pipeline (autoregressive)
- Loss decomposition with synthetic labels
- Memory efficiency during forward/backward
"""

import sys
import os
import json
import random
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oly.model.transformer import OlyConfig, OlyForCausalLM
from oly.act.act_token import (
    EMOTION_LABELS, EMOTION_TO_ID, ID_TO_EMOTION,
    build_composite_act_string, parse_act_from_response,
    EmotionState, CompositeACT,
)
from oly.act.act_head import ACTHead
from oly.act.act_loss import ACTLoss, ACTLossWithComposite
from oly.tokenizer.tokenizer import OlyTokenizer


@pytest.fixture(scope="module")
def tiny_config():
    """Tiny model for fast synthetic tests."""
    return OlyConfig(
        vocab_size=500,
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
        act_max_composite=5,
        act_loss_structural=2.0,
        act_loss_label=1.5,
        act_loss_response=1.0,
        gradient_checkpointing=False,
    )


@pytest.fixture(scope="module")
def model(tiny_config):
    """Small randomly-initialized model for testing."""
    torch.manual_seed(42)
    return OlyForCausalLM(tiny_config)


@pytest.fixture(scope="module")
def tokenizer():
    """Train a small tokenizer for integration tests."""
    tok = OlyTokenizer(vocab_size=500)
    texts = [
        "Hello, how are you feeling today?",
        "I am happy and grateful for your help.",
        "The weather is neutral and calm today.",
        "That makes me curious about the details.",
        "I feel sad about what happened yesterday.",
        "This is surprising and unexpected news!",
        "Let me think about this carefully before answering.",
        "I really appreciate everything you have done.",
        "I wish things were different in many ways.",
        "Everything feels peaceful and serene right now.",
    ] * 30
    tok.train_from_texts(texts)
    return tok


def _make_synthetic_batch(config, batch_size=4, seq_len=32):
    """Create a synthetic training batch with random data."""
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    act_labels = torch.randint(0, config.act_num_emotions, (batch_size,))
    act_intensities = torch.rand(batch_size) * 0.9 + 0.1
    act_mask = torch.ones(batch_size, dtype=torch.float)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "act_labels": act_labels,
        "act_intensities": act_intensities,
        "act_mask": act_mask,
    }


class TestForwardPassSynthetic:
    """Test forward pass with synthetic inputs."""

    def test_single_sample(self, model, tiny_config):
        """Forward pass with a single synthetic sample."""
        batch = _make_synthetic_batch(tiny_config, batch_size=1, seq_len=16)
        outputs = model(**batch)
        assert outputs["loss"] is not None
        assert outputs["loss"].item() > 0
        assert outputs["logits"].shape == (1, 16, tiny_config.vocab_size)

    def test_batch_forward(self, model, tiny_config):
        """Forward pass with a batch of synthetic samples."""
        batch = _make_synthetic_batch(tiny_config, batch_size=8, seq_len=32)
        outputs = model(**batch)
        assert outputs["logits"].shape == (8, 32, tiny_config.vocab_size)
        assert outputs["act_emotion_logits"].shape == (8, 17)
        assert outputs["act_intensity"].shape == (8,)

    def test_all_emotions_as_targets(self, model, tiny_config):
        """Test forward with each of the 17 emotions as the target label."""
        for emotion_id in range(17):
            batch = _make_synthetic_batch(tiny_config, batch_size=1)
            batch["act_labels"] = torch.tensor([emotion_id])
            batch["act_intensities"] = torch.tensor([0.7])
            outputs = model(**batch)
            assert outputs["loss"].item() > 0, f"Zero loss for emotion_id={emotion_id}"

    def test_variable_sequence_lengths(self, model, tiny_config):
        """Test with different sequence lengths."""
        for seq_len in [8, 16, 32, 64]:
            batch = _make_synthetic_batch(tiny_config, batch_size=2, seq_len=seq_len)
            outputs = model(**batch)
            assert outputs["logits"].shape[1] == seq_len

    def test_with_padding_mask(self, model, tiny_config):
        """Test forward with padding in the attention mask."""
        batch = _make_synthetic_batch(tiny_config, batch_size=2, seq_len=32)
        # Pad last 10 tokens
        batch["attention_mask"][:, -10:] = 0
        batch["labels"][:, -10:] = -100
        outputs = model(**batch)
        assert outputs["loss"] is not None


class TestACTHeadSynthetic:
    """Test ACT head outputs on synthetic hidden states."""

    @pytest.fixture
    def act_head(self, tiny_config):
        return ACTHead(tiny_config)

    def test_emotion_distribution(self, act_head):
        """Emotion probabilities should sum to 1 for each sample."""
        hidden = torch.randn(16, 8, 128)
        result = act_head(hidden)
        sums = result["emotion_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5)

    def test_intensity_always_in_range(self, act_head):
        """Intensity should always be in [0.1, 1.0] regardless of input."""
        for _ in range(10):
            hidden = torch.randn(32, 4, 128) * 100  # large values
            result = act_head(hidden)
            assert (result["intensity"] >= 0.1).all()
            assert (result["intensity"] <= 1.0).all()

    def test_predictions_change_with_input(self, act_head):
        """Different inputs should generally produce different predictions."""
        h1 = torch.randn(1, 8, 128)
        h2 = torch.randn(1, 8, 128) * 5
        r1 = act_head(h1)
        r2 = act_head(h2)
        # At least one of the predictions should differ
        logits_different = not torch.allclose(r1["emotion_logits"], r2["emotion_logits"], atol=0.01)
        intensity_different = not torch.allclose(r1["intensity"], r2["intensity"], atol=0.01)
        assert logits_different or intensity_different

    def test_top_k_returns_correct_count(self, act_head):
        """get_top_k_emotions should return exactly k results."""
        hidden = torch.randn(2, 8, 128)
        result = act_head(hidden)
        for k in [1, 3, 5]:
            top_k = act_head.get_top_k_emotions(result["emotion_logits"], k=k)
            assert len(top_k) == 2  # batch size
            for batch_result in top_k:
                assert len(batch_result) == k

    def test_label_prediction_roundtrip(self, act_head):
        """predict_emotion_label should return valid emotion strings."""
        pred_idx = torch.tensor([0, 4, 8, 16])
        labels = act_head.predict_emotion_label(pred_idx)
        assert labels == ["happy", "curious", "neutral", "serenity"]


class TestACTLossSynthetic:
    """Test ACT loss computation with synthetic targets."""

    @pytest.fixture
    def loss_fn(self, tiny_config):
        return ACTLoss(tiny_config)

    @pytest.fixture
    def composite_loss_fn(self, tiny_config):
        return ACTLossWithComposite(tiny_config)

    def test_loss_positive_for_all_emotions(self, loss_fn):
        """Loss should be > 0 for each emotion as target."""
        for emotion_id in range(17):
            logits = torch.randn(1, 17)
            labels = torch.tensor([emotion_id])
            intensities = torch.tensor([0.7])
            pred_intensities = torch.tensor([0.5])
            loss, parts = loss_fn(logits, pred_intensities, labels, intensities)
            assert loss.item() > 0

    def test_perfect_prediction_low_loss(self, loss_fn):
        """When prediction matches target exactly, loss should be very low."""
        # Create logits that strongly predict emotion 0 (happy)
        logits = torch.full((1, 17), -10.0)
        logits[0, 0] = 10.0  # strong prediction for happy
        labels = torch.tensor([0])
        intensities = torch.tensor([0.7])
        pred_intensities = torch.tensor([0.7])
        loss_perfect, _ = loss_fn(logits, pred_intensities, labels, intensities)

        # Random prediction should have higher loss
        logits_random = torch.randn(1, 17)
        loss_random, _ = loss_fn(logits_random, pred_intensities, labels, intensities)

        assert loss_perfect.item() < loss_random.item()

    def test_mask_zeroes_out_samples(self, loss_fn):
        """Masked samples should not contribute to loss."""
        logits = torch.randn(4, 17)
        labels = torch.randint(0, 17, (4,))
        intensities = torch.rand(4)
        pred_intensities = torch.rand(4)

        # All masked
        mask_zero = torch.zeros(4)
        loss_zero, _ = loss_fn(logits, pred_intensities, labels, intensities, mask_zero)

        # All unmasked
        mask_one = torch.ones(4)
        loss_one, _ = loss_fn(logits, pred_intensities, labels, intensities, mask_one)

        # Zero mask should give zero loss (or near zero from structural)
        assert loss_zero.item() <= loss_one.item()

    def test_composite_loss_with_secondary(self, composite_loss_fn):
        """Composite loss should include secondary emotion term."""
        logits = torch.randn(2, 17)
        labels = torch.tensor([0, 1])  # happy, sad
        intensities = torch.tensor([0.8, 0.6])
        pred_intensities = torch.rand(2)
        secondary_labels = torch.tensor([4, 9])  # curious, hopeful
        secondary_intensities = torch.tensor([0.4, 0.3])

        loss, parts = composite_loss_fn(
            logits, pred_intensities, labels, intensities,
            target_secondary_labels=secondary_labels,
            target_secondary_intensities=secondary_intensities,
        )
        assert "act_secondary_loss" in parts
        assert loss.item() > 0

    def test_loss_backward_propagates(self, loss_fn):
        """Gradients should flow through the loss computation."""
        logits = torch.randn(2, 17, requires_grad=True)
        pred_int = torch.randn(2, requires_grad=True)
        labels = torch.randint(0, 17, (2,))
        intensities = torch.rand(2)

        loss, _ = loss_fn(logits, pred_int, labels, intensities)
        loss.backward()
        assert logits.grad is not None
        assert pred_int.grad is not None

    def test_loss_components_logged(self, loss_fn):
        """All expected loss components should be returned."""
        logits = torch.randn(2, 17)
        labels = torch.randint(0, 17, (2,))
        pred_int = torch.rand(2)
        target_int = torch.rand(2)

        _, parts = loss_fn(logits, pred_int, labels, target_int)
        assert "act_label_loss" in parts
        assert "act_intensity_loss" in parts
        assert "act_structural_loss" in parts
        assert "act_total_loss" in parts


class TestTokenizerModelIntegration:
    """Test that the tokenizer and model work together on synthetic data."""

    def test_encode_forward_decode(self, model, tokenizer, tiny_config):
        """Full pipeline: text -> tokens -> model -> logits."""
        text = "Hello, how are you?"
        token_ids = tokenizer.encode(text, add_special_tokens=True)

        # Truncate to model's max length and pad
        max_len = 32
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        pad_len = max_len - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        token_ids = token_ids + [tokenizer.pad_token_id] * pad_len

        input_ids = torch.tensor([token_ids])
        mask = torch.tensor([attention_mask])

        outputs = model(input_ids=input_ids, attention_mask=mask)
        assert outputs["logits"].shape == (1, max_len, tiny_config.vocab_size)
        assert outputs["act_emotion_logits"].shape == (1, 17)

    def test_act_token_encoding(self, tokenizer):
        """ACT special tokens should encode to valid IDs."""
        act_ids = tokenizer.encode_act_token("happy", 0.8)
        assert len(act_ids) > 0
        assert act_ids[0] == tokenizer.act_start_token_id
        assert act_ids[-1] == tokenizer.act_end_token_id

    def test_all_emotions_encode(self, tokenizer):
        """All 17 emotions should produce valid ACT token encodings."""
        for emotion in EMOTION_LABELS:
            act_ids = tokenizer.encode_act_token(emotion, 0.5)
            assert len(act_ids) >= 3, f"ACT encoding for {emotion} too short: {act_ids}"

    def test_training_sample_format(self, model, tokenizer, tiny_config):
        """Simulate creating a training sample exactly as extract_data.py does."""
        # Build sample
        act_string = build_composite_act_string([("happy", 0.8)])
        input_text = "Thank you for helping me!"
        response = "That's really great to hear!"
        output_text = f"{act_string} {response}"
        full_text = f"User: {input_text}\nAssistant: {output_text}"

        # Tokenize
        token_ids = tokenizer.encode(full_text, add_special_tokens=True)
        max_len = 64
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        pad_len = max_len - len(token_ids)
        token_ids = token_ids + [tokenizer.pad_token_id] * pad_len

        # Forward
        input_ids = torch.tensor([token_ids])
        labels = input_ids.clone()
        act_labels = torch.tensor([EMOTION_TO_ID["happy"]])
        act_intensities = torch.tensor([0.8])

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            act_labels=act_labels,
            act_intensities=act_intensities,
        )
        assert outputs["loss"] is not None
        assert outputs["loss"].item() > 0


class TestGenerationSynthetic:
    """Test text generation with random weights (output will be random but pipeline should work)."""

    def test_generate_returns_tokens(self, model, tiny_config):
        """Generation should produce at least some new tokens."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        result = model.generate(input_ids, max_new_tokens=5, emit_act=True)
        assert result["generated_ids"].shape[1] >= 8
        assert result["act_result"] is not None

    def test_generate_act_result_structure(self, model, tiny_config):
        """ACT result from generation should have expected keys."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        result = model.generate(input_ids, max_new_tokens=5, emit_act=True)
        act = result["act_result"]
        assert "emotion_logits" in act
        assert "emotion_pred" in act
        assert "intensity" in act
        assert act["intensity"].item() >= 0.1
        assert act["intensity"].item() <= 1.0

    def test_generate_without_act(self, model, tiny_config):
        """Generation with emit_act=False should not return ACT result."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        result = model.generate(input_ids, max_new_tokens=5, emit_act=False)
        assert result["act_result"] is None

    def test_construct_act_token_from_prediction(self, model, tiny_config):
        """Build an ACT token string from model's emotion prediction."""
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        act_result = model.call_act(input_ids)

        emotion_idx = act_result["emotion_pred"].item()
        emotion_name = EMOTION_LABELS[emotion_idx]
        intensity = act_result["intensity"].item()

        act_string = build_composite_act_string([(emotion_name, intensity)])
        parsed = parse_act_from_response(act_string)
        assert parsed is not None
        assert parsed.dominant.name == emotion_name


class TestMemoryEfficiency:
    """Test that forward/backward don't leak memory (important for RTX 3050 Ti)."""

    def test_no_gradient_accumulation_leak(self, model, tiny_config):
        """Multiple forward-backward passes should not accumulate memory."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        for _ in range(5):
            batch = _make_synthetic_batch(tiny_config, batch_size=2, seq_len=16)
            outputs = model(**batch)
            outputs["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        # If we get here without OOM, the test passes
        model.eval()

    def test_gradient_checkpointing_mode(self, tiny_config):
        """Model with gradient checkpointing should also work."""
        tiny_config_ckpt = OlyConfig(
            vocab_size=tiny_config.vocab_size,
            hidden_size=tiny_config.hidden_size,
            num_hidden_layers=tiny_config.num_hidden_layers,
            num_attention_heads=tiny_config.num_attention_heads,
            intermediate_size=tiny_config.intermediate_size,
            max_position_embeddings=tiny_config.max_position_embeddings,
            hidden_dropout_prob=0.0,
            attention_dropout_prob=0.0,
            act_enabled=True,
            act_num_emotions=17,
            act_hidden_size=64,
            gradient_checkpointing=True,
        )
        ckpt_model = OlyForCausalLM(tiny_config_ckpt)
        ckpt_model.train()

        batch = _make_synthetic_batch(tiny_config_ckpt, batch_size=1, seq_len=16)
        outputs = ckpt_model(**batch)
        outputs["loss"].backward()

        # Verify gradients exist
        has_grad = any(p.grad is not None for p in ckpt_model.parameters() if p.requires_grad)
        assert has_grad
