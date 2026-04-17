"""
Tests for the Llama 8B ACT integration.

Since the full Llama 3.1 8B model requires significant compute resources,
these tests validate the ACT head component and integration logic without
loading the actual Llama model.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integrations.llama_8b.llama_act import LlamaACTHead
from oly.act.act_token import EMOTION_LABELS


class TestLlamaACTHead:
    """Tests for the lightweight ACT head attached to Llama."""

    @pytest.fixture
    def head(self):
        return LlamaACTHead(hidden_size=4096, num_emotions=17, act_hidden=1024)

    def test_forward_shape(self, head):
        hidden = torch.randn(2, 4096)  # pooled hidden state
        result = head(hidden)
        assert result["emotion_logits"].shape == (2, 17)
        assert result["emotion_probs"].shape == (2, 17)
        assert result["emotion_pred"].shape == (2,)
        assert result["intensity"].shape == (2,)

    def test_intensity_range(self, head):
        """Intensity must be in [0.1, 1.0]."""
        hidden = torch.randn(20, 4096)
        result = head(hidden)
        assert (result["intensity"] >= 0.1).all()
        assert (result["intensity"] <= 1.0).all()

    def test_emotion_probs_normalized(self, head):
        hidden = torch.randn(2, 4096)
        result = head(hidden)
        sums = result["emotion_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_parameter_count(self, head):
        total = sum(p.numel() for p in head.parameters())
        # ACT head should be relatively small (<10M params)
        assert total < 10_000_000
        print(f"LlamaACTHead params: {total:,}")

    def test_gradient_flow(self, head):
        """Verify gradients flow through the ACT head."""
        hidden = torch.randn(1, 4096, requires_grad=True)
        result = head(hidden)
        loss = result["emotion_logits"].sum() + result["intensity"].sum()
        loss.backward()
        assert hidden.grad is not None

    def test_save_load_weights(self, head, tmp_path):
        """Test ACT head weight save/load."""
        save_path = str(tmp_path / "act_head.pt")
        torch.save(head.state_dict(), save_path)

        new_head = LlamaACTHead(hidden_size=4096, num_emotions=17, act_hidden=1024)
        new_head.load_state_dict(torch.load(save_path, weights_only=True))

        # Verify identical predictions
        hidden = torch.randn(1, 4096)
        with torch.no_grad():
            orig_result = head(hidden)
            loaded_result = new_head(hidden)

        assert torch.allclose(orig_result["emotion_logits"], loaded_result["emotion_logits"])
        assert torch.allclose(orig_result["intensity"], loaded_result["intensity"])


class TestLlamaACTIntegration:
    """Integration tests (mock Llama -- does not load the real model)."""

    def test_llama_act_class_exists(self):
        from integrations.llama_8b.llama_act import LlamaACT
        assert LlamaACT is not None

    def test_act_token_construction(self):
        """Test that ACT tokens can be constructed from head predictions."""
        from oly.act.act_token import build_composite_act_string

        head = LlamaACTHead(hidden_size=128, num_emotions=17, act_hidden=64)
        hidden = torch.randn(1, 128)
        result = head(hidden)

        emotion_idx = result["emotion_pred"].item()
        intensity = result["intensity"].item()
        label = EMOTION_LABELS[emotion_idx]

        act_string = build_composite_act_string([(label, intensity)])
        assert "<|ACT:" in act_string
        assert label in act_string
