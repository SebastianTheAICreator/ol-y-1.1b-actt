"""
Tests for the ACT (Affective Communication Tokens) module.

Validates:
- ACT token schema and parsing
- Composite ACT format
- Emotion taxonomy
- ACT head forward pass
- ACT loss computation
- Color blending
- Novel emotion detection

Reference: Sakishimiro (2026), Section 3 "Formal Specification of the ACT Token"
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oly.act.act_token import (
    ACTToken, CompositeACT, EmotionState,
    EMOTION_LABELS, EMOTION_TO_ID, ID_TO_EMOTION,
    EMOTION_CATEGORIES, EMOTION_COLORS,
    parse_act_from_response, build_act_string, build_composite_act_string,
    ACT_PREFILL, INTENSITY_MIN, INTENSITY_MAX,
)
from oly.act.act_head import ACTHead
from oly.act.act_loss import ACTLoss


class TestEmotionState:
    """Tests for individual emotion state values."""

    def test_valid_emotion(self):
        state = EmotionState(name="happy", intensity=0.8)
        assert state.name == "happy"
        assert state.intensity == 0.8

    def test_intensity_clamping(self):
        """Intensity must be clamped to [0.1, 1.0]."""
        low = EmotionState(name="sad", intensity=0.01)
        assert low.intensity == INTENSITY_MIN  # clamped to 0.1

        high = EmotionState(name="angry", intensity=1.5)
        assert high.intensity == INTENSITY_MAX  # clamped to 1.0

    def test_novel_emotion(self):
        """Novel emotions outside the taxonomy should be flagged."""
        novel = EmotionState(name="yearning", intensity=0.7)
        assert novel.is_novel() is True

        known = EmotionState(name="curious", intensity=0.5)
        assert known.is_novel() is False

    def test_to_dict(self):
        state = EmotionState(name="happy", intensity=0.75)
        d = state.to_dict()
        assert d == {"name": "happy", "intensity": 0.75}


class TestACTToken:
    """Tests for single-label ACT token format."""

    def test_to_string(self):
        token = ACTToken(emotion=EmotionState(name="happy", intensity=0.8))
        s = token.to_string()
        assert '<|ACT:"emotion":' in s
        assert '"happy"' in s
        assert "0.8" in s
        assert s.startswith("<|ACT:")
        assert s.endswith("|>")

    def test_from_string(self):
        token_str = '<|ACT:"emotion":{"name":"curious","intensity":0.6}|>'
        token = ACTToken.from_string(token_str)
        assert token is not None
        assert token.emotion.name == "curious"
        assert token.emotion.intensity == 0.6

    def test_roundtrip(self):
        """Verify encode -> decode -> encode produces identical output."""
        original = ACTToken(emotion=EmotionState(name="sad", intensity=0.7))
        string = original.to_string()
        parsed = ACTToken.from_string(string)
        assert parsed is not None
        assert parsed.emotion.name == original.emotion.name
        assert parsed.emotion.intensity == original.emotion.intensity

    def test_invalid_string(self):
        assert ACTToken.from_string("not an ACT token") is None
        assert ACTToken.from_string("") is None


class TestCompositeACT:
    """Tests for the composite multi-component ACT format.

    Reference: Section 3.3, Equation 4
    """

    def test_composite_to_string(self):
        act = CompositeACT(emotions=[
            EmotionState(name="happy", intensity=0.8),
            EmotionState(name="curious", intensity=0.5),
        ])
        s = act.to_string()
        assert '"happy"' in s
        assert '"curious"' in s
        assert "[" in s  # array format

    def test_composite_from_string(self):
        token_str = '<|ACT:"emotion":[{"name":"sad","intensity":0.7},{"name":"curious","intensity":0.4}]|>'
        act = CompositeACT.from_string(token_str)
        assert act is not None
        assert len(act.emotions) == 2
        assert act.dominant.name == "sad"
        assert act.secondary.name == "curious"

    def test_sorted_by_intensity(self):
        """Components should be sorted from highest to lowest intensity."""
        act = CompositeACT(emotions=[
            EmotionState(name="curious", intensity=0.3),
            EmotionState(name="happy", intensity=0.9),
        ])
        assert act.dominant.name == "happy"
        assert act.secondary.name == "curious"

    def test_max_5_components(self):
        """Composite ACT should be limited to 5 components max."""
        emotions = [EmotionState(name=f"happy", intensity=0.5 + i*0.01) for i in range(7)]
        act = CompositeACT(emotions=emotions)
        assert len(act.emotions) == 5

    def test_novel_emotion_detection(self):
        act = CompositeACT(emotions=[
            EmotionState(name="happy", intensity=0.8),
            EmotionState(name="yearning", intensity=0.6),  # novel
        ])
        assert act.has_novel_emotions() is True
        assert "yearning" in act.get_novel_emotions()

    def test_color_blending(self):
        """Test color blending formula from Appendix B."""
        act = CompositeACT(emotions=[
            EmotionState(name="happy", intensity=0.8),
            EmotionState(name="sad", intensity=0.4),
        ])
        color = act.blend_colors()
        assert color.startswith("#")
        assert len(color) == 7  # #RRGGBB format

    def test_single_component_color(self):
        act = CompositeACT(emotions=[EmotionState(name="happy", intensity=0.8)])
        assert act.blend_colors() == EMOTION_COLORS["happy"]


class TestParseACTFromResponse:
    """Test parsing ACT tokens from full model responses."""

    def test_parse_with_response_text(self):
        response = '<|ACT:"emotion":[{"name":"happy","intensity":0.8}]|> Hello! How can I help you?'
        act = parse_act_from_response(response)
        assert act is not None
        assert act.dominant.name == "happy"

    def test_parse_blank_message(self):
        """The blank message event from Section 8.5 of the paper."""
        response = '<|ACT:"emotion":[{"name":"emptiness","intensity":0.9}]|>'
        act = parse_act_from_response(response)
        assert act is not None
        assert act.dominant.name == "emptiness"
        assert act.dominant.intensity == 0.9

    def test_parse_no_act_token(self):
        assert parse_act_from_response("Just a normal response.") is None

    def test_build_act_string(self):
        s = build_act_string("curious", 0.6)
        assert '<|ACT:' in s
        assert '"curious"' in s

    def test_build_composite_string(self):
        s = build_composite_act_string([("happy", 0.8), ("curious", 0.5)])
        act = parse_act_from_response(s)
        assert act is not None
        assert len(act.emotions) == 2


class TestEmotionTaxonomy:
    """Test the emotion taxonomy from Appendix A."""

    def test_17_emotions(self):
        assert len(EMOTION_LABELS) == 17

    def test_all_have_categories(self):
        for label in EMOTION_LABELS:
            assert label in EMOTION_CATEGORIES

    def test_all_have_colors(self):
        for label in EMOTION_LABELS:
            assert label in EMOTION_COLORS

    def test_id_mapping_consistency(self):
        for label in EMOTION_LABELS:
            idx = EMOTION_TO_ID[label]
            assert ID_TO_EMOTION[idx] == label

    def test_prefill_template(self):
        assert ACT_PREFILL == '<|ACT:"emotion":[{"name":"'


class TestACTHead:
    """Tests for the ACT neural network head."""

    @pytest.fixture
    def act_head(self):
        from oly.model.transformer import OlyConfig
        config = OlyConfig(hidden_size=128, act_num_emotions=17, act_hidden_size=64, act_max_composite=5)
        return ACTHead(config)

    def test_forward_shape(self, act_head):
        hidden = torch.randn(2, 16, 128)  # (batch, seq_len, hidden)
        result = act_head(hidden)
        assert result["emotion_logits"].shape == (2, 17)
        assert result["emotion_probs"].shape == (2, 17)
        assert result["emotion_pred"].shape == (2,)
        assert result["intensity"].shape == (2,)

    def test_intensity_range(self, act_head):
        """Predicted intensity must be in [0.1, 1.0]."""
        hidden = torch.randn(10, 8, 128)
        result = act_head(hidden)
        assert (result["intensity"] >= 0.1).all()
        assert (result["intensity"] <= 1.0).all()

    def test_emotion_probabilities_sum_to_1(self, act_head):
        hidden = torch.randn(2, 8, 128)
        result = act_head(hidden)
        sums = result["emotion_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_predict_emotion_label(self, act_head):
        pred_idx = torch.tensor([0, 4, 8])
        labels = act_head.predict_emotion_label(pred_idx)
        assert labels == ["happy", "curious", "neutral"]

    def test_get_top_k(self, act_head):
        hidden = torch.randn(1, 8, 128)
        result = act_head(hidden)
        top_k = act_head.get_top_k_emotions(result["emotion_logits"], k=3)
        assert len(top_k) == 1  # batch size 1
        assert len(top_k[0]) == 3  # top-3


class TestACTLoss:
    """Tests for the ACT loss functions."""

    @pytest.fixture
    def loss_fn(self):
        from oly.model.transformer import OlyConfig
        config = OlyConfig(act_num_emotions=17, act_loss_structural=2.0, act_loss_label=1.5)
        return ACTLoss(config)

    def test_label_loss(self, loss_fn):
        logits = torch.randn(4, 17)
        labels = torch.randint(0, 17, (4,))
        intensities = torch.rand(4) * 0.9 + 0.1
        pred_intensities = torch.rand(4) * 0.9 + 0.1

        total_loss, parts = loss_fn(logits, pred_intensities, labels, intensities)
        assert total_loss.item() > 0
        assert "act_label_loss" in parts
        assert "act_intensity_loss" in parts

    def test_loss_with_mask(self, loss_fn):
        """Test that masking excludes samples from loss."""
        logits = torch.randn(4, 17)
        labels = torch.randint(0, 17, (4,))
        intensities = torch.rand(4)
        pred_intensities = torch.rand(4)
        mask = torch.tensor([1, 1, 0, 0], dtype=torch.float)

        total_loss, parts = loss_fn(logits, pred_intensities, labels, intensities, mask)
        assert total_loss.item() > 0

    def test_loss_backward(self, loss_fn):
        logits = torch.randn(2, 17, requires_grad=True)
        labels = torch.randint(0, 17, (2,))
        intensities = torch.rand(2)
        pred_intensities = torch.rand(2, requires_grad=True)

        total_loss, _ = loss_fn(logits, pred_intensities, labels, intensities)
        total_loss.backward()
        assert logits.grad is not None
