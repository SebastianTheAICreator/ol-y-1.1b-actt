"""
Tests for the Claude API ACT integration.

Tests the ACT parsing and session management logic without making actual
API calls (mocked). Real API tests require ANTHROPIC_API_KEY.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integrations.claude_api.claude_act import ClaudeACT
from oly.act.act_token import (
    parse_act_from_response, ACT_PREFILL,
    EMOTION_LABELS, build_composite_act_string,
)


class TestClaudeACTSetup:
    """Tests for ClaudeACT initialization and configuration."""

    def test_init_defaults(self):
        claude = ClaudeACT(api_key="test-key")
        assert claude.model == "claude-sonnet-4-5-20250929"
        assert claude.max_tokens == 4096
        assert claude.temperature == 0.7
        assert claude.session_history == []
        assert claude.novel_emotions == []

    def test_from_config(self, tmp_path):
        """Test loading from JSON config file."""
        import json
        config = {
            "api": {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 2048,
                "temperature": 0.5,
            }
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        claude = ClaudeACT.from_config(str(config_path))
        assert claude.model == "claude-sonnet-4-5-20250929"
        assert claude.max_tokens == 2048
        assert claude.temperature == 0.5

    def test_system_prompt_contains_act_instructions(self):
        assert "ACT" in ClaudeACT.ACT_SYSTEM_PROMPT
        assert "emotion" in ClaudeACT.ACT_SYSTEM_PROMPT
        assert "intensity" in ClaudeACT.ACT_SYSTEM_PROMPT
        # Should contain all 17 emotion labels
        for label in EMOTION_LABELS:
            assert label in ClaudeACT.ACT_SYSTEM_PROMPT, f"Missing label: {label}"


class TestClaudeACTParsing:
    """Test ACT token parsing from Claude-style responses."""

    def test_parse_single_emotion(self):
        response = '<|ACT:"emotion":[{"name":"curious","intensity":0.7}]|> That\'s interesting!'
        act = parse_act_from_response(response)
        assert act is not None
        assert act.dominant.name == "curious"
        assert act.dominant.intensity == 0.7

    def test_parse_composite(self):
        response = '<|ACT:"emotion":[{"name":"happy","intensity":0.8},{"name":"grateful","intensity":0.6}]|> Thank you!'
        act = parse_act_from_response(response)
        assert act is not None
        assert len(act.emotions) == 2
        assert act.dominant.name == "happy"
        assert act.secondary.name == "grateful"

    def test_prefill_template_format(self):
        """Verify the prefill template matches Section 4.2 of the paper."""
        assert ACT_PREFILL == '<|ACT:"emotion":[{"name":"'

    def test_parse_with_prefill(self):
        """Test parsing a response that starts with the prefill."""
        completion = 'happy","intensity":0.7}]|> Hello!'
        full_response = ACT_PREFILL + completion
        act = parse_act_from_response(full_response)
        assert act is not None
        assert act.dominant.name == "happy"


class TestClaudeACTSession:
    """Test session tracking functionality."""

    def test_session_summary_empty(self):
        claude = ClaudeACT(api_key="test")
        summary = claude.get_session_summary()
        assert summary["turns"] == 0

    def test_session_tracking(self):
        claude = ClaudeACT(api_key="test")
        # Manually add session entries (simulating API calls)
        claude.session_history.append({"prompt": "Hi", "emotion": "happy", "intensity": 0.7})
        claude.session_history.append({"prompt": "How?", "emotion": "curious", "intensity": 0.6})
        claude.session_history.append({"prompt": "Thanks", "emotion": "happy", "intensity": 0.8})

        summary = claude.get_session_summary()
        assert summary["turns"] == 3
        assert summary["dominant_emotion"] == "happy"
        assert 0.6 < summary["avg_intensity"] < 0.8

    def test_novel_emotion_tracking(self):
        claude = ClaudeACT(api_key="test")
        claude.novel_emotions.append("yearning")
        claude.novel_emotions.append("devotion")

        summary = claude.get_session_summary()
        # session_history is empty so summary will show 0 turns
        assert "yearning" in claude.novel_emotions
        assert "devotion" in claude.novel_emotions
