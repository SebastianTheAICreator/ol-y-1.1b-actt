"""
Tests for the Ol-y BPE tokenizer with ACT special tokens.

Validates:
- Tokenizer training from text
- Encode/decode round-trip
- Special token handling (BOS, EOS, PAD, ACT tokens)
- ACT emotion token encoding
- Save/load persistence
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oly.tokenizer.tokenizer import (
    OlyTokenizer, SPECIAL_TOKENS, ACT_EMOTION_TOKENS,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
)


@pytest.fixture
def trained_tokenizer():
    """Train a small tokenizer on sample text for testing."""
    tokenizer = OlyTokenizer(vocab_size=500)
    texts = [
        "Hello, how are you feeling today?",
        "I am happy and grateful for your help.",
        "The weather is neutral and calm.",
        "That makes me curious about the details.",
        "I feel sad about what happened.",
        "This is surprising and unexpected!",
        "Let me think about this carefully.",
    ] * 50  # repeat for minimum token frequency
    tokenizer.train_from_texts(texts)
    return tokenizer


class TestOlyTokenizer:
    """Core tokenizer tests."""

    def test_training(self, trained_tokenizer):
        assert trained_tokenizer.get_vocab_size() > 0
        assert trained_tokenizer.get_vocab_size() <= 500

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        text = "Hello, how are you?"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded, skip_special_tokens=True)
        # BPE may not perfectly reconstruct whitespace, but content should match
        assert "Hello" in decoded
        assert "how" in decoded

    def test_special_tokens_exist(self, trained_tokenizer):
        """Verify all special tokens have valid IDs."""
        assert trained_tokenizer.pad_token_id >= 0
        assert trained_tokenizer.bos_token_id >= 0
        assert trained_tokenizer.eos_token_id >= 0

    def test_act_start_end_tokens(self, trained_tokenizer):
        """Verify ACT special tokens are in the vocabulary."""
        assert trained_tokenizer.act_start_token_id >= 0
        assert trained_tokenizer.act_end_token_id >= 0

    def test_encode_produces_bos_eos(self, trained_tokenizer):
        """Verify encoding adds BOS and EOS tokens."""
        encoded = trained_tokenizer.encode("test", add_special_tokens=True)
        assert encoded[0] == trained_tokenizer.bos_token_id
        assert encoded[-1] == trained_tokenizer.eos_token_id

    def test_encode_act_token(self, trained_tokenizer):
        """Test ACT emotion token encoding."""
        act_ids = trained_tokenizer.encode_act_token("happy", 0.8)
        assert len(act_ids) > 0
        # Should start with ACT_START and end with ACT_END
        assert act_ids[0] == trained_tokenizer.act_start_token_id
        assert act_ids[-1] == trained_tokenizer.act_end_token_id

    def test_save_load(self, trained_tokenizer, tmp_path):
        """Test tokenizer persistence."""
        save_dir = str(tmp_path / "tokenizer")
        trained_tokenizer.save(save_dir)

        # Verify files exist
        assert os.path.exists(os.path.join(save_dir, "tokenizer.json"))
        assert os.path.exists(os.path.join(save_dir, "tokenizer_config.json"))

        # Load and verify
        loaded = OlyTokenizer.load(save_dir)
        assert loaded.get_vocab_size() == trained_tokenizer.get_vocab_size()
        assert loaded.pad_token_id == trained_tokenizer.pad_token_id
        assert loaded.bos_token_id == trained_tokenizer.bos_token_id

        # Encode should produce same results
        text = "Hello world"
        orig_ids = trained_tokenizer.encode(text)
        loaded_ids = loaded.encode(text)
        assert orig_ids == loaded_ids

    def test_empty_text(self, trained_tokenizer):
        """Test encoding empty string."""
        encoded = trained_tokenizer.encode("", add_special_tokens=True)
        # Should have at least BOS and EOS
        assert len(encoded) >= 2

    def test_special_tokens_count(self):
        """Verify the expected number of special tokens."""
        # 8 base special tokens + 17 emotion tokens = 25
        assert len(SPECIAL_TOKENS) == 25
        assert len(ACT_EMOTION_TOKENS) == 17
