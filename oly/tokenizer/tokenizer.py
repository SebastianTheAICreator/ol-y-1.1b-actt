"""
Ol-y BPE Tokenizer with ACT special tokens.

Builds a Byte-Pair Encoding tokenizer using the HuggingFace `tokenizers` library.
Includes special tokens for ACT (Affective Communication Tokens) so the model
can natively produce and parse emotion annotations in the output stream.

Special tokens added beyond standard BOS/EOS/PAD/UNK:
- <|ACT_START|>   : marks beginning of ACT token
- <|ACT_END|>     : marks end of ACT token
- <|EMOTION|>     : generic emotion marker
- Each emotion label as a special token (e.g., <|happy|>, <|sad|>, ...)

This allows the model to learn to emit ACT tokens as atomic units rather than
having to spell them out character-by-character, significantly improving
structural compliance (99.3% with prefill enforcement, per the paper).

Reference: Sakishimiro (2026), Section 4 "Probabilistic Formulation"
"""

import os
import json
from typing import List, Optional, Dict

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC

from oly.act.act_token import EMOTION_LABELS


# === Special token definitions ===
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

# ACT-specific special tokens
ACT_START_TOKEN = "<|ACT_START|>"
ACT_END_TOKEN = "<|ACT_END|>"
ACT_EMOTION_TOKEN = "<|EMOTION|>"
ACT_INTENSITY_TOKEN = "<|INTENSITY|>"

# Create special tokens for each emotion label
ACT_EMOTION_TOKENS = [f"<|{label}|>" for label in EMOTION_LABELS]

# Complete list of all special tokens
SPECIAL_TOKENS = [
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
    ACT_START_TOKEN, ACT_END_TOKEN, ACT_EMOTION_TOKEN, ACT_INTENSITY_TOKEN,
] + ACT_EMOTION_TOKENS


class OlyTokenizer:
    """BPE tokenizer for the Ol-y 1.1B ACTT model.

    Wraps the HuggingFace `tokenizers` library to provide a fast BPE tokenizer
    with integrated ACT special tokens. Can be trained from scratch on a text
    corpus or loaded from a saved vocabulary.
    """

    def __init__(self, vocab_size: int = 32000):
        """Initialize tokenizer with BPE model.

        Args:
            vocab_size: target vocabulary size (default 32000)
        """
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        self.tokenizer.normalizer = NFKC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

        # Token ID cache (populated after training or loading)
        self._special_token_ids: Dict[str, int] = {}

    def train(self, files: List[str], min_frequency: int = 2):
        """Train BPE tokenizer on a corpus of text files.

        Args:
            files: list of file paths to train on
            min_frequency: minimum frequency for a token to be included
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self.tokenizer.train(files, trainer)

        # Add post-processor for BOS/EOS
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
            special_tokens=[
                (BOS_TOKEN, self.tokenizer.token_to_id(BOS_TOKEN)),
                (EOS_TOKEN, self.tokenizer.token_to_id(EOS_TOKEN)),
            ],
        )

        self._cache_special_ids()
        print(f"[Tokenizer] Trained with vocab_size={self.tokenizer.get_vocab_size()}")
        print(f"[Tokenizer] Special tokens: {len(SPECIAL_TOKENS)}")
        print(f"[Tokenizer] ACT emotion tokens: {len(ACT_EMOTION_TOKENS)}")

    def train_from_texts(self, texts: List[str], min_frequency: int = 2):
        """Train BPE tokenizer directly from a list of text strings.

        Convenient alternative to train() when data is already in memory.

        Args:
            texts: list of text strings to train on
            min_frequency: minimum frequency for token inclusion
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self.tokenizer.train_from_iterator(texts, trainer)

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
            special_tokens=[
                (BOS_TOKEN, self.tokenizer.token_to_id(BOS_TOKEN)),
                (EOS_TOKEN, self.tokenizer.token_to_id(EOS_TOKEN)),
            ],
        )

        self._cache_special_ids()

    def _cache_special_ids(self):
        """Cache special token IDs for fast lookup."""
        for token in SPECIAL_TOKENS:
            tid = self.tokenizer.token_to_id(token)
            if tid is not None:
                self._special_token_ids[token] = tid

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: input text string
            add_special_tokens: whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to text.

        Args:
            ids: list of token IDs
            skip_special_tokens: whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_act_token(self, emotion: str, intensity: float) -> List[int]:
        """Encode an ACT emotion annotation using special tokens.

        Instead of encoding the raw ACT string character-by-character, this uses
        the dedicated special tokens for more reliable parsing:
        <|ACT_START|> <|emotion_label|> <|INTENSITY|> intensity_tokens <|ACT_END|>

        Args:
            emotion: emotion label (must be in EMOTION_LABELS)
            intensity: intensity value in [0.1, 1.0]

        Returns:
            List of token IDs representing the ACT annotation
        """
        tokens = []
        # ACT start marker
        start_id = self._special_token_ids.get(ACT_START_TOKEN)
        if start_id is not None:
            tokens.append(start_id)

        # Emotion label as special token
        emotion_token = f"<|{emotion}|>"
        emotion_id = self._special_token_ids.get(emotion_token)
        if emotion_id is not None:
            tokens.append(emotion_id)

        # Intensity marker
        intensity_marker_id = self._special_token_ids.get(ACT_INTENSITY_TOKEN)
        if intensity_marker_id is not None:
            tokens.append(intensity_marker_id)

        # Intensity value as regular tokens (e.g., "0.8")
        intensity_str = f"{intensity:.1f}"
        intensity_ids = self.tokenizer.encode(intensity_str, add_special_tokens=False).ids
        tokens.extend(intensity_ids)

        # ACT end marker
        end_id = self._special_token_ids.get(ACT_END_TOKEN)
        if end_id is not None:
            tokens.append(end_id)

        return tokens

    def save(self, directory: str):
        """Save tokenizer to directory.

        Args:
            directory: path to save directory
        """
        os.makedirs(directory, exist_ok=True)
        self.tokenizer.save(os.path.join(directory, "tokenizer.json"))

        # Save metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "emotion_labels": EMOTION_LABELS,
            "special_token_ids": self._special_token_ids,
        }
        with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Tokenizer] Saved to {directory}")

    @classmethod
    def load(cls, directory: str) -> "OlyTokenizer":
        """Load tokenizer from directory.

        Args:
            directory: path to saved tokenizer directory

        Returns:
            Loaded OlyTokenizer instance
        """
        tokenizer_path = os.path.join(directory, "tokenizer.json")
        config_path = os.path.join(directory, "tokenizer_config.json")

        with open(config_path, "r") as f:
            metadata = json.load(f)

        instance = cls(vocab_size=metadata["vocab_size"])
        instance.tokenizer = Tokenizer.from_file(tokenizer_path)
        instance._special_token_ids = {
            k: int(v) for k, v in metadata.get("special_token_ids", {}).items()
        }
        return instance

    @property
    def pad_token_id(self) -> int:
        return self._special_token_ids.get(PAD_TOKEN, 0)

    @property
    def unk_token_id(self) -> int:
        return self._special_token_ids.get(UNK_TOKEN, 1)

    @property
    def bos_token_id(self) -> int:
        return self._special_token_ids.get(BOS_TOKEN, 2)

    @property
    def eos_token_id(self) -> int:
        return self._special_token_ids.get(EOS_TOKEN, 3)

    @property
    def act_start_token_id(self) -> int:
        return self._special_token_ids.get(ACT_START_TOKEN, -1)

    @property
    def act_end_token_id(self) -> int:
        return self._special_token_ids.get(ACT_END_TOKEN, -1)

    def get_vocab_size(self) -> int:
        """Return the actual vocabulary size."""
        return self.tokenizer.get_vocab_size()
