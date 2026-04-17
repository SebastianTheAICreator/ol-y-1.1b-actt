"""
Tokenizer training script for Ol-y 1.1B ACTT.

Trains a BPE tokenizer with ACT special tokens on the training data.
This should be run before training if no tokenizer exists yet.

Usage:
    python scripts/train_tokenizer.py --data data/train.jsonl --output outputs/tokenizer
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oly.tokenizer.tokenizer import OlyTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Ol-y BPE tokenizer")
    parser.add_argument("--data", default="data/train.jsonl", help="Training data JSONL")
    parser.add_argument("--output", default="outputs/tokenizer", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Min token frequency")
    args = parser.parse_args()

    print(f"[Tokenizer] Training BPE tokenizer (vocab_size={args.vocab_size})")
    print(f"[Tokenizer] Data: {args.data}")

    # Extract texts from JSONL
    texts = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            texts.append(sample.get("input", ""))
            texts.append(sample.get("output", ""))

    print(f"[Tokenizer] Extracted {len(texts)} text segments")

    # Train tokenizer
    tokenizer = OlyTokenizer(vocab_size=args.vocab_size)
    tokenizer.train_from_texts(texts, min_frequency=args.min_frequency)

    # Save
    tokenizer.save(args.output)

    # Test encoding/decoding
    test_text = "Hello, how are you feeling today?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n[Tokenizer] Test:")
    print(f"  Input:   '{test_text}'")
    print(f"  Encoded: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
    print(f"  Decoded: '{decoded}'")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Test ACT encoding
    act_ids = tokenizer.encode_act_token("happy", 0.8)
    print(f"\n[Tokenizer] ACT test:")
    print(f"  ACT(happy, 0.8) -> {act_ids}")

    print("\n[Tokenizer] Done!")


if __name__ == "__main__":
    main()
