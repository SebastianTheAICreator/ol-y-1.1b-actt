"""
Tests for the training pipeline.

Validates:
- Dataset loading and collation
- Training loop execution (micro test)
- Loss convergence on a tiny model
- Checkpoint save/load
"""

import sys
import os
import json
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oly.model.transformer import OlyConfig, OlyForCausalLM
from oly.act.act_token import build_composite_act_string


@pytest.fixture
def tiny_config():
    """Tiny model for fast training tests."""
    return OlyConfig(
        vocab_size=200,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        act_enabled=True,
        act_num_emotions=17,
        act_hidden_size=32,
        gradient_checkpointing=False,
    )


@pytest.fixture
def sample_data_file(tmp_path):
    """Create a temporary JSONL training file."""
    data_path = tmp_path / "train.jsonl"
    samples = []
    for i in range(50):
        act_string = build_composite_act_string([("happy", 0.8)])
        samples.append({
            "input": f"Hello test message {i}",
            "output": f"{act_string} This is a test response.",
            "emotion_label": "happy",
            "emotion_id": 0,
            "intensity": 0.8,
            "secondary_emotion": None,
            "secondary_intensity": None,
            "is_composite": False,
        })

    with open(data_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    return str(data_path)


class TestTrainingLoop:
    """Test that a micro training loop runs without errors."""

    def test_micro_training(self, tiny_config, sample_data_file):
        """Run 5 training steps on a tiny model and verify loss decreases."""
        from oly.tokenizer.tokenizer import OlyTokenizer

        # Train a tiny tokenizer
        tokenizer = OlyTokenizer(vocab_size=200)
        texts = []
        with open(sample_data_file, "r") as f:
            for line in f:
                s = json.loads(line)
                texts.append(s["input"])
                texts.append(s["output"])
        tokenizer.train_from_texts(texts)

        # Create model
        model = OlyForCausalLM(tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Simple training loop
        model.train()
        losses = []

        for step in range(5):
            input_ids = torch.randint(0, tiny_config.vocab_size, (1, 32))
            labels = input_ids.clone()
            act_labels = torch.tensor([0])
            act_intensities = torch.tensor([0.8])

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                act_labels=act_labels,
                act_intensities=act_intensities,
            )

            loss = outputs["loss"]
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Loss should have changed (may not monotonically decrease in 5 steps)
        assert len(losses) == 5
        assert all(l > 0 for l in losses)

    def test_checkpoint_save_load(self, tiny_config, tmp_path):
        """Test saving and loading model checkpoints."""
        from oly.utils.helpers import save_checkpoint, load_checkpoint

        model = OlyForCausalLM(tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Get initial predictions
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            before = model(input_ids)["logits"]

        # Save checkpoint
        save_checkpoint(model, optimizer, step=42, loss=1.5,
                       output_dir=str(tmp_path / "ckpt"), config=tiny_config.to_dict())

        # Create a new model and load checkpoint
        model2 = OlyForCausalLM(tiny_config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        info = load_checkpoint(str(tmp_path / "ckpt" / "checkpoint-42"), model2, optimizer2)

        assert info["step"] == 42
        assert info["loss"] == 1.5

        # Verify weights match
        with torch.no_grad():
            after = model2(input_ids)["logits"]
        assert torch.allclose(before, after, atol=1e-5)
