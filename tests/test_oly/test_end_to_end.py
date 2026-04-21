"""
End-to-end integration test: data generation -> tokenizer -> training -> inference.

This test runs the FULL pipeline on a tiny model with synthetic data:
1. Generate synthetic dataset (all 17 emotions)
2. Train a BPE tokenizer on the data
3. Create the ACTDataset from JSONL
4. Run a micro training loop (10 steps)
5. Verify loss decreases or stays bounded
6. Save and reload checkpoint
7. Run inference on the reloaded model
8. Verify ACT token can be constructed from the output

This test uses NO pre-trained weights -- everything is from scratch.
Designed to verify the complete pipeline before committing to long training runs.
"""

import sys
import os
import json
import random
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oly.model.transformer import OlyConfig, OlyForCausalLM
from oly.tokenizer.tokenizer import OlyTokenizer
from oly.act.act_token import (
    EMOTION_LABELS, EMOTION_TO_ID, build_composite_act_string,
    parse_act_from_response,
)
from oly.utils.helpers import save_checkpoint, load_checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from extract_data import generate_synthetic_dataset, save_dataset


@pytest.fixture(scope="module")
def e2e_config():
    """Tiny config for end-to-end test (~200K params instead of 1.1B)."""
    return OlyConfig(
        vocab_size=500,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        act_enabled=True,
        act_num_emotions=17,
        act_hidden_size=32,
        act_max_composite=5,
        act_loss_structural=2.0,
        act_loss_label=1.5,
        act_loss_response=1.0,
        gradient_checkpointing=False,
    )


@pytest.fixture(scope="module")
def synthetic_dataset():
    """Generate 340 synthetic samples (20 per emotion)."""
    random.seed(42)
    return generate_synthetic_dataset(num_samples=340, composite_ratio=0.3)


@pytest.fixture(scope="module")
def data_files(synthetic_dataset, tmp_path_factory):
    """Save synthetic data to JSONL files."""
    tmp_dir = tmp_path_factory.mktemp("e2e_data")
    train_path = str(tmp_dir / "train.jsonl")
    save_dataset(synthetic_dataset, train_path, split_ratio=0.85)
    val_path = train_path.replace("train", "val")
    return train_path, val_path


@pytest.fixture(scope="module")
def trained_tokenizer(data_files):
    """Train a tokenizer on the synthetic data."""
    train_path, _ = data_files
    texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            texts.append(sample["input"])
            texts.append(sample["output"])
    tokenizer = OlyTokenizer(vocab_size=500)
    tokenizer.train_from_texts(texts)
    return tokenizer


class ACTDatasetMini(torch.utils.data.Dataset):
    """Minimal ACT dataset for testing (mirrors scripts/train.py ACTDataset)."""

    def __init__(self, data_path, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_text = f"User: {sample['input']}\nAssistant: {sample['output']}"
        token_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        input_prefix = self.tokenizer.encode(
            f"User: {sample['input']}\nAssistant: ", add_special_tokens=True
        )
        input_len = min(len(input_prefix), self.max_length)

        labels = token_ids.copy()
        for i in range(input_len):
            labels[i] = -100

        pad_len = self.max_length - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        emotion_id = sample.get("emotion_id", 8)
        intensity = sample.get("intensity", 0.5)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "act_labels": torch.tensor(emotion_id, dtype=torch.long),
            "act_intensities": torch.tensor(intensity, dtype=torch.float),
            "act_mask": torch.tensor(1.0, dtype=torch.float),
        }


class TestEndToEndPipeline:
    """Complete pipeline test: data -> tokenizer -> train -> inference."""

    def test_step1_data_generated(self, synthetic_dataset):
        """Synthetic dataset should have 340 samples covering all emotions."""
        assert len(synthetic_dataset) == 340
        emotions_found = {s["emotion_label"] for s in synthetic_dataset}
        assert len(emotions_found) == 17

    def test_step2_data_saved(self, data_files):
        """JSONL files should exist and be non-empty."""
        train_path, val_path = data_files
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        assert os.path.getsize(train_path) > 0
        assert os.path.getsize(val_path) > 0

    def test_step3_tokenizer_trained(self, trained_tokenizer):
        """Tokenizer should have a valid vocabulary."""
        assert trained_tokenizer.get_vocab_size() > 0
        assert trained_tokenizer.pad_token_id >= 0
        assert trained_tokenizer.bos_token_id >= 0
        assert trained_tokenizer.eos_token_id >= 0
        assert trained_tokenizer.act_start_token_id >= 0

    def test_step4_dataset_loads(self, data_files, trained_tokenizer):
        """ACTDataset should load and produce valid batches."""
        train_path, _ = data_files
        dataset = ACTDatasetMini(train_path, trained_tokenizer, max_length=64)
        assert len(dataset) > 0

        sample = dataset[0]
        assert sample["input_ids"].shape == (64,)
        assert sample["attention_mask"].shape == (64,)
        assert sample["labels"].shape == (64,)
        assert sample["act_labels"].shape == ()
        assert sample["act_intensities"].shape == ()

    def test_step5_micro_training(self, e2e_config, data_files, trained_tokenizer):
        """Run 10 training steps and verify loss is reasonable."""
        torch.manual_seed(42)
        model = OlyForCausalLM(e2e_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        train_path, _ = data_files
        dataset = ACTDatasetMini(train_path, trained_tokenizer, max_length=64)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

        losses = []
        for step, batch in enumerate(loader):
            if step >= 10:
                break

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                act_labels=batch["act_labels"],
                act_intensities=batch["act_intensities"],
                act_mask=batch["act_mask"],
            )

            loss = outputs["loss"]
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert len(losses) == 10
        assert all(l > 0 for l in losses), "All losses should be positive"
        assert all(l < 100 for l in losses), "Losses should be bounded (no explosion)"

    def test_step6_checkpoint_roundtrip(self, e2e_config, tmp_path):
        """Save and reload a checkpoint, verify weights match."""
        torch.manual_seed(42)
        model = OlyForCausalLM(e2e_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Do one training step to change weights from initialization
        input_ids = torch.randint(0, e2e_config.vocab_size, (1, 16))
        labels = input_ids.clone()
        act_labels = torch.tensor([0])
        act_intensities = torch.tensor([0.8])
        outputs = model(input_ids=input_ids, labels=labels,
                       act_labels=act_labels, act_intensities=act_intensities)
        outputs["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        # Get predictions before save
        model.eval()
        with torch.no_grad():
            before = model(input_ids=input_ids)["logits"]

        # Save checkpoint
        save_checkpoint(model, optimizer, step=1, loss=1.0,
                       output_dir=str(tmp_path / "ckpt"),
                       config=e2e_config.to_dict())

        # Load into new model
        model2 = OlyForCausalLM(e2e_config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        info = load_checkpoint(str(tmp_path / "ckpt" / "checkpoint-1"), model2, optimizer2)

        assert info["step"] == 1
        assert info["loss"] == 1.0

        model2.eval()
        with torch.no_grad():
            after = model2(input_ids=input_ids)["logits"]

        assert torch.allclose(before, after, atol=1e-5), "Weights don't match after reload"

    def test_step7_inference_after_training(self, e2e_config):
        """After micro-training, model should generate tokens and emit ACT."""
        torch.manual_seed(42)
        model = OlyForCausalLM(e2e_config)

        # Quick training
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(3):
            ids = torch.randint(0, e2e_config.vocab_size, (1, 16))
            outputs = model(input_ids=ids, labels=ids,
                          act_labels=torch.tensor([0]),
                          act_intensities=torch.tensor([0.7]))
            outputs["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

        # Inference
        model.eval()
        prompt = torch.randint(0, e2e_config.vocab_size, (1, 8))
        result = model.generate(prompt, max_new_tokens=10, emit_act=True)

        assert result["generated_ids"].shape[1] >= 8
        assert result["act_result"] is not None
        assert result["act_result"]["intensity"].item() >= 0.1
        assert result["act_result"]["intensity"].item() <= 1.0

    def test_step8_act_token_from_model(self, e2e_config):
        """Construct a complete ACT token string from model output."""
        torch.manual_seed(42)
        model = OlyForCausalLM(e2e_config)
        model.eval()

        input_ids = torch.randint(0, e2e_config.vocab_size, (1, 8))
        act_result = model.call_act(input_ids)

        # Extract prediction
        emotion_idx = act_result["emotion_pred"].item()
        emotion_name = EMOTION_LABELS[emotion_idx]
        intensity = round(act_result["intensity"].item(), 1)

        # Build ACT token
        act_string = build_composite_act_string([(emotion_name, intensity)])
        assert '<|ACT:"emotion":' in act_string

        # Parse it back
        parsed = parse_act_from_response(act_string)
        assert parsed is not None
        assert parsed.dominant.name == emotion_name
        assert abs(parsed.dominant.intensity - intensity) < 0.01


class TestDatasetCoverage:
    """Verify that the synthetic dataset provides adequate training signal."""

    def test_emotion_balance(self, synthetic_dataset):
        """Each emotion should have at least 15 samples (target=20 per emotion)."""
        counts = {}
        for s in synthetic_dataset:
            e = s["emotion_label"]
            counts[e] = counts.get(e, 0) + 1

        for emotion in EMOTION_LABELS:
            assert counts.get(emotion, 0) >= 15, (
                f"{emotion}: only {counts.get(emotion, 0)} samples"
            )

    def test_intensity_spread(self, synthetic_dataset):
        """Intensities should span the valid range, not cluster."""
        intensities = [s["intensity"] for s in synthetic_dataset]
        assert min(intensities) <= 0.3, f"Min intensity too high: {min(intensities)}"
        assert max(intensities) >= 0.7, f"Max intensity too low: {max(intensities)}"

    def test_composite_samples_exist(self, synthetic_dataset):
        """Dataset should contain composite (multi-emotion) samples."""
        composite_count = sum(1 for s in synthetic_dataset if s["is_composite"])
        assert composite_count > 0, "No composite samples found"

    def test_all_act_tokens_parseable(self, synthetic_dataset):
        """Every sample's ACT token should be parseable."""
        for i, sample in enumerate(synthetic_dataset):
            act = parse_act_from_response(sample["output"])
            assert act is not None, (
                f"Sample {i} ({sample['emotion_label']}): unparseable ACT token"
            )

    def test_input_diversity(self, synthetic_dataset):
        """Inputs should have reasonable diversity (not all identical)."""
        unique_inputs = set(s["input"] for s in synthetic_dataset)
        # With 340 samples and 7+ prompts per emotion, we expect many unique inputs
        assert len(unique_inputs) >= 50, (
            f"Only {len(unique_inputs)} unique inputs in 340 samples"
        )
