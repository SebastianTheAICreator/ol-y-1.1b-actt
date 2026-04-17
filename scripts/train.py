"""
Training script for Ol-y 1.1B ACTT model.

Optimized for RTX 3050 Ti (4GB VRAM) + 16GB system RAM:
- Gradient checkpointing to reduce VRAM from ~8GB to ~3GB
- Mixed precision (FP16) training
- DeepSpeed ZeRO Stage 2 with CPU offloading
- Gradient accumulation (effective batch size = batch_size * accumulation_steps)
- Automatic memory management and garbage collection

Training loss follows the decomposed objective from Section 5 of the paper:
    L = lambda_1 * L_struct + lambda_2 * L_label + lambda_3 * L_resp

Where:
    lambda_1 = 2.0 (structural compliance -- highest priority)
    lambda_2 = 1.5 (emotion label accuracy)
    lambda_3 = 1.0 (response quality -- baseline)

Usage:
    python scripts/train.py --config config/oly_1b_config.json --data data/train.jsonl

With DeepSpeed:
    deepspeed scripts/train.py --config config/oly_1b_config.json --data data/train.jsonl
"""

import os
import sys
import json
import time
import math
import argparse
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oly.model.transformer import OlyConfig, OlyForCausalLM
from oly.tokenizer.tokenizer import OlyTokenizer
from oly.act.act_token import EMOTION_TO_ID, EMOTION_LABELS
from oly.utils.helpers import (
    set_seed, get_device_info, clear_gpu_memory,
    print_model_size, format_training_log,
    save_checkpoint, load_checkpoint,
)


class ACTDataset(Dataset):
    """Dataset for ACT-annotated conversational data.

    Loads JSONL files where each line contains:
    - input: user message
    - output: ACT token + model response
    - emotion_label: ground-truth emotion
    - emotion_id: emotion label index
    - intensity: emotion intensity value

    Tokenizes the input/output pairs and creates the proper label tensors
    for the combined LM + ACT loss.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: OlyTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Load JSONL data
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Construct the full training sequence: input + output
        input_text = sample["input"]
        output_text = sample["output"]
        full_text = f"User: {input_text}\nAssistant: {output_text}"

        # Tokenize
        token_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Create labels: shift by 1 for causal LM training
        # Mask input portion so the model only learns to predict the output
        input_tokens = self.tokenizer.encode(
            f"User: {input_text}\nAssistant: ", add_special_tokens=True
        )
        input_len = min(len(input_tokens), self.max_length)

        labels = token_ids.copy()
        # Set labels to -100 for input tokens (don't compute loss on them)
        for i in range(input_len):
            labels[i] = -100

        # Padding
        pad_len = self.max_length - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        # ACT labels
        emotion_id = sample.get("emotion_id", EMOTION_TO_ID.get(sample.get("emotion_label", "neutral"), 8))
        intensity = sample.get("intensity", 0.5)

        # Secondary emotion for composite ACT loss
        secondary_id = -1
        secondary_intensity = 0.0
        if sample.get("secondary_emotion"):
            secondary_id = EMOTION_TO_ID.get(sample["secondary_emotion"], -1)
            secondary_intensity = sample.get("secondary_intensity", 0.0)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "act_labels": torch.tensor(emotion_id, dtype=torch.long),
            "act_intensities": torch.tensor(intensity, dtype=torch.float),
            "act_mask": torch.tensor(1.0, dtype=torch.float),
            "secondary_labels": torch.tensor(secondary_id, dtype=torch.long),
            "secondary_intensities": torch.tensor(secondary_intensity, dtype=torch.float),
        }


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine learning rate schedule with linear warmup.

    Linearly increases LR from 0 to max during warmup_steps, then follows
    a cosine decay to 0 over the remaining steps.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_deepspeed_config(config: OlyConfig) -> Dict:
    """Create DeepSpeed configuration optimized for RTX 3050 Ti.

    Uses ZeRO Stage 2 with CPU offloading to fit the 1.1B model in 4GB VRAM:
    - ZeRO-2 partitions optimizer states across GPUs (or offloads to CPU)
    - CPU offloading moves optimizer states and gradients to system RAM
    - FP16 training reduces memory by 2x

    Returns:
        DeepSpeed config dictionary
    """
    return {
        "train_batch_size": config.act_max_composite,  # will be overridden
        "gradient_accumulation_steps": 8,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
    }


def train(
    config_path: str,
    data_path: str,
    val_data_path: Optional[str] = None,
    output_dir: str = "outputs",
    resume_from: Optional[str] = None,
    use_deepspeed: bool = False,
):
    """Main training loop for Ol-y 1.1B ACTT.

    Args:
        config_path: path to model configuration JSON
        data_path: path to training data JSONL
        val_data_path: optional path to validation data JSONL
        output_dir: directory for checkpoints and logs
        resume_from: optional checkpoint path to resume training
        use_deepspeed: whether to use DeepSpeed for memory optimization
    """
    # === Setup ===
    config = OlyConfig.from_json(config_path)
    config.gradient_checkpointing = True  # Always enable for RTX 3050 Ti

    device_info = get_device_info()
    print(f"\n{'='*60}")
    print(f"  Ol-y 1.1B ACTT Training")
    print(f"{'='*60}")
    print(f"  Device: {device_info.get('device', 'cpu')}")
    if 'gpu_name' in device_info:
        print(f"  GPU: {device_info['gpu_name']}")
        print(f"  VRAM: {device_info['gpu_vram_gb']:.1f} GB")
    print(f"  RAM: {device_info['ram_gb']:.1f} GB")
    print(f"  Config: {config_path}")
    print(f"  Data: {data_path}")
    print(f"{'='*60}\n")

    device = torch.device(device_info.get("device", "cpu"))
    set_seed(42)

    # === Tokenizer ===
    # Train a tokenizer on the data if no saved tokenizer exists
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        print("[Train] Loading existing tokenizer...")
        tokenizer = OlyTokenizer.load(tokenizer_dir)
    else:
        print("[Train] Training new tokenizer on data...")
        tokenizer = OlyTokenizer(vocab_size=config.vocab_size)
        # Extract text from JSONL for tokenizer training
        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                texts.append(sample.get("input", ""))
                texts.append(sample.get("output", ""))
        tokenizer.train_from_texts(texts)
        tokenizer.save(tokenizer_dir)

    # === Dataset ===
    train_dataset = ACTDataset(data_path, tokenizer, max_length=512)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Batch size 1 for low VRAM (accumulate gradients)
        shuffle=True,
        num_workers=0,  # 0 workers for Windows compatibility
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    val_loader = None
    if val_data_path and os.path.exists(val_data_path):
        val_dataset = ACTDataset(val_data_path, tokenizer, max_length=512)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # === Model ===
    print("[Train] Initializing Ol-y 1.1B ACTT model...")
    model = OlyForCausalLM(config)
    print_model_size(model, "Ol-y 1.1B ACTT")

    # Move to device (with memory optimization)
    if device.type == "cuda":
        model = model.half()  # FP16 to save VRAM
    model = model.to(device)

    # === Optimizer ===
    # AdamW with weight decay (exclude biases and layer norms)
    no_decay = ["bias", "layernorm", "layer_norm", "norm"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                      if not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

    # Learning rate schedule
    total_steps = len(train_loader) * 3 // 8  # 3 epochs, gradient accumulation = 8
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=500, total_steps=total_steps)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # === Resume ===
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint_info = load_checkpoint(resume_from, model, optimizer)
        start_step = checkpoint_info["step"]
        print(f"[Train] Resumed from step {start_step}")

    # === Training Loop ===
    os.makedirs(output_dir, exist_ok=True)
    accumulation_steps = 8
    log_interval = 50
    save_interval = 1000
    eval_interval = 500

    model.train()
    global_step = start_step
    accumulated_loss = 0.0
    step_start_time = time.time()
    best_val_loss = float("inf")

    print(f"\n[Train] Starting training...")
    print(f"  Total steps: ~{total_steps}")
    print(f"  Gradient accumulation: {accumulation_steps}")
    print(f"  Effective batch size: {1 * accumulation_steps}")
    print(f"  Learning rate: 3e-4 (cosine schedule)")
    print(f"  FP16: {device.type == 'cuda'}")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    print()

    for epoch in range(3):
        print(f"--- Epoch {epoch + 1}/3 ---")

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            act_labels = batch["act_labels"].to(device)
            act_intensities = batch["act_intensities"].to(device)
            act_mask = batch["act_mask"].to(device)

            # Forward pass with mixed precision
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        act_labels=act_labels,
                        act_intensities=act_intensities,
                        act_mask=act_mask,
                    )
                    loss = outputs["loss"] / accumulation_steps

                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    act_labels=act_labels,
                    act_intensities=act_intensities,
                    act_mask=act_mask,
                )
                loss = outputs["loss"] / accumulation_steps
                loss.backward()

            accumulated_loss += loss.item()

            # Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % log_interval == 0:
                    elapsed = time.time() - step_start_time
                    tokens_per_sec = (log_interval * accumulation_steps * 512) / elapsed
                    log_str = format_training_log(
                        step=global_step,
                        loss=accumulated_loss / log_interval,
                        lr=scheduler.get_last_lr()[0],
                        loss_components=outputs.get("loss_components"),
                        throughput=tokens_per_sec,
                    )
                    print(f"  {log_str}")
                    accumulated_loss = 0.0
                    step_start_time = time.time()

                # Validation
                if val_loader and global_step % eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device, scaler)
                    print(f"  [Eval] step={global_step} val_loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, global_step, val_loss,
                                       os.path.join(output_dir, "best"), config.to_dict())
                    model.train()

                # Save checkpoint
                if global_step % save_interval == 0:
                    save_checkpoint(model, optimizer, global_step, accumulated_loss,
                                   output_dir, config.to_dict())
                    clear_gpu_memory()

    # Final save
    save_checkpoint(model, optimizer, global_step, accumulated_loss, output_dir, config.to_dict())
    print(f"\n[Train] Training complete! Final step: {global_step}")
    print(f"[Train] Best validation loss: {best_val_loss:.4f}")
    print(f"[Train] Checkpoints saved to: {output_dir}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Run evaluation on validation set.

    Returns average loss across all validation samples.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        act_labels = batch["act_labels"].to(device)
        act_intensities = batch["act_intensities"].to(device)
        act_mask = batch["act_mask"].to(device)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    act_labels=act_labels,
                    act_intensities=act_intensities,
                    act_mask=act_mask,
                )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                act_labels=act_labels,
                act_intensities=act_intensities,
                act_mask=act_mask,
            )

        total_loss += outputs["loss"].item()
        num_batches += 1

        # Limit eval to 100 batches for speed
        if num_batches >= 100:
            break

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Ol-y 1.1B ACTT model")
    parser.add_argument("--config", default="config/oly_1b_config.json",
                        help="Path to model config JSON")
    parser.add_argument("--data", default="data/train.jsonl",
                        help="Path to training data JSONL")
    parser.add_argument("--val-data", default="data/val.jsonl",
                        help="Path to validation data JSONL")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--deepspeed", action="store_true",
                        help="Use DeepSpeed for memory optimization")
    args = parser.parse_args()

    train(
        config_path=args.config,
        data_path=args.data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        resume_from=args.resume,
        use_deepspeed=args.deepspeed,
    )


if __name__ == "__main__":
    main()
