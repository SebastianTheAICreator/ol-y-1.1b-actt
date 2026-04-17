"""
Utility functions for the Ol-y 1.1B ACTT project.

Memory management, device detection, and training helpers optimized
for consumer GPUs (RTX 3050 Ti with 4GB VRAM + 16GB system RAM).
"""

import os
import gc
import json
import random
from typing import Optional, Dict, Any

import numpy as np
import torch
import psutil


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: random seed value (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations (slight performance cost)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, Any]:
    """Detect available hardware and return device configuration.

    Returns a dictionary with device type, memory info, and recommended
    training settings for the detected hardware.
    """
    info = {
        "cpu_count": os.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / (1024 ** 3),
        "ram_available_gb": psutil.virtual_memory().available / (1024 ** 3),
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        info["gpu_vram_free_gb"] = (
            torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)
        ) / (1024 ** 3)

        # Recommend training settings based on VRAM
        vram = info["gpu_vram_gb"]
        if vram <= 4:
            # RTX 3050 Ti / low VRAM cards
            info["recommended"] = {
                "batch_size": 1,
                "gradient_accumulation": 8,
                "fp16": True,
                "gradient_checkpointing": True,
                "cpu_offload": True,
                "deepspeed_stage": 2,
            }
        elif vram <= 8:
            info["recommended"] = {
                "batch_size": 2,
                "gradient_accumulation": 4,
                "fp16": True,
                "gradient_checkpointing": True,
                "cpu_offload": False,
                "deepspeed_stage": 2,
            }
        else:
            info["recommended"] = {
                "batch_size": 4,
                "gradient_accumulation": 2,
                "fp16": True,
                "gradient_checkpointing": False,
                "cpu_offload": False,
                "deepspeed_stage": 0,
            }
    else:
        info["device"] = "cpu"
        info["recommended"] = {
            "batch_size": 1,
            "gradient_accumulation": 16,
            "fp16": False,
            "gradient_checkpointing": True,
            "cpu_offload": False,
            "deepspeed_stage": 0,
        }

    return info


def clear_gpu_memory():
    """Force garbage collection and clear GPU memory cache.

    Essential for training large models on limited VRAM -- call between
    training phases or when switching between model operations.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_model_size(model: torch.nn.Module, name: str = "Model"):
    """Print model parameter count and estimated memory footprint.

    Args:
        model: PyTorch model
        name: display name for the model
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Estimated memory: fp32 = 4 bytes/param, fp16 = 2 bytes/param
    mem_fp32 = total * 4 / (1024 ** 3)
    mem_fp16 = total * 2 / (1024 ** 3)

    print(f"[{name}] Parameters:")
    print(f"  Total:     {total:>15,} ({total / 1e9:.2f}B)")
    print(f"  Trainable: {trainable:>15,}")
    print(f"  Memory (fp32): {mem_fp32:.2f} GB")
    print(f"  Memory (fp16): {mem_fp16:.2f} GB")


def format_training_log(
    step: int,
    loss: float,
    lr: float,
    loss_components: Optional[Dict[str, float]] = None,
    throughput: Optional[float] = None,
) -> str:
    """Format a training log line with loss components.

    Args:
        step: current training step
        loss: total loss value
        lr: current learning rate
        loss_components: breakdown of loss (lm, act_label, act_intensity, etc.)
        throughput: tokens per second

    Returns:
        Formatted log string
    """
    parts = [f"step={step:>6d}", f"loss={loss:.4f}", f"lr={lr:.2e}"]

    if loss_components:
        for key, val in loss_components.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            short_key = key.replace("act_", "").replace("_loss", "")
            parts.append(f"{short_key}={val:.4f}")

    if throughput is not None:
        parts.append(f"tok/s={throughput:.0f}")

    return " | ".join(parts)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    output_dir: str,
    config: Optional[Dict] = None,
):
    """Save a training checkpoint.

    Args:
        model: the model to save
        optimizer: optimizer state
        step: current training step
        loss: current loss value
        output_dir: directory to save checkpoint
        config: optional model config to save alongside
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }, os.path.join(checkpoint_path, "training_state.pt"))

    if config is not None:
        with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    print(f"[Checkpoint] Saved step {step} to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        checkpoint_path: path to checkpoint directory
        model: model to load weights into
        optimizer: optional optimizer to restore state

    Returns:
        Dictionary with step and loss from the checkpoint
    """
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=False)

    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    print(f"[Checkpoint] Loaded step {state['step']} from {checkpoint_path}")
    return {"step": state["step"], "loss": state["loss"]}
