"""
One-click training pipeline for Ol-y 1.1B ACTT.

Runs the complete pipeline with a single command:
1. Install dependencies
2. Generate/extract training data
3. Train the BPE tokenizer
4. Train the Ol-y 1.1B model with ACT integration

Optimized for RTX 3050 Ti (4GB VRAM) + 16GB RAM.

Usage:
    python scripts/run_all.py

    # Skip dependency installation:
    python scripts/run_all.py --skip-install

    # Custom data size:
    python scripts/run_all.py --num-samples 50000

    # Resume training from checkpoint:
    python scripts/run_all.py --resume outputs/checkpoint-1000
"""

import os
import sys
import subprocess
import argparse
import time

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def run_command(cmd: str, description: str, check: bool = True) -> int:
    """Execute a shell command with a descriptive header.

    Args:
        cmd: command to run
        description: human-readable description of the step
        check: whether to raise on non-zero exit code

    Returns:
        Process return code
    """
    print(f"\n{'='*60}")
    print(f"  STEP: {description}")
    print(f"  CMD:  {cmd}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)

    if check and result.returncode != 0:
        print(f"\n[ERROR] Step failed: {description}")
        print(f"[ERROR] Return code: {result.returncode}")
        print(f"[ERROR] You can try running the command manually:")
        print(f"        {cmd}")
        return result.returncode

    return 0


def check_gpu():
    """Check if CUDA GPU is available and print info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"[GPU] Detected: {gpu_name} ({vram:.1f} GB VRAM)")
            if vram < 4:
                print(f"[GPU] WARNING: Low VRAM ({vram:.1f} GB). Training may be slow.")
            return True
        else:
            print("[GPU] No CUDA GPU detected. Training will use CPU (very slow).")
            return False
    except ImportError:
        print("[GPU] PyTorch not installed yet. Will check after installation.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="One-click Ol-y 1.1B ACTT training pipeline"
    )
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip pip install step")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of synthetic training samples")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory for model checkpoints")
    parser.add_argument("--resume", default=None,
                        help="Resume training from checkpoint path")
    parser.add_argument("--download-hf", action="store_true",
                        help="Download additional data from HuggingFace")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data extraction (use existing data)")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Skip tokenizer training (use existing)")
    args = parser.parse_args()

    start_time = time.time()

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║      Ol-y 1.1B ACTT - One-Click Training Pipeline        ║
    ║      Affective Communication Token Transformer           ║
    ║                                                          ║
    ║      Optimized for RTX 3050 Ti (4GB) + 16GB RAM          ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # === Step 0: GPU check ===
    check_gpu()

    # === Step 1: Install dependencies ===
    if not args.skip_install:
        rc = run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing Python dependencies",
            check=False,
        )
        if rc != 0:
            print("[WARN] Some dependencies may have failed. Continuing anyway...")

    # === Step 2: Create data directory ===
    os.makedirs("data", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # === Step 3: Extract/generate training data ===
    if not args.skip_data:
        data_cmd = (
            f"{sys.executable} scripts/extract_data.py "
            f"--output data/train.jsonl "
            f"--num-synthetic {args.num_samples}"
        )
        if args.download_hf:
            data_cmd += " --download-hf"

        rc = run_command(data_cmd, "Extracting and generating training data")
        if rc != 0:
            print("[ERROR] Data extraction failed. Cannot continue.")
            sys.exit(1)

    # Verify data exists
    if not os.path.exists("data/train.jsonl"):
        print("[ERROR] Training data not found at data/train.jsonl")
        print("[ERROR] Run with: python scripts/extract_data.py --output data/train.jsonl")
        sys.exit(1)

    # === Step 4: Train tokenizer ===
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    if not args.skip_tokenizer and not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        rc = run_command(
            f"{sys.executable} scripts/train_tokenizer.py "
            f"--data data/train.jsonl "
            f"--output {tokenizer_dir}",
            "Training BPE tokenizer with ACT special tokens",
        )
        if rc != 0:
            print("[ERROR] Tokenizer training failed. Cannot continue.")
            sys.exit(1)

    # === Step 5: Train model ===
    train_cmd = (
        f"{sys.executable} scripts/train.py "
        f"--config config/oly_1b_config.json "
        f"--data data/train.jsonl "
        f"--val-data data/val.jsonl "
        f"--output-dir {args.output_dir}"
    )
    if args.resume:
        train_cmd += f" --resume {args.resume}"

    rc = run_command(train_cmd, "Training Ol-y 1.1B ACTT model")

    # === Summary ===
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                    Training Complete                      ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Time elapsed: {hours:02d}h {minutes:02d}m                                    ║
    ║  Checkpoints:  {args.output_dir}/                                ║
    ║  Tokenizer:    {tokenizer_dir}/                        ║
    ║                                                          ║
    ║  Next steps:                                             ║
    ║  1. Run tests:  python -m pytest tests/                  ║
    ║  2. Inference:  See README.md for usage examples         ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    sys.exit(rc)


if __name__ == "__main__":
    main()
