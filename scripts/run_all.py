"""
One-click training pipeline for Ol-y 1.1B ACTT.

Runs the complete pipeline with a single command:
1. System check (GPU, RAM, disk)
2. Install dependencies
3. Run tests on synthetic data (verify everything works)
4. Extract/download training datasets
5. Train the BPE tokenizer
6. Train the Ol-y 1.1B model with ACT integration

Optimized for RTX 3050 Ti (4GB VRAM) + 16GB RAM.

Usage:
    python scripts/run_all.py

    # Skip dependency installation:
    python scripts/run_all.py --skip-install

    # Custom data size + all HuggingFace datasets:
    python scripts/run_all.py --num-samples 50000 --download-hf

    # Resume training from checkpoint:
    python scripts/run_all.py --resume outputs/checkpoint-1000

    # Only run tests (no training):
    python scripts/run_all.py --test-only
"""

import os
import sys
import subprocess
import argparse
import time
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Track timing for each step
step_timings = {}


def run_command(cmd: str, description: str, step_num: int, total_steps: int,
                check: bool = True) -> int:
    """Execute a shell command with detailed progress output.

    Args:
        cmd: command to run
        description: human-readable description of the step
        step_num: current step number
        total_steps: total number of steps
        check: whether to raise on non-zero exit code

    Returns:
        Process return code
    """
    step_start = time.time()

    print(f"\n{'='*64}")
    print(f"  [{step_num}/{total_steps}] {description}")
    print(f"  {'─'*60}")
    print(f"  CMD: {cmd}")
    print(f"{'='*64}\n")

    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)

    elapsed = time.time() - step_start
    step_timings[description] = elapsed

    if result.returncode == 0:
        print(f"\n  [{step_num}/{total_steps}] {description} -- DONE ({elapsed:.1f}s)")
    else:
        print(f"\n  [{step_num}/{total_steps}] {description} -- FAILED (exit code {result.returncode})")
        if check:
            print(f"  [ERROR] You can try running the command manually:")
            print(f"          {cmd}")

    return result.returncode


def check_system():
    """Comprehensive system check before training."""
    print(f"\n{'='*64}")
    print(f"  SYSTEM CHECK")
    print(f"{'='*64}")

    # Python version
    py_ver = sys.version.split()[0]
    print(f"\n  Python:  {py_ver}")
    print(f"  Path:    {sys.executable}")

    # OS
    import platform
    print(f"  OS:      {platform.system()} {platform.release()}")

    # RAM
    try:
        import psutil
        ram_total = psutil.virtual_memory().total / (1024**3)
        ram_available = psutil.virtual_memory().available / (1024**3)
        ram_pct = psutil.virtual_memory().percent
        print(f"\n  RAM:     {ram_total:.1f} GB total, {ram_available:.1f} GB free ({ram_pct}% used)")
        if ram_available < 8:
            print(f"  WARNING: Low available RAM. Close other applications for best performance.")
    except ImportError:
        print(f"\n  RAM:     (psutil not installed, cannot check)")

    # GPU
    gpu_available = False
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            vram_total = total_memory / (1024**3)
            vram_free = (total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
            print(f"\n  GPU:     {gpu_name}")
            print(f"  VRAM:    {vram_total:.1f} GB total, {vram_free:.1f} GB free")
            print(f"  CUDA:    {torch.version.cuda}")
            print(f"  PyTorch: {torch.__version__}")
            gpu_available = True

            if vram_total <= 4:
                print(f"\n  GPU Mode: LOW VRAM (<=4GB)")
                print(f"    - Batch size: 1")
                print(f"    - Gradient accumulation: 8 steps")
                print(f"    - Gradient checkpointing: ON")
                print(f"    - FP16 mixed precision: ON")
                print(f"    - CPU offloading: ON")
            elif vram_total <= 8:
                print(f"\n  GPU Mode: MEDIUM VRAM (4-8GB)")
                print(f"    - Batch size: 2")
                print(f"    - Gradient accumulation: 4 steps")
                print(f"    - Gradient checkpointing: ON")
                print(f"    - FP16 mixed precision: ON")
            else:
                print(f"\n  GPU Mode: HIGH VRAM (>8GB)")
                print(f"    - Batch size: 4")
                print(f"    - Gradient accumulation: 2 steps")
                print(f"    - FP16 mixed precision: ON")
        else:
            print(f"\n  GPU:     Not detected (CUDA unavailable)")
            print(f"  PyTorch: {torch.__version__}")
            print(f"  WARNING: Training on CPU will be VERY slow.")
    except ImportError:
        print(f"\n  GPU:     (PyTorch not installed yet)")

    # Disk space
    try:
        import shutil
        disk = shutil.disk_usage(PROJECT_ROOT)
        free_gb = disk.free / (1024**3)
        print(f"\n  Disk:    {free_gb:.1f} GB free")
        if free_gb < 10:
            print(f"  WARNING: Low disk space. Training may need 10-50GB for data + checkpoints.")
    except Exception:
        pass

    # Check data directory
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if os.path.exists(os.path.join(data_dir, "train.jsonl")):
        train_size = os.path.getsize(os.path.join(data_dir, "train.jsonl")) / (1024**2)
        # Count lines
        with open(os.path.join(data_dir, "train.jsonl"), "r") as f:
            num_samples = sum(1 for _ in f)
        print(f"\n  Existing data: {num_samples:,} samples ({train_size:.1f} MB)")
    else:
        print(f"\n  Existing data: None (will generate)")

    print(f"\n{'='*64}")
    return gpu_available


def count_data_samples(path):
    """Count samples in a JSONL file."""
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser(
        description="One-click Ol-y 1.1B ACTT training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Default:     Generate synthetic data + train model
  --download-hf: Also download GoEmotions, EmpatheticDialogues, DailyDialog, Saravia
  --test-only: Only run tests (no training)
  --data-only: Only extract/download data (no training)

Examples:
  python scripts/run_all.py                              # Synthetic only
  python scripts/run_all.py --download-hf                # All datasets
  python scripts/run_all.py --test-only                  # Just run tests
  python scripts/run_all.py --num-samples 50000 --download-hf  # Full pipeline
        """
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
                        help="Download ALL recommended HuggingFace datasets")
    parser.add_argument("--download-goemo", action="store_true",
                        help="Download GoEmotions only")
    parser.add_argument("--download-empathetic", action="store_true",
                        help="Download EmpatheticDialogues only")
    parser.add_argument("--oly-logs", default=None,
                        help="Path to cleaned Ol-y logs")
    parser.add_argument("--shinon-logs", dest="legacy_oly_logs", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--memory-state", default=None,
                        help="Optional JSON path for EMA emotional memory")
    parser.add_argument("--surface-act-logs", action="store_true",
                        help="Use surface ACT tokens instead of internal probe distributions")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data extraction (use existing data)")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Skip tokenizer training (use existing)")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip pre-training test validation")
    parser.add_argument("--pretrain-train-steps", type=int, default=12,
                        help="Tiny mini-training steps for the synthetic pre-train check")
    parser.add_argument("--pretrain-eval-batches", type=int, default=4,
                        help="Evaluation batches for the synthetic pre-train check")
    parser.add_argument("--pretrain-batch-size", type=int, default=4,
                        help="Batch size for the synthetic pre-train check")
    parser.add_argument("--pretrain-print-every", type=int, default=3,
                        help="Print tiny training metrics every N synthetic steps")
    parser.add_argument("--pretrain-generation-tokens", type=int, default=6,
                        help="Generated tokens in the synthetic preview")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run tests, do not train")
    parser.add_argument("--data-only", action="store_true",
                        help="Only extract/download data, do not train")
    args = parser.parse_args()

    start_time = time.time()

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║      Ol-y 1.1B ACTT - Complete Training Pipeline             ║
    ║      Affective Communication Token Transformer               ║
    ║                                                              ║
    ║      17 emotions | 1.1B parameters | ACT compliance          ║
    ║      Optimized for RTX 3050 Ti (4GB) + 16GB RAM              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Determine total steps
    total_steps = 0
    if not args.skip_install:
        total_steps += 1  # install deps
    total_steps += 1  # system check
    if not args.skip_tests:
        total_steps += 1  # run tests
    if not args.skip_data and not args.test_only:
        total_steps += 1  # data extraction
    if not args.skip_tokenizer and not args.test_only and not args.data_only:
        total_steps += 1  # tokenizer
    if not args.test_only and not args.data_only:
        total_steps += 1  # training

    current_step = 0

    # ═══════════════════════════════════════════════════════
    # STEP: System check
    # ═══════════════════════════════════════════════════════
    current_step += 1
    print(f"\n  [{current_step}/{total_steps}] System Check")
    gpu_available = check_system()

    # ═══════════════════════════════════════════════════════
    # STEP: Install dependencies
    # ═══════════════════════════════════════════════════════
    if not args.skip_install:
        current_step += 1
        rc = run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing Python dependencies",
            current_step, total_steps,
            check=False,
        )
        if rc != 0:
            print("  [WARN] Some dependencies may have failed. Continuing...")

    # ═══════════════════════════════════════════════════════
    # STEP: Run tests (verify code works before training)
    # ═══════════════════════════════════════════════════════
    if not args.skip_tests:
        current_step += 1
        test_cmd = (
            f"{sys.executable} scripts/pretrain_synthetic_check.py "
            f"--train-steps {args.pretrain_train_steps} "
            f"--eval-batches {args.pretrain_eval_batches} "
            f"--batch-size {args.pretrain_batch_size} "
            f"--print-every {args.pretrain_print_every} "
            f"--generation-tokens {args.pretrain_generation_tokens}"
        )
        rc = run_command(
            test_cmd,
            "Running synthetic pre-training integration check",
            current_step, total_steps,
            check=False,
        )
        if rc != 0:
            print("\n  [WARN] Some tests failed. Review the output above.")
            print("  [WARN] You can fix issues and re-run, or continue with --skip-tests")
            if args.test_only:
                sys.exit(rc)
        else:
            print("  All tests passed! Pipeline is verified.")

        if args.test_only:
            _print_summary(start_time, args, test_only=True)
            sys.exit(0)

    # ═══════════════════════════════════════════════════════
    # STEP: Create directories
    # ═══════════════════════════════════════════════════════
    os.makedirs("data", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════
    # STEP: Extract/download training data
    # ═══════════════════════════════════════════════════════
    if not args.skip_data:
        current_step += 1
        data_cmd = (
            f"{sys.executable} scripts/extract_data.py "
            f"--output data/train.jsonl "
            f"--num-synthetic {args.num_samples}"
        )
        if args.download_hf:
            data_cmd += " --download-hf"
        if args.download_goemo:
            data_cmd += " --download-goemo"
        if args.download_empathetic:
            data_cmd += " --download-empathetic"
        log_dir = args.oly_logs or args.legacy_oly_logs
        if log_dir:
            data_cmd += f' --oly-logs "{log_dir}"'
        if args.memory_state:
            data_cmd += f' --memory-state "{args.memory_state}"'
        if args.surface_act_logs:
            data_cmd += " --surface-act-logs"

        rc = run_command(
            data_cmd,
            "Extracting and downloading training datasets",
            current_step, total_steps,
        )
        if rc != 0:
            print("  [ERROR] Data extraction failed. Cannot continue.")
            sys.exit(1)

    # Verify data exists and report
    if not os.path.exists("data/train.jsonl"):
        print("  [ERROR] Training data not found at data/train.jsonl")
        print("  [ERROR] Run: python scripts/extract_data.py --output data/train.jsonl")
        sys.exit(1)

    train_count = count_data_samples("data/train.jsonl")
    val_count = count_data_samples("data/val.jsonl")
    train_mb = os.path.getsize("data/train.jsonl") / (1024**2)
    print(f"\n  Dataset ready:")
    print(f"    Training:   {train_count:>8,d} samples ({train_mb:.1f} MB)")
    print(f"    Validation: {val_count:>8,d} samples")

    if args.data_only:
        _print_summary(start_time, args, data_only=True)
        sys.exit(0)

    # ═══════════════════════════════════════════════════════
    # STEP: Train tokenizer
    # ═══════════════════════════════════════════════════════
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    if not args.skip_tokenizer and not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        current_step += 1
        rc = run_command(
            f"{sys.executable} scripts/train_tokenizer.py "
            f"--data data/train.jsonl "
            f"--output {tokenizer_dir}",
            "Training BPE tokenizer with ACT special tokens",
            current_step, total_steps,
        )
        if rc != 0:
            print("  [ERROR] Tokenizer training failed. Cannot continue.")
            sys.exit(1)
    else:
        print(f"\n  Tokenizer already exists at {tokenizer_dir}/")

    # ═══════════════════════════════════════════════════════
    # STEP: Train model
    # ═══════════════════════════════════════════════════════
    current_step += 1
    train_cmd = (
        f"{sys.executable} scripts/train.py "
        f"--config config/oly_1b_config.json "
        f"--data data/train.jsonl "
        f"--val-data data/val.jsonl "
        f"--output-dir {args.output_dir}"
    )
    if args.resume:
        train_cmd += f" --resume {args.resume}"

    print(f"\n  Training will begin with:")
    print(f"    Model:       Ol-y 1.1B ACTT (~1.1B parameters)")
    print(f"    Data:        {train_count:,} training samples")
    print(f"    Config:      config/oly_1b_config.json")
    print(f"    Output:      {args.output_dir}/")
    print(f"    Loss:        L = 2.0*L_struct + 1.5*L_label + 1.0*L_resp")
    if args.resume:
        print(f"    Resume from: {args.resume}")

    rc = run_command(
        train_cmd,
        "Training Ol-y 1.1B ACTT model",
        current_step, total_steps,
    )

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    _print_summary(start_time, args, training_rc=rc)
    sys.exit(rc)


def _print_summary(start_time, args, test_only=False, data_only=False, training_rc=0):
    """Print final pipeline summary."""
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")

    if test_only:
        mode = "TEST ONLY"
    elif data_only:
        mode = "DATA EXTRACTION ONLY"
    elif training_rc == 0:
        mode = "TRAINING COMPLETE"
    else:
        mode = "TRAINING FAILED"

    print(f"""

    ╔══════════════════════════════════════════════════════════════╗
    ║  {mode:^60s}  ║
    ╠══════════════════════════════════════════════════════════════╣""")

    # Timing breakdown
    print(f"    ║  Time elapsed: {hours:02d}h {minutes:02d}m {seconds:02d}s{' ' * 39}║")
    if step_timings:
        print(f"    ║{'─'*62}║")
        for step_name, step_time in step_timings.items():
            mins = int(step_time // 60)
            secs = int(step_time % 60)
            short_name = step_name[:45]
            time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            print(f"    ║    {short_name:<45s} {time_str:>10s}   ║")

    if not test_only:
        print(f"""    ║{'─'*62}║
    ║  Output:                                                     ║
    ║    Checkpoints:  {args.output_dir + '/':.<43s}║
    ║    Tokenizer:    {tokenizer_dir + '/':.<43s}║
    ║    Train data:   data/train.jsonl                            ║
    ║    Val data:     data/val.jsonl                               ║""")

    print(f"""    ║{'─'*62}║
    ║  Next steps:                                                 ║
    ║    1. Run tests:   python -m pytest tests/ -v                ║
    ║    2. Inference:    See README.md for usage examples          ║
    ║    3. Fine-tune:    python scripts/train.py --resume <ckpt>  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
