"""
Pre-training synthetic integration check for Ol-y ACTT.

This script exercises the main components without external datasets or large
weights. It is intended to run before real training:

1. ACT token parsing/building
2. Synthetic data generation and JSONL split
3. Ol-y log compression/extraction with probe-preferred labels
4. Async probe aggregation and EMA emotional memory
5. Tokenizer training/save/load
6. ACTDataset batching
7. Tiny OlyForCausalLM forward/loss/backward/generate
8. Mini-training diagnostics with before/after metrics
9. Generation-time EMA update
10. Checkpoint save/load
11. Quantization/config asset sanity checks

Usage:
    python scripts/pretrain_synthetic_check.py
"""

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from extract_data import generate_synthetic_dataset, save_dataset
from train import ACTDataset

from oly.act.act_loss import ACTLoss
from oly.act.act_token import (
    EMOTION_LABELS,
    build_composite_act_string,
    parse_act_from_response,
)
from oly.act.async_probe import AsyncProbeRunner
from oly.act.emotional_memory import EmotionalMemoryEMA
from oly.data.oly_logs import (
    compress_log_directory,
    extract_samples_from_session_log,
    load_oly_logs,
)
from oly.model.transformer import OlyConfig, OlyForCausalLM
from oly.tokenizer.tokenizer import OlyTokenizer
from oly.utils.helpers import load_checkpoint, save_checkpoint


REAL_TRAINING_DATA_SOURCES = [
    "Synthetic ACT conversations generated from the 17-label Ol-y ACT taxonomy.",
    "Cleaned Ol-y session logs with internal probe distributions and EMA emotional-memory annotations.",
    "GoEmotions by Google Research, mapped to the Ol-y ACT taxonomy for fine-grained and composite emotion labels.",
    "EmpatheticDialogues by Facebook AI, mapped to ACT labels for emotionally grounded response generation.",
    "DailyDialog, mapped to ACT labels for everyday conversational flow and neutral coverage.",
    "dair-ai/emotion / Saravia emotion dataset, mapped to ACT labels as a basic emotion baseline.",
]


class SmokeContext:
    def __init__(self, tmp_dir: Path, args: argparse.Namespace):
        self.tmp_dir = tmp_dir
        self.args = args
        self.samples: List[Dict] = []
        self.train_path = tmp_dir / "train.jsonl"
        self.val_path = tmp_dir / "val.jsonl"
        self.tokenizer = None
        self.config = None
        self.model = None
        self.memory = EmotionalMemoryEMA(alpha=0.3)
        self.metrics: Dict[str, Any] = {}


def _print_header() -> None:
    print("\nOl-y ACTT pre-training synthetic check")
    print("=" * 48)


def _run_step(name: str, fn: Callable[[SmokeContext], None], ctx: SmokeContext) -> None:
    print(f"\n[CHECK] {name}")
    started = time.perf_counter()
    fn(ctx)
    elapsed = time.perf_counter() - started
    print(f"[PASS]  {name} ({elapsed:.2f}s)")


def _label_name(label_id: int) -> str:
    if 0 <= int(label_id) < len(EMOTION_LABELS):
        return EMOTION_LABELS[int(label_id)]
    return "unknown"


@torch.no_grad()
def evaluate_tiny_model(
    model: OlyForCausalLM,
    loader: torch.utils.data.DataLoader,
    max_batches: int,
) -> Dict[str, Any]:
    """Evaluate a tiny model on a few synthetic batches."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_items = 0
    correct = 0
    intensity_abs_error = 0.0
    target_counts: Counter = Counter()
    pred_counts: Counter = Counter()
    sample_pairs = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            act_labels=batch["act_labels"],
            act_intensities=batch["act_intensities"],
            act_mask=batch["act_mask"],
        )
        batch_size = batch["input_ids"].shape[0]
        total_loss += outputs["loss"].item() * batch_size
        total_items += batch_size

        pred = outputs["act_emotion_logits"].argmax(dim=-1).cpu()
        target = batch["act_labels"].cpu()
        correct += (pred == target).sum().item()
        intensity_abs_error += (
            outputs["act_intensity"].detach().cpu() - batch["act_intensities"].cpu()
        ).abs().sum().item()

        target_counts.update(_label_name(i.item()) for i in target)
        pred_counts.update(_label_name(i.item()) for i in pred)

        if len(sample_pairs) < 5:
            for target_id, pred_id, pred_intensity in zip(
                target.tolist(),
                pred.tolist(),
                outputs["act_intensity"].detach().cpu().tolist(),
            ):
                sample_pairs.append({
                    "target": _label_name(target_id),
                    "predicted": _label_name(pred_id),
                    "pred_intensity": round(float(pred_intensity), 3),
                })
                if len(sample_pairs) >= 5:
                    break

    if was_training:
        model.train()

    total_items = max(total_items, 1)
    return {
        "loss": total_loss / total_items,
        "act_accuracy": correct / total_items,
        "intensity_mae": intensity_abs_error / total_items,
        "target_distribution": dict(target_counts),
        "prediction_distribution": dict(pred_counts),
        "examples": sample_pairs,
        "items": total_items,
    }


def print_eval_report(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n  {title}:")
    print(f"    loss={metrics['loss']:.4f}")
    print(f"    act_accuracy={metrics['act_accuracy'] * 100:.1f}%")
    print(f"    intensity_mae={metrics['intensity_mae']:.4f}")
    print(f"    evaluated_items={metrics['items']}")
    top_preds = Counter(metrics["prediction_distribution"]).most_common(5)
    if top_preds:
        formatted = ", ".join(f"{name}:{count}" for name, count in top_preds)
        print(f"    top_predictions={formatted}")


def check_project_assets(ctx: SmokeContext) -> None:
    modelfile = PROJECT_ROOT / "Modelfile.optimized"
    quant_config = PROJECT_ROOT / "config" / "quantization_presets.json"
    oly_3b_config = PROJECT_ROOT / "config" / "oly_3b_config.json"
    roadmap = PROJECT_ROOT / "ROADMAP_SAKISHIMIRO.txt"

    assert modelfile.exists(), "Missing Modelfile.optimized"
    assert "FROM" in modelfile.read_text(encoding="utf-8"), "Modelfile lacks FROM"

    with quant_config.open("r", encoding="utf-8") as f:
        quant = json.load(f)
    assert quant["default"] in quant["presets"], "Invalid quantization default"

    with oly_3b_config.open("r", encoding="utf-8") as f:
        oly_3b = json.load(f)
    assert oly_3b["act"]["num_emotions"] == 17
    assert oly_3b["model_name"] == "Ol-y-3B-ACTT"

    assert "Sakishimiro" in roadmap.read_text(encoding="utf-8")
    print(f"  Quantization default: {quant['default']}")
    print(f"  Coordination target: {oly_3b['model_name']}")


def check_act_schema(ctx: SmokeContext) -> None:
    act_string = build_composite_act_string([("sad", 0.7), ("hopeful", 0.4)])
    parsed = parse_act_from_response(f"{act_string} I can hold both feelings.")
    assert parsed is not None
    assert parsed.dominant.name == "sad"
    assert parsed.secondary.name == "hopeful"
    assert len(EMOTION_LABELS) == 17
    print(f"  Parsed composite ACT: {parsed.dominant.name}+{parsed.secondary.name}")


def check_synthetic_data(ctx: SmokeContext) -> None:
    ctx.samples = generate_synthetic_dataset(
        num_samples=ctx.args.num_synthetic,
        composite_ratio=ctx.args.composite_ratio,
    )
    assert len(ctx.samples) == ctx.args.num_synthetic
    emotion_counts = Counter(sample["emotion_label"] for sample in ctx.samples)
    assert len(emotion_counts) == 17

    save_dataset(ctx.samples, str(ctx.train_path), split_ratio=0.85)
    assert ctx.train_path.exists()
    assert ctx.val_path.exists()
    ctx.metrics["synthetic_emotions"] = dict(emotion_counts)
    print(f"  Synthetic samples: {len(ctx.samples)}")
    print(f"  Unique emotions: {len(emotion_counts)}/17")


def check_oly_logs(ctx: SmokeContext) -> None:
    raw_log = {
        "session_id": "pretrain-smoke",
        "messages": [
            {"role": "user", "content": "Are you actually okay?"},
            {
                "role": "assistant",
                "content": '<|ACT:"emotion":[{"name":"hopeful","intensity":0.8}]|> I think I can be okay.',
                "probe_distribution": {"sad": 0.82, "hopeful": 0.18},
                "hidden_states": [[0.1, 0.2]],
                "raw_logits": [1.0, 2.0],
            },
        ],
    }

    extracted = extract_samples_from_session_log(
        raw_log,
        prefer_probe=True,
        memory=ctx.memory,
    )
    assert len(extracted) == 1
    assert extracted[0]["emotion_label"] == "sad"
    assert extracted[0]["probe_valence"] < 0

    raw_dir = ctx.tmp_dir / "raw_logs"
    clean_dir = ctx.tmp_dir / "clean_logs"
    raw_dir.mkdir()
    (raw_dir / "session.json").write_text(json.dumps(raw_log), encoding="utf-8")

    count, written = compress_log_directory(str(raw_dir), str(clean_dir), memory=ctx.memory)
    assert count == 1
    assert len(written) == 1

    loaded = load_oly_logs(str(clean_dir), prefer_probe=True, memory=ctx.memory)
    assert len(loaded) == 1
    assert loaded[0]["emotion_label"] == "sad"
    print("  Probe-preferred label: hopeful surface -> sad internal")
    print(f"  EMA after log checks: value={ctx.memory.value:.4f}, sessions={ctx.memory.sessions}")


def check_async_probe_and_memory(ctx: SmokeContext) -> None:
    async def sad_probe(_context):
        await asyncio.sleep(0.01)
        return {"emotion_probs": {"sad": 0.75, "hopeful": 0.25}, "intensity": 0.7}

    def curious_probe(_context):
        return {"emotion_probs": {"curious": 1.0}, "intensity": 0.5}

    async def run():
        runner = AsyncProbeRunner(timeout_s=1.0, max_concurrency=2)
        return await runner.run_all(
            {"sad_probe": sad_probe, "curious_probe": curious_probe},
            {"prompt": "synthetic"},
        )

    result = asyncio.run(run())
    assert result["aggregate"]["num_probes"] == 2
    assert -1.0 <= result["aggregate"]["valence"] <= 1.0
    aggregate = result["aggregate"]
    print(
        "  Probe aggregate: "
        f"dominant={aggregate['dominant_emotion']}, "
        f"valence={aggregate['valence']:.4f}, "
        f"intensity={aggregate['intensity']:.4f}"
    )

    previous_sessions = ctx.memory.sessions
    value = ctx.memory.update_from_probe_distribution(
        result["aggregate"]["emotion_probs"],
        session_id="async-smoke",
    )
    assert -1.0 <= value <= 1.0
    assert ctx.memory.sessions == previous_sessions + 1
    print(f"  EMA after async probes: value={value:.4f}, tone={ctx.memory.tone}")


def check_tokenizer(ctx: SmokeContext) -> None:
    texts = []
    for sample in ctx.samples:
        texts.append(sample["input"])
        texts.append(sample["output"])

    tokenizer = OlyTokenizer(vocab_size=ctx.args.vocab_size)
    tokenizer.train_from_texts(texts, min_frequency=1)
    assert tokenizer.get_vocab_size() > len(EMOTION_LABELS)
    assert tokenizer.act_start_token_id >= 0

    encoded = tokenizer.encode_act_token("happy", 0.8)
    assert encoded[0] == tokenizer.act_start_token_id
    assert encoded[-1] == tokenizer.act_end_token_id

    tokenizer_dir = ctx.tmp_dir / "tokenizer"
    tokenizer.save(str(tokenizer_dir))
    ctx.tokenizer = OlyTokenizer.load(str(tokenizer_dir))
    assert ctx.tokenizer.get_vocab_size() == tokenizer.get_vocab_size()
    print(f"  Tokenizer vocab size: {ctx.tokenizer.get_vocab_size()}")
    print(f"  Encoded happy ACT token length: {len(encoded)}")


def check_dataset_and_model(ctx: SmokeContext) -> None:
    dataset = ACTDataset(str(ctx.train_path), ctx.tokenizer, max_length=ctx.args.max_length)
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["input_ids"].shape[0] == ctx.args.max_length
    assert sample["labels"].shape[0] == ctx.args.max_length

    vocab_size = max(ctx.tokenizer.get_vocab_size(), 128)
    ctx.config = OlyConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=ctx.args.max_length,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        act_enabled=True,
        act_num_emotions=17,
        act_hidden_size=32,
        act_max_composite=5,
        gradient_checkpointing=False,
    )
    ctx.model = OlyForCausalLM(ctx.config)
    ctx.model.set_emotional_memory(ctx.memory)

    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ctx.args.batch_size,
        shuffle=False,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ctx.args.batch_size,
        shuffle=True,
    )

    initial_eval = evaluate_tiny_model(ctx.model, eval_loader, ctx.args.eval_batches)
    print_eval_report("Before mini-train", initial_eval)

    batch = next(iter(train_loader))
    outputs = ctx.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        act_labels=batch["act_labels"],
        act_intensities=batch["act_intensities"],
        act_mask=batch["act_mask"],
    )
    assert outputs["loss"] is not None
    assert outputs["loss"].item() > 0
    assert outputs["act_emotion_logits"].shape == (batch["input_ids"].shape[0], 17)

    loss_fn = ACTLoss(ctx.config)
    act_loss, parts = loss_fn(
        outputs["act_emotion_logits"],
        outputs["act_intensity"],
        batch["act_labels"],
        batch["act_intensities"],
        batch["act_mask"],
    )
    assert act_loss.item() > 0
    assert "act_total_loss" in parts

    optimizer = torch.optim.AdamW(ctx.model.parameters(), lr=ctx.args.learning_rate)
    train_losses = []
    train_accuracies = []
    started = time.perf_counter()

    # Reuse the same tiny batch on purpose: this is a pre-train wiring check,
    # so a small overfit loop gives a clearer before/after signal.
    for step in range(ctx.args.train_steps):
        ctx.model.train()
        outputs = ctx.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            act_labels=batch["act_labels"],
            act_intensities=batch["act_intensities"],
            act_mask=batch["act_mask"],
        )
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        pred = outputs["act_emotion_logits"].argmax(dim=-1).detach().cpu()
        target = batch["act_labels"].cpu()
        train_losses.append(loss.item())
        train_accuracies.append((pred == target).float().mean().item())

        if ctx.args.print_every and (step + 1) % ctx.args.print_every == 0:
            print(
                f"    train_step={step + 1:02d} "
                f"loss={loss.item():.4f} "
                f"act_acc={train_accuracies[-1] * 100:.1f}%"
            )

    train_elapsed = time.perf_counter() - started
    final_eval = evaluate_tiny_model(ctx.model, eval_loader, ctx.args.eval_batches)
    print_eval_report("After mini-train", final_eval)

    loss_delta = final_eval["loss"] - initial_eval["loss"]
    acc_delta = final_eval["act_accuracy"] - initial_eval["act_accuracy"]
    fixed_batch_delta = train_losses[-1] - train_losses[0]
    print("\n  Mini-train deltas:")
    print(f"    fixed_batch_loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f} ({fixed_batch_delta:+.4f})")
    print(f"    eval_loss_delta: {loss_delta:+.4f}")
    print(f"    eval_act_accuracy_delta: {acc_delta * 100:+.1f} pp")
    print(f"    train_time: {train_elapsed:.2f}s for {ctx.args.train_steps} steps")
    print("    before_examples:")
    for example in initial_eval["examples"][:ctx.args.show_examples]:
        print(
            f"      target={example['target']:<10s} "
            f"pred={example['predicted']:<10s} "
            f"intensity={example['pred_intensity']:.3f}"
        )
    print("    after_examples:")
    for example in final_eval["examples"][:ctx.args.show_examples]:
        print(
            f"      target={example['target']:<10s} "
            f"pred={example['predicted']:<10s} "
            f"intensity={example['pred_intensity']:.3f}"
        )
    ctx.metrics["mini_train"] = {
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "fixed_batch_loss_delta": fixed_batch_delta,
        "eval_loss_delta": loss_delta,
        "eval_act_accuracy_delta": acc_delta,
    }

    generated = ctx.model.generate(
        batch["input_ids"][:1, :8],
        max_new_tokens=ctx.args.generation_tokens,
        emit_act=True,
        memory_session_id="generate-smoke",
    )
    assert generated["generated_ids"].shape[1] >= 8
    assert generated["act_result"] is not None
    assert generated["emotional_memory"] is not None
    act_pred = generated["act_result"]["emotion_pred"].item()
    act_intensity = generated["act_result"]["intensity"].item()
    decoded_preview = ctx.tokenizer.decode(
        generated["generated_ids"][0].detach().cpu().tolist(),
        skip_special_tokens=False,
    )
    print("\n  Generation preview:")
    print(f"    act_prediction={_label_name(act_pred)} intensity={act_intensity:.3f}")
    print(f"    memory={generated['emotional_memory']}")
    print(f"    decoded={decoded_preview[:180]!r}")

    checkpoint_dir = ctx.tmp_dir / "checkpoints"
    save_checkpoint(
        ctx.model,
        optimizer,
        step=1,
        loss=float(outputs["loss"].item()),
        output_dir=str(checkpoint_dir),
        config=ctx.config.to_dict(),
    )
    model2 = OlyForCausalLM(ctx.config)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    info = load_checkpoint(str(checkpoint_dir / "checkpoint-1"), model2, optimizer2)
    assert info["step"] == 1


def run_all_checks(ctx: SmokeContext) -> None:
    steps = [
        ("project assets", check_project_assets),
        ("ACT token schema", check_act_schema),
        ("synthetic data generation", check_synthetic_data),
        ("Ol-y log compression/extraction", check_oly_logs),
        ("async probes and EMA memory", check_async_probe_and_memory),
        ("tokenizer train/save/load", check_tokenizer),
        ("dataset, model, generation, checkpoint", check_dataset_and_model),
    ]
    for name, fn in steps:
        _run_step(name, fn, ctx)


def print_data_sources() -> None:
    print("\nReal training data sources:")
    for source in REAL_TRAINING_DATA_SOURCES:
        print(f"- {source}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic end-to-end pre-training check for Ol-y ACTT"
    )
    parser.add_argument("--num-synthetic", type=int, default=170,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--composite-ratio", type=float, default=0.3,
                        help="Fraction of synthetic samples with composite emotions")
    parser.add_argument("--vocab-size", type=int, default=512,
                        help="Tokenizer vocab size for the smoke run")
    parser.add_argument("--max-length", type=int, default=64,
                        help="Sequence length for dataset/model checks")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Tiny model batch size for diagnostic batches")
    parser.add_argument("--eval-batches", type=int, default=4,
                        help="Number of batches to evaluate before/after mini-training")
    parser.add_argument("--train-steps", type=int, default=12,
                        help="Number of tiny overfit steps to run before real training")
    parser.add_argument("--learning-rate", type=float, default=3e-3,
                        help="Learning rate for the tiny mini-training diagnostic")
    parser.add_argument("--print-every", type=int, default=3,
                        help="Print mini-training metrics every N steps; 0 disables")
    parser.add_argument("--show-examples", type=int, default=3,
                        help="Number of before/after ACT examples to print")
    parser.add_argument("--generation-tokens", type=int, default=6,
                        help="Number of generated tokens for the preview")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep the temporary smoke-test directory")
    parser.add_argument("--print-data-sources", action="store_true",
                        help="Print real training data sources after checks pass")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_steps < 1:
        raise SystemExit("--train-steps must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _print_header()

    temp_context = tempfile.TemporaryDirectory(prefix="oly-pretrain-smoke-")
    tmp_dir = Path(temp_context.name)

    try:
        ctx = SmokeContext(tmp_dir=tmp_dir, args=args)
        print(f"Temporary workspace: {tmp_dir}")
        run_all_checks(ctx)
        print("\nAll synthetic pre-training checks passed.")
        if args.print_data_sources:
            print_data_sources()
        return 0
    except Exception:
        print("\nPre-training synthetic check failed.")
        traceback.print_exc()
        return 1
    finally:
        if args.keep_temp:
            print(f"\nKeeping temporary workspace: {tmp_dir}")
        else:
            temp_context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
