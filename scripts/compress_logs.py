"""
Compress Ol-y session logs for ACT training.

This removes bulky debug tensors and keeps the conversational text, ACT tokens,
internal probe distributions, per-session probe valence, and optional EMA
emotional memory state.

Usage:
    python scripts/compress_logs.py --input logs/raw --output logs/clean
    python scripts/compress_logs.py --input logs/raw --output logs/clean --memory-state data/emotional_memory.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oly.act.emotional_memory import EmotionalMemoryEMA
from oly.data.oly_logs import compress_log_directory


def main():
    parser = argparse.ArgumentParser(description="Compress Ol-y logs for ACT training")
    parser.add_argument("--input", required=True, help="Directory with raw Ol-y JSON logs")
    parser.add_argument("--output", required=True, help="Directory for compressed JSON logs")
    parser.add_argument("--memory-state", default=None,
                        help="Optional JSON path for persistent EMA emotional memory")
    args = parser.parse_args()

    memory = EmotionalMemoryEMA.load(args.memory_state) if args.memory_state else None
    count, written = compress_log_directory(args.input, args.output, memory=memory)

    if args.memory_state and memory is not None:
        memory.save(args.memory_state)
        print(f"[EMA] Saved emotional memory to {args.memory_state}")
        print(f"[EMA] value={memory.value:.4f} tone={memory.tone} sessions={memory.sessions}")

    print(f"[Logs] Compressed {count} log file(s) into {args.output}")
    for path in written[:10]:
        print(f"  - {path}")
    if len(written) > 10:
        print(f"  ... {len(written) - 10} more")


if __name__ == "__main__":
    main()
