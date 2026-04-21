"""
Ol-y log parsing, compression, and training-sample extraction.

The extraction path can prefer internal probe distributions over the surface ACT
token. That supports Sakishimiro's EMA roadmap: logs can capture what Ol-y felt
internally even when the assistant response expressed a softer emotion.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from oly.act.act_token import (
    ACT_RE,
    EMOTION_TO_ID,
    build_composite_act_string,
    parse_act_from_response,
)
from oly.act.emotional_memory import (
    EmotionalMemoryEMA,
    dominant_emotion,
    dominant_probability,
    normalize_distribution,
    probe_valence,
)


DROP_MESSAGE_KEYS = {
    "hidden_states",
    "raw_logits",
    "logits",
    "attentions",
    "past_key_values",
    "kv_cache",
    "debug",
    "trace",
    "tokens",
    "token_ids",
    "embedding",
    "embeddings",
}

KEEP_MESSAGE_KEYS = {
    "role",
    "content",
    "timestamp",
    "created_at",
    "session_id",
    "turn_id",
    "message_id",
    "act_token",
    "emotion",
    "emotion_label",
    "intensity",
    "probe",
    "probe_distribution",
    "probe_probs",
    "emotion_probs",
    "internal_probe",
    "metadata",
}


def _json_files(log_dir: Union[str, os.PathLike]) -> Iterable[Path]:
    root = Path(log_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() == ".json")


def load_json_log(path: Union[str, os.PathLike]) -> Union[Mapping[str, object], List[object]]:
    """Load a JSON log file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_messages(data: Union[Mapping[str, object], Sequence[object]]) -> List[Mapping[str, object]]:
    """Normalize known Ol-y log shapes into a list of message dictionaries."""
    if isinstance(data, list):
        entries = data
    elif isinstance(data, Mapping) and isinstance(data.get("messages"), list):
        entries = data["messages"]
    elif isinstance(data, Mapping) and isinstance(data.get("turns"), list):
        entries = data["turns"]
    else:
        entries = []
    return [entry for entry in entries if isinstance(entry, Mapping)]


def get_probe_distribution(entry: Mapping[str, object]) -> Optional[Dict[str, float]]:
    """Find and normalize an internal probe distribution from a log entry."""
    direct_keys = (
        "probe_distribution",
        "probe_probs",
        "emotion_probs",
        "internal_probe",
    )
    for key in direct_keys:
        value = entry.get(key)
        if isinstance(value, Mapping):
            if "emotion_probs" in value and isinstance(value["emotion_probs"], Mapping):
                return normalize_distribution(value["emotion_probs"])
            if "distribution" in value and isinstance(value["distribution"], Mapping):
                return normalize_distribution(value["distribution"])
            return normalize_distribution(value)

    probe = entry.get("probe")
    if isinstance(probe, Mapping):
        for key in ("emotion_probs", "distribution", "probe_distribution", "probs"):
            value = probe.get(key)
            if isinstance(value, Mapping):
                return normalize_distribution(value)

    return None


def _probe_intensity(entry: Mapping[str, object], distribution: Mapping[str, float]) -> float:
    for key in ("probe_intensity", "intensity"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return max(0.1, min(1.0, float(value)))

    probe = entry.get("probe")
    if isinstance(probe, Mapping):
        value = probe.get("intensity")
        if isinstance(value, (int, float)):
            return max(0.1, min(1.0, float(value)))

    return max(0.1, min(1.0, dominant_probability(distribution)))


def _replace_or_prepend_act(content: str, emotion: str, intensity: float) -> str:
    act_string = build_composite_act_string([(emotion, intensity)])
    match = ACT_RE.search(content)
    if match:
        body = content[match.end():].lstrip()
        return f"{act_string} {body}" if body else act_string
    return f"{act_string} {content}".strip()


def _sample_from_assistant_turn(
    entries: Sequence[Mapping[str, object]],
    idx: int,
    source: str,
    prefer_probe: bool,
    memory_value: Optional[float],
) -> Optional[Dict[str, object]]:
    entry = entries[idx]
    content = str(entry.get("content", ""))
    distribution = get_probe_distribution(entry)
    act = parse_act_from_response(content)

    if not act and distribution is None:
        return None

    user_msg = ""
    for prev in reversed(entries[:idx]):
        if prev.get("role") == "user":
            user_msg = str(prev.get("content", ""))
            break
    if not user_msg:
        return None

    if prefer_probe and distribution:
        emotion = dominant_emotion(distribution)
        intensity = _probe_intensity(entry, distribution)
        output = _replace_or_prepend_act(content, emotion, intensity)
        valence = probe_valence(distribution)
        secondary_emotion = None
        secondary_intensity = None
        is_composite = False
    elif act and act.dominant:
        emotion = act.dominant.name
        intensity = act.dominant.intensity
        output = content
        valence = None
        memory_value = None
        secondary_emotion = act.secondary.name if act.secondary else None
        secondary_intensity = act.secondary.intensity if act.secondary else None
        is_composite = len(act.emotions) > 1
    else:
        return None

    sample = {
        "input": user_msg,
        "output": output,
        "emotion_label": emotion,
        "emotion_id": EMOTION_TO_ID.get(emotion, EMOTION_TO_ID["neutral"]),
        "intensity": intensity,
        "secondary_emotion": secondary_emotion,
        "secondary_intensity": secondary_intensity,
        "is_composite": is_composite,
        "source": source,
    }
    if distribution:
        sample["probe_distribution"] = distribution
        sample["probe_valence"] = valence if valence is not None else probe_valence(distribution)
        if memory_value is not None:
            sample["emotional_memory"] = memory_value
    return sample


def _average_probe_distributions(
    distributions: Iterable[Mapping[str, float]],
) -> Optional[Dict[str, float]]:
    distributions = [distribution for distribution in distributions if distribution]
    if not distributions:
        return None
    avg = {
        emotion: sum(d.get(emotion, 0.0) for d in distributions) / len(distributions)
        for emotion in EMOTION_TO_ID
    }
    return normalize_distribution(avg)


def extract_samples_from_session_log(
    data: Union[Mapping[str, object], Sequence[object]],
    source: str = "oly",
    prefer_probe: bool = True,
    memory: Optional[EmotionalMemoryEMA] = None,
) -> List[Dict[str, object]]:
    """Extract training samples from one Ol-y session log."""
    entries = get_messages(data)
    session_id = None
    if isinstance(data, Mapping):
        raw_session_id = data.get("session_id") or data.get("id")
        session_id = str(raw_session_id) if raw_session_id is not None else None

    session_distribution = _average_probe_distributions(
        get_probe_distribution(entry)
        for entry in entries
        if entry.get("role") == "assistant"
    )
    memory_value = None
    if memory is not None and session_distribution is not None:
        memory_value = memory.update_from_probe_distribution(
            session_distribution,
            session_id=session_id,
            metadata={"source": source},
        )

    samples = []
    for idx, entry in enumerate(entries):
        if entry.get("role") != "assistant":
            continue
        sample = _sample_from_assistant_turn(
            entries,
            idx,
            source=source,
            prefer_probe=prefer_probe,
            memory_value=memory_value,
        )
        if sample:
            samples.append(sample)
    return samples


def load_oly_logs(
    log_dir: str,
    prefer_probe: bool = True,
    memory: Optional[EmotionalMemoryEMA] = None,
) -> List[Dict[str, object]]:
    """Load Ol-y logs from a directory and convert them to training samples."""
    samples: List[Dict[str, object]] = []
    if not os.path.exists(log_dir):
        print(f"[Data] Ol-y log directory not found: {log_dir}")
        return samples

    for path in _json_files(log_dir):
        try:
            data = load_json_log(path)
            samples.extend(
                extract_samples_from_session_log(
                    data,
                    source="oly",
                    prefer_probe=prefer_probe,
                    memory=memory,
                )
            )
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"[Data] Error reading {path.name}: {exc}")
            continue

    print(f"[Data] Loaded {len(samples)} samples from Ol-y logs")
    return samples


def compress_message(entry: Mapping[str, object]) -> Dict[str, object]:
    """Drop bulky debug tensors while preserving conversational/probe signal."""
    compressed: Dict[str, object] = {}
    for key, value in entry.items():
        if key in DROP_MESSAGE_KEYS:
            continue
        if key in KEEP_MESSAGE_KEYS or not isinstance(value, (list, dict)):
            compressed[key] = value

    distribution = get_probe_distribution(entry)
    if distribution is not None:
        compressed["probe_distribution"] = distribution
        compressed["probe_valence"] = probe_valence(distribution)
    return compressed


def compress_session_log(
    data: Union[Mapping[str, object], Sequence[object]],
    memory: Optional[EmotionalMemoryEMA] = None,
) -> Dict[str, object]:
    """Compress one session log into a clean, training-friendly JSON shape."""
    messages = [compress_message(entry) for entry in get_messages(data)]
    compressed: Dict[str, object] = {"messages": messages}

    if isinstance(data, Mapping):
        for key in ("session_id", "id", "created_at", "timestamp", "metadata"):
            if key in data:
                compressed[key] = data[key]

    normalized = _average_probe_distributions(
        get_probe_distribution(message)
        for message in messages
        if message.get("role") == "assistant"
    )
    if normalized:
        compressed["session_probe_distribution"] = normalized
        compressed["session_probe_valence"] = probe_valence(normalized)
        compressed["session_internal_emotion"] = dominant_emotion(normalized)
        if memory is not None:
            session_id = str(compressed.get("session_id", compressed.get("id", "")) or "")
            compressed["emotional_memory"] = memory.update_from_probe_distribution(
                normalized,
                session_id=session_id,
                metadata={"source": "compressed_log"},
            )

    compressed["compression"] = {
        "format": "oly-clean-log-v1",
        "credit": "Log compression roadmap by Sakishimiro",
    }
    return compressed


def compress_log_directory(
    input_dir: str,
    output_dir: str,
    memory: Optional[EmotionalMemoryEMA] = None,
) -> Tuple[int, List[str]]:
    """Compress every JSON log in a directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    if not os.path.exists(input_dir):
        return 0, written

    for path in _json_files(input_dir):
        try:
            data = load_json_log(path)
            compressed = compress_session_log(data, memory=memory)
            target = Path(output_dir) / path.name
            with target.open("w", encoding="utf-8") as f:
                json.dump(compressed, f, indent=2, ensure_ascii=False)
            written.append(str(target))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[Logs] Skipping {path.name}: {exc}")

    return len(written), written
