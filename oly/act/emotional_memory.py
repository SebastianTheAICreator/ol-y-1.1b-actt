"""
Persistent emotional memory based on probe distributions.

The memory tracks Ol-y's internal emotional trajectory across sessions using
an exponential moving average (EMA):

    M(s) = alpha * V_probe(s) + (1 - alpha) * M(s - 1)

The probe distribution is treated as the internal signal. This keeps the memory
anchored to what the model felt internally, even when the expressed ACT token is
more hopeful or socially softened.

Credit: EMA emotional-memory idea and ACT framing by Sakishimiro.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Union

from oly.act.act_token import EMOTION_LABELS


DEFAULT_EMOTION_VALENCE: Dict[str, float] = {
    "happy": 0.9,
    "sad": -0.8,
    "angry": -0.7,
    "surprised": 0.2,
    "curious": 0.35,
    "awkward": -0.35,
    "question": 0.0,
    "think": 0.05,
    "neutral": 0.0,
    "hopeful": 0.75,
    "nostalgic": 0.15,
    "regret": -0.7,
    "grateful": 0.85,
    "relieved": 0.55,
    "emptiness": -0.9,
    "reflective": 0.05,
    "serenity": 0.8,
}


def clamp_valence(value: float) -> float:
    """Clamp a valence value to the canonical [-1, 1] range."""
    return max(-1.0, min(1.0, float(value)))


def normalize_distribution(distribution: Mapping[str, float]) -> Dict[str, float]:
    """Return a probability-like distribution over known ACT emotions."""
    cleaned = {
        emotion: max(0.0, float(distribution.get(emotion, 0.0)))
        for emotion in EMOTION_LABELS
    }
    total = sum(cleaned.values())
    if total <= 0:
        return {"neutral": 1.0}
    return {emotion: value / total for emotion, value in cleaned.items() if value > 0}


def probe_valence(
    distribution: Mapping[str, float],
    valence_map: Optional[Mapping[str, float]] = None,
) -> float:
    """Compute a scalar valence score from an internal probe distribution."""
    normalized = normalize_distribution(distribution)
    valences = valence_map or DEFAULT_EMOTION_VALENCE
    score = sum(
        probability * float(valences.get(emotion, 0.0))
        for emotion, probability in normalized.items()
    )
    return clamp_valence(score)


def dominant_emotion(distribution: Mapping[str, float]) -> str:
    """Return the highest-probability known ACT emotion."""
    normalized = normalize_distribution(distribution)
    return max(normalized.items(), key=lambda item: item[1])[0]


def dominant_probability(distribution: Mapping[str, float]) -> float:
    """Return the probability of the dominant ACT emotion."""
    normalized = normalize_distribution(distribution)
    return max(normalized.values()) if normalized else 0.0


@dataclass
class EmotionalMemoryEMA:
    """Persistent EMA state for internal emotional valence."""

    alpha: float = 0.3
    value: float = 0.0
    sessions: int = 0
    history: list = field(default_factory=list)
    max_history: int = 100

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1]")
        self.value = clamp_valence(self.value)

    def update_from_valence(
        self,
        valence: float,
        session_id: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> float:
        """Update memory with an already-computed probe valence."""
        valence = clamp_valence(valence)
        if self.sessions == 0 and self.value == 0.0:
            new_value = valence
        else:
            new_value = self.alpha * valence + (1.0 - self.alpha) * self.value

        self.value = clamp_valence(new_value)
        self.sessions += 1

        entry = {
            "session_id": session_id,
            "probe_valence": valence,
            "memory": self.value,
        }
        if metadata:
            entry["metadata"] = dict(metadata)
        self.history.append(entry)
        if self.max_history > 0 and len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return self.value

    def update_from_probe_distribution(
        self,
        distribution: Mapping[str, float],
        session_id: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> float:
        """Compute probe valence from a distribution and update EMA memory."""
        valence = probe_valence(distribution)
        meta = dict(metadata or {})
        meta.setdefault("dominant_emotion", dominant_emotion(distribution))
        meta.setdefault("dominant_probability", dominant_probability(distribution))
        return self.update_from_valence(valence, session_id=session_id, metadata=meta)

    @property
    def tone(self) -> str:
        """Coarse readable state for logs and summaries."""
        if self.value <= -0.35:
            return "negative"
        if self.value >= 0.35:
            return "positive"
        return "mixed"

    def to_dict(self) -> Dict[str, object]:
        """Serialize memory state to a JSON-compatible dictionary."""
        return {
            "alpha": self.alpha,
            "value": self.value,
            "sessions": self.sessions,
            "tone": self.tone,
            "history": list(self.history),
            "credit": "EMA emotional-memory idea by Sakishimiro",
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EmotionalMemoryEMA":
        """Create memory state from serialized data."""
        return cls(
            alpha=float(data.get("alpha", 0.3)),
            value=float(data.get("value", 0.0)),
            sessions=int(data.get("sessions", 0)),
            history=list(data.get("history", [])),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EmotionalMemoryEMA":
        """Load memory state from disk, or return a fresh memory if missing."""
        state_path = Path(path)
        if not state_path.exists():
            return cls()
        with state_path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: Union[str, Path]) -> None:
        """Persist memory state to disk."""
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def update_memory_from_sessions(
    distributions: Iterable[Mapping[str, float]],
    alpha: float = 0.3,
) -> EmotionalMemoryEMA:
    """Convenience helper for replaying a sequence of probe distributions."""
    memory = EmotionalMemoryEMA(alpha=alpha)
    for distribution in distributions:
        memory.update_from_probe_distribution(distribution)
    return memory
