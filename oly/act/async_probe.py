"""
Async emotional probe runner.

The runner lets multiple internal probes execute concurrently and aggregates
their emotion distributions into one normalized ACT distribution. Probe callables
can be regular functions or async functions; each receives the same context
object and returns a mapping with an ``emotion_probs``/``distribution`` field.

Credit: async probe roadmap direction by Sakishimiro.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Optional, Union

from oly.act.act_token import EMOTION_LABELS
from oly.act.emotional_memory import normalize_distribution, probe_valence


ProbeCallable = Callable[[Any], Union[Mapping[str, Any], Awaitable[Mapping[str, Any]]]]


@dataclass
class ProbeResult:
    """Normalized output from one internal probe."""

    probe_id: str
    emotion_probs: Dict[str, float]
    intensity: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def valence(self) -> float:
        return probe_valence(self.emotion_probs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "emotion_probs": dict(self.emotion_probs),
            "intensity": self.intensity,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
            "error": self.error,
            "valence": self.valence,
        }


def _extract_distribution(raw: Mapping[str, Any]) -> Dict[str, float]:
    """Accept common probe output shapes and return normalized emotion probs."""
    candidate = (
        raw.get("emotion_probs")
        or raw.get("distribution")
        or raw.get("probe_distribution")
        or raw.get("probs")
    )
    if not isinstance(candidate, Mapping):
        emotion = raw.get("emotion") or raw.get("emotion_label") or raw.get("label")
        if emotion in EMOTION_LABELS:
            candidate = {str(emotion): 1.0}
        else:
            candidate = {"neutral": 1.0}
    return normalize_distribution(candidate)


def _extract_intensity(raw: Mapping[str, Any], distribution: Mapping[str, float]) -> float:
    intensity = raw.get("intensity")
    if intensity is None:
        intensity = raw.get("probe_intensity")
    if intensity is None:
        intensity = max(distribution.values()) if distribution else 0.0
    return max(0.0, min(1.0, float(intensity)))


def _coerce_result(probe_id: str, raw: Mapping[str, Any], latency_ms: float) -> ProbeResult:
    distribution = _extract_distribution(raw)
    reserved = {
        "emotion_probs", "distribution", "probe_distribution", "probs",
        "emotion", "emotion_label", "label", "intensity", "probe_intensity",
    }
    metadata = {key: value for key, value in raw.items() if key not in reserved}
    return ProbeResult(
        probe_id=probe_id,
        emotion_probs=distribution,
        intensity=_extract_intensity(raw, distribution),
        latency_ms=latency_ms,
        metadata=metadata,
    )


class AsyncProbeRunner:
    """Run internal emotional probes concurrently with timeouts."""

    def __init__(self, timeout_s: float = 10.0, max_concurrency: int = 4):
        self.timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run_probe(
        self,
        probe_id: str,
        probe: ProbeCallable,
        context: Any,
    ) -> ProbeResult:
        """Run one probe and normalize its output."""
        start = time.perf_counter()
        try:
            async with self._semaphore:
                if inspect.iscoroutinefunction(probe):
                    raw = await asyncio.wait_for(probe(context), timeout=self.timeout_s)
                else:
                    raw = await asyncio.wait_for(
                        asyncio.to_thread(probe, context),
                        timeout=self.timeout_s,
                    )
                    if inspect.isawaitable(raw):
                        raw = await asyncio.wait_for(raw, timeout=self.timeout_s)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if not isinstance(raw, Mapping):
                raise TypeError("probe must return a mapping")
            return _coerce_result(probe_id, raw, latency_ms)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return ProbeResult(
                probe_id=probe_id,
                emotion_probs={"neutral": 1.0},
                intensity=0.0,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def run_all(
        self,
        probes: Mapping[str, ProbeCallable],
        context: Any,
    ) -> Dict[str, Any]:
        """Run all probes and return raw plus aggregate results."""
        results = await asyncio.gather(
            *[
                self.run_probe(probe_id, probe, context)
                for probe_id, probe in probes.items()
            ]
        )
        aggregate = aggregate_probe_results(results)
        return {
            "results": [result.to_dict() for result in results],
            "aggregate": aggregate,
        }


def aggregate_probe_results(results: Iterable[ProbeResult]) -> Dict[str, Any]:
    """Average successful probe distributions into one internal signal."""
    ok_results = [result for result in results if result.ok]
    if not ok_results:
        distribution = {"neutral": 1.0}
        return {
            "emotion_probs": distribution,
            "dominant_emotion": "neutral",
            "intensity": 0.0,
            "valence": 0.0,
            "num_probes": 0,
        }

    totals = {emotion: 0.0 for emotion in EMOTION_LABELS}
    intensity = 0.0
    for result in ok_results:
        for emotion, probability in result.emotion_probs.items():
            totals[emotion] = totals.get(emotion, 0.0) + probability
        intensity += result.intensity

    distribution = normalize_distribution(totals)
    dominant = max(distribution.items(), key=lambda item: item[1])[0]
    return {
        "emotion_probs": distribution,
        "dominant_emotion": dominant,
        "intensity": intensity / len(ok_results),
        "valence": probe_valence(distribution),
        "num_probes": len(ok_results),
    }
