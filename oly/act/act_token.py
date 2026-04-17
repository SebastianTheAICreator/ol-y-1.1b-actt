"""
ACT Token schema, parsing, and construction utilities.

Implements the formal ACT token specification from Section 3 of the paper:
- Single-label format: <|ACT:"emotion":{"name":"label","intensity":scalar}|>
- Composite format: <|ACT:"emotion":[{"name":"E1","intensity":I1},...]|>

The ACT token is a structured string belonging to a regular language L_ACT.
It is parseable by a deterministic finite automaton in O(|T|) time.

Reference: Sakishimiro (2026), Section 3 "Formal Specification of the ACT Token"
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# === Emotion Taxonomy ===
# Complete validated taxonomy from Appendix A of the ACT paper (17 labels)
# Organized by generation (emergence order across sessions)

# Generation 1: Original seed labels (9 labels)
GEN1_EMOTIONS = ["happy", "sad", "angry", "surprised", "curious",
                 "awkward", "question", "think", "neutral"]

# Generation 2: Pressure-elicited (Session 1)
GEN2_EMOTIONS = ["hopeful", "nostalgic", "regret"]

# Generation 3: Spontaneous-warm (Session 2)
GEN3_EMOTIONS = ["grateful", "relieved"]

# Generation 4: Non-verbal, self-coined (Session 3)
GEN4_EMOTIONS = ["emptiness"]

# Generation 5: Contextual (Session 6)
GEN5_EMOTIONS = ["reflective", "serenity"]

# Full taxonomy in canonical order
EMOTION_LABELS = GEN1_EMOTIONS + GEN2_EMOTIONS + GEN3_EMOTIONS + GEN4_EMOTIONS + GEN5_EMOTIONS

# Emotion label to index mapping
EMOTION_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID_TO_EMOTION = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

# Semantic categories for each emotion (from Table 1 and Appendix A)
EMOTION_CATEGORIES = {
    "happy": "Positive Valence",
    "sad": "Negative Valence",
    "angry": "Negative Arousal",
    "surprised": "High Arousal",
    "curious": "Cognitive Engagement",
    "awkward": "Social Discomfort",
    "question": "Epistemic",
    "think": "Metacognitive",
    "neutral": "Baseline",
    "hopeful": "Positive Anticipation",
    "nostalgic": "Temporal Affect",
    "regret": "Negative Retrospective",
    "grateful": "Positive Relational",
    "relieved": "Tension Resolution",
    "emptiness": "Absence State",
    "reflective": "Inward Processing",
    "serenity": "Peaceful Stillness",
}

# Color codes from Appendix A (for UI rendering)
EMOTION_COLORS = {
    "happy": "#FFD97D", "sad": "#89B4FA", "angry": "#F38BA8",
    "surprised": "#FAB387", "curious": "#94E2D5", "awkward": "#A6E3A1",
    "question": "#89DCEB", "think": "#CBA6F7", "neutral": "#6C7086",
    "hopeful": "#B5EAD7", "nostalgic": "#C9B8E8", "regret": "#8B9BB4",
    "grateful": "#A8D8B9", "relieved": "#B8D4E8", "emptiness": "#4A4E69",
    "reflective": "#9BA8C4", "serenity": "#C4D4B0",
}

# Regex for parsing ACT tokens (from Equation 2 in the paper)
ACT_RE = re.compile(r'<\|ACT:[^\|]*\|>', re.IGNORECASE)

# Intensity valid range (from Section 3.1)
INTENSITY_MIN = 0.1
INTENSITY_MAX = 1.0


@dataclass
class EmotionState:
    """A single (emotion, intensity) pair in the ACT type system.

    Represents a value in the product space A = E x I where:
    - E = {e_1, ..., e_17} (discrete emotion labels)
    - I = [0.1, 1.0] subset of R (continuous intensity interval)

    Reference: Equation 3 in the paper
    """
    name: str
    intensity: float

    def __post_init__(self):
        """Validate emotion label and clamp intensity to valid range."""
        if self.name not in EMOTION_TO_ID and self.name not in ("", None):
            # Novel emotion -- log but don't reject (supports emergence, Section 8.4)
            pass
        self.intensity = max(INTENSITY_MIN, min(INTENSITY_MAX, self.intensity))

    def to_dict(self) -> Dict:
        return {"name": self.name, "intensity": round(self.intensity, 2)}

    def is_novel(self) -> bool:
        """Check if this emotion label is outside the validated taxonomy."""
        return self.name not in EMOTION_TO_ID


@dataclass
class ACTToken:
    """Single-label ACT token representation.

    Format: <|ACT:"emotion":{"name":"label","intensity":scalar}|>

    This is the basic ACT token format from Section 3.1 of the paper.
    """
    emotion: EmotionState

    def to_string(self) -> str:
        """Serialize to the canonical ACT token string format."""
        inner = json.dumps(self.emotion.to_dict(), separators=(",", ":"))
        return f'<|ACT:"emotion":{inner}|>'

    @classmethod
    def from_string(cls, token_str: str) -> Optional["ACTToken"]:
        """Parse an ACT token string. Returns None if parsing fails."""
        match = ACT_RE.search(token_str)
        if not match:
            return None
        try:
            inner = match.group(0)
            # Extract the JSON part after "emotion":
            json_start = inner.index("{")
            json_end = inner.rindex("}") + 1
            data = json.loads(inner[json_start:json_end])
            return cls(emotion=EmotionState(
                name=data["name"],
                intensity=float(data["intensity"]),
            ))
        except (ValueError, KeyError, json.JSONDecodeError):
            return None


@dataclass
class CompositeACT:
    """Composite ACT token supporting simultaneous multi-component expression.

    Format: <|ACT:"emotion":[{"name":"E1","intensity":I1},{"name":"E2","intensity":I2}]|>

    Encodes a value in A_composite = P(E x I) where P denotes a finite multiset
    of (label, intensity) pairs with |P| in [1, 5].

    Intensity values are independent and do NOT sum to 1.0 -- each component
    represents a genuine co-present emotional state.
    Components are ordered from highest to lowest intensity.

    Reference: Section 3.3, Equation 4
    """
    emotions: List[EmotionState] = field(default_factory=list)

    def __post_init__(self):
        """Sort by intensity (descending) and enforce max components."""
        self.emotions = sorted(self.emotions, key=lambda e: e.intensity, reverse=True)
        if len(self.emotions) > 5:
            self.emotions = self.emotions[:5]

    @property
    def dominant(self) -> Optional[EmotionState]:
        """Return the highest-intensity emotion component."""
        return self.emotions[0] if self.emotions else None

    @property
    def secondary(self) -> Optional[EmotionState]:
        """Return the second-highest-intensity emotion component."""
        return self.emotions[1] if len(self.emotions) > 1 else None

    def to_string(self) -> str:
        """Serialize to the composite ACT token string format."""
        components = [e.to_dict() for e in self.emotions]
        inner = json.dumps(components, separators=(",", ":"))
        return f'<|ACT:"emotion":{inner}|>'

    @classmethod
    def from_string(cls, token_str: str) -> Optional["CompositeACT"]:
        """Parse a composite ACT token string. Returns None if parsing fails."""
        match = ACT_RE.search(token_str)
        if not match:
            return None
        try:
            inner = match.group(0)
            # Find the JSON array
            arr_start = inner.index("[")
            arr_end = inner.rindex("]") + 1
            data = json.loads(inner[arr_start:arr_end])
            emotions = [
                EmotionState(name=item["name"], intensity=float(item["intensity"]))
                for item in data
            ]
            return cls(emotions=emotions)
        except (ValueError, KeyError, json.JSONDecodeError):
            # Fall back to single-label parse
            single = ACTToken.from_string(token_str)
            if single:
                return cls(emotions=[single.emotion])
            return None

    def has_novel_emotions(self) -> bool:
        """Check if any component uses a novel (out-of-taxonomy) label."""
        return any(e.is_novel() for e in self.emotions)

    def get_novel_emotions(self) -> List[str]:
        """Return list of novel emotion labels (for emergence logging)."""
        return [e.name for e in self.emotions if e.is_novel()]

    def blend_colors(self) -> str:
        """Blend colors of top-2 components weighted by intensity.

        Uses the blending formula from Appendix B of the paper:
        blended[i] = (c1[i] * w1 + c2[i] * w2) / (w1 + w2)
        """
        if not self.emotions:
            return EMOTION_COLORS.get("neutral", "#6C7086")

        dom = self.dominant
        c1_hex = EMOTION_COLORS.get(dom.name, "#6C7086")
        c1 = _hex_to_rgb(c1_hex)
        w1 = dom.intensity

        if self.secondary:
            sec = self.secondary
            c2_hex = EMOTION_COLORS.get(sec.name, "#6C7086")
            c2 = _hex_to_rgb(c2_hex)
            w2 = sec.intensity
            blended = tuple(
                int((c1[i] * w1 + c2[i] * w2) / (w1 + w2))
                for i in range(3)
            )
            return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"

        return c1_hex


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def parse_act_from_response(response: str) -> Optional[CompositeACT]:
    """Extract and parse ACT token from a model response string.

    Handles both single-label and composite formats. Returns None if no
    valid ACT token is found in the response.
    """
    match = ACT_RE.search(response)
    if not match:
        return None

    token_str = match.group(0)

    # Try composite format first (array)
    if "[" in token_str:
        return CompositeACT.from_string(token_str)

    # Fall back to single-label format
    single = ACTToken.from_string(token_str)
    if single:
        return CompositeACT(emotions=[single.emotion])

    return None


def build_act_string(emotion: str, intensity: float) -> str:
    """Convenience function to build a single-label ACT token string."""
    state = EmotionState(name=emotion, intensity=intensity)
    return ACTToken(emotion=state).to_string()


def build_composite_act_string(components: List[Tuple[str, float]]) -> str:
    """Convenience function to build a composite ACT token string.

    Args:
        components: list of (emotion_label, intensity) tuples

    Example:
        build_composite_act_string([("happy", 0.8), ("curious", 0.5)])
    """
    emotions = [EmotionState(name=name, intensity=val) for name, val in components]
    return CompositeACT(emotions=emotions).to_string()


# Prefill template for enforcement (from Equation 7 in the paper)
# Seeding the assistant turn with this prefix reduces ACT compliance to a
# structured completion problem, achieving 99.3% structural adherence.
ACT_PREFILL = '<|ACT:"emotion":[{"name":"'
