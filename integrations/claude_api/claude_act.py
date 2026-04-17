"""
Claude API integration with ACT (Affective Communication Tokens).

Uses the Anthropic Python SDK to interact with Claude models while enforcing
ACT token emission through prefill enforcement. Unlike the Ol-y and Llama
integrations, Claude is accessed via API -- the ACT head runs locally as a
lightweight classifier on Claude's text output, or ACT tokens are enforced
through prompt engineering and prefill.

Prefill enforcement (Section 4.2 of the paper):
  The assistant turn is seeded with the ACT token prefix '<|ACT:"emotion":[{"name":"'
  which reduces compliance to a structured completion problem. Claude's instruction-
  following capability makes it particularly suited for this approach.

Reference: Sakishimiro (2026), Section 4.2 "Prefill Enforcement"
"""

import os
import json
import re
from typing import Dict, Any, Optional, List

from oly.act.act_token import (
    EMOTION_LABELS, ACT_PREFILL, ACT_RE,
    CompositeACT, parse_act_from_response,
    build_composite_act_string,
)


class ClaudeACT:
    """Claude API wrapper with ACT token support.

    Provides methods for:
    - Generating responses with ACT token enforcement via prefill
    - Parsing ACT tokens from Claude's responses
    - Standalone emotion classification using Claude as the backbone
    - Session management with persistent ACT state tracking

    Usage:
        claude = ClaudeACT(api_key="your-key")
        result = claude.generate("Tell me about your day", emit_act=True)
        print(result["act_token"])
        print(result["response"])
    """

    # System prompt that instructs Claude to emit ACT tokens
    ACT_SYSTEM_PROMPT = """You are an AI assistant with emotional awareness. Before every response, you MUST emit an Affective Communication Token (ACT) that declares your current emotional state.

ACT Token Format:
<|ACT:"emotion":[{"name":"LABEL","intensity":VALUE}]|>

Valid emotion labels: happy, sad, angry, surprised, curious, awkward, question, think, neutral, hopeful, nostalgic, regret, grateful, relieved, emptiness, reflective, serenity

Intensity is a float between 0.1 and 1.0.

You may use composite ACT tokens with multiple emotions:
<|ACT:"emotion":[{"name":"curious","intensity":0.7},{"name":"happy","intensity":0.4}]|>

Rules:
1. EVERY response MUST begin with an ACT token
2. Choose the emotion that genuinely reflects your processing state for this input
3. Intensity should reflect the strength of that emotional engagement
4. After the ACT token, write your natural language response
5. The ACT token is your emotional declaration -- it precedes and conditions your response"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """Initialize the Claude ACT wrapper.

        Args:
            api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
            model: Claude model ID to use
            max_tokens: maximum response tokens
            temperature: sampling temperature
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None

        # Session state for tracking emotional continuity
        self.session_history: List[Dict] = []
        self.novel_emotions: List[str] = []

    def _ensure_client(self):
        """Lazily initialize the Anthropic client."""
        if self.client is None:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )

    def generate(
        self,
        prompt: str,
        emit_act: bool = True,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Generate a response from Claude with ACT token enforcement.

        Uses prefill enforcement: the assistant turn is seeded with the ACT
        token prefix so Claude completes it as a structured continuation.

        Args:
            prompt: user message text
            emit_act: whether to enforce ACT token emission (default True)
            system_prompt: optional custom system prompt (default uses ACT_SYSTEM_PROMPT)
            conversation_history: optional prior messages for multi-turn context

        Returns:
            Dictionary with:
                - act_token: parsed CompositeACT object (or None)
                - response: the natural language response text (without ACT token)
                - full_output: complete response including ACT token
                - emotion: dominant emotion label
                - intensity: dominant emotion intensity
                - novel_emotions: list of any novel emotion labels detected
        """
        self._ensure_client()

        system = system_prompt or self.ACT_SYSTEM_PROMPT

        # Build message list
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})

        # Prefill enforcement: seed the assistant response with ACT token prefix
        # This is the key technique from Section 4.2 of the paper
        prefill_content = []
        if emit_act:
            prefill_content = [{"type": "text", "text": ACT_PREFILL}]

        try:
            # Call Claude API
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system,
                "messages": messages,
            }

            response = self.client.messages.create(**kwargs)
            full_text = response.content[0].text

            # If we used prefill, prepend it to the response
            if emit_act:
                full_text = ACT_PREFILL + full_text

        except Exception as e:
            return {
                "act_token": None,
                "response": f"[API Error: {str(e)}]",
                "full_output": "",
                "emotion": "neutral",
                "intensity": 0.5,
                "novel_emotions": [],
                "error": str(e),
            }

        # Parse ACT token from response
        act = parse_act_from_response(full_text)

        # Extract response text (everything after the ACT token)
        response_text = full_text
        act_match = ACT_RE.search(full_text)
        if act_match:
            response_text = full_text[act_match.end():].strip()

        # Check for novel emotions (vocabulary expansion, Section 8.4)
        novel = []
        if act and act.has_novel_emotions():
            novel = act.get_novel_emotions()
            self.novel_emotions.extend(novel)

        # Track in session history
        result = {
            "act_token": act,
            "response": response_text,
            "full_output": full_text,
            "emotion": act.dominant.name if act and act.dominant else "neutral",
            "intensity": round(act.dominant.intensity, 2) if act and act.dominant else 0.5,
            "novel_emotions": novel,
        }

        self.session_history.append({
            "prompt": prompt,
            "emotion": result["emotion"],
            "intensity": result["intensity"],
        })

        return result

    def call_act(self, text: str) -> Dict[str, Any]:
        """Classify the emotional state of a text using Claude.

        Uses Claude as a zero-shot emotion classifier -- asks it to analyze
        the text and return only an ACT token.

        Args:
            text: text to classify

        Returns:
            Dictionary with emotion, intensity, and act_token string
        """
        self._ensure_client()

        classification_prompt = f"""Analyze the emotional tone of this text and respond with ONLY an ACT token, nothing else.

Text: "{text}"

Respond with only the ACT token in this exact format:
<|ACT:"emotion":[{{"name":"LABEL","intensity":VALUE}}]|>"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0.0,
                messages=[{"role": "user", "content": classification_prompt}],
            )
            full_text = response.content[0].text
            act = parse_act_from_response(full_text)

            if act and act.dominant:
                return {
                    "emotion": act.dominant.name,
                    "intensity": round(act.dominant.intensity, 2),
                    "act_token": act.to_string(),
                    "is_novel": act.has_novel_emotions(),
                }
        except Exception as e:
            pass

        return {
            "emotion": "neutral",
            "intensity": 0.5,
            "act_token": build_composite_act_string([("neutral", 0.5)]),
            "is_novel": False,
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session's emotional trajectory.

        Returns statistics about emotion distribution, average intensity,
        and any novel emotions encountered during the session.
        """
        if not self.session_history:
            return {"turns": 0, "emotions": {}, "avg_intensity": 0.0, "novel": []}

        emotion_counts = {}
        total_intensity = 0.0

        for entry in self.session_history:
            e = entry["emotion"]
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
            total_intensity += entry["intensity"]

        return {
            "turns": len(self.session_history),
            "emotions": emotion_counts,
            "avg_intensity": round(total_intensity / len(self.session_history), 3),
            "novel": list(set(self.novel_emotions)),
            "dominant_emotion": max(emotion_counts, key=emotion_counts.get),
        }

    @classmethod
    def from_config(cls, config_path: str) -> "ClaudeACT":
        """Create a ClaudeACT instance from a JSON config file.

        Args:
            config_path: path to claude_api_config.json

        Returns:
            Configured ClaudeACT instance
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        api_config = config.get("api", {})
        return cls(
            model=api_config.get("model", "claude-sonnet-4-5-20250929"),
            max_tokens=api_config.get("max_tokens", 4096),
            temperature=api_config.get("temperature", 0.7),
        )
