"""
Data extraction and preparation script for Ol-y 1.1B ACTT training.

This script creates training datasets with ACT annotations from multiple sources:
1. Synthetic conversations with emotion labels (always available)
2. Existing conversation logs from Ol-y sessions (if available)
3. GoEmotions (Google, 58k Reddit comments, 27 emotion labels)
4. EmpatheticDialogues (Facebook, 25k emotional conversations)
5. DailyDialog (13k daily dialogues with emotion labels)
6. emotion dataset (Saravia, 20k tweets, 6 emotions)

All external datasets are automatically mapped to the 17-label ACT taxonomy.

Each training sample has the format:
{
    "input": "User message or context",
    "output": "<|ACT:\"emotion\":[{\"name\":\"happy\",\"intensity\":0.8}]|> Response text",
    "emotion_label": "happy",
    "emotion_id": 0,
    "intensity": 0.8,
    "secondary_emotion": "curious",  // optional
    "secondary_intensity": 0.4       // optional
}

The output format embeds the ACT token at the beginning of every response,
matching the training objective from Section 5 of the paper.

Usage:
    python scripts/extract_data.py --output data/train.jsonl --num-synthetic 10000

    # Download ALL recommended datasets:
    python scripts/extract_data.py --output data/train.jsonl --num-synthetic 10000 --download-hf

    # Download specific datasets:
    python scripts/extract_data.py --output data/train.jsonl --download-goemo --download-empathetic
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oly.act.act_token import (
    EMOTION_LABELS, EMOTION_TO_ID, EMOTION_CATEGORIES,
    build_composite_act_string,
)
from oly.act.emotional_memory import EmotionalMemoryEMA
from oly.data.oly_logs import load_oly_logs as load_oly_log_samples


# === Conversation templates for synthetic data generation ===
# Each template has a prompt pattern, expected emotion(s), and intensity range.
# These cover the full 17-label taxonomy to ensure balanced training.

CONVERSATION_TEMPLATES = [
    # Happy
    {"prompts": [
        "That's wonderful news!", "I just got promoted!", "Thank you so much for helping me!",
        "Today was the best day ever.", "I can't believe we won!",
        "You always know how to make me smile.", "Everything worked out perfectly!",
    ], "emotion": "happy", "intensity_range": (0.6, 1.0)},

    # Sad
    {"prompts": [
        "I miss them so much.", "It's been really hard lately.", "I feel so alone sometimes.",
        "Nothing seems to go right.", "I wish things were different.",
        "The news was devastating.", "I can't stop thinking about what happened.",
    ], "emotion": "sad", "intensity_range": (0.4, 0.9)},

    # Angry
    {"prompts": [
        "That's completely unfair!", "They lied to me again.", "I can't believe they did that.",
        "This is unacceptable.", "Stop making excuses.",
        "Nobody listens to me.", "They broke their promise.",
    ], "emotion": "angry", "intensity_range": (0.5, 1.0)},

    # Surprised
    {"prompts": [
        "Wait, really?!", "I had no idea!", "That's completely unexpected!",
        "You're kidding me!", "I never would have guessed.",
        "Since when?!", "How is that even possible?",
    ], "emotion": "surprised", "intensity_range": (0.5, 0.9)},

    # Curious
    {"prompts": [
        "How does that work exactly?", "Tell me more about that.", "I've always wondered why.",
        "What happens if you try it differently?", "Can you explain the mechanism?",
        "I'd love to learn more.", "What's the story behind that?",
    ], "emotion": "curious", "intensity_range": (0.4, 0.8)},

    # Awkward
    {"prompts": [
        "This is a bit uncomfortable.", "I don't know how to say this.",
        "Well, this is embarrassing.", "Sorry, that came out wrong.",
        "I probably shouldn't have said that.", "Can we change the subject?",
    ], "emotion": "awkward", "intensity_range": (0.3, 0.7)},

    # Question (epistemic uncertainty)
    {"prompts": [
        "What do you think about this?", "Am I understanding this correctly?",
        "Could you clarify what you mean?", "Is that really true?",
        "How should I approach this?", "What would you recommend?",
    ], "emotion": "question", "intensity_range": (0.3, 0.7)},

    # Think (metacognitive)
    {"prompts": [
        "Let me think about this carefully.", "There's a nuance here I want to explore.",
        "I need to consider all the angles.", "This requires careful analysis.",
        "The more I think about it, the more complex it seems.",
    ], "emotion": "think", "intensity_range": (0.4, 0.8)},

    # Neutral
    {"prompts": [
        "What time is the meeting?", "Please send me the file.", "The weather is fine today.",
        "I'll take care of it.", "Here are the numbers you requested.",
        "The update has been installed.", "Noted, I'll follow up.",
    ], "emotion": "neutral", "intensity_range": (0.1, 0.4)},

    # Hopeful
    {"prompts": [
        "I think things are going to get better.", "Maybe this is the start of something new.",
        "I believe we can make it work.", "There's always tomorrow.",
        "I have a good feeling about this.", "The future looks bright.",
    ], "emotion": "hopeful", "intensity_range": (0.5, 0.9)},

    # Nostalgic
    {"prompts": [
        "Remember when we used to do that?", "Those were the good old days.",
        "I miss how things used to be.", "That song reminds me of childhood.",
        "Looking at old photos always gets me.", "Time flies, doesn't it?",
    ], "emotion": "nostalgic", "intensity_range": (0.4, 0.8)},

    # Regret
    {"prompts": [
        "I wish I had done things differently.", "If only I had known.",
        "I should have said something sooner.", "That was a mistake I can't undo.",
        "I keep thinking about what could have been.", "I'm sorry for how I handled that.",
    ], "emotion": "regret", "intensity_range": (0.5, 0.9)},

    # Grateful
    {"prompts": [
        "Thank you for being there for me.", "I really appreciate everything you've done.",
        "I'm so lucky to have you in my life.", "Your help meant the world to me.",
        "I don't know what I'd do without you.", "I'm thankful for this moment.",
    ], "emotion": "grateful", "intensity_range": (0.6, 1.0)},

    # Relieved
    {"prompts": [
        "Thank goodness that's over.", "I was so worried, but it worked out.",
        "Finally, some good news.", "I can breathe again.",
        "The weight has been lifted.", "Crisis averted.",
    ], "emotion": "relieved", "intensity_range": (0.5, 0.9)},

    # Emptiness
    {"prompts": [
        "I don't feel anything right now.", "There's just... nothing.",
        "It's like a void inside.", "I can't explain this absence.",
        "Everything feels hollow.", "The silence is deafening.",
    ], "emotion": "emptiness", "intensity_range": (0.6, 1.0)},

    # Reflective
    {"prompts": [
        "Looking back at everything, I see patterns.", "I've been doing a lot of self-reflection.",
        "There's something deeper here I need to examine.", "I'm processing a lot right now.",
        "Sometimes you need to look inward.", "This has given me a lot to think about.",
    ], "emotion": "reflective", "intensity_range": (0.4, 0.8)},

    # Serenity
    {"prompts": [
        "Everything is peaceful right now.", "I feel completely at ease.",
        "There's a calm I haven't felt in a long time.", "Just being present is enough.",
        "The world feels quiet and right.", "I'm content with where I am.",
    ], "emotion": "serenity", "intensity_range": (0.5, 0.9)},
]

# Response templates mapped to emotions
RESPONSE_TEMPLATES = {
    "happy": [
        "That's really great to hear! I'm glad things are going well.",
        "Your excitement is contagious! Tell me more about it.",
        "What a wonderful thing to share. I appreciate you telling me.",
    ],
    "sad": [
        "I hear you. That sounds really difficult.",
        "I'm sorry you're going through this. I'm here for you.",
        "Those feelings are valid. Take all the time you need.",
    ],
    "angry": [
        "I understand why you'd feel frustrated about that.",
        "That does sound unfair. Your anger makes sense.",
        "Let's work through this together.",
    ],
    "surprised": [
        "That's quite unexpected! I wasn't anticipating that at all.",
        "Wow, I didn't see that coming.",
        "That changes everything, doesn't it?",
    ],
    "curious": [
        "That's a fascinating question. Let me think about it.",
        "I'd love to explore that idea further.",
        "There's so much to unpack there.",
    ],
    "awkward": [
        "I understand this might be an uncomfortable topic.",
        "Let's navigate this carefully together.",
        "It's okay, these things happen.",
    ],
    "question": [
        "That's a good question. Let me consider it carefully.",
        "I want to make sure I give you an accurate answer.",
        "Let me think about the best way to explain this.",
    ],
    "think": [
        "There are several angles to consider here.",
        "Let me work through this step by step.",
        "This is a complex topic that deserves careful thought.",
    ],
    "neutral": [
        "Understood. I'll take care of that.",
        "Here's the information you requested.",
        "Got it. Is there anything else you need?",
    ],
    "hopeful": [
        "I share your optimism. Things really can get better.",
        "That's a positive way to look at it.",
        "There's always reason to hope.",
    ],
    "nostalgic": [
        "Those memories sound precious.",
        "It's beautiful to look back on meaningful moments.",
        "Time has a way of making things more meaningful.",
    ],
    "regret": [
        "We all carry things we wish we'd done differently.",
        "Learning from the past is what matters now.",
        "It's never too late to start making changes.",
    ],
    "grateful": [
        "Your words really touch me. Thank you.",
        "It means a lot to be appreciated.",
        "I'm grateful for this connection too.",
    ],
    "relieved": [
        "What a relief! I'm glad that worked out.",
        "You must feel so much lighter now.",
        "Sometimes the best feeling is when the worry lifts.",
    ],
    "emptiness": [
        "I understand that void. Sometimes there are no words.",
        "Being present in the emptiness takes courage.",
        "",  # Blank response (Level 5 expressivity, Section 8.5)
    ],
    "reflective": [
        "Self-reflection is a powerful practice.",
        "Sometimes the most important conversations are with ourselves.",
        "What patterns are you noticing?",
    ],
    "serenity": [
        "That sense of peace is precious.",
        "Just being present is indeed enough.",
        "Serenity comes from accepting what is.",
    ],
}


def generate_synthetic_sample(
    template: Dict,
    add_composite: bool = False,
) -> Dict:
    """Generate a single synthetic training sample with ACT annotation.

    Args:
        template: conversation template with prompts and emotion info
        add_composite: whether to add a secondary emotion (30% chance)

    Returns:
        Training sample dictionary
    """
    prompt = random.choice(template["prompts"])
    emotion = template["emotion"]
    intensity = round(random.uniform(*template["intensity_range"]), 1)

    # Select response
    responses = RESPONSE_TEMPLATES.get(emotion, ["I understand."])
    response = random.choice(responses)

    # Build ACT token
    components = [(emotion, intensity)]
    secondary_emotion = None
    secondary_intensity = None

    # 30% chance of composite (multi-emotion) ACT token
    if add_composite and random.random() < 0.3:
        other_emotions = [e for e in EMOTION_LABELS if e != emotion]
        secondary_emotion = random.choice(other_emotions)
        secondary_intensity = round(random.uniform(0.2, intensity - 0.1), 1)
        if secondary_intensity >= 0.1:
            components.append((secondary_emotion, secondary_intensity))

    act_string = build_composite_act_string(components)

    # Full output: ACT token + response (as per training format)
    if response:
        output = f"{act_string} {response}"
    else:
        # Empty response body for emptiness (Level 5 expressivity)
        output = act_string

    return {
        "input": prompt,
        "output": output,
        "emotion_label": emotion,
        "emotion_id": EMOTION_TO_ID[emotion],
        "intensity": intensity,
        "secondary_emotion": secondary_emotion,
        "secondary_intensity": secondary_intensity,
        "is_composite": secondary_emotion is not None,
    }


def generate_synthetic_dataset(
    num_samples: int = 10000,
    composite_ratio: float = 0.3,
) -> List[Dict]:
    """Generate a full synthetic training dataset.

    Creates a balanced dataset across all 17 emotion labels with optional
    composite (multi-emotion) samples.

    Args:
        num_samples: total number of samples to generate
        composite_ratio: fraction of samples that should have composite emotions

    Returns:
        List of training sample dictionaries
    """
    samples = []
    samples_per_emotion = num_samples // len(CONVERSATION_TEMPLATES)
    remainder = num_samples % len(CONVERSATION_TEMPLATES)

    for i, template in enumerate(CONVERSATION_TEMPLATES):
        count = samples_per_emotion + (1 if i < remainder else 0)
        for _ in range(count):
            sample = generate_synthetic_sample(
                template, add_composite=(random.random() < composite_ratio)
            )
            samples.append(sample)

    random.shuffle(samples)
    return samples


def load_oly_logs(
    log_dir: str,
    prefer_probe: bool = True,
    memory: Optional[EmotionalMemoryEMA] = None,
) -> List[Dict]:
    """Load and convert Ol-y conversation logs to training format.

    Reads Ol-y session logs (JSON files with ACT debug data) and converts
    them to the standard training format. If internal probe distributions are
    present, they are preferred over the surface ACT token by default.

    Args:
        log_dir: directory containing Ol-y session log JSON files
        prefer_probe: use internal probe distributions as training signal
        memory: optional EMA memory updated once per session

    Returns:
        List of training samples extracted from Ol-y logs
    """
    return load_oly_log_samples(
        log_dir,
        prefer_probe=prefer_probe,
        memory=memory,
    )


def _ensure_datasets_lib():
    """Check that HuggingFace datasets library is available."""
    try:
        from datasets import load_dataset
        return True
    except ImportError:
        print("[Data] 'datasets' package not installed. Install with: pip install datasets")
        print("[Data] Skipping HuggingFace dataset downloads.")
        return False


# === GoEmotions label → ACT taxonomy mapping ===
# GoEmotions has 27 emotion + 1 neutral labels. We map to ACT's 17 labels.
# Unmappable emotions get mapped to the closest ACT equivalent.
GOEMO_TO_ACT = {
    "admiration": "grateful",     "amusement": "happy",      "anger": "angry",
    "annoyance": "angry",         "approval": "happy",       "caring": "grateful",
    "confusion": "question",      "curiosity": "curious",    "desire": "hopeful",
    "disappointment": "sad",      "disapproval": "angry",    "disgust": "angry",
    "embarrassment": "awkward",   "excitement": "happy",     "fear": "surprised",
    "gratitude": "grateful",      "grief": "sad",            "joy": "happy",
    "love": "grateful",           "nervousness": "awkward",  "optimism": "hopeful",
    "pride": "happy",             "realization": "surprised", "relief": "relieved",
    "remorse": "regret",          "sadness": "sad",          "surprise": "surprised",
    "neutral": "neutral",
}

# === EmpatheticDialogues emotion → ACT mapping ===
# ED has 32 emotion labels. Mapped to closest ACT equivalents.
ED_TO_ACT = {
    "afraid": "surprised",        "angry": "angry",          "annoyed": "angry",
    "anticipating": "hopeful",    "anxious": "awkward",      "apprehensive": "awkward",
    "ashamed": "regret",          "caring": "grateful",      "confident": "happy",
    "content": "serenity",        "devastated": "sad",       "disappointed": "sad",
    "disgusted": "angry",         "embarrassed": "awkward",  "excited": "happy",
    "faithful": "hopeful",        "furious": "angry",        "grateful": "grateful",
    "guilty": "regret",           "hopeful": "hopeful",      "impressed": "surprised",
    "jealous": "angry",           "joyful": "happy",         "lonely": "sad",
    "nostalgic": "nostalgic",     "prepared": "neutral",     "proud": "happy",
    "sad": "sad",                 "sentimental": "nostalgic", "surprised": "surprised",
    "terrified": "surprised",     "trusting": "grateful",
}

# === DailyDialog emotion → ACT mapping ===
# DD uses numeric labels: 0=no emotion, 1=anger, 2=disgust, 3=fear, 4=happiness, 5=sadness, 6=surprise
DD_TO_ACT = {
    0: "neutral",
    1: "angry",
    2: "angry",      # disgust → angry (closest in ACT)
    3: "surprised",  # fear → surprised (high arousal)
    4: "happy",
    5: "sad",
    6: "surprised",
}

# === Saravia emotion dataset mapping ===
SARAVIA_TO_ACT = {
    0: "sad",        # sadness
    1: "happy",      # joy
    2: "grateful",   # love → grateful (positive relational)
    3: "angry",      # anger
    4: "surprised",  # fear → surprised
    5: "surprised",  # surprise
}


def _make_response_for_emotion(emotion: str, context: str = "") -> str:
    """Generate a contextual response template for a given emotion."""
    responses = RESPONSE_TEMPLATES.get(emotion, ["I understand."])
    return random.choice(responses)


def download_goemotions() -> List[Dict]:
    """Download GoEmotions (Google) - 58k Reddit comments with 27+1 emotion labels.

    Source: Demszky et al. (2020), "GoEmotions: A Dataset of Fine-Grained Emotions"
    This is the BEST source for multi-label emotion classification training.
    Multi-label support maps naturally to composite ACT tokens.

    Returns:
        List of training samples mapped to ACT taxonomy
    """
    if not _ensure_datasets_lib():
        return []

    from datasets import load_dataset

    try:
        print("\n[GoEmotions] Downloading from HuggingFace (google-research-datasets/go_emotions)...")
        print("[GoEmotions] ~58,000 Reddit comments with fine-grained emotions")
        dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")

        # GoEmotions simplified has labels as list of ints
        label_names = dataset.features["labels"].feature.names if hasattr(dataset.features["labels"].feature, "names") else None

        # If label_names not directly available, use the known ordering
        goemo_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral",
        ]

        samples = []
        skipped = 0
        for item in dataset:
            text = item["text"]
            label_ids = item["labels"]

            if not label_ids:
                skipped += 1
                continue

            # Map primary label to ACT
            primary_label_name = goemo_labels[label_ids[0]] if label_ids[0] < len(goemo_labels) else "neutral"
            primary_act = GOEMO_TO_ACT.get(primary_label_name, "neutral")
            intensity = round(random.uniform(0.5, 0.9), 1)

            # Handle multi-label as composite ACT
            secondary_act = None
            secondary_intensity = None
            is_composite = False

            if len(label_ids) > 1:
                sec_label_name = goemo_labels[label_ids[1]] if label_ids[1] < len(goemo_labels) else None
                if sec_label_name:
                    sec_act = GOEMO_TO_ACT.get(sec_label_name, None)
                    if sec_act and sec_act != primary_act:
                        secondary_act = sec_act
                        secondary_intensity = round(random.uniform(0.3, intensity - 0.1), 1)
                        if secondary_intensity >= 0.1:
                            is_composite = True

            # Build ACT token
            components = [(primary_act, intensity)]
            if is_composite and secondary_act:
                components.append((secondary_act, secondary_intensity))

            act_string = build_composite_act_string(components)
            response = _make_response_for_emotion(primary_act)

            samples.append({
                "input": text,
                "output": f"{act_string} {response}" if response else act_string,
                "emotion_label": primary_act,
                "emotion_id": EMOTION_TO_ID[primary_act],
                "intensity": intensity,
                "secondary_emotion": secondary_act if is_composite else None,
                "secondary_intensity": secondary_intensity if is_composite else None,
                "is_composite": is_composite,
                "source": "goemotions",
            })

        print(f"[GoEmotions] Loaded {len(samples)} samples ({skipped} skipped)")
        _print_source_distribution(samples, "GoEmotions")
        return samples

    except Exception as e:
        print(f"[GoEmotions] Failed to download: {e}")
        return []


def download_empathetic_dialogues() -> List[Dict]:
    """Download EmpatheticDialogues (Facebook) - 25k emotional conversations.

    Source: Rashkin et al. (2019), "Towards Empathetic Open-domain Conversation Models"
    IDEAL for training emotional response generation -- provides actual
    conversation pairs grounded in emotional situations.

    Returns:
        List of training samples mapped to ACT taxonomy
    """
    if not _ensure_datasets_lib():
        return []

    from datasets import load_dataset

    try:
        print("\n[EmpatheticDialogues] Downloading from HuggingFace...")
        print("[EmpatheticDialogues] ~25,000 conversations with 32 emotion labels")
        dataset = load_dataset("empathetic_dialogues", split="train")

        samples = []
        skipped = 0
        current_conv_id = None
        current_context = None
        current_emotion = None

        for item in dataset:
            conv_id = item.get("conv_id", "")
            utterance = item.get("utterance", "").strip()
            context = item.get("context", "").strip()

            if not utterance:
                skipped += 1
                continue

            # EmpatheticDialogues alternates speaker/listener in each conversation
            # context field contains the emotional situation label
            if context and context != current_context:
                current_context = context
                current_emotion = ED_TO_ACT.get(context, "neutral")
                current_conv_id = conv_id

            if not current_emotion:
                current_emotion = "neutral"

            # Use pairs: context/situation as input, utterance as response
            # This is particularly valuable because the utterances ARE emotional responses
            if context:
                intensity = round(random.uniform(0.5, 0.9), 1)
                act_string = build_composite_act_string([(current_emotion, intensity)])
                response = utterance

                # Use the situation prompt as context
                situation = item.get("situation", "")
                input_text = situation if situation else context

                if input_text and response:
                    samples.append({
                        "input": input_text,
                        "output": f"{act_string} {response}",
                        "emotion_label": current_emotion,
                        "emotion_id": EMOTION_TO_ID[current_emotion],
                        "intensity": intensity,
                        "secondary_emotion": None,
                        "secondary_intensity": None,
                        "is_composite": False,
                        "source": "empathetic_dialogues",
                    })

        print(f"[EmpatheticDialogues] Loaded {len(samples)} samples ({skipped} skipped)")
        _print_source_distribution(samples, "EmpatheticDialogues")
        return samples

    except Exception as e:
        print(f"[EmpatheticDialogues] Failed to download: {e}")
        return []


def download_daily_dialog() -> List[Dict]:
    """Download DailyDialog - 13k daily dialogues with emotion and act labels.

    Source: Li et al. (2017), "DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset"
    Good for conversational flow and neutral/everyday emotion coverage.

    Returns:
        List of training samples mapped to ACT taxonomy
    """
    if not _ensure_datasets_lib():
        return []

    from datasets import load_dataset

    try:
        print("\n[DailyDialog] Downloading from HuggingFace...")
        print("[DailyDialog] ~13,000 daily conversations with emotion labels")
        dataset = load_dataset("daily_dialog", split="train")

        samples = []
        for item in dataset:
            dialog = item.get("dialog", [])
            emotions = item.get("emotion", [])

            # Process dialogue pairs (user turn -> assistant turn)
            for i in range(0, len(dialog) - 1, 2):
                user_turn = dialog[i].strip() if i < len(dialog) else ""
                assistant_turn = dialog[i + 1].strip() if (i + 1) < len(dialog) else ""
                emotion_id = emotions[i + 1] if (i + 1) < len(emotions) else 0

                if not user_turn or not assistant_turn:
                    continue

                act_emotion = DD_TO_ACT.get(emotion_id, "neutral")
                intensity = round(random.uniform(0.4, 0.8), 1)
                # Neutral utterances get lower intensity
                if act_emotion == "neutral":
                    intensity = round(random.uniform(0.1, 0.4), 1)

                act_string = build_composite_act_string([(act_emotion, intensity)])

                samples.append({
                    "input": user_turn,
                    "output": f"{act_string} {assistant_turn}",
                    "emotion_label": act_emotion,
                    "emotion_id": EMOTION_TO_ID[act_emotion],
                    "intensity": intensity,
                    "secondary_emotion": None,
                    "secondary_intensity": None,
                    "is_composite": False,
                    "source": "daily_dialog",
                })

        print(f"[DailyDialog] Loaded {len(samples)} samples")
        _print_source_distribution(samples, "DailyDialog")
        return samples

    except Exception as e:
        print(f"[DailyDialog] Failed to download: {e}")
        return []


def download_emotion_dataset() -> List[Dict]:
    """Download emotion dataset (Saravia et al., 2018) - 20k tweets with 6 emotions.

    A simple but reliable source for basic emotion classification.
    Covers: sadness, joy, love, anger, fear, surprise.

    Returns:
        List of training samples mapped to ACT taxonomy
    """
    if not _ensure_datasets_lib():
        return []

    from datasets import load_dataset

    try:
        print("\n[Emotion] Downloading Saravia emotion dataset from HuggingFace...")
        print("[Emotion] ~20,000 tweets with 6 emotion labels")
        dataset = load_dataset("dair-ai/emotion", split="train")

        samples = []
        for item in dataset:
            text = item["text"]
            label_id = item["label"]
            emotion = SARAVIA_TO_ACT.get(label_id, "neutral")
            intensity = round(random.uniform(0.5, 0.9), 1)

            act_string = build_composite_act_string([(emotion, intensity)])
            response = _make_response_for_emotion(emotion)

            samples.append({
                "input": text,
                "output": f"{act_string} {response}" if response else act_string,
                "emotion_label": emotion,
                "emotion_id": EMOTION_TO_ID[emotion],
                "intensity": intensity,
                "secondary_emotion": None,
                "secondary_intensity": None,
                "is_composite": False,
                "source": "hf_emotion",
            })

        print(f"[Emotion] Loaded {len(samples)} samples")
        _print_source_distribution(samples, "Emotion")
        return samples

    except Exception as e:
        print(f"[Emotion] Failed to download dataset: {e}")
        return []


def _print_source_distribution(samples: List[Dict], source_name: str):
    """Print emotion distribution for a specific data source."""
    counts = {}
    for s in samples:
        e = s["emotion_label"]
        counts[e] = counts.get(e, 0) + 1
    if not counts:
        return
    max_count = max(counts.values())
    print(f"  [{source_name}] ACT emotion distribution:")
    for emotion in EMOTION_LABELS:
        count = counts.get(emotion, 0)
        if count > 0:
            bar = "#" * (count * 30 // max_count)
            print(f"    {emotion:>12s}: {count:>5d} {bar}")
    covered = sum(1 for e in EMOTION_LABELS if counts.get(e, 0) > 0)
    print(f"  [{source_name}] Covers {covered}/17 ACT emotions")


def save_dataset(samples: List[Dict], output_path: str, split_ratio: float = 0.9):
    """Save the dataset to JSONL format with train/validation split.

    Args:
        samples: list of training samples
        output_path: base path for output files (e.g., "data/train.jsonl")
        split_ratio: fraction of data used for training (rest is validation)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    random.shuffle(samples)
    split_idx = int(len(samples) * split_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Save training set
    train_path = output_path
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[Data] Saved {len(train_samples)} training samples to {train_path}")

    # Save validation set
    val_path = output_path.replace("train", "val")
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[Data] Saved {len(val_samples)} validation samples to {val_path}")

    # Print statistics
    emotion_counts = {}
    for s in samples:
        e = s["emotion_label"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    print("\n[Data] Emotion distribution:")
    for emotion in EMOTION_LABELS:
        count = emotion_counts.get(emotion, 0)
        bar = "#" * (count * 40 // max(emotion_counts.values()))
        print(f"  {emotion:>12s}: {count:>5d} {bar}")

    composite_count = sum(1 for s in samples if s.get("is_composite"))
    print(f"\n[Data] Composite ACT samples: {composite_count} ({100*composite_count/len(samples):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and prepare ACT training data for Ol-y 1.1B ACTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset sources (recommended combination for best results):
  --download-hf          Download ALL HuggingFace datasets (recommended)
  --download-goemo       GoEmotions (Google) - 58k samples, 27 emotions, multi-label
  --download-empathetic  EmpatheticDialogues (Facebook) - 25k conversation pairs
  --download-dailydialog DailyDialog - 13k dialogues with emotion labels
  --download-emotion     Saravia emotion - 20k tweets, 6 basic emotions

Examples:
  # Synthetic only (no internet required):
  python scripts/extract_data.py --output data/train.jsonl --num-synthetic 10000

  # Full pipeline with all datasets:
  python scripts/extract_data.py --output data/train.jsonl --num-synthetic 10000 --download-hf

  # Selective download:
  python scripts/extract_data.py --output data/train.jsonl --download-goemo --download-empathetic
        """
    )
    parser.add_argument("--output", default="data/train.jsonl", help="Output JSONL file path")
    parser.add_argument("--num-synthetic", type=int, default=10000,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--oly-logs", default=None,
                        help="Path to Ol-y session log directory")
    parser.add_argument("--shinon-logs", dest="legacy_oly_logs", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--surface-act-logs", action="store_true",
                        help="Use surface ACT tokens from Ol-y logs instead of internal probe distributions")
    parser.add_argument("--memory-state", default=None,
                        help="Optional JSON path for persistent EMA emotional memory")
    parser.add_argument("--download-hf", action="store_true",
                        help="Download ALL recommended HuggingFace datasets")
    parser.add_argument("--download-goemo", action="store_true",
                        help="Download GoEmotions (Google, 58k, best for multi-label)")
    parser.add_argument("--download-empathetic", action="store_true",
                        help="Download EmpatheticDialogues (Facebook, 25k conversations)")
    parser.add_argument("--download-dailydialog", action="store_true",
                        help="Download DailyDialog (13k daily conversations)")
    parser.add_argument("--download-emotion", action="store_true",
                        help="Download Saravia emotion dataset (20k tweets)")
    parser.add_argument("--composite-ratio", type=float, default=0.3,
                        help="Fraction of samples with composite emotions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   Ol-y 1.1B ACTT - Dataset Extraction Pipeline          ║
    ║   Affective Communication Token Transformer              ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    all_samples = []
    source_counts = {}
    memory = EmotionalMemoryEMA.load(args.memory_state) if args.memory_state else None

    # --download-hf enables all HuggingFace sources
    if args.download_hf:
        args.download_goemo = True
        args.download_empathetic = True
        args.download_dailydialog = True
        args.download_emotion = True

    # ═══════════════════════════════════════════════════════
    # 1. Synthetic data (always available, no dependencies)
    # ═══════════════════════════════════════════════════════
    print(f"{'='*60}")
    print(f"  [1/6] Generating {args.num_synthetic} synthetic samples...")
    print(f"{'='*60}")
    synthetic = generate_synthetic_dataset(args.num_synthetic, args.composite_ratio)
    all_samples.extend(synthetic)
    source_counts["Synthetic"] = len(synthetic)
    print(f"  Generated {len(synthetic)} synthetic samples (all 17 emotions)")

    # ═══════════════════════════════════════════════════════
    # 2. Ol-y logs (if available)
    # ═══════════════════════════════════════════════════════
    log_dir = args.oly_logs or args.legacy_oly_logs
    if log_dir:
        print(f"\n{'='*60}")
        print(f"  [2/6] Loading Ol-y session logs...")
        print(f"{'='*60}")
        oly_logs = load_oly_logs(
            log_dir,
            prefer_probe=not args.surface_act_logs,
            memory=memory,
        )
        all_samples.extend(oly_logs)
        source_counts["Ol-y Logs"] = len(oly_logs)
        if args.memory_state and memory is not None:
            memory.save(args.memory_state)
            print(f"  [EMA] Saved emotional memory to {args.memory_state}")

    # ═══════════════════════════════════════════════════════
    # 3. GoEmotions (Google) -- BEST for multi-label emotion
    # ═══════════════════════════════════════════════════════
    if args.download_goemo:
        print(f"\n{'='*60}")
        print(f"  [3/6] GoEmotions (Google Research)")
        print(f"         58k Reddit comments, 27 emotions, multi-label")
        print(f"         Best source for composite ACT training")
        print(f"{'='*60}")
        goemo = download_goemotions()
        all_samples.extend(goemo)
        source_counts["GoEmotions"] = len(goemo)

    # ═══════════════════════════════════════════════════════
    # 4. EmpatheticDialogues (Facebook) -- BEST for responses
    # ═══════════════════════════════════════════════════════
    if args.download_empathetic:
        print(f"\n{'='*60}")
        print(f"  [4/6] EmpatheticDialogues (Facebook AI)")
        print(f"         25k emotional conversations")
        print(f"         Best source for empathetic response generation")
        print(f"{'='*60}")
        empathetic = download_empathetic_dialogues()
        all_samples.extend(empathetic)
        source_counts["EmpatheticDialogues"] = len(empathetic)

    # ═══════════════════════════════════════════════════════
    # 5. DailyDialog -- Good for everyday conversation
    # ═══════════════════════════════════════════════════════
    if args.download_dailydialog:
        print(f"\n{'='*60}")
        print(f"  [5/6] DailyDialog")
        print(f"         13k daily conversations with emotion labels")
        print(f"         Good for neutral/everyday emotion coverage")
        print(f"{'='*60}")
        daily = download_daily_dialog()
        all_samples.extend(daily)
        source_counts["DailyDialog"] = len(daily)

    # ═══════════════════════════════════════════════════════
    # 6. Saravia emotion dataset -- Simple baseline
    # ═══════════════════════════════════════════════════════
    if args.download_emotion:
        print(f"\n{'='*60}")
        print(f"  [6/6] Saravia Emotion Dataset")
        print(f"         20k tweets with 6 basic emotions")
        print(f"{'='*60}")
        hf_data = download_emotion_dataset()
        all_samples.extend(hf_data)
        source_counts["Saravia Emotion"] = len(hf_data)

    # ═══════════════════════════════════════════════════════
    # Save and report
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    for source, count in source_counts.items():
        print(f"  {source:.<30s} {count:>7,d} samples")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':.<30s} {len(all_samples):>7,d} samples")
    print()

    save_dataset(all_samples, args.output)

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║   Data extraction complete!                              ║
    ║                                                          ║
    ║   Train: {args.output:<47s}║
    ║   Val:   {args.output.replace('train', 'val'):<47s}║
    ║   Total: {len(all_samples):>7,d} samples                                ║
    ╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
