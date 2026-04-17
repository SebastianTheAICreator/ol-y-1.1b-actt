"""
Data extraction and preparation script for Ol-y 1.1B ACTT training.

This script creates training datasets with ACT annotations from multiple sources:
1. Synthetic conversations with emotion labels (always available)
2. Existing conversation logs from Project Shinon (if available)
3. Public emotion-tagged datasets from HuggingFace (optional download)

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


def load_shinon_logs(log_dir: str) -> List[Dict]:
    """Load and convert Project Shinon conversation logs to training format.

    Reads Shinon's session logs (JSON files with ACT debug data) and converts
    them to the standard training format.

    Args:
        log_dir: directory containing Shinon session log JSON files

    Returns:
        List of training samples extracted from Shinon logs
    """
    samples = []

    if not os.path.exists(log_dir):
        print(f"[Data] Shinon log directory not found: {log_dir}")
        return samples

    for filename in os.listdir(log_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different Shinon log formats
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict) and "messages" in data:
                entries = data["messages"]
            else:
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                role = entry.get("role", "")
                content = entry.get("content", "")
                if role == "assistant" and "<|ACT:" in content:
                    # This is a Shinon response with ACT token
                    from oly.act.act_token import parse_act_from_response, ACT_RE
                    act = parse_act_from_response(content)
                    if act and act.dominant:
                        # Find the preceding user message
                        idx = entries.index(entry)
                        user_msg = ""
                        for prev in reversed(entries[:idx]):
                            if prev.get("role") == "user":
                                user_msg = prev.get("content", "")
                                break

                        if user_msg:
                            samples.append({
                                "input": user_msg,
                                "output": content,
                                "emotion_label": act.dominant.name,
                                "emotion_id": EMOTION_TO_ID.get(
                                    act.dominant.name, 8  # default to neutral
                                ),
                                "intensity": act.dominant.intensity,
                                "secondary_emotion": (
                                    act.secondary.name if act.secondary else None
                                ),
                                "secondary_intensity": (
                                    act.secondary.intensity if act.secondary else None
                                ),
                                "is_composite": len(act.emotions) > 1,
                                "source": "shinon",
                            })

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Data] Error reading {filename}: {e}")
            continue

    print(f"[Data] Loaded {len(samples)} samples from Shinon logs")
    return samples


def download_emotion_dataset() -> List[Dict]:
    """Download and process a public emotion-tagged dataset from HuggingFace.

    Uses the 'emotion' dataset (Saravia et al., 2018) which contains
    text samples labeled with 6 emotion categories. These are mapped
    to the ACT taxonomy for additional training data.

    Returns:
        List of training samples from the public dataset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Data] 'datasets' package not available. Skipping public dataset download.")
        return []

    # Mapping from the 'emotion' dataset labels to ACT labels
    emotion_map = {
        0: "sad",       # sadness
        1: "happy",     # joy
        2: "angry",     # anger (mapped to love->grateful makes less sense, use angry)
        3: "angry",     # anger
        4: "surprised", # fear -> surprised (closest in ACT taxonomy)
        5: "surprised", # surprise
    }

    try:
        print("[Data] Downloading emotion dataset from HuggingFace...")
        dataset = load_dataset("emotion", split="train")

        samples = []
        for item in dataset:
            text = item["text"]
            label_id = item["label"]
            emotion = emotion_map.get(label_id, "neutral")
            intensity = round(random.uniform(0.5, 0.9), 1)

            act_string = build_composite_act_string([(emotion, intensity)])
            samples.append({
                "input": text,
                "output": f"{act_string} I understand what you're expressing.",
                "emotion_label": emotion,
                "emotion_id": EMOTION_TO_ID[emotion],
                "intensity": intensity,
                "secondary_emotion": None,
                "secondary_intensity": None,
                "is_composite": False,
                "source": "hf_emotion",
            })

        print(f"[Data] Loaded {len(samples)} samples from HuggingFace emotion dataset")
        return samples

    except Exception as e:
        print(f"[Data] Failed to download dataset: {e}")
        return []


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
    parser = argparse.ArgumentParser(description="Extract and prepare ACT training data")
    parser.add_argument("--output", default="data/train.jsonl", help="Output JSONL file path")
    parser.add_argument("--num-synthetic", type=int, default=10000,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--shinon-logs", default=None,
                        help="Path to Shinon session log directory")
    parser.add_argument("--download-hf", action="store_true",
                        help="Download public emotion dataset from HuggingFace")
    parser.add_argument("--composite-ratio", type=float, default=0.3,
                        help="Fraction of samples with composite emotions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    all_samples = []

    # 1. Generate synthetic data (always available, no dependencies)
    print(f"[Data] Generating {args.num_synthetic} synthetic samples...")
    synthetic = generate_synthetic_dataset(args.num_synthetic, args.composite_ratio)
    all_samples.extend(synthetic)
    print(f"[Data] Generated {len(synthetic)} synthetic samples")

    # 2. Load Shinon logs (if available)
    if args.shinon_logs:
        shinon = load_shinon_logs(args.shinon_logs)
        all_samples.extend(shinon)

    # 3. Download public dataset (optional)
    if args.download_hf:
        hf_data = download_emotion_dataset()
        all_samples.extend(hf_data)

    # Save everything
    print(f"\n[Data] Total samples: {len(all_samples)}")
    save_dataset(all_samples, args.output)
    print("\n[Data] Data extraction complete!")


if __name__ == "__main__":
    main()
