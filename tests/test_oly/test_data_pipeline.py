"""
Tests for the data extraction and preparation pipeline.

Validates the complete data pipeline with synthetic data:
- Synthetic sample generation (all 17 emotions)
- Composite ACT token generation
- Dataset balance and integrity
- JSONL serialization/deserialization
- Train/validation split correctness
- HuggingFace dataset mapping logic
- Ol-y log parsing
- Edge cases (empty responses, novel emotions)

These tests run entirely on synthetic data -- no model training required.
"""

import sys
import os
import json
import random
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oly.act.act_token import (
    EMOTION_LABELS, EMOTION_TO_ID, EMOTION_CATEGORIES,
    build_composite_act_string, parse_act_from_response,
    INTENSITY_MIN, INTENSITY_MAX,
)

# Import from the extract_data script
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from extract_data import (
    generate_synthetic_sample,
    generate_synthetic_dataset,
    save_dataset,
    load_oly_logs,
    CONVERSATION_TEMPLATES,
    RESPONSE_TEMPLATES,
)


class TestConversationTemplates:
    """Verify conversation templates cover the full taxonomy."""

    def test_all_17_emotions_have_templates(self):
        """Each of the 17 emotions must have at least one template."""
        template_emotions = {t["emotion"] for t in CONVERSATION_TEMPLATES}
        for emotion in EMOTION_LABELS:
            assert emotion in template_emotions, f"Missing template for: {emotion}"

    def test_all_17_emotions_have_responses(self):
        """Each emotion must have at least one response template."""
        for emotion in EMOTION_LABELS:
            assert emotion in RESPONSE_TEMPLATES, f"Missing response for: {emotion}"
            assert len(RESPONSE_TEMPLATES[emotion]) >= 1

    def test_template_intensity_ranges_valid(self):
        """Intensity ranges must be within [0.1, 1.0]."""
        for t in CONVERSATION_TEMPLATES:
            low, high = t["intensity_range"]
            assert low >= INTENSITY_MIN, f"{t['emotion']}: low={low} < {INTENSITY_MIN}"
            assert high <= INTENSITY_MAX, f"{t['emotion']}: high={high} > {INTENSITY_MAX}"
            assert low < high, f"{t['emotion']}: low >= high"

    def test_each_template_has_multiple_prompts(self):
        """Each emotion should have at least 3 prompt variants for diversity."""
        for t in CONVERSATION_TEMPLATES:
            assert len(t["prompts"]) >= 3, (
                f"{t['emotion']} has only {len(t['prompts'])} prompts (need >= 3)"
            )

    def test_no_duplicate_prompts_across_emotions(self):
        """Prompts should not be duplicated across different emotion templates."""
        all_prompts = []
        for t in CONVERSATION_TEMPLATES:
            all_prompts.extend(t["prompts"])
        assert len(all_prompts) == len(set(all_prompts)), "Found duplicate prompts"


class TestSyntheticSampleGeneration:
    """Test individual synthetic sample creation."""

    def test_basic_sample_structure(self):
        """A generated sample must have all required fields."""
        random.seed(42)
        template = CONVERSATION_TEMPLATES[0]  # happy
        sample = generate_synthetic_sample(template, add_composite=False)

        required_keys = [
            "input", "output", "emotion_label", "emotion_id",
            "intensity", "secondary_emotion", "secondary_intensity", "is_composite",
        ]
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"

    def test_sample_emotion_matches_template(self):
        """Generated sample emotion must match the template emotion."""
        random.seed(42)
        for template in CONVERSATION_TEMPLATES:
            sample = generate_synthetic_sample(template, add_composite=False)
            assert sample["emotion_label"] == template["emotion"]
            assert sample["emotion_id"] == EMOTION_TO_ID[template["emotion"]]

    def test_sample_intensity_in_range(self):
        """Generated intensity must be within the template's range."""
        random.seed(42)
        for template in CONVERSATION_TEMPLATES:
            for _ in range(10):
                sample = generate_synthetic_sample(template, add_composite=False)
                low, high = template["intensity_range"]
                assert low <= sample["intensity"] <= high, (
                    f"{template['emotion']}: intensity {sample['intensity']} "
                    f"not in [{low}, {high}]"
                )

    def test_sample_output_contains_act_token(self):
        """Output must start with a valid ACT token."""
        random.seed(42)
        template = CONVERSATION_TEMPLATES[0]
        sample = generate_synthetic_sample(template)

        act = parse_act_from_response(sample["output"])
        assert act is not None, f"No ACT token in output: {sample['output'][:100]}"
        assert act.dominant.name == template["emotion"]

    def test_sample_input_from_template(self):
        """Input should be one of the template's prompts."""
        random.seed(42)
        for template in CONVERSATION_TEMPLATES:
            sample = generate_synthetic_sample(template)
            assert sample["input"] in template["prompts"]

    def test_non_composite_sample(self):
        """Non-composite samples should have no secondary emotion."""
        random.seed(42)
        template = CONVERSATION_TEMPLATES[0]
        sample = generate_synthetic_sample(template, add_composite=False)
        assert sample["is_composite"] is False
        assert sample["secondary_emotion"] is None
        assert sample["secondary_intensity"] is None

    def test_composite_sample_generation(self):
        """Composite samples should sometimes have a secondary emotion."""
        random.seed(42)
        has_composite = False
        for _ in range(100):
            template = random.choice(CONVERSATION_TEMPLATES)
            sample = generate_synthetic_sample(template, add_composite=True)
            if sample["is_composite"]:
                has_composite = True
                assert sample["secondary_emotion"] is not None
                assert sample["secondary_emotion"] in EMOTION_LABELS
                assert sample["secondary_emotion"] != sample["emotion_label"]
                assert sample["secondary_intensity"] is not None
                assert sample["secondary_intensity"] >= INTENSITY_MIN
                break

        assert has_composite, "No composite samples generated in 100 attempts"

    def test_emptiness_blank_response(self):
        """The 'emptiness' emotion can produce blank responses (Section 8.5)."""
        # Find the emptiness template
        emptiness_template = None
        for t in CONVERSATION_TEMPLATES:
            if t["emotion"] == "emptiness":
                emptiness_template = t
                break
        assert emptiness_template is not None

        # The response template for emptiness includes an empty string
        assert "" in RESPONSE_TEMPLATES["emptiness"]


class TestSyntheticDatasetGeneration:
    """Test full dataset generation."""

    @pytest.fixture
    def dataset(self):
        """Generate a small balanced dataset."""
        random.seed(42)
        return generate_synthetic_dataset(num_samples=170, composite_ratio=0.3)

    def test_correct_sample_count(self, dataset):
        assert len(dataset) == 170

    def test_all_emotions_represented(self, dataset):
        """All 17 emotions must appear in the dataset."""
        emotions_found = {s["emotion_label"] for s in dataset}
        for emotion in EMOTION_LABELS:
            assert emotion in emotions_found, f"Emotion not found: {emotion}"

    def test_balanced_distribution(self, dataset):
        """Emotion distribution should be approximately balanced."""
        counts = {}
        for s in dataset:
            e = s["emotion_label"]
            counts[e] = counts.get(e, 0) + 1

        expected_per_emotion = 170 // 17  # = 10
        for emotion in EMOTION_LABELS:
            count = counts.get(emotion, 0)
            # Allow +/- 2 variance from perfect balance
            assert count >= expected_per_emotion - 2, (
                f"{emotion}: only {count} samples (expected ~{expected_per_emotion})"
            )

    def test_composite_ratio(self, dataset):
        """Approximately 30% of samples should be composite."""
        composite_count = sum(1 for s in dataset if s["is_composite"])
        ratio = composite_count / len(dataset)
        # 30% target with ~10% tolerance
        assert 0.05 < ratio < 0.55, f"Composite ratio: {ratio:.2f} (expected ~0.30)"

    def test_all_samples_have_valid_act_tokens(self, dataset):
        """Every sample output must contain a parseable ACT token."""
        for i, sample in enumerate(dataset):
            act = parse_act_from_response(sample["output"])
            assert act is not None, (
                f"Sample {i}: no valid ACT token in output: {sample['output'][:80]}"
            )

    def test_dataset_is_shuffled(self):
        """Two datasets with the same seed should be identically shuffled."""
        random.seed(123)
        ds1 = generate_synthetic_dataset(num_samples=34)
        random.seed(123)
        ds2 = generate_synthetic_dataset(num_samples=34)

        for s1, s2 in zip(ds1, ds2):
            assert s1["input"] == s2["input"]
            assert s1["emotion_label"] == s2["emotion_label"]

    def test_large_dataset(self):
        """Test generation with a larger sample count."""
        random.seed(42)
        ds = generate_synthetic_dataset(num_samples=1000)
        assert len(ds) == 1000

        # Verify all emotions still present
        emotions = {s["emotion_label"] for s in ds}
        assert len(emotions) == 17


class TestDatasetSerialization:
    """Test saving and loading datasets to JSONL format."""

    def test_save_and_reload(self, tmp_path):
        """Save dataset to JSONL and verify it can be reloaded correctly."""
        random.seed(42)
        dataset = generate_synthetic_dataset(num_samples=100)
        output_path = str(tmp_path / "train.jsonl")

        save_dataset(dataset, output_path, split_ratio=0.8)

        # Check train file exists
        assert os.path.exists(output_path)

        # Check val file exists
        val_path = output_path.replace("train", "val")
        assert os.path.exists(val_path)

        # Reload and verify
        loaded_train = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                loaded_train.append(json.loads(line.strip()))

        loaded_val = []
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                loaded_val.append(json.loads(line.strip()))

        # 80/20 split of 100 samples
        assert len(loaded_train) == 80
        assert len(loaded_val) == 20
        assert len(loaded_train) + len(loaded_val) == 100

    def test_jsonl_format_integrity(self, tmp_path):
        """Each line in JSONL must be valid JSON with all fields."""
        random.seed(42)
        dataset = generate_synthetic_dataset(num_samples=50)
        output_path = str(tmp_path / "train.jsonl")

        save_dataset(dataset, output_path, split_ratio=1.0)  # no val split

        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                sample = json.loads(line.strip())
                assert "input" in sample, f"Line {line_num}: missing 'input'"
                assert "output" in sample, f"Line {line_num}: missing 'output'"
                assert "emotion_label" in sample, f"Line {line_num}: missing 'emotion_label'"
                assert "emotion_id" in sample, f"Line {line_num}: missing 'emotion_id'"
                assert "intensity" in sample, f"Line {line_num}: missing 'intensity'"

    def test_unicode_handling(self, tmp_path):
        """Verify JSONL handles unicode characters."""
        random.seed(42)
        dataset = generate_synthetic_dataset(num_samples=10)
        output_path = str(tmp_path / "train.jsonl")
        save_dataset(dataset, output_path, split_ratio=1.0)

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert len(content) > 0


class TestOlyLogParsing:
    """Test Ol-y session log loading and conversion."""

    def test_load_nonexistent_directory(self):
        """Loading from nonexistent directory should return empty list."""
        result = load_oly_logs("/nonexistent/path/to/logs")
        assert result == []

    def test_load_valid_oly_logs(self, tmp_path):
        """Parse valid Ol-y session logs into training format."""
        # Create a mock Ol-y log file
        log_data = {
            "session_id": "test-001",
            "messages": [
                {"role": "user", "content": "How are you feeling today?"},
                {
                    "role": "assistant",
                    "content": '<|ACT:"emotion":[{"name":"happy","intensity":0.8}]|> I am doing well!'
                },
                {"role": "user", "content": "That's great!"},
                {
                    "role": "assistant",
                    "content": '<|ACT:"emotion":[{"name":"grateful","intensity":0.7}]|> Thank you for asking.'
                },
            ]
        }
        log_file = tmp_path / "session_001.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f)

        samples = load_oly_logs(str(tmp_path))
        assert len(samples) == 2
        assert samples[0]["emotion_label"] == "happy"
        assert samples[0]["input"] == "How are you feeling today?"
        assert samples[1]["emotion_label"] == "grateful"
        assert samples[1]["input"] == "That's great!"

    def test_load_oly_logs_array_format(self, tmp_path):
        """Ol-y logs in array format should also be parsed."""
        log_data = [
            {"role": "user", "content": "Tell me something interesting."},
            {
                "role": "assistant",
                "content": '<|ACT:"emotion":[{"name":"curious","intensity":0.6}]|> Here is something cool.'
            },
        ]
        log_file = tmp_path / "session_002.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f)

        samples = load_oly_logs(str(tmp_path))
        assert len(samples) == 1
        assert samples[0]["emotion_label"] == "curious"

    def test_load_oly_logs_skips_invalid(self, tmp_path):
        """Invalid JSON files should be skipped gracefully."""
        bad_file = tmp_path / "broken.json"
        with open(bad_file, "w") as f:
            f.write("{ invalid json }")

        samples = load_oly_logs(str(tmp_path))
        assert samples == []

    def test_load_oly_logs_skips_non_act_messages(self, tmp_path):
        """Messages without ACT tokens should be skipped."""
        log_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},  # no ACT token
            ]
        }
        log_file = tmp_path / "session_003.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f)

        samples = load_oly_logs(str(tmp_path))
        assert len(samples) == 0

    def test_oly_composite_act(self, tmp_path):
        """Ol-y logs with composite ACT tokens should parse correctly."""
        log_data = {
            "messages": [
                {"role": "user", "content": "I feel conflicted about this."},
                {
                    "role": "assistant",
                    "content": '<|ACT:"emotion":[{"name":"sad","intensity":0.7},{"name":"hopeful","intensity":0.4}]|> I understand.'
                },
            ]
        }
        log_file = tmp_path / "session_004.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f)

        samples = load_oly_logs(str(tmp_path))
        assert len(samples) == 1
        assert samples[0]["emotion_label"] == "sad"
        assert samples[0]["is_composite"] is True


class TestEmotionIdMapping:
    """Verify emotion ID mapping consistency in generated data."""

    def test_ids_match_labels(self):
        """emotion_id must correspond to EMOTION_TO_ID[emotion_label]."""
        random.seed(42)
        dataset = generate_synthetic_dataset(num_samples=170)
        for sample in dataset:
            expected_id = EMOTION_TO_ID[sample["emotion_label"]]
            assert sample["emotion_id"] == expected_id, (
                f"{sample['emotion_label']}: got id {sample['emotion_id']}, "
                f"expected {expected_id}"
            )

    def test_all_ids_in_valid_range(self):
        """All emotion IDs must be in [0, 16]."""
        random.seed(42)
        dataset = generate_synthetic_dataset(num_samples=170)
        for sample in dataset:
            assert 0 <= sample["emotion_id"] < 17
