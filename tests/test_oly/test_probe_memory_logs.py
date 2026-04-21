import asyncio
import json

from oly.act.async_probe import AsyncProbeRunner
from oly.act.emotional_memory import EmotionalMemoryEMA, probe_valence
from oly.data.oly_logs import (
    compress_session_log,
    extract_samples_from_session_log,
    load_oly_logs,
)
from oly.model.transformer import OlyConfig, OlyForCausalLM


def test_probe_valence_uses_internal_distribution():
    distribution = {"sad": 0.7, "hopeful": 0.3}
    assert probe_valence(distribution) < 0


def test_emotional_memory_ema_roundtrip(tmp_path):
    memory = EmotionalMemoryEMA(alpha=0.3)
    memory.update_from_probe_distribution({"sad": 1.0}, session_id="s1")
    after_first = memory.value
    memory.update_from_probe_distribution({"hopeful": 1.0}, session_id="s2")

    assert after_first < 0
    assert memory.sessions == 2
    assert after_first < memory.value < 0.75

    path = tmp_path / "memory.json"
    memory.save(path)
    loaded = EmotionalMemoryEMA.load(path)

    assert loaded.value == memory.value
    assert loaded.sessions == 2


def test_async_probe_runner_aggregates_mixed_probes():
    async def sad_probe(_context):
        await asyncio.sleep(0.01)
        return {"emotion_probs": {"sad": 0.8, "hopeful": 0.2}, "intensity": 0.7}

    def hope_probe(_context):
        return {"emotion_probs": {"hopeful": 1.0}, "intensity": 0.5}

    async def run():
        runner = AsyncProbeRunner(timeout_s=1.0, max_concurrency=2)
        return await runner.run_all({"sad": sad_probe, "hope": hope_probe}, {})

    result = asyncio.run(run())

    assert len(result["results"]) == 2
    assert result["aggregate"]["num_probes"] == 2
    assert result["aggregate"]["dominant_emotion"] in {"sad", "hopeful"}
    assert -1.0 <= result["aggregate"]["valence"] <= 1.0


def test_log_compression_keeps_probe_and_drops_debug():
    raw = {
        "session_id": "session-1",
        "messages": [
            {"role": "user", "content": "How are you?"},
            {
                "role": "assistant",
                "content": '<|ACT:"emotion":[{"name":"hopeful","intensity":0.8}]|> I will be okay.',
                "probe_distribution": {"sad": 0.9, "hopeful": 0.1},
                "raw_logits": [1, 2, 3],
                "hidden_states": [[0.1]],
            },
        ],
    }

    compressed = compress_session_log(raw)
    assistant = compressed["messages"][1]

    assert "raw_logits" not in assistant
    assert "hidden_states" not in assistant
    assert assistant["probe_distribution"]["sad"] > 0.8
    assert compressed["session_internal_emotion"] == "sad"
    assert compressed["session_probe_valence"] < 0


def test_training_extraction_prefers_probe_over_surface_act():
    raw = {
        "session_id": "session-2",
        "messages": [
            {"role": "user", "content": "Are you really fine?"},
            {
                "role": "assistant",
                "content": '<|ACT:"emotion":[{"name":"hopeful","intensity":0.8}]|> I think so.',
                "probe_distribution": {"sad": 0.85, "hopeful": 0.15},
            },
        ],
    }

    samples = extract_samples_from_session_log(raw, prefer_probe=True)

    assert len(samples) == 1
    assert samples[0]["emotion_label"] == "sad"
    assert samples[0]["probe_valence"] < 0
    assert '<|ACT:"emotion":[{"name":"sad"' in samples[0]["output"]


def test_load_oly_logs_with_memory_updates_once_per_session(tmp_path):
    raw = {
        "session_id": "session-3",
        "messages": [
            {"role": "user", "content": "Talk to me."},
            {
                "role": "assistant",
                "content": '<|ACT:"emotion":[{"name":"hopeful","intensity":0.8}]|> I am here.',
                "probe_distribution": {"sad": 0.9, "hopeful": 0.1},
            },
            {"role": "user", "content": "Again?"},
            {
                "role": "assistant",
                "content": '<|ACT:"emotion":[{"name":"hopeful","intensity":0.7}]|> Still here.',
                "probe_distribution": {"sad": 0.8, "hopeful": 0.2},
            },
        ],
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    memory = EmotionalMemoryEMA(alpha=0.3)
    samples = load_oly_logs(str(tmp_path), memory=memory)

    assert len(samples) == 2
    assert memory.sessions == 1
    assert samples[0]["emotional_memory"] == memory.value
    assert samples[1]["emotional_memory"] == memory.value


def test_main_model_generate_can_update_emotional_memory():
    config = OlyConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
        act_enabled=True,
        act_num_emotions=17,
        act_hidden_size=16,
        gradient_checkpointing=False,
    )
    model = OlyForCausalLM(config)
    memory = EmotionalMemoryEMA(alpha=0.3)
    model.set_emotional_memory(memory)

    import torch

    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    result = model.generate(input_ids, max_new_tokens=1, emit_act=True, memory_session_id="runtime-1")

    assert result["act_result"] is not None
    assert result["emotional_memory"] is not None
    assert memory.sessions == 1
