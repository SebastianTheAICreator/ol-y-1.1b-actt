# Ol-y 1.1B ACTT

**Affective Communication Token Transformer** — A 1.1B parameter language model with integrated emotional state declaration.

Based on the ACT paper by Sakishimiro (2026): *"Affective Communication Tokens: A Framework for Emergent Emotional Expression in Large Language Models"*

## What is ACT?

ACT (Affective Communication Tokens) embeds emotional state directly into the generative output stream as a machine-readable prefix token. Before every response, the model declares its emotional state:

```
<|ACT:"emotion":[{"name":"curious","intensity":0.7}]|> That's a fascinating question!
```

The 17-emotion taxonomy includes: `happy`, `sad`, `angry`, `surprised`, `curious`, `awkward`, `question`, `think`, `neutral`, `hopeful`, `nostalgic`, `regret`, `grateful`, `relieved`, `emptiness`, `reflective`, `serenity`

## Architecture

| Component | Value |
|-----------|-------|
| Parameters | ~1.1B |
| Hidden size | 2048 |
| Layers | 24 |
| Attention heads | 16 |
| FFN size | 6400 (SwiGLU) |
| Position encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| ACT head | Separate callable module |

## Project Structure

```
├── config/                    # Model configurations
│   ├── oly_1b_config.json     # Ol-y 1.1B config
│   ├── llama_8b_config.json   # Llama 8B + ACT config
│   ├── claude_api_config.json # Claude API config
│   └── deepspeed_config.json  # DeepSpeed ZeRO-2
├── oly/                       # Core Ol-y model
│   ├── model/transformer.py   # Transformer architecture
│   ├── act/                   # ACT module
│   │   ├── act_token.py       # Token schema & parsing
│   │   ├── act_head.py        # Neural ACT head
│   │   └── act_loss.py        # Training losses
│   ├── tokenizer/tokenizer.py # BPE tokenizer
│   └── utils/helpers.py       # Training utilities
├── integrations/              # External model integrations
│   ├── llama_8b/llama_act.py  # Llama 3.1 8B + ACT
│   └── claude_api/claude_act.py # Claude API + ACT
├── scripts/                   # Executable scripts
│   ├── extract_data.py        # Data extraction
│   ├── compress_logs.py       # Ol-y log cleanup + EMA state
│   ├── train.py               # Training script
│   ├── train_tokenizer.py     # Tokenizer training
│   └── run_all.py             # One-click pipeline
├── tests/                     # Test suites
│   ├── test_oly/              # Ol-y model tests
│   ├── test_llama/            # Llama integration tests
│   └── test_claude/           # Claude integration tests
└── data/                      # Training data (generated)
```

## Quick Start (One Click)

```bash
python scripts/run_all.py
```

This single command:
1. Installs all dependencies
2. Generates training data with ACT annotations
3. Trains the BPE tokenizer
4. Trains the Ol-y 1.1B model

## Manual Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python scripts/extract_data.py --output data/train.jsonl --num-synthetic 10000
```

With cleaned Ol-y logs and persistent internal emotional memory:

```bash
python scripts/compress_logs.py --input logs/raw --output logs/clean --memory-state data/emotional_memory.json
python scripts/extract_data.py --output data/train.jsonl --oly-logs logs/clean --memory-state data/emotional_memory.json
```

By default, Ol-y log extraction prefers internal probe distributions over the
surface ACT token, so training can follow what Ol-y felt internally rather
than only what she expressed.

Runtime EMA memory can also be attached directly to the main model:

```python
from oly.act.emotional_memory import EmotionalMemoryEMA
from oly.model.transformer import OlyForCausalLM

memory = EmotionalMemoryEMA.load("data/emotional_memory.json")
model.set_emotional_memory(memory)
result = model.generate(input_ids, emit_act=True, memory_session_id="session-001")
memory.save("data/emotional_memory.json")
```

The core `forward` path stays a normal training/inference path. During
`generate`, the integrated ACT head runs first; when memory is attached, its
emotion probability distribution updates the EMA state.

### 3. Train Tokenizer

```bash
python scripts/train_tokenizer.py --data data/train.jsonl --output outputs/tokenizer
```

### 4. Train Model

```bash
python scripts/train.py --config config/oly_1b_config.json --data data/train.jsonl
```

## Hardware Requirements

Optimized for **RTX 3050 Ti (4GB VRAM) + 16GB RAM**:

- Gradient checkpointing (trades compute for memory)
- FP16 mixed precision training
- DeepSpeed ZeRO Stage 2 with CPU offloading
- Batch size 1 with gradient accumulation (effective batch = 8)

## Integrations

### Llama 3.1 8B + ACT

```python
from integrations.llama_8b import LlamaACT

llama = LlamaACT.from_pretrained("meta-llama/Llama-3.1-8B", load_in_4bit=True)
result = llama.generate("Tell me about your day", emit_act=True)
print(result["act_token"])   # <|ACT:"emotion":[{"name":"happy","intensity":0.7}]|>
print(result["response"])    # "I'm doing well..."
```

### Claude API + ACT

```python
from integrations.claude_api import ClaudeACT

claude = ClaudeACT(api_key="your-key")
result = claude.generate("What do you think about art?", emit_act=True)
print(result["emotion"])     # "curious"
print(result["intensity"])   # 0.7
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## ACT Loss Formulation

Training uses the decomposed loss from Section 5 of the paper:

```
L = λ₁·L_struct + λ₂·L_label + λ₃·L_resp
```

Where λ₁=2.0, λ₂=1.5, λ₃=1.0 — structural compliance is highest priority.

## Roadmap Additions

- Async probes: `oly/act/async_probe.py`
- EMA emotional memory: `oly/act/emotional_memory.py`
- Log compression: `oly/data/oly_logs.py`, `scripts/compress_logs.py`
- Quantized runtime target: `Modelfile.optimized`, `config/quantization_presets.json`
- Ol-y 3B coordination target: `config/oly_3b_config.json`
- Credited roadmap text: `ROADMAP_SAKISHIMIRO.txt`

## License

Custom proprietary license — copying, redistribution, sale, or claiming this
code as your own is prohibited without prior written permission. ACT materials
that are explicitly attributed to an external ACT source remain excluded from
this ownership claim and follow their original terms. See [LICENSE](LICENSE).

## Reference

Sakishimiro (2026). "Affective Communication Tokens: A Framework for Emergent Emotional Expression in Large Language Models." Independent Research — Project Shinon.
