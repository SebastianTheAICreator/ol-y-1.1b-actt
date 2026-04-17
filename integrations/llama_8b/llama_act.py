"""
Llama 3.1 8B with ACT (Affective Communication Tokens) integration.

Wraps a Llama 3.1 8B model (loaded via HuggingFace Transformers) with:
1. An ACT head bolted onto the transformer's hidden states
2. Prefill enforcement for structural compliance
3. QLoRA fine-tuning support for training on RTX 3050 Ti

The Llama model serves as the backbone, and the ACT head is added as a
separate callable module. During fine-tuning, only the LoRA adapters and
ACT head are trained -- the base Llama weights remain frozen.

This follows the approach used in Project Shinon (Section 8.1 of the paper):
"A locally-hosted companion AI system built on Llama 3.1 8B served via Ollama"

Reference: Sakishimiro (2026), Sections 8 and 14
"""

import json
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn

from oly.act.act_token import (
    EMOTION_LABELS, EMOTION_TO_ID, ID_TO_EMOTION,
    CompositeACT, EmotionState, ACT_PREFILL,
    parse_act_from_response, build_composite_act_string,
)


class LlamaACTHead(nn.Module):
    """Lightweight ACT head designed to attach to a Llama model.

    This head takes the last hidden state from Llama's transformer layers
    and produces emotion classification + intensity prediction.
    It is intentionally small so it can be trained without unfreezing the
    base model.
    """

    def __init__(self, hidden_size: int = 4096, num_emotions: int = 17, act_hidden: int = 1024):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, act_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(act_hidden, act_hidden),
            nn.SiLU(),
        )
        self.emotion_classifier = nn.Linear(act_hidden, num_emotions)
        self.intensity_regressor = nn.Sequential(
            nn.Linear(act_hidden, act_hidden // 2),
            nn.SiLU(),
            nn.Linear(act_hidden // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run ACT prediction on the last hidden state.

        Args:
            hidden_states: pooled hidden state from Llama (batch, hidden_size)

        Returns:
            Dictionary with emotion_logits and intensity predictions
        """
        projected = self.projection(hidden_states)
        emotion_logits = self.emotion_classifier(projected)
        raw_intensity = self.intensity_regressor(projected).squeeze(-1)
        intensity = 0.1 + torch.sigmoid(raw_intensity) * 0.9  # scale to [0.1, 1.0]
        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": torch.softmax(emotion_logits, dim=-1),
            "emotion_pred": emotion_logits.argmax(dim=-1),
            "intensity": intensity,
        }


class LlamaACT:
    """Llama 3.1 8B with ACT integration.

    Provides a high-level interface for:
    - Loading Llama 3.1 8B with QLoRA quantization (fits in 4GB VRAM)
    - Attaching the ACT head for emotion prediction
    - Generating responses with ACT token prefill enforcement
    - Fine-tuning with LoRA adapters + ACT head

    Usage:
        llama = LlamaACT.from_pretrained("meta-llama/Llama-3.1-8B")
        result = llama.generate("Hello, how are you?", emit_act=True)
        print(result["act_token"])  # <|ACT:"emotion":[{"name":"happy","intensity":0.7}]|>
        print(result["response"])   # "I'm doing well, thank you!"
    """

    def __init__(self, model=None, tokenizer=None, act_head=None, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.act_head = act_head
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "meta-llama/Llama-3.1-8B",
        load_in_4bit: bool = True,
        device: str = "cuda",
    ) -> "LlamaACT":
        """Load Llama 3.1 8B with QLoRA quantization and attach ACT head.

        Args:
            model_name: HuggingFace model ID or local path
            load_in_4bit: whether to use 4-bit NF4 quantization (required for <8GB VRAM)
            device: target device

        Returns:
            Initialized LlamaACT instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # 4-bit quantization config for RTX 3050 Ti
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        print(f"[LlamaACT] Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Determine hidden size from model config
        hidden_size = model.config.hidden_size

        # Initialize ACT head
        act_head = LlamaACTHead(hidden_size=hidden_size).to(device)

        print(f"[LlamaACT] Model loaded. Hidden size: {hidden_size}")
        print(f"[LlamaACT] ACT head parameters: {sum(p.numel() for p in act_head.parameters()):,}")

        return cls(model=model, tokenizer=tokenizer, act_head=act_head, device=device)

    def prepare_for_training(self, lora_r: int = 16, lora_alpha: int = 32):
        """Prepare the model for QLoRA fine-tuning.

        Freezes the base model, attaches LoRA adapters to attention layers,
        and keeps the ACT head fully trainable.

        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
        """
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # ACT head stays fully trainable (it's separate from the base model)
        for param in self.act_head.parameters():
            param.requires_grad = True

        print("[LlamaACT] Ready for QLoRA training with ACT head")

    def call_act(self, input_text: str) -> Dict[str, Any]:
        """Call the ACT function to predict emotional state from input.

        This is the standalone emotion prediction interface. It runs the
        input through the Llama backbone and then the ACT head, returning
        the predicted emotion and intensity without generating a response.

        Args:
            input_text: the input prompt/context

        Returns:
            Dictionary with emotion prediction, intensity, and ACT token string
        """
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get last hidden state, pool to last token
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            act_result = self.act_head(last_hidden)

        emotion_idx = act_result["emotion_pred"].item()
        emotion_label = EMOTION_LABELS[emotion_idx]
        intensity = act_result["intensity"].item()

        return {
            "emotion": emotion_label,
            "intensity": round(intensity, 2),
            "emotion_probs": {
                EMOTION_LABELS[i]: round(p, 4)
                for i, p in enumerate(act_result["emotion_probs"][0].cpu().tolist())
            },
            "act_token": build_composite_act_string([(emotion_label, intensity)]),
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        emit_act: bool = True,
    ) -> Dict[str, Any]:
        """Generate a response with optional ACT token emission.

        When emit_act=True:
        1. First calls the ACT head to predict emotional state
        2. Constructs the ACT token prefix
        3. Uses prefill enforcement to seed the response with the ACT token
        4. Generates the natural language response

        Args:
            prompt: input prompt text
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            emit_act: whether to prepend ACT token (default True)

        Returns:
            Dictionary with act_token, response text, and metadata
        """
        # Step 1: Get ACT prediction
        act_info = None
        prefill = ""
        if emit_act:
            act_info = self.call_act(prompt)
            prefill = act_info["act_token"]

        # Step 2: Generate response (with optional ACT prefill)
        full_prompt = prompt
        if prefill:
            # Prefill enforcement: seed the assistant response with the ACT token
            full_prompt = f"{prompt}\n\nAssistant: {prefill}"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the newly generated tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "act_token": prefill if emit_act else None,
            "response": response.strip(),
            "act_info": act_info,
            "full_output": prefill + response if emit_act else response,
        }

    def save_act_head(self, path: str):
        """Save the ACT head weights separately."""
        torch.save(self.act_head.state_dict(), path)
        print(f"[LlamaACT] ACT head saved to {path}")

    def load_act_head(self, path: str):
        """Load ACT head weights."""
        self.act_head.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        print(f"[LlamaACT] ACT head loaded from {path}")
