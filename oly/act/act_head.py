"""
ACT Head: Separate callable function for emotion classification and intensity prediction.

This module implements the ACT head as a standalone neural network component that
operates on the transformer's hidden states. It is architecturally separate from the
language modeling head, implementing the "calling function" pattern where:

1. The transformer backbone produces contextual hidden states
2. The ACT head is called as a separate function to predict emotional state
3. The result is used to construct the ACT token prefix
4. The LM head then generates the natural language response

The ACT head contains:
- A pooling layer to aggregate sequence-level information
- A projection network to map from hidden_size to act_hidden_size
- An emotion classifier (discrete categorical over E)
- An intensity regressor (continuous scalar in [0.1, 1.0])

Reference: Sakishimiro (2026), Sections 3-5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any

from oly.act.act_token import EMOTION_LABELS, EMOTION_TO_ID, INTENSITY_MIN, INTENSITY_MAX


class ACTHead(nn.Module):
    """ACT classification and regression head.

    This is the "calling function" -- a separate module that can be invoked
    independently of the LM head to predict the model's emotional state.

    Architecture:
        hidden_states -> pool -> project -> [emotion_classifier, intensity_regressor]

    The emotion classifier models P(e | c; theta) = softmax(f_theta(c))_e
    (Equation 9 from the paper).

    The intensity regressor models I ~ Beta(alpha, beta) rescaled to [0.1, 1.0]
    (Equation 10), approximated here as a sigmoid output scaled to the valid range.
    """

    def __init__(self, config):
        """Initialize the ACT head.

        Args:
            config: OlyConfig with act_* parameters
        """
        super().__init__()
        self.num_emotions = config.act_num_emotions
        self.act_hidden_size = config.act_hidden_size
        self.intensity_min = config.act_intensity_range[0]
        self.intensity_max = config.act_intensity_range[1]

        # Projection from transformer hidden size to ACT internal dimension
        # This creates a separate representation space for emotional processing
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.act_hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(config.act_hidden_size, config.act_hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        # Emotion classifier: maps ACT hidden state to logits over emotion labels
        # Implements f_theta(c) from Equation 9
        self.emotion_classifier = nn.Linear(config.act_hidden_size, self.num_emotions)

        # Intensity regressor: predicts continuous intensity value
        # Output is passed through sigmoid and rescaled to [0.1, 1.0]
        self.intensity_regressor = nn.Sequential(
            nn.Linear(config.act_hidden_size, config.act_hidden_size // 2),
            nn.SiLU(),
            nn.Linear(config.act_hidden_size // 2, 1),
        )

        # Composite component count predictor (1-5 components)
        self.component_predictor = nn.Linear(config.act_hidden_size, config.act_max_composite)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run the ACT head on transformer hidden states.

        This is the main "call" interface for the ACT function.

        Args:
            hidden_states: output from the transformer backbone (batch, seq_len, hidden_size)
            attention_mask: padding mask for proper pooling (batch, seq_len)

        Returns:
            Dictionary with:
                - emotion_logits: raw logits over emotion labels (batch, num_emotions)
                - emotion_probs: probability distribution over emotions (batch, num_emotions)
                - emotion_pred: predicted emotion index (batch,)
                - intensity: predicted intensity value in [0.1, 1.0] (batch,)
                - component_logits: logits for number of composite components (batch, max_composite)
                - act_hidden: internal ACT representation (batch, act_hidden_size)
        """
        # Pool hidden states to get a single vector per sequence
        # Uses masked mean pooling (average over non-padding tokens)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            # Use last token position (common for causal LMs)
            pooled = hidden_states[:, -1, :]

        # Project to ACT hidden space
        act_hidden = self.projection(pooled)  # (batch, act_hidden_size)

        # Emotion classification: P(e | c; theta)
        emotion_logits = self.emotion_classifier(act_hidden)  # (batch, num_emotions)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        emotion_pred = emotion_logits.argmax(dim=-1)  # (batch,)

        # Intensity regression: I in [0.1, 1.0]
        raw_intensity = self.intensity_regressor(act_hidden).squeeze(-1)  # (batch,)
        # Sigmoid + rescale to [intensity_min, intensity_max]
        intensity = torch.sigmoid(raw_intensity)
        intensity = self.intensity_min + intensity * (self.intensity_max - self.intensity_min)

        # Composite component count prediction
        component_logits = self.component_predictor(act_hidden)

        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "emotion_pred": emotion_pred,
            "intensity": intensity,
            "component_logits": component_logits,
            "act_hidden": act_hidden,
        }

    def predict_emotion_label(self, emotion_idx: torch.Tensor) -> list:
        """Convert emotion index tensor to human-readable labels."""
        return [EMOTION_LABELS[idx.item()] for idx in emotion_idx]

    def get_top_k_emotions(
        self, emotion_logits: torch.Tensor, k: int = 3
    ) -> list:
        """Get top-k emotion predictions with probabilities.

        Useful for constructing composite ACT tokens where multiple
        emotions are co-present.
        """
        probs = F.softmax(emotion_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(k, self.num_emotions), dim=-1)

        results = []
        for batch_idx in range(emotion_logits.shape[0]):
            batch_emotions = []
            for j in range(top_indices.shape[-1]):
                idx = top_indices[batch_idx, j].item()
                prob = top_probs[batch_idx, j].item()
                batch_emotions.append({
                    "label": EMOTION_LABELS[idx],
                    "probability": prob,
                    "index": idx,
                })
            results.append(batch_emotions)
        return results
