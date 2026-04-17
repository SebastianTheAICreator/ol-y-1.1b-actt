"""
ACT Loss formulation for fine-tuning compliance.

Implements the decomposed training objective from Section 5 of the paper:

    L = lambda_1 * L_struct + lambda_2 * L_label + lambda_3 * L_resp

Where:
    L_struct: Structural compliance loss -- penalises responses without valid ACT tokens
    L_label:  Emotion label cross-entropy against gold-standard annotations
    L_resp:   Standard causal LM loss (handled by the main model, not here)

Recommended weights from the paper (Section 5.1):
    lambda_1 = 2.0  (structural compliance is highest priority)
    lambda_2 = 1.5  (emotion accuracy is second priority)
    lambda_3 = 1.0  (response quality is baseline priority)

Reference: Sakishimiro (2026), Section 5 "Loss Formulation for Fine-Tuning Compliance"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oly.act.act_token import INTENSITY_MIN, INTENSITY_MAX


class ACTLoss(nn.Module):
    """Combined ACT loss for training emotion classification and intensity prediction.

    This module computes L_struct and L_label from the paper.
    L_resp (language modeling loss) is computed separately in the main model.
    """

    def __init__(self, config):
        """Initialize ACT loss with weights from config.

        Args:
            config: OlyConfig with act_loss_* parameters
        """
        super().__init__()
        self.structural_weight = config.act_loss_structural  # lambda_1 = 2.0
        self.label_weight = config.act_loss_label              # lambda_2 = 1.5
        self.num_emotions = config.act_num_emotions

    def forward(
        self,
        emotion_logits: Optional[torch.Tensor],
        predicted_intensity: Optional[torch.Tensor],
        target_labels: Optional[torch.Tensor],
        target_intensities: Optional[torch.Tensor],
        act_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the combined ACT loss.

        Args:
            emotion_logits: predicted emotion logits (batch, num_emotions)
            predicted_intensity: predicted intensity values (batch,)
            target_labels: ground-truth emotion label indices (batch,)
            target_intensities: ground-truth intensity values (batch,)
            act_mask: binary mask -- 1 for samples with ACT annotations,
                      0 for samples without. Only masked samples contribute
                      to the ACT loss. Shape: (batch,)

        Returns:
            Tuple of:
                - total ACT loss (scalar tensor)
                - dictionary of individual loss components for logging
        """
        device = emotion_logits.device if emotion_logits is not None else torch.device("cpu")
        loss_parts = {}

        # Default mask: all samples have ACT annotations
        if act_mask is None and target_labels is not None:
            act_mask = torch.ones(target_labels.shape[0], device=device)

        total_loss = torch.tensor(0.0, device=device)

        # === L_label: Emotion label cross-entropy (Equation 13) ===
        # L_label = -E[ log softmax(f_theta(c))_{e*} ]
        if emotion_logits is not None and target_labels is not None:
            label_loss = F.cross_entropy(
                emotion_logits, target_labels, reduction="none"
            )
            # Apply mask: only count samples with ACT annotations
            if act_mask is not None:
                label_loss = (label_loss * act_mask).sum() / act_mask.sum().clamp(min=1)
            else:
                label_loss = label_loss.mean()

            loss_parts["act_label_loss"] = label_loss
            total_loss = total_loss + self.label_weight * label_loss

        # === L_intensity: Intensity regression loss ===
        # Not explicitly in the paper but necessary for training the intensity head.
        # Uses Smooth L1 loss which is robust to outliers.
        if predicted_intensity is not None and target_intensities is not None:
            intensity_loss = F.smooth_l1_loss(
                predicted_intensity, target_intensities, reduction="none"
            )
            if act_mask is not None:
                intensity_loss = (intensity_loss * act_mask).sum() / act_mask.sum().clamp(min=1)
            else:
                intensity_loss = intensity_loss.mean()

            loss_parts["act_intensity_loss"] = intensity_loss
            # Intensity loss is weighted as part of structural compliance
            total_loss = total_loss + self.structural_weight * intensity_loss

        # === L_struct: Structural compliance (Equation 12) ===
        # This is implicitly enforced by the ACT head architecture (the model
        # always produces valid emotion logits and intensity), but we add a
        # confidence penalty to encourage high-confidence predictions.
        if emotion_logits is not None:
            # Entropy of the emotion distribution -- lower entropy = more confident
            probs = F.softmax(emotion_logits, dim=-1)
            entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)
            # Normalize by max possible entropy
            max_entropy = torch.log(torch.tensor(float(self.num_emotions), device=device))
            structural_loss = (entropy / max_entropy)
            if act_mask is not None:
                structural_loss = (structural_loss * act_mask).sum() / act_mask.sum().clamp(min=1)
            else:
                structural_loss = structural_loss.mean()

            loss_parts["act_structural_loss"] = structural_loss
            total_loss = total_loss + 0.5 * structural_loss  # moderate weight

        loss_parts["act_total_loss"] = total_loss
        return total_loss, loss_parts


class ACTLossWithComposite(ACTLoss):
    """Extended ACT loss supporting composite multi-component emotions.

    Handles the case where the target has multiple (label, intensity) pairs
    per sample, as in the Composite ACT format (Section 3.3).
    """

    def forward(
        self,
        emotion_logits: Optional[torch.Tensor],
        predicted_intensity: Optional[torch.Tensor],
        target_labels: Optional[torch.Tensor],
        target_intensities: Optional[torch.Tensor],
        act_mask: Optional[torch.Tensor] = None,
        target_secondary_labels: Optional[torch.Tensor] = None,
        target_secondary_intensities: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute composite ACT loss with support for secondary emotions.

        Extends the base loss by adding a secondary emotion prediction target.
        The primary loss is computed as in the base class. The secondary loss
        is optional and only applied when secondary targets are provided.
        """
        total_loss, loss_parts = super().forward(
            emotion_logits, predicted_intensity,
            target_labels, target_intensities, act_mask
        )

        # Secondary emotion loss (if composite targets are provided)
        if (target_secondary_labels is not None
                and emotion_logits is not None):
            # For composite, we want the model's top-2 predictions to match
            # both the primary and secondary targets
            probs = F.softmax(emotion_logits, dim=-1)
            # Penalise low probability on the secondary label
            secondary_loss = F.cross_entropy(
                emotion_logits, target_secondary_labels, reduction="none"
            )
            if act_mask is not None:
                secondary_mask = act_mask * (target_secondary_labels >= 0).float()
                secondary_loss = (secondary_loss * secondary_mask).sum() / secondary_mask.sum().clamp(min=1)
            else:
                secondary_loss = secondary_loss.mean()

            loss_parts["act_secondary_loss"] = secondary_loss
            total_loss = total_loss + 0.5 * self.label_weight * secondary_loss

        loss_parts["act_total_loss"] = total_loss
        return total_loss, loss_parts
