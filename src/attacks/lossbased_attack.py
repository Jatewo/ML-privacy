"""Loss-based Membership Inference Attack."""

import torch
import torch.nn.functional as f
from .membership_inference_attack import MembershipInferenceAttack


class LossBasedMIA(MembershipInferenceAttack):
    """Attacks the model based on the Cross-Entropy Loss values.

    The intuition is that the model will have significantly lower loss
    on data it has seen during training (members) compared to unseen
    data (non-members) due to overfitting.
    """

    def _calculate_signals(
        self, images: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Cross-Entropy Loss for each sample.

        Args:
            images (torch.Tensor): Input images.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Array of loss values per sample.

        """
        outputs = self.target_model(images)

        loss = f.cross_entropy(outputs, labels, reduction="none")

        return loss
