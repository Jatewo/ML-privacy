"""Abstract base class for Membership Inference Attacks."""

from abc import ABC, abstractmethod
import torch
import numpy as np
from torch.utils.data import DataLoader
from ..utils.logger import get_colored_logger

log = get_colored_logger(__name__)


class MembershipInferenceAttack(ABC):
    """Abstract base class for membership inference attacks.

    Implements the Template Method Pattern. Subclasses must implement
    _calculate_signals to define the specific attack metric (loss, confidence, etc.).

    Attributes:
        target_model (nn.Module): The trained model to attack.
        device (torch.device): The device the model is currently on.

    """

    def __init__(self, target_model: torch.nn.Module) -> None:
        """Initialize the attack.

        Args:
            target_model (torch.nn.Module): The trained target model.

        """
        self.target_model = target_model

        try:
            self.device = next(target_model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        self.target_model.eval()

    def run_attack(
        self, members: DataLoader, non_members: DataLoader,
    ) -> dict[str, np.ndarray]:
        """Execute the attack on member and non-member datasets.

        Args:
            members (DataLoader): Loader containing member samples (training data).
            non_members (DataLoader): Loader containing non-member samples (test data).

        Returns:
            dict[str, np.ndarray]: A dictionary containing:
                - "member_signals": Array of attack signals for members.
                - "non_member_signals": Array of attack signals for non-members.

        """
        log.info(f"Running {self.__class__.__name__}...")

        log.debug("Calculating signals for Members...")
        member_signals = self._compute_dataset_signals(members)

        log.debug("Calculating signals for Non-Members...")
        non_member_signals = self._compute_dataset_signals(non_members)

        return {
            "member_signals": member_signals,
            "non_member_signals": non_member_signals,
        }

    def _compute_dataset_signals(self, loader: DataLoader) -> np.ndarray:
        """Iterate over a loader and collect signals.

        Args:
            loader (DataLoader): The data loader to iterate.

        Returns:
            np.ndarray: A 1D array of calculated signals.

        """
        all_signals = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                batch_signals = self._calculate_signals(images, labels)

                if isinstance(batch_signals, torch.Tensor):
                    batch_signals = batch_signals.cpu().numpy()

                all_signals.extend(batch_signals)

        return np.array(all_signals)

    @abstractmethod
    def _calculate_signals(
        self, images: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the attack signal (metric) for a batch of data.

        Must be implemented by subclasses.

        Args:
            images (torch.Tensor): Batch of input images.
            labels (torch.Tensor): Batch of true labels.

        Returns:
            torch.Tensor: A tensor of signals (one scalar per image).

        """
        pass
