"""Module for evaluating membership inference attacks."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from ..utils.logger import get_colored_logger

log = get_colored_logger(__name__)


class AttackEvaluator:
    """Evaluates the performance of a Membership Inference Attack.

    Generates plots and calculates metrics as defined in the course slides.

    Attributes:
        member_signals (np.ndarray): Signals (loss/confidence) for member samples.
        non_member_signals (np.ndarray): Signals for non-member samples.
        labels (np.ndarray): Ground truth labels (1 for member, 0 for non-member).
        signals (np.ndarray): Combined signals for all samples.

    """

    def __init__(
        self,
        member_signals: np.ndarray,
        non_member_signals: np.ndarray,
        lower_is_better: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            member_signals (np.ndarray): Array of signals for members.
            non_member_signals (np.ndarray): Array of signals for non-members.
            lower_is_better (bool, optional):
                If True, lower signal values indicate membership (e.g., Loss).
                If False, higher values indicate membership (e.g., Confidence).
                Defaults to True.

        """
        self.member_signals = member_signals
        self.non_member_signals = non_member_signals
        self.lower_is_better = lower_is_better

        self.labels = np.concatenate(
            [np.ones_like(member_signals), np.zeros_like(non_member_signals)],
        )
        self.signals = np.concatenate([member_signals, non_member_signals])

        self.score_signals = -self.signals if lower_is_better else self.signals

    def plot_distributions(
        self, save_path: str = "results/loss_distribution.png",
    ) -> None:
        """Plot histograms of the signals for members vs non-members.

        Corresponds to the visualization in Slide 17[cite: 297].

        Args:
            save_path (str, optional): Path to save the plot.

        """
        plt.figure(figsize=(10, 6))

        plt.hist(
            self.member_signals,
            bins=50,
            alpha=0.5,
            label="Members (Train)",
            density=True,
            log=True,
        )
        plt.hist(
            self.non_member_signals,
            bins=50,
            alpha=0.5,
            label="Non-Members (Test)",
            density=True,
            log=True,
        )

        signal_name = "Loss" if self.lower_is_better else "Confidence"
        plt.title(f"Signal Distribution ({signal_name})")
        plt.xlabel(f"{signal_name} Value")
        plt.ylabel("Density (Log Scale)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(save_path)
        log.info(f"Distribution plot saved to {save_path}")
        plt.close()

    def plot_roc_curve(self, save_path: str = "results/roc_curve.png") -> None:
        """Plot the ROC curve.

        Corresponds to the visualization in Slide 18[cite: 329].

        Args:
            save_path (str, optional): Path to save the plot.

        """
        fpr, tpr, _ = metrics.roc_curve(self.labels, self.score_signals)
        roc_auc = metrics.auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.savefig(save_path)
        log.info(f"ROC curve saved to {save_path}")
        plt.close()

    def get_metrics(self) -> dict[str, float]:
        """Calculate AUC and find the optimal threshold for accuracy.

        This performs the "Threshold Selection and Analysis" required by the project.

        Returns:
            dict: Dictionary containing AUC, Best Accuracy, and Best Threshold.

        """
        fpr, tpr, thresholds = metrics.roc_curve(self.labels, self.score_signals)
        roc_auc = metrics.auc(fpr, tpr)

        best_acc = 0.0
        best_thresh = 0.0

        for _, thresh in enumerate(thresholds):
            real_thresh = -thresh if self.lower_is_better else thresh

            if self.lower_is_better:
                pred_members = self.signals <= real_thresh
            else:
                pred_members = self.signals >= real_thresh

            acc = metrics.accuracy_score(self.labels, pred_members)

            if acc > best_acc:
                best_acc = acc
                best_thresh = real_thresh

        return {
            "auc": roc_auc,
            "best_accuracy": best_acc,
            "best_threshold": best_thresh,
        }
