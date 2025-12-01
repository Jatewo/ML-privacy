"""
Main script to execute the Loss-based Membership Inference Attack (MIA).
This script orchestrates the Data, Model, Attack, and Analysis domains.
"""

from src.data.dataset_handler import DatasetHandler
from src.models.model_wrapper import TargetModelWrapper
from src.attacks import LossBasedMIA
from src.analysis import AttackEvaluator
from src.utils.logger import get_colored_logger
from typing import Sized, cast
import config


log = get_colored_logger(__name__)


def main():
    log.info("Starting Mini Project 2: Loss-based Membership Inference Attack.")

    log.info("Phase 1: Preparing Datasets (Downloading/Loading CIFAR-10).")
    data_handler = DatasetHandler(dataset_name="cifar10", batch_size=config.BATCH_SIZE)

    train_loader, test_loader = data_handler.get_loaders()

    train_size = len(cast(Sized, train_loader.dataset))
    test_size = len(cast(Sized, test_loader.dataset))
    log.info(f"Loaded datasets. Train size: {train_size}, Test size: {test_size}.")

    log.info("Phase 2: Training the Target Model.")

    model_wrapper = TargetModelWrapper(dataset_name="cifar10")

    model_wrapper.train(train_loader, test_loader, epochs=config.EPOCHS)

    target_model = model_wrapper.get_model()

    log.info(f"Phase 3: Running Loss-Based MIA on {config.ATTACK_SIZE} samples.")

    member_loader, non_member_loader = data_handler.sample_attack_subsets(
        n_samples=config.ATTACK_SIZE
    )

    loss_attack = LossBasedMIA(target_model=target_model)
    attack_results = loss_attack.run_attack(
        members=member_loader, non_members=non_member_loader
    )

    member_losses = attack_results["member_signals"]
    non_member_losses = attack_results["non_member_signals"]

    log.info("Phase 4: Analyzing results and finding optimal threshold.")

    evaluator = AttackEvaluator(
        member_signals=member_losses, non_member_signals=non_member_losses
    )

    metrics_result = evaluator.get_metrics()

    evaluator.plot_distributions(save_path="results/loss_distribution.png")
    evaluator.plot_roc_curve(save_path="results/roc_curve.png")

    log.critical("--- MIA Results Summary ---")
    log.critical(f"Total Attack Samples: {config.ATTACK_SIZE * 2}")
    log.critical(f"Attack AUC: {metrics_result['auc']:.4f}")
    log.critical(f"Best Attack Accuracy: {metrics_result['best_accuracy']:.4f}")
    log.critical(f"Optimal Loss Threshold: {metrics_result['best_threshold']:.4f}")
    log.critical("--- Analysis Complete. Check the 'results/' directory for plots. ---")


if __name__ == "__main__":
    main()
