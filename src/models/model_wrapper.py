"""TargetModelWrapper class for managing model training."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .architecture import SimpleCNN

from ..utils.logger import get_colored_logger

log = get_colored_logger(__name__)


class TargetModelWrapper:
    """Wraps the PyTorch model to handle training and device management.

    Attributes:
        device (torch.device): The device (CPU, CUDA, or MPS) to run on.
        model (SimpleCNN): The underlying neural network.

    Methods:
        train: Execute the training loop.
        get_model: Return the trained model instance.

    """

    def __init__(self, dataset_name: str = "cifar10") -> None:
        """Initialize the TargetModelWrapper.

        Automatically detects available hardware (CUDA, MPS, or CPU).

        Args:
            dataset_name (str, optional): The name of the dataset to configure
                input channels. Defaults to "cifar10".

        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        log.info(f"Model initialized on device: {self.device}")

        channels = 1 if dataset_name.lower() == "mnist" else 3

        self.model = SimpleCNN(num_channels=channels, num_classes=10).to(self.device)

    def train(
        self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 10,
    ) -> None:
        """Train the model using the provided data loaders.

        Args:
            train_loader (DataLoader): Loader for training data.
            test_loader (DataLoader): Loader for validation data.
            epochs (int, optional): Number of training epochs. Defaults to 10.

        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        log.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total

            val_acc = self._evaluate(test_loader)

            log.info(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}%",
            )

    def _evaluate(self, loader: DataLoader) -> float:
        """Evaluate the model on a given dataset.

        Args:
            loader (DataLoader): The data loader to evaluate.

        Returns:
            float: The accuracy percentage (0-100).

        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def get_model(self) -> SimpleCNN:
        """Return the underlying PyTorch model.

        Returns:
            SimpleCNN: The trained model instance.

        """
        return self.model
