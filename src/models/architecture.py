"""SimpleCNN architecture class."""
import torch
import torch.nn as nn
import torch.nn.functional as f

class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network.

    This architecture uses an AdaptiveAvgPool2d layer to handle variable input sizes
    (e.g., MNIST's 28x28 vs CIFAR-10's 32x32) without changing the fully connected
    layers.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization for first layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization for second layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        bn3 (nn.BatchNorm2d): Batch normalization for third layer.
        pool (nn.MaxPool2d): Max pooling layer.
        adaptive_pool (nn.AdaptiveAvgPool2d): Adaptive pooling to enforce output size.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer (output).
        dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
        forward: Performs the forward pass of the network.

    """

    def __init__(self, num_channels: int = 3, num_classes: int = 10) -> None:
        """Initialize the SimpleCNN.

        Args:
            num_channels (int, optional): Number of input channels (1 for MNIST,
                3 for CIFAR). Defaults to 3.
            num_classes (int, optional): Number of output classes. Defaults to 10.

        """
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output logits.

        """
        x = self.pool(f.relu(self.bn1(self.conv1(x))))
        x = self.pool(f.relu(self.bn2(self.conv2(x))))
        x = self.pool(f.relu(self.bn3(self.conv3(x))))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
