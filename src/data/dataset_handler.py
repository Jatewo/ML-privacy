"""DatasetHandler class for handling datasets."""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

type DatasetType = datasets.CIFAR10 | datasets.MNIST


class DatasetHandler:
    """Class for handling datasets.

    Attributes:
        dataset_name (str): The name of the dataset.
        batch_size (int): The batch size.
        root_dir (str): The root directory for the dataset.
        transform (torchvision.transforms.Compose): The transformation to apply to
            the dataset.

    Methods:
        get_loaders: Get the training and testing DataLoaders.
        sample_attack_subsets: Sample subsets of the training dataset for the attack.

    """

    def __init__(
        self,
        dataset_name: str = "cifar10",
        batch_size: int = 64,
        root_dir: str = "./data",
    ) -> None:
        """Initialize the DatasetHandler.

        Args:
            dataset_name (str, optional): The name of the dataset.
                Defaults to "cifar10".
            batch_size (int, optional): The batch size. Defaults to 64.
            root_dir (str, optional): The root directory for the dataset.
                Defaults to "./data".

        """
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.root_dir = root_dir

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        )

    def get_loaders(
        self,
    ) -> tuple[
        DataLoader[DatasetType], DataLoader[DatasetType],
    ]:
        """Get the training and testing DataLoaders.

        Downloads the dataset if not present. Wraps the dataset in DataLoaders.

        Returns:
            tuple: A tuple containing the training and testing DataLoaders.

        """
        train_set = self._load_dataset(train=True)
        test_set = self._load_dataset(train=False)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def sample_attack_subsets(
        self,
        n_samples: int = 5000,
    ) -> tuple[DataLoader[Subset[DatasetType]], DataLoader[Subset[DatasetType]]]:
        """Randomly sample N instances from Train and N from Test for the Attack phase.

        Args:
            n_samples (int, optional): The number of samples to sample.
                Defaults to 5000.

        Returns:
            train_subset (Subset): A random slice of the training data (Members)
            test_subset (Subset): A random slice of the test data (Non-Members)

        """
        train_set = self._load_dataset(train=True)
        test_set = self._load_dataset(train=False)

        train_indices = np.random.choice(
            len(train_set), n_samples, replace=False,
        )
        test_indices = np.random.choice(len(test_set), n_samples, replace=False)

        train_subset = Subset(train_set, list(train_indices))
        test_subset = Subset(test_set, list(test_indices))

        attack_train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        attack_test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return attack_train_loader, attack_test_loader

    def _load_dataset(self, train: bool) -> DatasetType:
        """Load the dataset from disk or download it if not present.

        Args:
            train (bool): Whether to load the training or test dataset.

        Returns:
            torch.utils.data.Dataset: The dataset.

        """
        if self.dataset_name == "cifar10":
            return datasets.CIFAR10(
                root=self.root_dir,
                train=train,
                download=True,
                transform=self.transform,
            )
        elif self.dataset_name == "mnist":
            mnist_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ],
            )
            return datasets.MNIST(
                root=self.root_dir,
                train=train,
                download=True,
                transform=mnist_transform,
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")
