from src.data import DatasetHandler
from typing import Sized, cast


def main():
    data_handler = DatasetHandler(dataset_name="cifar10")

    train_loader, test_loader = data_handler.get_loaders()
    print("Datasets downloaded and loaders ready for training.")
    
    print(f"Training set size: {len(cast(Sized, train_loader.dataset))}")
    print(f"Test set size: {len(cast(Sized, test_loader.dataset))}")

if __name__ == "__main__":
    main()
