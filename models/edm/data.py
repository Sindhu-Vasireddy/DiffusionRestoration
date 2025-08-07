import torch
import xarray as xr
from torch.utils.data import DataLoader
import lightning
import numpy as np
from typing import Optional, Dict, Any

from models.transforms import apply_transforms
import models.xarray_utils as xu
from models.edm.diffusion_dataset import DiffusionDatasetWrapper


class DataModule(lightning.LightningDataModule):
    """
    A LightningDataModule for managing data loading and preprocessing for training, validation, and testing.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the DataModule with the given configuration.

        Args:
            config (dict): Configuration dictionary containing data loader and dataset settings.
        """

        super().__init__()

        self.config = config
        self.num_workers = config["data_loader"]["num_workers"]
        self.batch_size = config["data_loader"]["batch_size"]

    def setup(self, stage: str = None):
        """
        stage: fit /  test
        """

        if stage == "fit" or stage is None:
            self.train = DiffusionDatasetWrapper(
                DiffusionDataset("train", self.config["dataset"]),
                conditional=self.config["dataset"]["conditional"],
            )
            self.valid = DiffusionDatasetWrapper(
                DiffusionDataset("valid", self.config["dataset"]),
                conditional=self.config["dataset"]["conditional"],
            )

        if stage == "test":
            self.test = DiffusionDatasetWrapper(
                DiffusionDataset("test", self.config["dataset"]),
                conditional=self.config["dataset"]["conditional"],
            )

    def train_dataloader(self):
        return DataLoader(self.train, **self.config["data_loader"])

    def val_dataloader(self):
        return DataLoader(self.valid, **self.config["data_loader"])

    def test_dataloader(self):
        return DataLoader(self.test, **self.config["data_loader"])


class DiffusionDataset(torch.utils.data.Dataset):
    """Dataset for training.
    
    Args:
        stage: Stage of the dataset (train, valid, test).
        config: Configuration dictionary containing dataset settings.
        epsilon: Small value to avoid division by zero in normalization.
        transform_esm_with_target_reference: Whether to apply transformations with reference to the target data.
    """

    def __init__(
        self,
        stage: str,
        config: dict,
        epsilon=0.0001,
        transform_esm_with_target_reference=False,
    ):
        self.stage = stage
        self.config = config
        self.transforms = config["transforms"]
        self.epsilon = epsilon
        self.transform_esm_with_target_reference = transform_esm_with_target_reference

        self.target = None
        self.target_reference = None
        self.climate_model = None
        self.data = None
        self.pad = torch.nn.ZeroPad2d((0, 0, 2, 2))

        self.splits = {
            "train": [str(config["train_start"]), str(config["train_end"])],
            "valid": [str(config["valid_start"]), str(config["valid_end"])],
            "test": [str(config["test_start"]), str(config["test_end"])],
        }

        self.prepare_target_data()

    def load_data(self, is_reference=False):
        """Loads data from file and applies some preprocessing.

        Args:
            is_reference: Loads data from the training period to be used as reference for transformations.
        """

        data_path: str = self.config["dataset_path"]
        target = xr.open_dataset(data_path)
        target = target[self.config["target_variable"]]

        if is_reference:
            target = target.sel(
                time=slice(self.splits["train"][0], self.splits["train"][1])
            )
        else:
            target = target.sel(
                time=slice(self.splits[self.stage][0], self.splits[self.stage][1])
            )
        if self.config["crop_lat_target"]:
            target = target[:, 1:]
        return target

    def prepare_target_data(self):
        """Calls the target data loading and applies transformations."""

        self.target = self.load_data()
        self.target_reference = self.load_data(is_reference=True)
        self.num_samples = len(self.target.time.values)
        self.data = apply_transforms(self.target, self.target_reference, self.config)

    def __getitem__(self, index):

        y = torch.from_numpy(self.data.isel(time=index).values).float().unsqueeze(0)

        if self.config["apply_padding"]:
            y = self.pad(y)

        return y

    def __len__(self):
        return self.num_samples
