from typing import List, Optional, Type, Union, Tuple
import torch

class StandardNormalNoiseDataset(torch.utils.data.Dataset):
    """A dataset that generates standard normal noise tensors.

    Args:
        shape: The shape of the noise tensors that will be generated.
        size: The total number of noise tensors in the dataset.
        fixed_noise: If True, the noise is generated during the first epoch and then reused.
    """

    def __init__(self, shape: tuple[int, ...], size: int, fixed_noise: bool = True):
        super().__init__()
        self.shape = shape
        self.size = size
        self.fixed_noise = fixed_noise
        self.noise_cache = {}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        # Generate noise tensor if it is not cached.
        if index not in self.noise_cache:
            noise = torch.randn(self.shape)
            if self.fixed_noise:
                self.noise_cache[index] = noise
        else:
            noise = self.noise_cache[index]
        return {"noise": noise}


class DiffusionDatasetWrapper(torch.utils.data.Dataset):
    """A wrapper for a PyTorch dataset that generates noise tensors.

    Args:
        dataset: A PyTorch dataset.
        fixed_noise: If True, the noise is generated during the first epoch and then reused.
            Using the same noise for each sample in the dataset is useful for validation.
        reduce_to_vector: samples noise with shape (batch size, 1, H*W//2)
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        fixed_noise: bool = False,
        conditional: bool = False
        ):
        super().__init__()
        self.dataset = dataset
        target_data_shape  = dataset[0].shape
        self.conditional = conditional
        
        self.noise_dataset = StandardNormalNoiseDataset(
            shape=target_data_shape, size=len(dataset), fixed_noise=fixed_noise
        )

    def __len__(self) -> int:
        return len(self.dataset)-2

    def __getitem__(self, index) -> dict[str, torch.Tensor]:

        if index < 2:
            index = 2

        batch_dict = {"target": self.dataset[index],
                      "noise": self.noise_dataset[index]["noise"]}
        if self.conditional:
            batch_dict["condition"] = self.dataset[index-2:index].squeeze(0)

        return batch_dict
