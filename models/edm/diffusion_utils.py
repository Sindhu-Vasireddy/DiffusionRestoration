import torch
import sys
import lightning 

from enum import Enum


class DiffEqDomain(str, Enum):
    TIME = "time"
    SIGMA = "sigma"
    LOGSNR = "logsnr"


def expand_dims(x: torch.Tensor, target_ndim):
    """Expands the dimensions of a tensor to match a target number of dimensions.

    Args:
        x: Input tensor of shape [N].
        target_ndim: Target number of dimensions.

    Returns:
        Tensor of shape [N, 1, ..., 1] with target_ndim dimensions and the same values as x.
    """
    return x.reshape(x.shape + (1,) * (target_ndim - x.ndim))


def _append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def sigma_to_logsnr(sigma: torch.Tensor, sigma_data: float):
    return 2 * torch.log(sigma_data / sigma)


def logsnr_to_sigma(logsnr: torch.Tensor, sigma_data: float):
    return sigma_data * torch.exp(-logsnr / 2)


class ProgressBar(lightning.pytorch.callbacks.TQDMProgressBar):
    """ Custom progress bar that removes the validation progress to avoid
        printing multiple lines during sampling.
    """
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True

        return bar
