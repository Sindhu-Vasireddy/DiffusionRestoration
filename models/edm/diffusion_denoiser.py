import lightning as pl
import torch
import abc

from models.edm.diffusion_utils import expand_dims, sigma_to_logsnr

class KarrasDenoiser(torch.nn.Module):
    """EDM denoiser from Karras et al. (2022) with preconditioned inputs and outputs.

    This denoiser wraps a trainable model and scales its inputs and outputs as follows:

        output = c_skip * input + c_out * model(c_in * input, c_noise)

    where Karras et al. (2022) originally defined c_skip, c_out, c_in, c_noise coefficients
    as functions of sigma and sigma_data as follows (see Table 1 in the paper):

        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
        c_noise = ln(sigma) / 4

    Note that these coefficients take a simpler form when computed from log-SNR instead of sigma,
    where log-SNR is deined as 2 log (sigma_data / sigma):

        c_skip = sigmoid(logsnr)
        c_out = sigma_data * sqrt(sigmoid(-logsnr))
        c_in = (1 / sigma_data) * sqrt(sigmoid(logsnr))
        c_noise = logsnr

    where sigmoid(x) = 1 / (1 + exp(-x)) is the sigmoid function. Note that definition of c_noise
    here is slightly different (logsnr is an affine transform of ln(sigma) / 4), but we keep it
    this way for simplicity, and it works equivalently well in practice.

    Reference: https://arxiv.org/abs/2206.00364.
    """

    def __init__(self, model: torch.nn.Module, sigma_data: torch.Tensor):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def _c_skip(self, logsnr: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logsnr)

    def _c_in(self, logsnr: torch.Tensor) -> torch.Tensor:
        return (1 / self.sigma_data) * torch.sqrt(torch.sigmoid(logsnr))

    def _c_out(self, logsnr: torch.Tensor) -> torch.Tensor:
        return self.sigma_data * torch.sqrt(torch.sigmoid(-logsnr))

    def forward(self,
                input: torch.Tensor,
                sigma: torch.Tensor,
                conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: noised image of shape (batch size, num. out channels, height)
            sigma: noise scale of shape (batch size)
            conditioning: additional conditioning of shape (batch size, channels, height)

        Returns:
            prediction: network predictions of shape (batch size, num. out channels, height)
        """
        logsnr = sigma_to_logsnr(sigma, sigma_data=self.sigma_data)
        c_in = expand_dims(self._c_in(logsnr), input.ndim)
        c_out = expand_dims(self._c_out(logsnr), input.ndim)
        c_skip = expand_dims(self._c_skip(logsnr), input.ndim)

        if conditioning is not None:
            out = c_skip * input + c_out * self.model(c_in * input, logsnr, conditioning)
        else:
            out = c_skip * input + c_out * self.model(c_in * input, logsnr, None)

        return out
