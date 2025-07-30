import functools
import torch

from models.edm.diffusion_utils import expand_dims

class KarrasLoss():
    """Computes preconditioned MSE loss between denoised and target inputs from Karras et al. (2022).

    The loss has the following form:
        loss = precond(sigma) * (D(y + n, sigma) - y) ** 2,
        where:
            - precond(sigma) is an element-wise function that assigns weights to different noise levels,
            - D is the Karras preconditioned denoiser (KarrasDenoiser),
            - y is the noiseless input and y + n is the noised input.
    """

    def loss(self,
              denoiser,
              batch_dict: dict[str, torch.Tensor],
              sigma: torch.Tensor,
              **model_kwargs) -> torch.Tensor:

        batch_dict = batch_dict.copy()  # Avoid modifying the original dict.

        target, noise = batch_dict["target"], batch_dict["noise"]

        noised_target = target + noise * expand_dims(sigma, target.ndim)

        if "condition" in batch_dict.keys():
            condition = batch_dict["condition"]
        else:
            condition = None

        denoised_target = denoiser(noised_target, sigma, conditioning = condition)
        precond = (sigma**2 + denoiser.sigma_data**2) / (sigma * denoiser.sigma_data) ** 2

        loss = precond * (denoised_target - target).pow(2).flatten(1).mean(1)

        return loss 

    def __call__(self,
                 denoiser, 
                 loss_weight_fn,
                 batch_dict: dict[str, torch.Tensor],
                 sigma: torch.Tensor,
                 **model_kwargs) -> torch.Tensor:

        """Computes weighted loss for a batch of inputs."""

        loss = self.loss(denoiser, batch_dict, sigma, **model_kwargs)  # shape: [batch_size]

        weight = loss_weight_fn(sigma, denoiser.sigma_data)  # shape: [batch_size]

        return (loss * weight).mean()