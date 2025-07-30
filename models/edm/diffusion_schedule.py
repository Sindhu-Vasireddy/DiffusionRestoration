import torch
#import abc
import math

from models.edm.diffusion_utils import _append_zero, DiffEqDomain


class KarrasNoiseSchedule():
    """Specifies noise schedule proposed by Karras et al. (2022).

    The schedule is defined in terms of sigma (Eq. 5 in the paper):
        sigma_i = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho, i=0,...,n-1,
        sigma_n = 0.

    Reference: https://arxiv.org/abs/2206.00364.
    """

    def __init__(self, sigma_data: float, sigma_min: float, sigma_max: float, rho: float = 7.0):
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        # Precompute some constants.
        self.sigma_min_inv_rho = self.sigma_min ** (1 / self.rho)
        self.sigma_max_inv_rho = self.sigma_max ** (1 / self.rho)


    def sigma_fn(self, t: torch.Tensor) -> torch.Tensor:
        """Defines element-wise function sigma(t) = t."""
        return t


    def get_sigma_schedule(self, n_steps) -> tuple[torch.Tensor, torch.Tensor]:
        """Rerturns a tensor of sigma steps."""

        if type(n_steps) == torch.Tensor:
            n_steps = n_steps.item()

        steps = torch.linspace(0, 1, n_steps)
        sigma = (
            self.sigma_max_inv_rho + steps * (self.sigma_min_inv_rho - self.sigma_max_inv_rho)
        ) ** self.rho
        sigma = _append_zero(sigma)

        return sigma, sigma[0]


    def get_t_schedule(self, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a tensor of time steps calculated as t = sigma_inv(sigma)."""
        return self.get_sigma_schedule(n_steps)


    def get_logsnr_schedule(self, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Rerturns a tensor of log-SNR steps computed from sigma."""

        sigma, sigma0 = self.get_sigma_schedule(n_steps)
        sigma[-1] = 1e-8  # Avoid division by zero.

        return 2 * torch.log(self.sigma_data / sigma), sigma0


    def get_x_schedule(self, n_steps, domain: DiffEqDomain) -> tuple[torch.Tensor, torch.Tensor]:
        if domain == DiffEqDomain.TIME:
            return self.get_t_schedule(n_steps)
        elif domain == DiffEqDomain.SIGMA:
            return self.get_sigma_schedule(n_steps)
        elif domain == DiffEqDomain.LOGSNR:
            return self.get_logsnr_schedule(n_steps)
        else:
            raise ValueError(f"Unsupported domain: {domain}.")


    @torch.jit.ignore
    def compute_prior_logp(self, y: torch.Tensor) -> torch.Tensor:
        """Computes the prior log-probability of the specified y."""

        batch_size = y.shape[0]
        sigma0 = self.get_sigma_schedule(1, device=y.device)[1]
        log_prob_per_dim = (
            -0.5 * (y / sigma0) ** 2 - torch.log(sigma0) - 0.5 * math.log(2 * torch.pi)
        )
        return torch.sum(log_prob_per_dim.view(batch_size, -1), dim=1)

