import torch
from tqdm import tqdm
import numpy as np


class KarrasStochasticSampler(torch.nn.Module):
    """
    Implements the stochastic sampler from Karras et al. (2022), Algorithm 2.

    Reference: https://arxiv.org/abs/2206.00364.
    https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/example.py#L36
    """

    def __init__(
        self,
        denoiser,
        use_conditioning=True,
        boosting=False,
        num_diffusion_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=4,
        S_min=0.0,
        S_max=np.float64("inf"),
        S_noise=1.00,
    ) -> None:
        super().__init__()

        self.denoiser = denoiser
        self.use_conditioning = use_conditioning
        self.boosting = boosting
        self.num_diffusion_steps = num_diffusion_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.one = torch.ones(1).to(self.denoiser.device)

    def sample(self, x_current=None, x_past=None, index=0, show_progress=True):
        """Samples the diffusion model using conditioning or discriminator guidance.

        Args:
            x_init: Fields from the previous time step used as conditioning.
            index: Number of smaple used in the progress bar.
        """

        # Adjust noise levels based on what's supported by the network.
        sigma_min = self.sigma_min
        sigma_max = self.sigma_max

        latents = torch.randn_like(self.init_latents)

        # Time step discretization.
        step_indices = torch.arange(
            self.num_diffusion_steps, dtype=torch.float64, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_diffusion_steps - 1)
            * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat(
            [self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        if self.use_conditioning:
            conditioning = torch.cat([x_past, x_current], dim=1)
        else:
            conditioning = None

        ## Settings for boosting
        S_churn_manual = 4.0
        S_noise_manual = 1.000
        period = 5

        log_ratio = torch.tensor([np.inf] * latents.shape[0], device=latents.device)
        S_churn_vec = torch.tensor(
            [self.S_churn] * latents.shape[0], device=latents.device
        )
        S_churn_max = torch.tensor(
            [np.sqrt(2) - 1] * latents.shape[0], device=latents.device
        )
        S_noise_vec = torch.tensor(
            [self.S_noise] * latents.shape[0], device=latents.device
        )

        x_next = latents.to(torch.float64) * t_steps[0]

        integration_steps = zip(t_steps[:-1], t_steps[1:])

        if show_progress:   
            integration_steps = tqdm(
                integration_steps,
                total=len(t_steps[1:]),
                desc=f"Generating sample {index}",
            )

        for i, (t_cur, t_next) in enumerate(integration_steps): 
            x_cur = x_next

            S_churn_vec_ = S_churn_vec.clone()
            S_noise_vec_ = S_noise_vec.clone()

            if i % period == 0:
                if self.boosting:
                    S_churn_vec_[log_ratio < 0.0] = S_churn_manual
                    S_noise_vec_[log_ratio < 0.0] = S_noise_manual

            # Increase noise temporarily.
            if self.S_min <= t_cur <= self.S_max:
                gamma_vec = torch.minimum(
                    S_churn_vec_ / self.num_diffusion_steps, S_churn_max
                )
            else:
                gamma_vec = torch.zeros_like(S_churn_vec_)

            t_hat = self.round_sigma(t_cur + gamma_vec * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt()[
                :, None, None, None
            ] * S_noise_vec_[:, None, None, None] * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.denoiser(
                x=x_hat, sigma=t_hat, conditioning=conditioning
            ).to(torch.float64)

            d_cur = (x_hat - denoised) / t_hat[:, None, None, None] 
          
            x_next = x_hat + (t_next - t_hat)[:, None, None, None] * d_cur

            # Apply 2nd order correction.
            if i < self.num_diffusion_steps - 1:
                denoised = self.denoiser(
                    x=x_next, sigma=t_next * self.one, conditioning=conditioning
                ).to(torch.float64)

                d_prime = (x_next - denoised) / t_next

                x_next = x_hat + (t_next - t_hat)[:, None, None, None] * (
                    0.5 * d_cur + 0.5 * d_prime
                )

        return x_next

    def round_sigma(self, sigma):
        # return torch.tensor([sigma])
        return sigma

