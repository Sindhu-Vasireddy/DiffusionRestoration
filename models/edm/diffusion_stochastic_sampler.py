import sys
import torch
from tqdm.notebook import tqdm


class GuidedKarrasSampler(torch.nn.Module):
    """ Guided Karras Stochastic Sampler for diffusion models using Tweedies formular.

    Adapted from references:
    Yao and Mammadov et al. 2025: https://github.com/neuraloperator/FunDPS/blob/main/generation/dps.py
    Chung et al., 2023: https://github.com/DPS2022/diffusion-posterior-sampling
    """

    def __init__(
        self,
        denoiser,
        use_conditioning=True,
        guidance=None,
        num_diffusion_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.use_conditioning = use_conditioning
        self.guidance = guidance
        self.num_diffusion_steps = num_diffusion_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.one = torch.ones(1).to(self.denoiser.device)
        self.device = denoiser.device


    def sample(self, x_current, x_past, index=0, show_progress=True):
        """Generate a single batch of samples."""

        # Initialize latents
        latents = torch.randn_like(x_current)

        # Set up sigma schedule
        self.sigma_t_steps = self.get_sigma_schedule()

        # denoiser inputs
        x_next = latents.to(torch.float64) * self.sigma_t_steps[0]
        conditioning = torch.cat([x_past, x_current], dim=1)

        integration_steps = zip(self.sigma_t_steps[:-1], self.sigma_t_steps[1:])
        if show_progress:
            integration_steps = self.progress_bar(integration_steps, self.num_diffusion_steps, index)

        for i, (sigma_t_cur, sigma_t_next) in enumerate(integration_steps):
            x_cur = x_next.detach().clone()
            x_cur.requires_grad_(True)
            sigma_t = self.round_sigma(sigma_t_cur)

            # Euler step
            x_N = self.denoiser(x=x_cur, sigma=sigma_t, conditioning=conditioning).to(torch.float64)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

            # 2nd order correction
            if i < self.num_diffusion_steps - 1:
                x_N = self.denoiser(x=x_next, sigma=sigma_t_next, conditioning=conditioning).to(torch.float64)
                d_prime = (x_next - x_N) / sigma_t_next
                x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

            # Apply guidance
            if self.guidance is not None:
                grad = self.guidance.get_weighted_gradient(x_cur, x_N, retain_graph=False)
                x_next = x_next - self.sigma_t_steps[i].item() * grad

            if x_next.isnan().any():
                print(f"\nStep {i}: NaN detected!")
                break

        x_final = x_next.detach()
        return x_final
    
    def get_sigma_schedule(self):
        """Returns a tensor of sigma steps for the diffusion process."""
        step_indices = torch.arange(self.num_diffusion_steps, dtype=torch.float64, device=self.device)
        sigma_t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_diffusion_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        sigma_t_steps = torch.cat([self.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])
        return sigma_t_steps

    def progress_bar(self, steps, length, index):
        """Creates a progress bar for the sampling process."""
        integration_steps = tqdm(
            steps,
            total=length,
            desc=f"Generating sample {index}",
            dynamic_ncols=True,
            file=sys.stdout,
            leave=False
        )
        return integration_steps


    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
