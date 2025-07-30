import torch

from models.edm.diffusion_utils import expand_dims


class KarrasODE(torch.nn.Module):
    """
    Implements Heun's 2nd order method from Karras et al. (2022), Algorithm 1.

    Reference: https://arxiv.org/abs/2206.00364.
    """

    def __init__(
        self,
        denoiser,
        guidance=None,
        diagnostics=None) -> None:
        super().__init__()

        self.denoiser = denoiser
        self.guidance = guidance
        self.diagnostics = diagnostics

    def forward(self, y0, x, c):
        return self.solve(x, y0, c)
    
    def x_to_sigma(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def sigma_to_x(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma 

    def solve(
        self,
        x: torch.Tensor,
        y0: torch.Tensor,
        c: torch.Tensor) -> torch.Tensor:

        """Integrates the specified ODE over x and returns the y trajectory."""

        # Apply Euler step with 2nd order correction.
        x_is, x_ip1s = x[:-1], x[1:]
        y = y0
        for i in range(x_is.shape[0]-1):
            x_i = x_is[i]
            x_ip1 = x_ip1s[i]
            y = self.euler_step_correct(x_i, x_ip1 - x_i, y, c)

        ## Special case the last step.
        x_i, x_ip1 = x[-2], x[-1]
        y = self.euler_step(x_i, x_ip1 - x_i, y, c)

        return y

    def euler_step_correct(self,
                           x: torch.Tensor,
                           dx: torch.Tensor,
                           y: torch.Tensor,
                           c: torch.Tensor) -> torch.Tensor:
        """Computes Euler step with 2nd order correction."""

        dy_dx = self.dy_dx(x, y, c)

        # Compute Euler step for each element of y_tuple.
        y_new = self.update_fn(y, dy_dx, dx)

        # Appy 2nd order correction.
        dy_dx_new = self.dy_dx(x + dx, y_new, c)

        y =  self.correction_fn(y, dy_dx, dy_dx_new, dx)

        return y

    def euler_step(self,
                   x: torch.Tensor,
                   dx: torch.Tensor,
                   y: torch.Tensor,
                   c: torch.Tensor) -> torch.Tensor:
        """Computes Euler step."""

        dy_dx = self.dy_dx(x, y, c)
        y_new = self.update_fn(y, dy_dx, dx)

        return y_new

    def update_fn(self, y, dy_dx, dx):
        return y + dy_dx * dx

    def correction_fn(self, y, dy_dx, dy_dx_new, dx):
        return y + 0.5 * (dy_dx + dy_dx_new) * dx

    def dy_dx(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Computes dy/dx for the specified x and y, where x is supposed to be time."""

        batch_size = y.shape[0]

        # Compute sigma and d sigma / dt for each time point in the batch.
        # Assume sigma(t) = t, i.e., d sigma / dt = 1.
        sigma = expand_dims(x.repeat(batch_size), y.ndim)  # shape: [batch_size, 1, ...]
        dsigma_dt = 1.0

        # Compute dy/dx.
        dy_dx = (dsigma_dt / sigma) * (y - self.denoiser(y, sigma.squeeze(), conditioning=c))

        return dy_dx