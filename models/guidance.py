import torch 

def l2_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum((prediction - target)**2, dim=(-2, -1))).mean()

class Guidance:
    """Guidance class for computing the weighted gradient for the diffusion model."""
    def __init__(self, measurement, transforms, loss_type, gamma=0.0):
        self.transforms = transforms
        self.gamma = gamma
        self.measurement = measurement 
        self.loss = self.get_loss_fn(loss_type)

    def get_loss_fn(self, type: str):
        """Returns the loss function based on the specified type."""
        if type == "mse":
            return torch.nn.functional.mse_loss
        elif type == "l1":
            return torch.nn.functional.l1_loss
        elif type == "l2":
            return l2_loss

    def get_weighted_gradient(self, state, prediction, retain_graph=False):
        """Computes the weighted gradient of the denoised prediction with respect to the
        state of the diffusion process."""

        denorm_prediction = self.transforms.apply_inverse_transforms(prediction)
        loss = self.loss(denorm_prediction.mean()[None], self.measurement.double())
        grad = torch.autograd.grad(loss, state, retain_graph=retain_graph)[0]

        return (grad * self.gamma)