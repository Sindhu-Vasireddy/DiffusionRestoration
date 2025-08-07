import torch 

class Guidance:
    """Guidance class for computing the weighted gradient for the diffusion model.

    Args:
        measurement: Measurement tensor for the guidance.
        transforms: Transform object for applying inverse transformations.
        loss_type: Type of loss function to use (e.g., "mse", "l1").
        gamma: Weighting factor for the guidance.

    """
    def __init__(self, measurement, transforms, loss_type, gamma=0.0):
        self.transforms = transforms
        self.gamma = gamma
        self.measurement = measurement 
        self.loss = self.get_loss_fn(loss_type)

    def get_loss_fn(self, type: str):
        """Returns the loss function based on the specified type.

        Args:
            type: Type of loss function to return (e.g., "mse", "l1").
        """

        if type == "mse":
            return torch.nn.functional.mse_loss
        elif type == "l1":
            return torch.nn.functional.l1_loss

    def mean_constraint(self, prediction:torch.Tensor) -> torch.Tensor:
        """Computes a mean constraint and returns the loss.

        Args:
            prediction: Denoised prediction tensor from the diffusion model.
            Should be in the same units as the measurement tensor.
        """

        loss = self.loss(prediction.mean()[None], self.measurement.double())

        return loss

    def get_weighted_gradient(self, state, prediction, retain_graph=False):
        """Computes the weighted gradient of the denoised prediction with respect to the
        state of the diffusion process.

        Args:
            state: Current state of the diffusion process.
            prediction: Denoised prediction tensor from the diffusion model.        
            retain_graph: Whether to retain the computation graph for further backward passes.
        """

        prediction_no_trafo = self.transforms.apply_inverse_transforms(prediction)

        loss = self.mean_constraint(prediction_no_trafo)

        grad = torch.autograd.grad(loss, state, retain_graph=retain_graph)[0]

        return self.gamma * grad

