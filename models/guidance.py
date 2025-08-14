import torch 


class Constraint(torch.nn.Module):
    """Constraint module for enforcing measurement constraints on model predictions.

    Args: 
        measurement: Measurement tensor for the guidance.   
        loss_fn: Loss function to use for the constraint.
        constraint_type: Type of constraint to apply (e.g., "mean", "sparse").
    """

    def __init__(self, measurement, loss_type, constraint_type):
        super().__init__()  
        self.measurement = measurement
        self.constraint_type = constraint_type
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

    def mean_constraint(self, prediction: torch.Tensor) -> torch.Tensor:
        """Computes the mean measurement constraint for guidance.

        Args:
            prediction: model state prediction with shape [batch size, variables, height, width]
        
        Returns:
            The computed mean constraint loss.
        """

        return self.loss(prediction.mean(), self.measurement)

    def sparse_constraint(self, prediction: torch.Tensor, time_index: int) -> torch.Tensor:
        """Computes sparse measurement constraint for guidance.

        Args:
            prediction: model state prediction with shape [batch size, variables, height, width]
            time_index: Time index for the sparse measurement frames.

        Returns:
            The computed sparse constraint loss.
        """

        measurement_slice = self.measurement[time_index].to(prediction.device)[None,None]

        if torch.isnan(measurement_slice).sum() > 0:
            prediction = prediction[~torch.isnan(measurement_slice)]
            measurement_slice = measurement_slice[~torch.isnan(measurement_slice)]

        return self.loss(prediction, measurement_slice)

    def forward(self, prediction: torch.Tensor, time_index: int = None) -> torch.Tensor:
        if self.constraint_type == "mean":
            return self.mean_constraint(prediction)
        elif self.constraint_type == "sparse":
            return self.sparse_constraint(prediction, time_index)


class Guidance:
    """Guidance class for computing the weighted gradient for the diffusion model.

    Args:
        measurement: Measurement tensor for the guidance.
        transforms: Transform object for applying inverse transformations.
        loss_type: Type of loss function to use (e.g., "mse", "l1").
        gamma: Weighting factor for the guidance.

    """
    def __init__(self, constraint, transforms, gamma=0.0):
        self.transforms = transforms
        self.gamma = gamma
        self.constraint = constraint


    def get_weighted_gradient(self, state, prediction, time_index=None, retain_graph=False):
        """Computes the weighted gradient of the denoised prediction with respect to the
        state of the diffusion process.

        Args:
            state: Current state of the diffusion process.
            prediction: Denoised prediction tensor from the diffusion model.        
            retain_graph: Whether to retain the computation graph for further backward passes.
        """

        prediction_no_trafo = self.transforms.apply_inverse_transforms(prediction)

        loss = self.constraint(prediction_no_trafo, time_index)

        grad = torch.autograd.grad(loss, state, retain_graph=retain_graph)[0]
        grad = torch.clamp(grad, min=-1e8, max=1e8)

        return self.gamma * grad

