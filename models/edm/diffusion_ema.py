import torch

@torch.inference_mode()
def ema_update(model, model_ema, decay):
    """Incorporates updated model parameters into an EMA model.

    Should be called after each optimizer step.
    """
    model_params = dict(model.named_parameters())
    averaged_params = dict(model_ema.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        param = param.to(averaged_params[name].device)
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(model_ema.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class EMAWarmupSchedule:
    """Implements an EMA warmup using an inverse decay schedule.

    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(
        self,
        inv_gamma=1.0,
        power=1.0,
        min_value=0.0,
        max_value=1.0,
        start_at=0,
        last_epoch=0,
    ):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_step = last_epoch

    def state_dict(self):
        """Returns the state of the class as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the class's state.
        Args:
            state_dict (dict): scaler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        """Gets the current EMA decay rate."""
        step = max(0, self.last_step - self.start_at)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return 0.0 if step < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_step += 1
