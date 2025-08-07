import copy
import functools
import math
import sys

import matplotlib.pyplot as plt
import lightning
import torch
import wandb
import numpy as np

from models.edm.diffusion_ema import ema_update, EMAWarmupSchedule
from models.edm.diffusion_ode import KarrasODE
from models.edm.diffusion_loss import KarrasLoss
from models.edm.diffusion_schedule import KarrasNoiseSchedule
from models.edm.diffusion_denoiser import KarrasDenoiser
from models.edm.diffusion_unet_periodic import PeriodicSongUNet
from models.edm.diffusion_unet import SongUNet


class DiffusionModel(lightning.LightningModule):
    """A Pytorch Lightning module for denoising diffusion training and inference.

    Args:
        model: A denoising model.
        training_config: An optional dataclass that contains configuration for training.
        inference_config: An optional dataclass that contains configuration for inference.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.save_hyperparameters({**config})


        net = PeriodicSongUNet(img_resolution=60, **self.config["diffusion_network"])
        self.model = KarrasDenoiser(model=net, sigma_data=config["diffusion"]["sigma_data"]) 
        self.model_ema = copy.deepcopy(self.model).eval().requires_grad_(False)

        # training config
        self.setup_train_loss()
        self.sigma_sampler = LogNormalNoiseSampler(loc=-1.2, scale=1.2)
        self.ema_schedule = EMAWarmupSchedule(inv_gamma=1.0, power=0.6, max_value=0.9999)

        # Save validation parameters.
        self.val_loss_weight_fn = _loss_weighting_uniform
        self.validation_sigmas = [config["diffusion"]["sigma_data"]]
        self.validation_optimal_denoiser = None

        # Initialize cache for optimal validation losses.
        self._validation_optimal_loss_cache = {}
        self.validation_step_data = None

        # (optional) inference config:
        self.setup_inference(sigma_data=config["diffusion"]["sigma_data"])

    def setup_train_loss(self):

        self.loss_fn = KarrasLoss()
        weight_config = self.config["diffusion"]
        if weight_config["loss_weights"] == 'uniform':
            self.train_loss_weight_fn = _loss_weighting_uniform
        elif weight_config["loss_weights"] == 'snr':
            self.train_loss_weight_fn = _loss_weighting_snr
        elif weight_config["loss_weights"] == 'soft_min_snr':
            self.train_loss_weight_fn = _loss_weighting_soft_min_snr
        elif weight_config["loss_weights"] == 'soft_min_snr_gamma':
            self.train_loss_weight_fn = _loss_weighting_min_snr_gamma
        else:
            raise Exception(f'{weight_config} is not a valid option. Choose: uniform, soft_min_sr or soft_min_sr_gamma')

    def setup_inference(self,
                        inference_n_steps=50,
                        sigma_data=0.5,
                        sigma_min=0.002,
                        sigma_max=80.0,
                        rho=7.0,
                        guidance=None,
                        diagnostics=None
                        ):

        self.inference_n_steps = inference_n_steps
        self.inference_return_trajectory = False
        self.guidance = guidance
        self.diagnostics = diagnostics

        self.inference_ode = KarrasODE

        self.inference_noise_schedule =  KarrasNoiseSchedule(sigma_data=sigma_data,
                                                             sigma_max=sigma_max,
                                                             sigma_min=sigma_min,
                                                             rho=rho).get_sigma_schedule
 

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, conditioning=None) -> torch.Tensor:
        return self.model_ema(x, sigma, conditioning=conditioning)

    def configure_optimizers(self):

        # Save optimization parameters.
        self._optimizer_builder = functools.partial(
            torch.optim.AdamW, **{"lr":  self.config["scheduler"]["learning_rate"]}
        )

        optimizer = self._optimizer_builder(self.parameters())

        self._lr_scheduler_builder = functools.partial(
            torch.optim.lr_scheduler.ReduceLROnPlateau, **({"mode": "min",
                                                            "factor": 0.5,
                                                            "patience": 5,
                                                            "min_lr": 1e-7}))

        lr_scheduler = self._lr_scheduler_builder(optimizer)

        if lr_scheduler is None:
            return optimizer

        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                    "monitor": "validation/loss",
                },
            }

    def optimizer_step(self, *args, **kwargs):
        """Updates model parameters and EMA model parameters."""
        super().optimizer_step(*args, **kwargs)

        # Remove NaNs from gradients.
        for param in self.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        # Update EMA model.
        ema_decay = self.ema_schedule.get_value()
        ema_update(self.model, self.model_ema, ema_decay)
        self.ema_schedule.step()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.model_ema.state_dict()

    def training_step(self, batch_dict: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Samples noise level and computes loss."""

        del batch_idx  # Unused.

        batch_size = batch_dict["target"].shape[0]

        sigma = self.sigma_sampler(batch_size, device=batch_dict["target"].device)

        loss = self.loss_fn(self.model, self.train_loss_weight_fn, batch_dict, sigma)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch_dict: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Computes and logs validation metrics."""

        self.validation_step_data =  batch_dict
        
        if self.validation_sigmas is not None:
        
            total_loss = 0.0
            total_loss_ema = 0.0
        
            batch_size = batch_dict["target"].shape[0]
            sigma = torch.empty((batch_size,), device=batch_dict["target"].device)
        
            for sigma_value in self.validation_sigmas:
        
                sigma.fill_(sigma_value)
        
                loss = self.loss_fn(self.model, self.val_loss_weight_fn, batch_dict, sigma)
                loss_ema = self.loss_fn(self.model_ema, self.val_loss_weight_fn, batch_dict, sigma)
        
                if self.validation_optimal_denoiser is not None:
        
                    optimal_loss_idx = (batch_idx, sigma_value)
                    if optimal_loss_idx not in self._validation_optimal_loss_cache:
        
                        self._validation_optimal_loss_cache[optimal_loss_idx] = self.loss_fn(self.validation_optimal_denoiser,
                                                                                             self.val_loss_weight_fn,
                                                                                             batch_dict,
                                                                                             sigma)
        
                    loss -= self._validation_optimal_loss_cache[optimal_loss_idx]
                    loss_ema -= self._validation_optimal_loss_cache[optimal_loss_idx]
        
                self.log(f"validation/loss/sigma_{sigma_value:.1e}", loss, on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"validation/loss_ema/sigma_{sigma_value:.1e}", loss_ema, on_step=False, on_epoch=True, sync_dist=True)
        
                total_loss += loss
                total_loss_ema += loss_ema
        
            total_loss /= len(self.validation_sigmas)
            total_loss_ema /= len(self.validation_sigmas)

            self.log("validation/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("validation/loss_ema", total_loss_ema, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)


    def on_validation_epoch_end(self):

        if self.trainer.sanity_checking:  # optional skip
            return

        if self.config["plot_validation_samples"]:
            batch = self.validation_step_data
            x = batch["target"]
            pred = self.predict(batch)

            fig, ax = plt.subplots(1,3,figsize=(15, 6), constrained_layout=True)

            n_rows = 1
            n_samples = 2

            i = np.random.randint(len(x))
            ax[0].set_title(f'target: ep={self.current_epoch} | m={x[i,0].mean(): 2.2e}, s={x[i,0].std(): 2.2e}')
            ax[0].imshow(x[i,0].cpu())
            ax[0].set_axis_off()

            for j in range(n_samples):
                ax[j+1].set_title(f'prediction m={pred[j,0].mean(): 2.2e}, s={pred[j,0].std(): 2.2e}')
                ax[j+1].imshow(pred[j,0].cpu())
                ax[j+1].set_axis_off()

            wandb.log({"val_plot": wandb.Image(plt)})

            if self.config["plot_validation_samples"]:
                plt.show()

            plt.close()


    def predict(self,batch_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generates samples given a batch of noise and conditional inputs."""

        if self.diagnostics is not None:
            self.diagnostics.initialize() 

        with torch.no_grad():
            x, sigma0 = self.inference_noise_schedule(self.inference_n_steps)
            x = x.to(self.device)
            sigma0 = sigma0.to(self.device)

            y0 = sigma0 * batch_dict["noise"]
            ode = self.inference_ode(self.model, self.guidance, self.diagnostics) 
            if "condition" in batch_dict.keys():
                c = batch_dict["condition"]
            else:
                c = None
            y = ode(y0=y0, x=x, c=c)

        return y


def _loss_weighting_uniform(sigma, sigma_data):
    """Uniform weighting scheme that assigns equal weights to all noise levels."""

    del sigma_data  # Unused.
    return torch.ones_like(sigma)


def _loss_weighting_snr(sigma, sigma_data):
    """Weighting function that assigns weights proportional to the signal-to-noise ratio."""

    return (sigma_data / sigma) ** 2


def _loss_weighting_min_snr_gamma(sigma, sigma_data, gamma=5.0):
    """Weighting function based on the min-SNR-gamma weighting scheme from Hang et al. (2022).

    Reference: https://arxiv.org/abs/2303.09556.
    """

    snr = (sigma_data / sigma) ** 2
    return torch.minimum(snr, torch.ones_like(snr) * gamma)


def _loss_weighting_soft_min_snr(sigma, sigma_data):
    """Weighting function based on the soft-min-SNR: 4 * SNR / (1 + SNR) ** 2."""

    snr = (sigma_data / sigma) ** 2
    return 4 * snr / (1 + snr) ** 2


class LogUniformSigmaSampler():
    """Samples noise levels from a log-uniform distribution."""

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, batch_size: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """Generates a batch of sigmas."""

        rand_tensor = torch.rand(batch_size, device=device, dtype=dtype)
        log_min_value, log_max_value = math.log(self.min_value), math.log(self.max_value)

        return torch.exp(log_min_value + rand_tensor * (log_max_value - log_min_value))


class LogNormalNoiseSampler():
    """Samples noise levels from a log-normal distribution."""

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __call__(self, batch_size: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """Generates a batch of noise samples."""

        rand_tensor = torch.randn(batch_size, device=device, dtype=dtype)
        return torch.exp(self.loc + rand_tensor * self.scale)