import sys
import numpy as np
import torch
from tqdm.notebook import tqdm
import xarray as xr
import pandas as pd
from pathlib import Path
import os

from models.edm.diffusion_model import DiffusionModel
from models.edm.diffusion_stochastic_sampler import GuidedKarrasSampler
import models.xarray_utils as xu
from models.guidance import Guidance

class DiffusionInference:
    def __init__(
        self,
        transforms,
        config,
        checkpoint_path,
        diagnostics=None,
        noise_shape=(1, 1, 180, 360),
    ):
        """Performs inference with a saved diffusion model checkpoint.
        Args:
            config: Model configuration dictionionary
            checkpoint_path: Path to model checkpoint
        """

        self.transforms = transforms
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.diagnostics = diagnostics
        self.noise_shape = noise_shape
        self.init_latents = torch.randn(noise_shape)

    def get_model(self):
        """Load model from checkpoint."""

        checkpoint = torch.load(self.checkpoint_path)

        for param in self.config["diffusion_network"].keys():
            if param in checkpoint["hyper_parameters"]["diffusion_network"]:
                self.config["diffusion_network"][param] = checkpoint["hyper_parameters"]["diffusion_network"][param]

        for param in self.config["diffusion"].keys():
            if param in checkpoint["hyper_parameters"]["diffusion"]:
                self.config["diffusion"][param] = checkpoint["hyper_parameters"]["diffusion"][param]

        self.model = DiffusionModel.load_from_checkpoint(
            self.checkpoint_path, config=self.config
        )
        self.model.model_ema.load_state_dict(checkpoint["ema_state_dict"])
        self.model.to("cuda")

    def rollout(
        self,
        config,
        x_current,
        x_past,
    ):
        """Run the inference.

        Args:
            inference_n_steps: Number of ODE integration steps for sampling.
            num_batches: Number of batches to sample
        """
        guidance = Guidance(measurement=torch.tensor([4.0],
                            device=x_current.device),
                            transforms=self.transforms,
                            gamma=config.gamma,
                            loss_type="mse")

        sampler = GuidedKarrasSampler(
                num_diffusion_steps=config.num_diffusion_steps,
                denoiser=self.model,
                use_conditioning=config.use_conditioning,
                guidance=guidance,
            )


        num_steps = range(config.num_rollout_steps)
        if  config.show_rollout_progress:   
            i = 0
            num_steps = self.progress_bar(num_steps, config.num_rollout_steps, i)

        predictions = []
        for i in num_steps:
            prediction = sampler.sample(x_current=x_current,
                                        x_past=x_past,
                                        index=i+1,
                                        show_progress=config.show_progress)
            x_past = x_current
            x_current = prediction

            if self.noise_shape[0] > 1:
                predictions.append(prediction.unsqueeze(0).cpu())
            else:
                predictions.append(prediction.cpu())

            # free memory by writing predictions to disk
            if config.flush_output_dir is not None and i % 365 == 0:
                predictions = self.flush_output(config, i, predictions)

        self.predictions = torch.cat(predictions, dim=0)
        if config.to_xarray:
            self.predictions = self.convert_to_xarray(self.predictions)
        if config.to_physical:
            self.predictions = self.transforms.apply_inverse_transforms(self.predictions)

    def progress_bar(self, steps, length, index):
        num_steps = tqdm(
                steps,
                total=length,
                desc=f"Sample {index}",
                dynamic_ncols=True,
                file=sys.stdout,
                leave=False
            )
        return num_steps
    
    def flush_output(self, config, index, predictions):

        if index == 0:
            path = get_unique_filename(config.flush_output_dir)
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

        predictions = torch.cat(predictions, dim=0)
        predictions = self.convert_to_xarray(predictions)
        predictions = self.transforms.apply_inverse_transforms(predictions)
        xu.write_dataset(predictions.to_dataset(name="output"),
                          f"{path}/output_year_{index:06d}.nc")
        return predictions


    def convert_to_xarray(self, samples: torch.Tensor) -> xr.DataArray:
        """Covert samples to phyiscal space and xarray format."""

        samples = samples.numpy()
        lats = self.transforms.target_data.latitude
        lons = self.transforms.target_data.longitude

        start_date = f'{self.config["dataset"]["test_start"]}-01-01'
        num_days = len(samples)
        time = pd.date_range(start=start_date, periods=num_days, freq="D")

        if self.noise_shape[0] > 1:
            samples = xr.DataArray(
                data=samples[:, :, 0],
                dims=["time", "member", "latitude", "longitude"],
                coords=dict(
                    time=time,
                    member=np.arange(self.noise_shape[0]),
                    latitude=lats,
                    longitude=lons,
                ),
            )
        else:
            samples = xr.DataArray(
                data=samples[:, 0],
                dims=["time", "latitude", "longitude"],
                coords=dict(
                    time=time,
                    latitude=lats,
                    longitude=lons,
                ),
            )

        return samples
        

def get_unique_filename(save_path):
    counter = 1  
    base = save_path
    while os.path.exists(save_path):
        save_path = f"{base[:-1]}_run_{counter}"
        counter += 1
    return save_path