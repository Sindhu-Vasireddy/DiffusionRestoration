import sys
import numpy as np
import torch
#from tqdm.notebook import tqdm
from tqdm import tqdm
import xarray as xr
import pandas as pd
from pathlib import Path
import os
from typing import List, Iterable

from models.edm.diffusion_model import DiffusionModel
from models.edm.diffusion_stochastic_sampler import GuidedKarrasSampler
import models.xarray_utils as xu
from models.guidance import Guidance
from models.transforms import Transform

class DiffusionInference:
    """Handles inference using a saved diffusion model checkpoint.

    Args:
        transforms: Precomputed transforms for the dataset.
        config: Model configuration dictionary
        checkpoint_path: Path to model checkpoint .ckpt file.
        noise_shape: Shape of the noise tensor used for sampling.
    """
    def __init__(
        self,
        transforms: Transform,
        config: dict,
        checkpoint_path: str,
        noise_shape: tuple[int] = (1, 1, 180, 360),
    ):

        self.transforms = transforms
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.noise_shape = noise_shape
        self.init_latents = torch.randn(noise_shape)
        self.guidance = None

    def initialize_model(self):
        """Load diffusion model from checkpoint."""

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

    def initialize_guidance(self, measurement: torch.Tensor, gamma: float):
        """Initialize the guidance object for sampling."""

        self.guidance = Guidance(measurement=measurement,
                                 transforms=self.transforms,
                                 gamma=gamma,
                                 loss_type="mse")

    def rollout(
        self,
        sample_config,
        x_current: torch.Tensor,
        x_past: torch.Tensor,
    ):
        """Run the inference by rolling out the autoregressive diffusion model.

        Args:
            sample_config: Hyperparameter configuration for the sampling process.
            x_current: Initial condition of current state of the physical system.
            x_past: Initial condition of past state of the physical system.
        """

        # Initialize the sampler
        sampler = GuidedKarrasSampler(
                num_diffusion_steps=sample_config.num_diffusion_steps,
                denoiser=self.model,
                use_conditioning=sample_config.use_conditioning,
                guidance=self.guidance,
            )

        num_steps = range(sample_config.num_rollout_steps)
        if  sample_config.show_rollout_progress:   
            num_steps = self.progress_bar(num_steps, sample_config.num_rollout_steps)

        # Sample from the diffusion model autoregressively
        predictions = []
        for i in num_steps:
            prediction = sampler.sample(x_current=x_current,
                                        x_past=x_past,
                                        sample_index=i,
                                        show_progress=sample_config.show_progress)
            x_past = x_current
            x_current = prediction

            if self.noise_shape[0] > 1:
                predictions.append(prediction.unsqueeze(0).cpu())
            else:
                predictions.append(prediction.cpu())

            # free memory by writing predictions to disk
            if sample_config.flush_output_dir is not None and i % 365 == 0:
                predictions = self.flush_output(sample_config, i, predictions)

        # Post-process predictions
        self.predictions = torch.cat(predictions, dim=0)
        if sample_config.to_xarray:
            self.predictions = self.convert_to_xarray(self.predictions)
        if sample_config.to_physical:
            self.predictions = self.transforms.apply_inverse_transforms(self.predictions)

    def progress_bar(self, steps: Iterable, length: int) -> tqdm:
        """Create a progress bar for the sample count.
        Args:
            steps: Iterable of steps to iterate over.
            length: Total number of steps.
        Returns:
            num_steps: A tqdm progress bar object.
        """

        num_steps = tqdm(
                steps,
                total=length,
                desc=f"Sample count",
                dynamic_ncols=True,
                file=sys.stdout,
                #leave=False
            )
        return num_steps

    def flush_output(self, config, index: int, predictions: List[torch.Tensor]) -> List:
        """Save the output predictions to disk to save memory.
        
        Args:
            config: Configuration dictionary containing the output directory.
            index: Current index of the prediction.
            predictions: List of predictions to be saved.
        """

        if index == 0:
            path = get_unique_filename(config.flush_output_dir)
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

        predictions = torch.cat(predictions, dim=0)
        predictions = self.convert_to_xarray(predictions)
        predictions = self.transforms.apply_inverse_transforms(predictions)
        xu.write_dataset(predictions.to_dataset(name="output"),
                          f"{path}/output_year_{index:06d}.nc")
        predictions = []
        return predictions


    def convert_to_xarray(self, samples: torch.Tensor) -> xr.DataArray:
        """Covert samples to phyiscal space and xarray format.

        Args:
            samples: Samples from the diffusion model in physical space.

        Returns:
            samples: Samples in xarray format with time, latitude, and longitude dimensions.
        """

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
    """Generates a unique filename by appending a counter if the file already exists.
    """
    counter = 1
    base = save_path
    while os.path.exists(save_path):
        save_path = f"{base[:-1]}_run_{counter}"
        counter += 1
    return save_path