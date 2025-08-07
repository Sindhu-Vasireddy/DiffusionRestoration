import torch
import xarray as xr
import numpy as np
import yaml
import pandas as pd
import os
from dataclasses import dataclass

from models.config import read_yaml
from models.transforms import Transform
from models.edm.diffusion_inference import DiffusionInference
import models.xarray_utils as xu


@dataclass
class SamplingConfig:
    """ Default sampling configuration """

    use_conditioning: bool = False
    use_past_state: bool = True

    num_rollout_steps: int = 3
    num_diffusion_steps: int = 100

    boosting: bool = False

    show_progress: bool = True
    show_rollout_progress: bool = False
    to_physical: bool = True
    to_xarray: bool = True

    flush_output_dir: str = None


class Simulation():
    """ Samples from guided, conditional and unconditional diffusion models. 
    
    Args:
        diffusion_config_path: Path to the diffusion model configuration file.
        diffusion_model_checkpoint_path: Path to the diffusion model checkpoint file.
        noise_shape: Shape of the noise tensor used for sampling.
        variable_name: Name of the variable in the dataset to be sampled.
    """
    
    def __init__(self,
                 diffusion_config_path: str,
                 diffusion_model_checkpoint_path: str,
                 noise_shape: tuple,
                 variable_name: str,
                 ):
        
        self.diffusion_config = read_yaml(diffusion_config_path)
        self.noise_shape = noise_shape
        self.variable_name = variable_name
        self.diffusion_checkpoint_path = diffusion_model_checkpoint_path
        self.parameters = {} 

    def load_data(self):
        """ Loads the data and precomputes training transforms and initial condition. """
        
        # get target data for inverse transforms
        target_data = xr.open_dataset(self.diffusion_config["dataset"]["dataset_path"])[self.variable_name]
        if self.diffusion_config["dataset"]["crop_lat_target"]:
            target_data = target_data[:,1:] 
            
        self.target_train = target_data.sel(time=slice(str(self.diffusion_config["dataset"]["train_start"]),
                                                  str(self.diffusion_config["dataset"]["train_end"])))
        
        self.transforms = Transform(self.target_train, self.diffusion_config["dataset"])
        self.target_test = target_data.sel(time=slice(str(self.diffusion_config["dataset"]["test_start"]),
                                                  str(self.diffusion_config["dataset"]["test_end"])))

        self.target_test_transformed = self.transforms.apply_transforms(self.target_test)

        self.x_past = self.prepare_state(self.target_test_transformed[1])
        self.x_current = self.prepare_state(self.target_test_transformed[2])

    def prepare_state(self, state, device="cuda"):
        """ Prepares the state for the diffusion model initialization. """
        state = torch.from_numpy(state.values).unsqueeze(0).unsqueeze(0).to(device)
        state = state.repeat(self.noise_shape[0], 1, 1, 1)
        return state    

    def initialize(self):
        """ Initializes data and diffusion model."""

        self.load_data()

        self.inference = DiffusionInference(
                        transforms=self.transforms,
                        config=self.diffusion_config,
                        checkpoint_path=self.diffusion_checkpoint_path,
                        noise_shape=self.noise_shape,
                   )

        self.inference.initialize_model()

    def run(self, sampling_config):
        """ Samples from the diffusion model. 

        Args:
            sampling_config: Sampling configuration object.
        """

        self.inference.rollout(
            sampling_config,
            x_current=self.x_current,
            x_past=self.x_past,
        )

        return self.inference.predictions


    def save_output(self, dataset:xr.DataArray, save_path: str, config=None):
        """ Saves the output dataset to disk."""

        if config is not None:
            dataset.attrs["guidance"] = yaml.dump(recursive_convert(config))
        dataset = dataset.to_dataset(name=self.variable_name)

        save_path = get_unique_filename(save_path) 

        xu.write_dataset(ds=dataset, file_name=save_path)


def get_unique_filename(save_path):
    """Generates a unique filename by appending a counter if the file already exists."""
    base, ext = os.path.splitext(save_path)
    counter = 1  
    
    while os.path.exists(save_path):
        save_path = f"{base}_run_{counter}{ext}"
        counter += 1
    
    return save_path

def recursive_convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: recursive_convert(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [recursive_convert(item) for item in obj]
    return obj