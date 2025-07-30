import torch
import xarray as xr
import numpy as np
import yaml
import pandas as pd
import os
from dataclasses import dataclass

from src.config import read_yaml
from src.transforms import Transform
from src.edm.inference import DiffusionInference
from src.discriminator.guidance import DiscriminatorGuidance
from src.discriminator.diagnostics import Diagnostics
import src.xarray_utils as xu


@dataclass
class SamplingConfig:
    """ Default sampling configuration """

    sampler_type: str = "sde"
    use_conditioning: bool = False
    use_past_state: bool = False
    use_past_2_state: bool = False
    num_rollout_steps: int = 3
    num_diffusion_steps: int = 100
    s_noise: float = 1.0
    s_churn: float = 2.5
    s_min: float = 0.0
    s_max: float = 100.0
    epsilon_scaling: float = 1.0
    t_max: float = 1.0

    noise_delta: float = 0.0
    guidance_weight: float = 300
    t_min: float = 0.0
    boosting: bool = False
    guidance_sde_type: str = "edm"

    show_progress: bool = True
    show_rollout_progress: bool = False
    to_physical: bool = True
    to_xarray: bool = True
    flush_output_dir: str = None
    use_diagnostic: bool = False
    initial_condition_noise: float = None
    use_constraint: bool = False
    stabilization_parameter: float = 0.0


class Simulation():
    """ Samples from guided, conditional and unconditional diffusion models. """
    
    def __init__(self,
                 diffusion_config_path: str,
                 diffusion_model_id: str,
                 noise_shape: tuple,
                 variable_name: str,
                 discriminator_model_id: str = None,
                 ):
        
        self.diffusion_config = read_yaml(diffusion_config_path)
        self.noise_shape = noise_shape
        self.variable_name = variable_name
        if discriminator_model_id is not None:
            self.discriminator_checkpoint_path = f"/p/tmp/hess/scratch/discriminator-guidance/checkpoints/dis_{discriminator_model_id}/best.ckpt"
        else:
            self.discriminator_checkpoint_path = None
        self.diffusion_checkpoint_path=f"/p/tmp/hess/scratch/discriminator-guidance/checkpoints/dm_{diffusion_model_id}/best.ckpt"
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

        self.x_old_old = self.prepare_state(self.target_test_transformed[0])
        self.x_old = self.prepare_state(self.target_test_transformed[1])
        self.x_current = self.prepare_state(self.target_test_transformed[2])

    def prepare_state(self, state):
        """ Prepares the state for the diffusion model initialization. """
        state = torch.from_numpy(state.values).unsqueeze(0).unsqueeze(0).cuda()
        state = state.repeat(self.noise_shape[0], 1, 1, 1)
        return state    

    def load_models(self):
        """ Loads the discriminator and diffusion model. """

        if self.discriminator_checkpoint_path is not None: 
            self.guidance = DiscriminatorGuidance(
                                self.discriminator_checkpoint_path,
                                device="cuda"
                            )
        else:
            self.guidance = None



        self.inference = DiffusionInference(
                        transforms=self.transforms,
                        config=self.diffusion_config,
                        checkpoint_path=self.diffusion_checkpoint_path,
                        guidance=self.guidance,
                        diagnostics=Diagnostics(),
                        noise_shape=self.noise_shape,
                   )

        self.inference.get_model()
    
    def show_model_config(self):
        """ Prints the discriminator model hyperparameters. """
        df = pd.DataFrame(self.guidance.config.items(), columns=['Parameter', 'Value'])
        print(df)
    
    def show_parameter_count(self):
        """ Prints the number of parameters in the discriminator and diffusion model. """

        model = self.guidance.model
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Discriminator model parameter count: {num_trainable_params:,}")

        model = self.inference.model
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Diffusion model parameter count: {num_trainable_params:,}")

    def run(self, sampling_config):
        
        """ Samples from the diffusion model. """

        if self.inference.guidance is not None: 
            self.inference.guidance.guidance_sde_type = sampling_config.guidance_sde_type 

        self.inference.rollout(
            sampling_config,
            x_current=self.x_current,
            x_old=self.x_old,
            x_old_old=self.x_old_old if sampling_config.use_past_2_state else None,
        )

        return self.inference.predictions

    def forecast(
            self,
            sampling_config,
            num_forecasts=1,
        ):

        if self.inference.guidance is not None: 
            self.inference.guidance.guidance_sde_type = sampling_config.guidance_sde_type 

        forecasts = []
        for i in range(num_forecasts):

            x_old_old = self.prepare_state(self.target_test_transformed[i])
            x_old = self.prepare_state(self.target_test_transformed[i+1])
            x_current = self.prepare_state(self.target_test_transformed[i+2])

            if sampling_config.initial_condition_noise is not None:
                if sampling_config.use_past_2_state:
                    x_old_old += torch.randn_like(x_old_old) * sampling_config.initial_condition_noise
                x_old += torch.randn_like(x_old) * sampling_config.initial_condition_noise
                x_current += torch.randn_like(x_current) * sampling_config.initial_condition_noise

            self.inference.rollout(
                sampling_config,
                x_current=x_current,
                x_old=x_old,
                x_old_old=x_old_old if sampling_config.use_past_2_state else None,
            )

            forecast = self.inference.predictions[:]
            forecasts.append(forecast)
        forecasts = xr.concat(forecasts, dim="forecast_index")
        return forecasts

    def save_output(self, dataset:xr.DataArray, save_path: str, config=None):
        #dataset = self.inference.predictions
        if config is not None:
            dataset.attrs["guidance"] = yaml.dump(recursive_convert(config))
        dataset = dataset.to_dataset(name=self.variable_name)

        save_path = get_unique_filename(save_path) 

        xu.write_dataset(ds=dataset,
                         file_name=save_path)

def get_unique_filename(save_path):
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