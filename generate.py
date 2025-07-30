from pathlib import Path
import numpy as np

from models.simulation import Simulation, SamplingConfig
from models.config import read_yaml, parse_filename

def main(config: dict):
    """Executes the sampling routines and writes results to disk."""

    print(config["experiment"])

    diffusion_config_path = (
        f"{config["paths"]["config_dir"]}/{config["paths"]["diffusion_config_fname"]}"
    )

    sim = Simulation(
        diffusion_config_path=diffusion_config_path,
        discriminator_model_id=config["models"]["discriminator_model_id"],
        diffusion_model_id=config["models"]["diffusion_model_id"],
        noise_shape=config["experiment"]["noise_shape"],
        variable_name=config["experiment"]["variable_name"],
    )

    sim.load_data()
    sim.load_models()

    sampling_config = SamplingConfig(**config["sampling"])

    if config["experiment"]["forecast"]:
        dataset = sim.forecast(sampling_config, num_forecasts=config["experiment"]["num_forecasts"])
    else:
        dataset = sim.run(sampling_config)

    save_path = f"{config["paths"]["output_dir"]}/{config["paths"]["output_fname"][:-3]}_length_{config["experiment"]["num_rollout_steps"]}_w_{config["experiment"]["guidance_weight"]}_c_{config["experiment"]["s_churn"]}.nc"
    sim.save_output(dataset=dataset,
                    save_path=save_path,
                    config={"models": config["models"]})

if __name__ == "__main__":
    fname = parse_filename()
    config = read_yaml(fname)
    main(config)
