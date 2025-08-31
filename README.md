# [ELLIS Summer School 2025](https://www.ellis-jena.eu/summer-school-2025/): AI for Earth and Climate Sciences <br><br> Research Challenge: <br><br> *Physically Consistent Reconstruction of Spatiotemporal Dynamics with Generative Diffusion Models*

This repository serves as a starting point and playground for solving inverse problems with diffusion models in a physically consistent manner.

## Installation

Install depedencies with

```bash
conda env create -f environment.yml
```

## Model training

To train a diffusion model from scratch, edit the configuration file in `config/` accordingly. Then execute

```bash
python training.py -c config/config_file.yml
```

## Model inference and sampling

See the Jyputer notebook [Constrained-vorticity-sampling.ipynb](notebooks/Constrained-vorticity-sampling.ipynb) for examples.
