# Constrained sampling with generative diffusion models

A playground for including constraints at inference in to generative diffusion models.

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