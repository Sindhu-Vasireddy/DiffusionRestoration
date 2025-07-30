import os
import torch
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger

from models.edm.data import DataModule
from models.config import read_yaml, parse_filename
from models.edm.diffusion_model import DiffusionModel

def main(config, return_model=False):

    lightning.seed_everything(config["rng_seed"], workers=True)
    torch.set_float32_matmul_precision(config["float32_matmul_precision"])

    data_module = DataModule(config)

    model = DiffusionModel(config)

    callbacks = []

    if config["logger"] == "wandb":
        os.environ["WANDB_DATA_DIR"] = config["wandb"]["save_dir"]
        os.environ["WANDB_CACHE_DIR"] = config["wandb"]["save_dir"]
        logger = WandbLogger(**config["wandb"], config=config)
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"/{config['checkpoints']['save_dir']}/dm_{logger.experiment.id}",
            filename="best",
            monitor="validation/loss",
            mode="min",
        )
        callbacks.append(checkpoint_callback)
    else:
        logger = None

    if config["trainer"]["enable_progress_bar"]:
        progress_bar_callback = RichProgressBar(
                                theme=RichProgressBarTheme(
                                description="green",
                                progress_bar="green",
                                batch_progress="black",
                                time="black",
                                processing_speed="black",),
                                refresh_rate=50 if config["trainer"]["enable_progress_bar"] else 0, 
        )

        callbacks.append(progress_bar_callback)

    trainer = lightning.Trainer(
        **config["trainer"],
        callbacks=callbacks,
        logger=logger,
        deterministic=False,
    )

    if config["logger"] == "wandb":
        run = trainer.logger.experiment
        run.name = f"{config['wandb']['name']}_{run.id}"

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=config["checkpoints"]["resume_ckpt_path"],
    )

    if return_model:
        return model, data_module


if __name__ == "__main__":
    fname = parse_filename()
    config = read_yaml(fname)
    main(config)
