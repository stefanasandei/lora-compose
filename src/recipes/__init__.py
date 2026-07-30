import os

from . import dop, dreambooth, joint, standard


methods = {
    "dop": dop,
    "standard": standard,
    "dreambooth": dreambooth,
    "joint": joint,
}


def prepare_dataset(pipe, cfg):
    image_dir = (
        os.path.join(cfg.dataset_dir, cfg.character)
        if cfg.get("character") else None
    )
    return methods[cfg.recipe.method].prepare_dataset(pipe, cfg, image_dir)


def batch_loss(transformer, batch, scheduler, cfg):
    return methods[cfg.recipe.method].batch_loss(transformer, batch, scheduler, cfg)


def collate_fn(cfg):
    return methods[cfg.recipe.method].collate
