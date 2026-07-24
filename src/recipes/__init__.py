from . import dop, dreambooth, standard


methods = {
    "dop": dop,
    "standard": standard,
    "dreambooth": dreambooth,
}


def prepare_dataset(pipe, cfg, image_dir):
    return methods[cfg.recipe.method].prepare_dataset(pipe, cfg, image_dir)


def batch_loss(transformer, batch, scheduler, cfg):
    return methods[cfg.recipe.method].batch_loss(transformer, batch, scheduler, cfg)


def collate_fn(cfg):
    return methods[cfg.recipe.method].collate
