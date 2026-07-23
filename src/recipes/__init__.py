from . import dreambooth, standard


methods = {
    "standard": standard,
    "dreambooth": dreambooth,
}


def prepare_dataset(pipe, cfg, image_dir):
    return methods[cfg.recipe.method].prepare_dataset(pipe, cfg, image_dir)


def compute_loss(pred, target, cfg):
    return methods[cfg.recipe.method].compute_loss(pred, target, cfg)


def collate_fn(cfg):
    return methods[cfg.recipe.method].collate
