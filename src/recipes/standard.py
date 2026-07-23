import torch.nn.functional as F

from data import build_cached_dataset, collate_fn, list_dataset_pairs


def prepare_dataset(pipe, cfg, image_dir):
    pairs = list_dataset_pairs(image_dir)
    if not pairs:
        raise ValueError(f"No training image/caption pairs found in {image_dir}")

    dataset = build_cached_dataset(
        pipe, pairs, image_dir, "standard", max_sequence_length=cfg.max_sequence_length
    )
    return dataset, len(pairs)


def compute_loss(pred, target, cfg):
    return F.mse_loss(pred.float(), target.float(), reduction="mean")


def collate(batch):
    return collate_fn(batch)
