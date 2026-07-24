import torch.nn.functional as F

from data import build_cached_dataset, collate_fn, list_dataset_pairs
from .flow_matching import predict, prepare_inputs


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


def batch_loss(transformer, batch, scheduler, cfg):
    inputs = prepare_inputs(transformer, batch, scheduler)
    pred = predict(transformer, inputs, batch["prompt_embeds"], batch["attention_mask"])
    return compute_loss(pred, inputs.target, cfg)


def collate(batch):
    return collate_fn(batch)
