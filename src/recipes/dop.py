import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data import build_cached_dataset, collate_fn, list_dataset_pairs
from .flow_matching import predict, prepare_inputs


class DifferentialOutputPreservationDataset(Dataset):
    def __init__(self, instance_dataset, class_dataset):
        self.instance_dataset = instance_dataset
        self.class_dataset = class_dataset

    def __getitem__(self, index):
        instance = self.instance_dataset[index]
        class_example = self.class_dataset[index]
        return instance | {
            "class_prompt_embeds": class_example["prompt_embeds"],
            "class_attention_mask": class_example["attention_mask"],
        }

    def __len__(self):
        return len(self.instance_dataset)


def prepare_dataset(pipe, cfg, image_dir):
    pairs = list_dataset_pairs(image_dir)
    if not pairs:
        raise ValueError(f"No training image/caption pairs found in {image_dir}")

    trigger_word = cfg.recipe.trigger_word
    preservation_class = cfg.recipe.preservation_class
    if not trigger_word:
        raise ValueError("recipe.trigger_word must not be empty")
    if not preservation_class:
        raise ValueError("recipe.preservation_class must not be empty")

    missing_trigger = [pair["basename"] for pair in pairs if trigger_word not in pair["prompt"]]
    if missing_trigger:
        names = ", ".join(missing_trigger[:5])
        raise ValueError(f"DOP trigger word {trigger_word!r} is missing from captions: {names}")

    class_pairs = [
        pair | {"prompt": pair["prompt"].replace(trigger_word, preservation_class)}
        for pair in pairs
    ]
    instance_dataset = build_cached_dataset(
        pipe, pairs, image_dir, "dop_instance", max_sequence_length=cfg.max_sequence_length
    )
    class_dataset = build_cached_dataset(
        pipe, class_pairs, image_dir, "dop_class", max_sequence_length=cfg.max_sequence_length
    )
    return DifferentialOutputPreservationDataset(instance_dataset, class_dataset), len(pairs)


def compute_loss(pred, target, cfg):
    return F.mse_loss(pred.float(), target.float(), reduction="mean")


def batch_loss(transformer, batch, scheduler, cfg):
    inputs = prepare_inputs(transformer, batch, scheduler)
    instance_pred = predict(
        transformer, inputs, batch["prompt_embeds"], batch["attention_mask"]
    )
    instance_loss = compute_loss(instance_pred, inputs.target, cfg)
    # Backward each term right after its forward pass. Gradients accumulate, so
    # this equals one backward on the summed loss, but only one autograd graph
    # is alive at a time: peak activation memory matches the standard recipe.
    instance_loss.backward()

    with torch.no_grad(), transformer.disable_adapter():
        prior_pred = predict(
            transformer,
            inputs,
            batch["class_prompt_embeds"],
            batch["class_attention_mask"],
        )
    preservation_pred = predict(
        transformer,
        inputs,
        batch["class_prompt_embeds"],
        batch["class_attention_mask"],
    )
    preservation_loss = F.mse_loss(
        preservation_pred.float(), prior_pred.float(), reduction="mean"
    )
    weighted_preservation_loss = cfg.recipe.preservation_loss_weight * preservation_loss
    weighted_preservation_loss.backward()
    return instance_loss.detach() + weighted_preservation_loss.detach()


def collate(batch):
    result = collate_fn(batch)
    class_batch = [
        {
            "latent": item["latent"],
            "prompt_embeds": item["class_prompt_embeds"],
            "attention_mask": item["class_attention_mask"],
        }
        for item in batch
    ]
    class_result = collate_fn(class_batch)
    result["class_prompt_embeds"] = class_result["prompt_embeds"]
    result["class_attention_mask"] = class_result["attention_mask"]
    return result
