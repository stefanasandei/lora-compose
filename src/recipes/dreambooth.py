import hashlib
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from data import build_cached_dataset, collate_fn, encode_image, list_dataset_pairs, load_or_create_cache


log = logging.getLogger(__name__)
_BATCH_KEYS = ("latent", "prompt_embeds", "attention_mask")


class DreamBoothDataset(Dataset):
    """Pairs each subject image with a generated class-prior image."""

    def __init__(self, instance_dataset, class_dataset):
        self.instance_dataset = instance_dataset
        self.class_dataset = class_dataset

    def __getitem__(self, index):
        instance = self.instance_dataset[index % len(self.instance_dataset)]
        class_example = self.class_dataset[index % len(self.class_dataset)]
        return instance | {f"class_{key}": value for key, value in class_example.items()}

    def __len__(self):
        return max(len(self.instance_dataset), len(self.class_dataset))


class InMemoryClassDataset(Dataset):
    def __init__(self, latents, prompt_embeds, attention_mask):
        self.latents = latents
        self.prompt_embeds = prompt_embeds
        self.attention_mask = attention_mask

    def __getitem__(self, index):
        return {
            "latent": self.latents[index],
            "prompt_embeds": self.prompt_embeds[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return len(self.latents)


@torch.no_grad()
def _generate_class_cache(pipe, cfg, instance_dataset):
    dreambooth_cfg = cfg.recipe
    prior_prompts = list(dreambooth_cfg.prior_prompts)
    if not prior_prompts:
        raise ValueError("recipe.prior_prompts must contain at least one prompt")

    log.info(f"Generating {dreambooth_cfg.num_class_images} class images with the frozen base model")
    latents = []
    prompt_embeds = []
    attention_mask = []
    encoded_prompts = {}
    for index in tqdm(range(dreambooth_cfg.num_class_images), desc="Generating class images"):
        prompt = prior_prompts[index % len(prior_prompts)]
        instance_latent = instance_dataset[index % len(instance_dataset)]["latent"]
        height = instance_latent.shape[-2] * pipe.vae_scale_factor
        width = instance_latent.shape[-1] * pipe.vae_scale_factor
        generator = torch.Generator(device=pipe.transformer.device).manual_seed(cfg.train.seed + index)
        image = pipe(
            prompt,
            generator=generator,
            num_inference_steps=20,
            height=height,
            width=width,
            guidance_scale=3.8,
        ).images[0]
        latents.append(encode_image(pipe, image))
        if prompt not in encoded_prompts:
            encoded_prompts[prompt] = pipe.encode_prompt(
                prompt,
                do_classifier_free_guidance=False,
                max_sequence_length=cfg.max_sequence_length,
            )[:2]
        embeds, mask = encoded_prompts[prompt]
        prompt_embeds.append(embeds.cpu())
        attention_mask.append(mask.cpu())

    return {
        "latents": latents,
        "prompt_embeds": prompt_embeds,
        "attention_mask": attention_mask,
    }


def _class_cache_path(cfg, image_dir):
    dreambooth_cfg = cfg.recipe
    cache_key = "\n".join((
        cfg.model_name_or_path,
        "\n".join(dreambooth_cfg.prior_prompts),
        str(dreambooth_cfg.num_class_images),
        str(cfg.train.seed),
    ))
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:8]
    return os.path.join(image_dir, f"dreambooth_class_ar_{cache_hash}.pth")


def _build_class_dataset(pipe, cfg, image_dir, instance_dataset):
    cache_path = _class_cache_path(cfg, image_dir)
    cache = load_or_create_cache(
        cache_path, lambda: _generate_class_cache(pipe, cfg, instance_dataset)
    )
    return InMemoryClassDataset(**cache)


def prepare_dataset(pipe, cfg, image_dir):
    instance_pairs = list_dataset_pairs(image_dir)

    instance_dataset = build_cached_dataset(
        pipe,
        instance_pairs,
        image_dir,
        "dreambooth_instance_v2",
        max_sequence_length=cfg.max_sequence_length,
    )
    class_dataset = _build_class_dataset(pipe, cfg, image_dir, instance_dataset)
    return DreamBoothDataset(instance_dataset, class_dataset), len(instance_pairs)


def compute_loss(pred, target, cfg):
    instance_pred, prior_pred = pred.chunk(2)
    instance_target, prior_target = target.chunk(2)
    instance_loss = F.mse_loss(instance_pred.float(), instance_target.float(), reduction="mean")
    prior_loss = F.mse_loss(prior_pred.float(), prior_target.float(), reduction="mean")
    return instance_loss + cfg.recipe.prior_loss_weight * prior_loss


def collate(batch):
    examples = [
        {key: item[prefix + key] for key in _BATCH_KEYS}
        for prefix in ("", "class_")
        for item in batch
    ]
    return collate_fn(examples)
