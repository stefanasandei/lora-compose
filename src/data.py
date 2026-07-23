import hashlib
import logging
import os

import torch
from diffusers import SanaPipeline
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


log = logging.getLogger(__name__)


def list_dataset_pairs(image_dir: str, prompt: str | None = None) -> list[dict]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image dataset not found: {image_dir}")

    pairs = []
    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        basename = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, filename)
        caption_path = os.path.join(image_dir, f"{basename}.txt")
        if prompt is None and not os.path.exists(caption_path):
            log.warning(f"Missing caption for {image_path}, skipping.")
            continue

        caption = prompt
        if caption is None:
            with open(caption_path, encoding="utf-8") as file:
                caption = file.read().strip()

        pairs.append({"basename": basename, "image_path": image_path, "prompt": caption})

    return pairs


def load_or_create_cache(cache_path, create):
    if os.path.exists(cache_path):
        log.info(f"Cache found at {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    cache = create()
    torch.save(cache, cache_path)
    log.info(f"Saved cache to {cache_path}")
    return cache


@torch.no_grad()
def encode_image(pipe: SanaPipeline, image: Image.Image) -> torch.Tensor:
    vae = pipe.vae
    dtype = vae.dtype
    vae.to(torch.float32)

    vae_scale = pipe.vae_scale_factor
    target = pipe.transformer.config.sample_size * vae_scale
    width, height = image.size
    scale = target / max(width, height)
    width = max(round(width * scale / vae_scale) * vae_scale, vae_scale)
    height = max(round(height * scale / vae_scale) * vae_scale, vae_scale)
    image_tensor = pipe.image_processor.preprocess(image, height=height, width=width)
    image_tensor = image_tensor.to(device=vae.device, dtype=torch.float32)

    latent = vae.encode(image_tensor).latent * vae.config.scaling_factor
    vae.to(dtype)
    return latent.cpu()


@torch.no_grad()
def _encode_examples(pipe, pairs, max_sequence_length):
    pipe.text_encoder.eval()
    pipe.vae.eval()
    examples = {}

    for pair in tqdm(pairs, desc="Caching training data"):
        prompt_embeds, attention_mask, _, _ = pipe.encode_prompt(
            pair["prompt"],
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )
        with Image.open(pair["image_path"]) as image:
            latent = encode_image(pipe, image.convert("RGB"))
        examples[pair["basename"]] = {
            "latent": latent,
            "prompt_embeds": prompt_embeds.cpu(),
            "attention_mask": attention_mask.cpu(),
        }

    return examples


class CachedDataset(Dataset):
    def __init__(self, pairs, examples):
        self.pairs = pairs
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[self.pairs[index]["basename"]]

    def __len__(self):
        return len(self.pairs)


def build_cached_dataset(pipe, pairs, cache_dir, prefix, max_sequence_length):
    prompts = "\n".join(pair["prompt"] for pair in pairs)
    cache_hash = hashlib.sha256(prompts.encode()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"{prefix}_{cache_hash}_cache.pth")
    examples = load_or_create_cache(
        cache_path, lambda: _encode_examples(pipe, pairs, max_sequence_length)
    )
    return CachedDataset(pairs, examples)


def collate_fn(batch):
    return {
        "latent": torch.cat([item["latent"] for item in batch]),
        "prompt_embeds": pad_sequence([item["prompt_embeds"].squeeze(0) for item in batch], batch_first=True),
        "attention_mask": pad_sequence([item["attention_mask"].squeeze(0) for item in batch], batch_first=True),
    }
