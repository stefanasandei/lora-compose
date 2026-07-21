import logging
import os

from diffusers import SanaPipeline
from torch.utils.data import Dataset
import torch

from PIL import Image
from tqdm import tqdm


log = logging.getLogger(__name__)


def list_dataset_pairs(character_dir: str) -> list[dict]:
    if not os.path.isdir(character_dir):
        raise FileNotFoundError(f"Character dataset not found: {character_dir}")

    pairs = []
    for fname in sorted(os.listdir(character_dir)):
        if not fname.lower().endswith(".png"):
            continue

        basename = os.path.splitext(fname)[0]
        image_path = os.path.join(character_dir, fname)
        caption_path = os.path.join(character_dir, f"{basename}.txt")

        if not os.path.exists(caption_path):
            log.warning(f"Missing caption for {image_path}, skipping.")
            continue

        with open(caption_path, encoding="utf-8") as f:
            caption = f.read().strip()

        pairs.append({
            "basename": basename,
            "image_path": image_path,
            "caption_path": caption_path,
            "caption": caption,
        })

    return pairs


@torch.no_grad()
def cache_text_embeddings(pipe, pairs: list[dict], cache_path: str, max_sequence_length):
    if os.path.exists(cache_path):
        log.info(f"Text embeddings cache found at {cache_path}")
        return

    pipe.text_encoder.eval()
    embeds = {}
    for item in tqdm(pairs, desc="Caching text embeddings"):
        prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
            item["caption"],
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )
        embeds[item["basename"]] = {
            "prompt_embeds": prompt_embeds.cpu(),
            "attention_mask": prompt_attention_mask.cpu(),
        }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(embeds, cache_path)
    log.info(f"Saved text embeddings to {cache_path}")


@torch.no_grad()
def cache_vae_latents(pipe: SanaPipeline, pairs: list[dict], cache_path: str):
    if os.path.exists(cache_path):
        log.info(f"VAE latents cache found at {cache_path}")
        return

    vae = pipe.vae
    vae.eval()
    orig_dtype = vae.dtype
    vae.to(torch.float32)

    vae_scale = pipe.vae_scale_factor
    target = pipe.transformer.config.sample_size * vae_scale

    latents = {}
    for item in tqdm(pairs, desc="Caching VAE latents"):
        image = Image.open(item["image_path"]).convert("RGB")
        w, h = image.size
        scale = target / max(w, h)
        new_w = max(round(w * scale / vae_scale) * vae_scale, vae_scale)
        new_h = max(round(h * scale / vae_scale) * vae_scale, vae_scale)
        image_tensor = pipe.image_processor.preprocess(image, height=new_h, width=new_w)
        image_tensor = image_tensor.to(device=vae.device, dtype=torch.float32)

        latent = vae.encode(image_tensor).latent
        latent = latent * vae.config.scaling_factor

        latents[item["basename"]] = latent.cpu()

    vae.to(orig_dtype)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(latents, cache_path)
    log.info(f"Saved VAE latents to {cache_path}")


def collate_fn(batch: list[dict]) -> dict:
    return {
        "latent": torch.cat([b["latent"] for b in batch], dim=0),
        "prompt_embeds": torch.cat([b["prompt_embeds"] for b in batch], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in batch], dim=0),
    }

class CachedSanaDataset(Dataset):
    def __init__(self, pairs: list[dict], text_cache_path: str, vae_cache_path: str):
        self.pairs = pairs
        self.text_embeds = torch.load(text_cache_path, map_location="cpu", weights_only=True)
        self.vae_latents = torch.load(vae_cache_path, map_location="cpu", weights_only=True)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        basename = item["basename"]
        return {
            "latent": self.vae_latents[basename],
            "prompt_embeds": self.text_embeds[basename]["prompt_embeds"],
            "attention_mask": self.text_embeds[basename]["attention_mask"],
        }

    def __len__(self):
        return len(self.pairs)