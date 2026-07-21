import logging
import os
import random

import hydra
import torch
import torch.nn.functional as F
import wandb
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CachedSanaDataset, cache_text_embeddings, cache_vae_latents, list_dataset_pairs, collate_fn
from lora import apply_lora
from models.sana import get_sana_pipeline
from sampling import sample_prompts
from utils import set_seed


log = logging.getLogger(__name__)


def compute_flow_matching_loss(
    transformer,
    scheduler_timesteps: torch.Tensor,
    scheduler_sigmas: torch.Tensor,
    num_train_timesteps: int,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    timestep_scale: float,
) -> torch.Tensor:
    batch_size = latents.size(0)
    noise = torch.randn_like(latents)

    u = torch.rand(batch_size, device=latents.device)
    indices = (u * num_train_timesteps).long()
    timesteps = scheduler_timesteps[indices].to(dtype=latents.dtype)
    sigmas = scheduler_sigmas[indices].to(dtype=latents.dtype).view(batch_size, 1, 1, 1)

    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    target = noise - latents

    scaled_timesteps = timesteps * timestep_scale

    pred = transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=prompt_embeds,
        timestep=scaled_timesteps,
        encoder_attention_mask=attention_mask,
        return_dict=False,
    )[0]

    if pred.shape[1] == 2 * target.shape[1]:
        pred = pred.chunk(2, dim=1)[0]

    return F.mse_loss(pred, target, reduction="mean")


def run_training(cfg: DictConfig) -> None:
    # 1. setup state
    set_seed(cfg.train.seed)

    run_name = f"{cfg.character}_{cfg.lora.method}_{os.urandom(3).hex()}"
    wandb.init(project="lora-composition", name=run_name, config=OmegaConf.to_container(cfg, resolve=True))

    character_dir = os.path.join(cfg.dataset_dir, cfg.character)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Character: {cfg.character}")
    log.info(f"Output dir: {output_dir}")

    pairs = list_dataset_pairs(character_dir)
    log.info(f"Found {len(pairs)} image/caption pairs")

    text_cache = os.path.join(character_dir, "text_embeddings.pth")
    vae_cache = os.path.join(character_dir, "vae_latents.pth")

    pipe = get_sana_pipeline(cache_dir=cfg.cache_dir)
    log.info("Pipeline loaded")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "Efficient-Large-Model/SANA_600M_1024px_diffusers",
        subfolder="scheduler",
        cache_dir=cfg.cache_dir,
    )
    noise_scheduler_timesteps = noise_scheduler.timesteps.to(pipe.transformer.device)
    noise_scheduler_sigmas = noise_scheduler.sigmas.to(pipe.transformer.device)
    num_train_timesteps = noise_scheduler.config.num_train_timesteps

    cache_text_embeddings(pipe, pairs, text_cache)
    cache_vae_latents(pipe, pairs, vae_cache)

    dataset = CachedSanaDataset(pairs, text_cache, vae_cache)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn)

    # 2. setup lora parameters for training
    pipe.transformer = apply_lora(pipe.transformer, cfg.lora)
    pipe.transformer.train()

    trainable_params = list(filter(lambda p: p.requires_grad, pipe.transformer.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.train.lr)

    device = pipe.transformer.device
    dtype = pipe.transformer.dtype
    timestep_scale = pipe.transformer.config.timestep_scale
    max_grad_norm = cfg.train.get("max_grad_norm", 1.0)

    # 3. actual training
    for epoch in range(1, cfg.train.epochs + 1):
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.train.epochs}"):
            latents = batch["latent"].to(device, dtype=dtype)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=dtype)
            attention_mask = batch["attention_mask"].to(device)

            loss = compute_flow_matching_loss(
                pipe.transformer,
                noise_scheduler_timesteps,
                noise_scheduler_sigmas,
                num_train_timesteps,
                latents,
                prompt_embeds,
                attention_mask,
                timestep_scale,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.transformer.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        # 3.1 stats for nerds after the end of each epoch
        avg_loss = epoch_loss / len(dataloader)
        log.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        wandb.log({"loss": avg_loss, "epoch": epoch}, step=epoch)

        if epoch % 50 == 0:
            pipe.transformer.eval()

            samples = []
            for i, prompt in enumerate(cfg.sample_prompts):
                generator = torch.manual_seed(cfg.train.seed + i)
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=20,
                    height=1024,
                    width=1024,
                    generator=generator,
                    guidance_scale=3.8,
                ).images[0]
                samples.append(wandb.Image(image, caption=f"epoch {epoch}: {prompt[:60]}"))
            wandb.log({"samples": samples, "epoch": epoch}, step=epoch)

            pipe.transformer.train()
            torch.cuda.empty_cache()

    # 4. after training, save lora and do final samples
    adapter_path = os.path.join(output_dir, f"epoch_{cfg.train.epochs}")
    pipe.transformer.save_pretrained(adapter_path)
    log.info(f"Saved adapter to {adapter_path}")

    log.info("Training complete. Sampling...")
    sample_dir = os.path.join(output_dir, "samples")
    sample_prompts(pipe, cfg.sample_prompts, sample_dir,seed=cfg.train.seed)
    
    wandb.finish()


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()