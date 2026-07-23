import logging
import os

import hydra
import torch
import wandb
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from adapters import apply_adapter
from recipes import collate_fn, compute_loss, prepare_dataset
from sana import get_sana_pipeline, offload_frozen_components, sampling_components
from sampling import sample_prompts
from utils import set_seed


log = logging.getLogger(__name__)


def flow_matching_prediction(
    transformer,
    scheduler_timesteps: torch.Tensor,
    scheduler_sigmas: torch.Tensor,
    num_train_timesteps: int,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    timestep_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    return pred, target


def run_training(cfg: DictConfig) -> None:
    # 1. setup state
    set_seed(cfg.train.seed)

    run_name = f"{cfg.character}_{cfg.recipe.method}_{cfg.adapter.method}_{os.urandom(3).hex()}"
    wandb.init(project="lora-composition", name=run_name, config=OmegaConf.to_container(cfg, resolve=True))

    character_dir = os.path.join(cfg.dataset_dir, cfg.character)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Character: {cfg.character}")
    log.info(f"Output dir: {output_dir}")

    pipe = get_sana_pipeline(model_name_or_path=cfg.model_name_or_path, cache_dir=cfg.cache_dir)
    log.info("Pipeline loaded")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model_name_or_path,
        subfolder="scheduler",
        cache_dir=cfg.cache_dir,
    )
    noise_scheduler_timesteps = noise_scheduler.timesteps.to(pipe.transformer.device)
    noise_scheduler_sigmas = noise_scheduler.sigmas.to(pipe.transformer.device)
    num_train_timesteps = noise_scheduler.config.num_train_timesteps

    dataset, num_images = prepare_dataset(pipe, cfg, character_dir)
    log.info(f"Found {num_images} training images")
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn(cfg))
    offload_frozen_components(pipe)
    log.info("Offloaded frozen text encoder and VAE to CPU")

    # 2. setup adapter for training
    if cfg.train.get("gradient_checkpointing", False):
        pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer = apply_adapter(pipe.transformer, cfg.adapter)
    pipe.transformer.train()

    trainable_params = list(filter(lambda p: p.requires_grad, pipe.transformer.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

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

            pred, target = flow_matching_prediction(
                pipe.transformer,
                noise_scheduler_timesteps,
                noise_scheduler_sigmas,
                num_train_timesteps,
                latents,
                prompt_embeds,
                attention_mask,
                timestep_scale,
            )
            loss = compute_loss(pred, target, cfg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.transformer.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            wandb.log({"loss": loss.item()})

        # 3.1 stats for nerds after the end of each epoch
        avg_loss = epoch_loss / len(dataloader)
        log.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")

        if epoch % cfg.train.get("sample_every_epochs", 50) == 0:
            pipe.transformer.eval()

            with sampling_components(pipe):
                images = sample_prompts(
                    pipe, cfg.sample_prompts, output_dir=None,
                    seed=cfg.train.seed,
                    num_inference_steps=20, height=1024, width=1024, guidance_scale=3.8,
                )
            samples = [
                wandb.Image(img, caption=f"epoch {epoch}: {prompt[:60]}")
                for img, prompt in zip(images, cfg.sample_prompts)
            ]
            wandb.log({"samples": samples, "epoch": epoch})
            
            pipe.transformer.train()
            torch.cuda.empty_cache()

    # 4. after training, save lora and do final samples
    pipe.transformer.eval()

    adapter_path = os.path.join(output_dir, f"epoch_{cfg.train.epochs}")
    pipe.transformer.save_pretrained(adapter_path)
    log.info(f"Saved adapter to {adapter_path}")

    log.info("Training complete. Sampling...")
    sample_dir = os.path.join(output_dir, "samples")
    with sampling_components(pipe):
        sample_prompts(pipe, cfg.sample_prompts, sample_dir, seed=cfg.train.seed)

    wandb.finish()


@hydra.main(version_base=None, config_path="../config", config_name="training/lora")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
