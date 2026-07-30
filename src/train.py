import logging
import os

import hydra
import torch
import wandb
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from adapters import apply_adapter, create_optimizer, finalize_adapter
from recipes import batch_loss, collate_fn, prepare_dataset
from sana import get_sana_pipeline, offload_frozen_components, sampling_components
from sampling import sample_prompts
from utils import set_seed


log = logging.getLogger(__name__)


def save_adapter_checkpoint(transformer, output_dir, epoch):
    adapter_path = os.path.join(output_dir, f"epoch_{epoch}")
    transformer.save_pretrained(adapter_path)
    log.info("Saved adapter checkpoint to %s", adapter_path)


def run_training(cfg: DictConfig) -> None:
    # 1. setup state
    set_seed(cfg.train.seed)

    characters = list(cfg.get("characters") or [cfg.character])
    subject_name = "-".join(characters)
    run_name = (
        f"{subject_name}_{cfg.recipe.method}_{cfg.adapter.method}_"
        f"{os.urandom(3).hex()}"
    )
    wandb.init(
        project="lora-composition",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Characters: {', '.join(characters)}")
    log.info(f"Output dir: {output_dir}")

    pipe = get_sana_pipeline(
        model_name_or_path=cfg.model_name_or_path,
        cache_dir=cfg.cache_dir,
    )
    if any(
        parameter.requires_grad
        for parameter in pipe.text_encoder.parameters()
    ):
        raise RuntimeError("The text encoder must remain frozen")
    log.info("Pipeline loaded")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model_name_or_path,
        subfolder="scheduler",
        cache_dir=cfg.cache_dir,
    )
    scheduler = {
        "timesteps": noise_scheduler.timesteps.to(pipe.transformer.device),
        "sigmas": noise_scheduler.sigmas.to(pipe.transformer.device),
        "num_train_timesteps": noise_scheduler.config.num_train_timesteps,
    }

    dataset, num_images = prepare_dataset(pipe, cfg)
    log.info(f"Found {num_images} training images")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn(cfg),
    )
    offload_frozen_components(pipe)
    log.info("Offloaded frozen text encoder and VAE to CPU")

    # 2. setup adapter for training
    if cfg.train.get("gradient_checkpointing", False):
        pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer = apply_adapter(
        pipe.transformer,
        cfg.adapter,
        dataloader=dataloader,
        loss_fn=lambda model, batch: batch_loss(model, batch, scheduler, cfg),
    )
    pipe.transformer.train()

    optimizer = create_optimizer(pipe.transformer, cfg.adapter, cfg.train)

    max_grad_norm = cfg.train.get("max_grad_norm", 1.0)

    # 3. actual training
    for epoch in range(1, cfg.train.epochs + 1):
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.train.epochs}"):
            loss = batch_loss(pipe.transformer, batch, scheduler, cfg)
            # Some recipes (e.g. DOP) backpropagate each loss term internally
            # to keep only one autograd graph in memory; they return a
            # detached total for logging.
            if loss.requires_grad:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.transformer.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            wandb.log({"loss": loss.item()})

        # 3.1 stats for nerds after the end of each epoch
        avg_loss = epoch_loss / len(dataloader)
        log.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")

        save_every_epochs = cfg.train.get("save_every_epochs")
        if (
            save_every_epochs
            and epoch % save_every_epochs == 0
            and epoch != cfg.train.epochs
        ):
            save_adapter_checkpoint(pipe.transformer, output_dir, epoch)

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

    # 4. after training, save adapter and do final samples
    pipe.transformer.eval()
    finalize_adapter(pipe.transformer, cfg.adapter)

    save_adapter_checkpoint(pipe.transformer, output_dir, cfg.train.epochs)

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
