import logging
import os

import torch
from tqdm import tqdm


log = logging.getLogger(__name__)


def sample_prompts(
    pipe,
    prompts: list[str],
    output_dir: str,
    seed: int = 42,
    num_inference_steps: int = 20,
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 3.8,
    complex_human_instruction=None,
):
    # generate one image per prompt and save them to output_dir
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(tqdm(prompts, desc="Sampling")):
        generator = torch.manual_seed(seed + i)
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator,
            guidance_scale=guidance_scale,
            complex_human_instruction=complex_human_instruction,
        ).images[0]
        image.save(os.path.join(output_dir, f"{i:02d}.png"))

    log.info(f"Saved {len(prompts)} samples to {output_dir}")
