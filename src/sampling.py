import logging
import os

import torch
from tqdm import tqdm


log = logging.getLogger(__name__)


def sample_prompts(pipe, prompts, output_dir=None, seed=42, num_seeds=1, **kwargs):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    images = []
    for i, prompt in enumerate(tqdm(prompts, desc="Sampling")):
        for s in range(num_seeds):
            generator = torch.manual_seed(seed + i * num_seeds + s)
            image = pipe(prompt=prompt, generator=generator, **kwargs).images[0]
            images.append(image)
            if output_dir is not None:
                fname = f"{i:02d}" if num_seeds == 1 else f"{i:02d}_{s}"
                image.save(os.path.join(output_dir, f"{fname}.png"))

    return images
