from diffusers.pipelines.sana.pipeline_sana import SanaPipeline
import torch

from tqdm import tqdm
import os

def gen_reference_dataset(pipe: SanaPipeline, prompts: list[dict], ref_dir: str):
    os.mkdir(ref_dir)

    for i, item in enumerate(tqdm(prompts)):
        for seed in range(0, 5):
            image = pipe(prompt=item["prompt"], generator=torch.manual_seed(seed))[0]
            image[0].save(f"{ref_dir}/{i:02}_{seed}.png")
