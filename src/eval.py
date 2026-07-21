from datetime import datetime
import logging
import json
import os

import hydra
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
from PIL import Image

from transformers import AutoImageProcessor, Dinov2Model, CLIPVisionModel
import torch.nn.functional as F
import torch
from peft import PeftModel
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_tensor

from models.sana import get_sana_pipeline
from reference import gen_reference_dataset

log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_dino_similarity(ref_dir, output_dir, prompts, num_seeds, indices=None):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()

    similarities = []
    indices = indices or range(len(prompts))
    for i in indices:
        for seed in range(num_seeds):
            ref = Image.open(f"{ref_dir}/{i:02}_{seed}.png").convert("RGB")
            out = Image.open(f"{output_dir}/{i:02}_{seed}.png").convert("RGB")
            inputs = processor([ref, out], return_tensors="pt").to(device)

            with torch.no_grad():
                embeds = model(**inputs).pooler_output
            
            sim = F.cosine_similarity(embeds[0:1], embeds[1:2]).item()
            similarities.append(sim)
   
    return [sum(similarities) / len(similarities)]

def compute_clip_similarity(ref_dir, output_dir, prompts, num_seeds, indices=None):
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()

    similarities = []
    indices = indices or range(len(prompts))
    for i in indices:
        for seed in range(num_seeds):
            ref = Image.open(f"{ref_dir}/{i:02}_{seed}.png").convert("RGB")
            out = Image.open(f"{output_dir}/{i:02}_{seed}.png").convert("RGB")
            inputs = processor([ref, out], return_tensors="pt").to(device)

            with torch.no_grad():
                embeds = model(**inputs).pooler_output

            sim = F.cosine_similarity(embeds[0:1], embeds[1:2]).item()
            similarities.append(sim)

    return [sum(similarities) / len(similarities)]

def compute_lpips(ref_dir, output_dir, prompts, num_seeds, indices=None):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device).eval()

    distances = []
    indices = indices or range(len(prompts))
    for i in indices:
        for seed in range(num_seeds):
            ref = Image.open(f"{ref_dir}/{i:02}_{seed}.png").convert("RGB")
            out = Image.open(f"{output_dir}/{i:02}_{seed}.png").convert("RGB")
            ref_t = to_tensor(ref.resize((224, 224))).unsqueeze(0).to(device)
            out_t = to_tensor(out.resize((224, 224))).unsqueeze(0).to(device)

            with torch.no_grad():
                dist = lpips(ref_t, out_t).item()
            distances.append(dist)

    return [sum(distances) / len(distances)]

def run_eval(cfg: DictConfig) -> None:
    log.info("Running evaluation.")

    # 1. setup lora and generated images for eval
    with open(f"{cfg.dataset_dir}/evals/prompts.json") as f:
        prompts = json.loads(f.read())

    lora_name = cfg.get("lora_name")
    if not lora_name:
        lora_name = os.path.basename(cfg.lora_path) if cfg.get("lora_path") else "nolora"

    ref_dir = f"{cfg.dataset_dir}/evals/reference"
    ref_ready = os.path.isdir(ref_dir) # images from base frozen model

    evals_dir = f"{cfg.dataset_dir}/evals" # images generated from the finetuned model
    existing = sorted([d for d in os.listdir(evals_dir) if d.startswith(lora_name + "_")])
    output_dir = f"{evals_dir}/{existing[-1]}" if existing else f"{evals_dir}/{lora_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 2. generate images and only then load the model
    if not (ref_ready and existing):
        pipe = get_sana_pipeline(cache_dir=cfg.cache_dir)
        if cfg.get("lora_path"):
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, cfg.lora_path)

        if not ref_ready:
            log.info("Reference dataset not present, generating...")
            gen_reference_dataset(pipe, prompts, ref_dir)

        if not existing:
            os.makedirs(output_dir, exist_ok=True)
            for i, item in enumerate(tqdm(prompts)):
                for seed in range(cfg.num_seeds):
                    image = pipe(prompt=item["prompt"], generator=torch.manual_seed(seed))[0]
                    image[0].save(f"{output_dir}/{i:02}_{seed}.png")

        del pipe
        torch.cuda.empty_cache()

    # 3. compute the metrics
    metrics = {}

    subject = cfg.get("character")
    indices = [i for i, p in enumerate(prompts) if p.get("character") != subject] if subject else None

    ref_dir = f"{cfg.dataset_dir}/evals/reference"
    metrics["DINO"] = compute_dino_similarity(ref_dir, output_dir, prompts, cfg.num_seeds, indices)
    metrics["CLIP"] = compute_clip_similarity(ref_dir, output_dir, prompts, cfg.num_seeds, indices)
    metrics["LPIPS"] = compute_lpips(ref_dir, output_dir, prompts, cfg.num_seeds, indices)

    log.info(f"Evaluation done.\n{'-'*10}")
    metrics = pd.DataFrame(metrics)

    metrics.to_csv(f"{output_dir}/metrics.csv", index=False)
    print(metrics)


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
