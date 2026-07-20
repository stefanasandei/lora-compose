from datetime import datetime
import logging
import json
import os

import hydra
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
from PIL import Image

from transformers import AutoImageProcessor, Dinov2Model
import torch.nn.functional as F
import torch
from peft import PeftModel

from models.sana import get_sana_pipeline
from reference import gen_reference_dataset

log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
 
def compute_dino_similarity(ref_dir, output_dir, prompts, num_seeds):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()

    similarities = []
    for i in range(len(prompts)):
        for seed in range(num_seeds):
            ref = Image.open(f"{ref_dir}/{i:02}_{seed}.png").convert("RGB")
            out = Image.open(f"{output_dir}/{i:02}_{seed}.png").convert("RGB")
            inputs = processor([ref, out], return_tensors="pt").to(device)

            with torch.no_grad():
                embeds = model(**inputs).pooler_output
            
            sim = F.cosine_similarity(embeds[0:1], embeds[1:2]).item()
            similarities.append(sim)
   
    return [sum(similarities) / len(similarities)]

def run_eval(cfg: DictConfig) -> None:
    pipe = get_sana_pipeline(cache_dir=cfg.cache_dir)
    log.info("Running evaluation.")

    if cfg.get("lora_path"):
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, cfg.lora_path)
        lora_name = cfg.get("lora_name") or os.path.basename(cfg.lora_path)
    else:
        lora_name = "nolora"

    # 1. get frozen model reference data
    with open(f"{cfg.dataset_dir}/evals/prompts.json") as f:
        prompts = json.loads(f.read())

    if not os.path.isdir(f"{cfg.dataset_dir}/evals/reference"):
        log.info("Reference dataset not present, generating...")
        gen_reference_dataset(pipe, prompts, f"{cfg.dataset_dir}/evals/reference")

    # 2. run the trained model the same prompts
    evals_dir = f"{cfg.dataset_dir}/evals"
    existing = sorted([d for d in os.listdir(evals_dir) if d.startswith(lora_name + "_")])
    output_dir = f"{evals_dir}/{existing[-1]}" if existing else f"{evals_dir}/{lora_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    ref_dir = f"{cfg.dataset_dir}/evals/reference"
    metrics["DINO"] = compute_dino_similarity(ref_dir, output_dir, prompts, cfg.num_seeds)

    log.info(f"Evaluation done.\n{'-'*10}")
    metrics = pd.DataFrame(metrics)

    metrics.to_csv(f"{output_dir}/metrics.csv", index=False)
    print(metrics)


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
