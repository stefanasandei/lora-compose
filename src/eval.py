from datetime import datetime
import logging
import json
import os

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from PIL import Image

from diffusers.pipelines.sana.pipeline_sana import SanaPipeline
from transformers import AutoImageProcessor, AutoProcessor, Dinov2Model, CLIPModel
import torch.nn.functional as F
import torch
from peft import PeftModel
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_tensor

from sana import get_sana_pipeline
from sampling import sample_prompts

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

def compute_clip_score(output_dir, prompts, num_seeds, indices=None):
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()

    scores = []
    indices = indices or range(len(prompts))
    for i in indices:
        prompt_text = prompts[i]["prompt"]
        for seed in range(num_seeds):
            img = Image.open(f"{output_dir}/{i:02}_{seed}.png").convert("RGB")
            inputs = processor(text=[prompt_text], images=[img], return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            sim = F.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()
            scores.append(sim)

    return [sum(scores) / len(scores)]

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

def compute_arcface_similarity(character_dir, output_dir, prompts, num_seeds, indices=None):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', root='/mnt/sda3/Documents/Models')
    app.prepare(ctx_id=0, det_size=(640, 640))

    real_embeds = []
    for fname in sorted(os.listdir(character_dir)):
        if not fname.endswith(".png"):
            continue
        img = np.array(Image.open(os.path.join(character_dir, fname)).convert("RGB"))[:, :, ::-1]
        faces = app.get(img)
        if len(faces) > 0:
            real_embeds.append(faces[0].embedding)
    if not real_embeds:
        return [0.0]
    mean_embed = np.mean(real_embeds, axis=0)

    similarities = []
    indices = indices or range(len(prompts))
    for i in indices:
        for seed in range(num_seeds):
            img = np.array(Image.open(f"{output_dir}/{i:02}_{seed}.png").convert("RGB"))[:, :, ::-1]
            faces = app.get(img)
            if len(faces) == 0:
                continue
            emb = faces[0].embedding
            sim = np.dot(emb, mean_embed) / (np.linalg.norm(emb) * np.linalg.norm(mean_embed))
            similarities.append(sim)

    if not similarities:
        return [0.0]
    return [sum(similarities) / len(similarities)]

def gen_reference_dataset(pipe: SanaPipeline, prompts: list[dict], ref_dir: str, num_seeds: int):
    sample_prompts(
        pipe, [p["prompt"] for p in prompts], ref_dir,
        seed=0, num_seeds=num_seeds,
    )

def run_eval(cfg: DictConfig) -> None:
    log.info("Running evaluation.")

    # 1. setup lora and generated images for eval
    with open(f"{cfg.dataset_dir}/evals/prompts.json") as f:
        prompts = json.loads(f.read())

    lora_name = cfg["lora_name"]
    ref_dir = f"{cfg.dataset_dir}/evals/reference"
    ref_ready = os.path.isdir(ref_dir) # images from base frozen model

    evals_dir = f"{cfg.dataset_dir}/evals" # images generated from the finetuned model
    existing = sorted([d for d in os.listdir(evals_dir) if d.startswith(lora_name + "_")])
    output_dir = f"{evals_dir}/{existing[-1]}" if existing else f"{evals_dir}/{lora_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 2. generate images and only then load the model
    if not (ref_ready and existing):
        pipe = get_sana_pipeline(model_name_or_path=cfg.model_name_or_path, cache_dir=cfg.cache_dir)

        if not ref_ready:
            log.info("Reference dataset not present, generating...")
            gen_reference_dataset(pipe, prompts, ref_dir, cfg.num_seeds)

        if cfg.get("lora_path"):
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, cfg.lora_path)

        if not existing:
            sample_prompts(pipe, [item["prompt"] for item in prompts], output_dir, seed=0, num_seeds=cfg.num_seeds)

        del pipe
        torch.cuda.empty_cache()

    # 3. compute the metrics
    metrics = {}

    subject = cfg.get("character")
    indices = [i for i, p in enumerate(prompts) if p.get("character") != subject] if subject else None
    character_indices = [i for i, p in enumerate(prompts) if p.get("character") == subject] if subject else None
    other_character_indices = [i for i, p in enumerate(prompts) if p.get("character") and p.get("character") != subject] if subject else None

    ref_dir = f"{cfg.dataset_dir}/evals/reference"
    metrics["DINO"] = compute_dino_similarity(ref_dir, output_dir, prompts, cfg.num_seeds, indices)
    metrics["CLIP_Score"] = compute_clip_score(output_dir, prompts, cfg.num_seeds, indices)
    metrics["LPIPS"] = compute_lpips(ref_dir, output_dir, prompts, cfg.num_seeds, indices)
    if subject:
        character_dir = os.path.join(cfg.dataset_dir, subject)
        if character_indices:
            metrics["ArcFace"] = compute_arcface_similarity(character_dir, output_dir, prompts, cfg.num_seeds, character_indices)
        if other_character_indices:
            metrics["PRES"] = compute_arcface_similarity(character_dir, output_dir, prompts, cfg.num_seeds, other_character_indices)

    log.info(f"Evaluation done.\n{'-'*10}")
    metrics = pd.DataFrame(metrics)

    metrics.to_csv(f"{output_dir}/metrics.csv", index=False)
    print(metrics)


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
