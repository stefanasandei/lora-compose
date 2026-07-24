from datetime import datetime
import json
import logging
import os

import hydra
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from peft import PeftModel
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_tensor
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    CLIPModel,
    Dinov2Model,
)

from sana import get_sana_pipeline
from sampling import sample_prompts


log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_dino_model():
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
    return processor, model


def get_dino_embeddings(images, processor, model):
    inputs = processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        embeds = model(**inputs).pooler_output
    return F.normalize(embeds, dim=-1)


def get_clip_model():
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
    return processor, model


def clip_similarity(image, prompt, processor, model):
    inputs = processor(
        text=[prompt], images=[image], return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return F.cosine_similarity(
        outputs.image_embeds, outputs.text_embeds
    ).item()


def get_arcface_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", root="/mnt/sda3/Documents/Models")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def face_embedding(app, image):
    """Return the largest detected face, avoiding detector-order ambiguity."""
    array = np.asarray(image.convert("RGB"))[:, :, ::-1]
    faces = app.get(array)
    if not faces:
        return None
    face = max(
        faces,
        key=lambda item: (
            (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
        ),
    )
    embedding = face.embedding
    return embedding / np.linalg.norm(embedding)


def character_embedding(app, character_dir):
    embeddings = []
    for filename in sorted(os.listdir(character_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        with Image.open(os.path.join(character_dir, filename)) as image:
            embedding = face_embedding(app, image)
        if embedding is not None:
            embeddings.append(embedding)
    if not embeddings:
        raise ValueError(f"No faces detected in subject references: {character_dir}")
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding / np.linalg.norm(mean_embedding), len(embeddings)


def prompt_group(prompt, subject):
    character = prompt.get("character")
    if character == subject:
        return "target"
    if character:
        return "other_identity"
    return "general"


def image_path(directory, prompt_index, seed, num_seeds):
    filename = (
        f"{prompt_index:02d}.png"
        if num_seeds == 1
        else f"{prompt_index:02d}_{seed}.png"
    )
    return os.path.join(directory, filename)


def required_samples_exist(directory, num_prompts, num_seeds):
    return os.path.isdir(directory) and all(
        os.path.isfile(image_path(directory, index, seed, num_seeds))
        for index in range(num_prompts)
        for seed in range(num_seeds)
    )


def evaluate_samples(
    prompts,
    output_dir,
    ref_dir,
    character_dir,
    subject,
    num_seeds,
):
    dino_processor, dino_model = get_dino_model()
    clip_processor, clip_model = get_clip_model()
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device).eval()
    face_app = get_arcface_app()
    subject_embedding, num_subject_references = character_embedding(
        face_app, character_dir
    )
    log.info(
        "Built identity reference from %d detected subject faces",
        num_subject_references,
    )

    rows = []
    for prompt_index, prompt_info in enumerate(prompts):
        prompt = prompt_info["prompt"]
        group = prompt_group(prompt_info, subject)
        for seed in range(num_seeds):
            output_path = image_path(
                output_dir, prompt_index, seed, num_seeds
            )
            reference_path = image_path(
                ref_dir, prompt_index, seed, num_seeds
            )
            with (
                Image.open(output_path) as output_file,
                Image.open(reference_path) as reference_file,
            ):
                output = output_file.convert("RGB")
                reference = reference_file.convert("RGB")

                dino_embeds = get_dino_embeddings(
                    [reference, output], dino_processor, dino_model
                )
                dino_preservation = torch.dot(
                    dino_embeds[0], dino_embeds[1]
                ).item()

                output_tensor = (
                    to_tensor(output.resize((224, 224)))
                    .unsqueeze(0)
                    .to(device)
                )
                reference_tensor = (
                    to_tensor(reference.resize((224, 224)))
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    lpips_preservation = lpips(
                        reference_tensor, output_tensor
                    ).item()

                output_face = face_embedding(face_app, output)
                reference_face = face_embedding(face_app, reference)

                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "seed": seed,
                        "prompt": prompt,
                        "prompt_type": prompt_info.get("type"),
                        "prompt_character": prompt_info.get("character"),
                        "group": group,
                        "output_path": output_path,
                        "reference_path": reference_path,
                        "CLIP_Score": clip_similarity(
                            output, prompt, clip_processor, clip_model
                        ),
                        "DINO_Base_Preservation": dino_preservation,
                        "LPIPS_Base_Distance": lpips_preservation,
                        "Face_Detected": output_face is not None,
                        "Base_Face_Detected": reference_face is not None,
                        "ArcFace_Target": (
                            float(np.dot(output_face, subject_embedding))
                            if output_face is not None
                            else np.nan
                        ),
                        "Base_ArcFace_Target": (
                            float(np.dot(reference_face, subject_embedding))
                            if reference_face is not None
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


def clustered_interval(frame, metric, num_bootstrap=5000, seed=0):
    """Bootstrap prompt means, not correlated images from the same prompt."""
    prompt_means = (
        frame.groupby("prompt_index", sort=False)[metric]
        .mean()
        .dropna()
        .to_numpy()
    )
    if len(prompt_means) == 0:
        return np.nan, np.nan, np.nan, 0
    estimate = float(prompt_means.mean())
    if len(prompt_means) == 1:
        return estimate, np.nan, np.nan, 1

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        prompt_means,
        size=(num_bootstrap, len(prompt_means)),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return estimate, float(lower), float(upper), len(prompt_means)


def prompt_mean(frame, metric):
    return frame.groupby("prompt_index")[metric].mean().dropna().mean()


def harmonic_mean(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-8, 1.0)
    return len(values) / np.reciprocal(values).sum()


def headline_metrics(samples):
    target = samples[samples["group"] == "target"]
    other_identities = samples[samples["group"] == "other_identity"]
    non_target = samples[samples["group"] != "target"]

    identity = prompt_mean(target, "ArcFace_Target")
    prompt_adherence = prompt_mean(target, "CLIP_Score")
    leakage = prompt_mean(other_identities, "ArcFace_Target")
    preservation = prompt_mean(non_target, "DINO_Base_Preservation")
    balanced = harmonic_mean((identity, 1.0 - leakage, preservation))

    return pd.DataFrame(
        [{
            "Identity": identity,
            "Prompt_Adherence": prompt_adherence,
            "Concept_Leakage": leakage,
            "Model_Preservation": preservation,
            "Balanced_Score": balanced,
        }]
    )


def summarize_samples(samples, num_bootstrap):
    rows = []
    metrics = (
        "CLIP_Score",
        "DINO_Base_Preservation",
        "LPIPS_Base_Distance",
        "ArcFace_Target",
        "Base_ArcFace_Target",
    )
    for group, group_frame in samples.groupby("group", sort=False):
        for metric_index, metric in enumerate(metrics):
            mean, lower, upper, n_prompts = clustered_interval(
                group_frame,
                metric,
                num_bootstrap=num_bootstrap,
                seed=metric_index,
            )
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "n_prompts": n_prompts,
                    "n_images": int(group_frame[metric].notna().sum()),
                }
            )

        rows.extend(
            (
                {
                    "group": group,
                    "metric": "Face_Detection_Rate",
                    "mean": float(group_frame["Face_Detected"].mean()),
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "n_prompts": int(group_frame["prompt_index"].nunique()),
                    "n_images": len(group_frame),
                },
                {
                    "group": group,
                    "metric": "Base_Face_Detection_Rate",
                    "mean": float(group_frame["Base_Face_Detected"].mean()),
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "n_prompts": int(group_frame["prompt_index"].nunique()),
                    "n_images": len(group_frame),
                },
            )
        )
    return pd.DataFrame(rows)


def gen_reference_dataset(pipe, prompts, ref_dir, num_seeds):
    sample_prompts(
        pipe,
        [prompt["prompt"] for prompt in prompts],
        ref_dir,
        seed=0,
        num_seeds=num_seeds,
    )


def run_eval(cfg: DictConfig):
    log.info("Running evaluation.")
    with open(
        os.path.join(cfg.dataset_dir, "evals", "prompts.json"),
        encoding="utf-8",
    ) as file:
        prompts = json.load(file)

    evals_dir = os.path.join(cfg.dataset_dir, "evals")
    ref_dir = os.path.join(evals_dir, "reference")
    existing = sorted(
        directory
        for directory in os.listdir(evals_dir)
        if directory.startswith(cfg.lora_name + "_")
        and os.path.isdir(os.path.join(evals_dir, directory))
    )
    output_dir = (
        os.path.join(evals_dir, existing[-1])
        if existing
        else os.path.join(
            evals_dir,
            f"{cfg.lora_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    )

    reference_ready = required_samples_exist(
        ref_dir, len(prompts), cfg.num_seeds
    )
    output_ready = required_samples_exist(
        output_dir, len(prompts), cfg.num_seeds
    )
    if not (reference_ready and output_ready):
        pipe = get_sana_pipeline(
            model_name_or_path=cfg.model_name_or_path,
            cache_dir=cfg.cache_dir,
        )
        if not reference_ready:
            log.info("Generating missing frozen-model reference samples")
            gen_reference_dataset(pipe, prompts, ref_dir, cfg.num_seeds)
        if cfg.get("lora_path"):
            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer, cfg.lora_path
            )
        if not output_ready:
            log.info("Generating missing adapted-model evaluation samples")
            sample_prompts(
                pipe,
                [item["prompt"] for item in prompts],
                output_dir,
                seed=0,
                num_seeds=cfg.num_seeds,
            )
        del pipe
        torch.cuda.empty_cache()

    character_dir = os.path.join(cfg.dataset_dir, cfg.character)
    samples = evaluate_samples(
        prompts,
        output_dir,
        ref_dir,
        character_dir,
        cfg.character,
        cfg.num_seeds,
    )
    summary = summarize_samples(
        samples, cfg.get("num_bootstrap", 5000)
    )
    headline = headline_metrics(samples)

    samples.to_csv(os.path.join(output_dir, "sample_metrics.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    headline.to_csv(os.path.join(output_dir, "headline_metrics.csv"), index=False)
    print(headline.to_string(index=False))
    log.info("Saved per-image and summary metrics to %s", output_dir)


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(cfg: DictConfig):
    run_eval(cfg)


if __name__ == "__main__":
    main()
