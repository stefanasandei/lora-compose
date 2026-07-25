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

from evaluation.identity import (
    build_reference_embeddings,
    evaluate_identity_assignments,
    face_embedding,
    normalize_prompts,
)
from evaluation.summary import headline_metrics, summarize_samples
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


def prompt_group(prompt, subjects):
    prompt_subjects = prompt["subjects"]
    if any(subject in subjects for subject in prompt_subjects):
        return "target"
    if prompt_subjects:
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
    subjects,
    primary_subject_embedding,
    face_app,
    num_seeds,
):
    dino_processor, dino_model = get_dino_model()
    clip_processor, clip_model = get_clip_model()
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device).eval()
    rows = []
    for prompt_index, prompt_info in enumerate(prompts):
        prompt = prompt_info["prompt"]
        group = prompt_group(prompt_info, subjects)
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
                            float(
                                np.dot(output_face, primary_subject_embedding)
                            )
                            if output_face is not None
                            else np.nan
                        ),
                        "Base_ArcFace_Target": (
                            float(
                                np.dot(reference_face, primary_subject_embedding)
                            )
                            if reference_face is not None
                            else np.nan
                        ),
                    }
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


def evaluation_subjects(cfg):
    """Resolve the optional multi-subject config with legacy compatibility."""

    configured = cfg.get("characters")
    if configured:
        subjects = list(configured)
    elif cfg.get("character"):
        subjects = [cfg.character]
    else:
        raise ValueError("Evaluation requires `character` or `characters`")
    if len(subjects) != len(set(subjects)):
        raise ValueError("Evaluation subjects must be unique")
    return subjects


def output_cache_prefix(cfg):
    """Key composed-adapter samples by their immutable manifest hash."""

    prefix = cfg.lora_name
    lora_path = cfg.get("lora_path")
    if not lora_path:
        return prefix
    manifest_path = os.path.join(lora_path, "composition.json")
    if not os.path.isfile(manifest_path):
        return prefix
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    manifest_hash = manifest.get("manifest_hash")
    if not manifest_hash:
        raise ValueError(f"Composition manifest has no manifest_hash: {manifest_path}")
    return f"{prefix}_{manifest_hash[:12]}"


def run_eval(cfg: DictConfig):
    log.info("Running evaluation.")
    with open(
        os.path.join(cfg.dataset_dir, "evals", "prompts.json"),
        encoding="utf-8",
    ) as file:
        prompts = normalize_prompts(json.load(file))

    evals_dir = os.path.join(cfg.dataset_dir, "evals")
    ref_dir = os.path.join(evals_dir, "reference")
    cache_prefix = output_cache_prefix(cfg)
    existing = sorted(
        directory
        for directory in os.listdir(evals_dir)
        if directory.startswith(cache_prefix + "_")
        and os.path.isdir(os.path.join(evals_dir, directory))
    )
    output_dir = (
        os.path.join(evals_dir, existing[-1])
        if existing
        else os.path.join(
            evals_dir,
            f"{cache_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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

    subjects = evaluation_subjects(cfg)
    subject_directories = {
        subject: os.path.join(cfg.dataset_dir, subject)
        for subject in subjects
    }
    face_app = get_arcface_app()
    reference_embeddings, reference_counts = build_reference_embeddings(
        face_app, subject_directories
    )
    for subject, count in reference_counts.items():
        log.info(
            "Built %s identity reference from %d detected faces",
            subject,
            count,
        )

    samples = evaluate_samples(
        prompts,
        output_dir,
        ref_dir,
        subjects,
        reference_embeddings[subjects[0]],
        face_app,
        cfg.num_seeds,
    )
    identity_prompts = [
        {
            **prompt,
            "subjects": [
                subject for subject in prompt["subjects"]
                if subject in reference_embeddings
            ],
        }
        for prompt in prompts
    ]
    identities = evaluate_identity_assignments(
        face_app,
        identity_prompts,
        output_dir,
        cfg.num_seeds,
        reference_embeddings,
        image_path,
        cfg.get("identity_threshold", 0.3),
    )
    summary = summarize_samples(
        samples, cfg.get("num_bootstrap", 5000)
    )
    headline = headline_metrics(samples)

    samples.to_csv(os.path.join(output_dir, "sample_metrics.csv"), index=False)
    identities.to_csv(
        os.path.join(output_dir, "identity_metrics.csv"), index=False
    )
    summary.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    headline.to_csv(os.path.join(output_dir, "headline_metrics.csv"), index=False)
    print(headline.to_string(index=False))
    log.info("Saved sample, identity, and summary metrics to %s", output_dir)


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(cfg: DictConfig):
    run_eval(cfg)


if __name__ == "__main__":
    main()
