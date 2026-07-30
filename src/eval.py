from datetime import datetime
import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from peft import PeftModel

from evaluation.identity import (
    build_reference_embeddings,
    evaluate_identity_assignments,
)
from evaluation.prompts import load_prompts, prompt_fingerprint
from evaluation.sample_metrics import evaluate_samples
from evaluation.summary import (
    composition_headline_metrics,
    headline_metrics,
    summarize_identities,
    summarize_samples,
)
from sana import get_sana_pipeline
from sampling import sample_prompts


log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_arcface_app(root):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", root=root)
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    return app


def image_path(directory, prompt_index, seed, num_seeds):
    filename = (
        f"{prompt_index:02d}.png"
        if num_seeds == 1
        else f"{prompt_index:02d}_{seed}.png"
    )
    return Path(directory) / filename


def required_samples_exist(directory, num_prompts, num_seeds):
    directory = Path(directory)
    return directory.is_dir() and all(
        image_path(directory, index, seed, num_seeds).is_file()
        for index in range(num_prompts)
        for seed in range(num_seeds)
    )


def evaluation_subjects(cfg):
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


def adapter_path(cfg):
    return cfg.get("adapter_path") or cfg.get("lora_path")


def run_name(cfg):
    return cfg.get("run_name") or cfg.get("lora_name") or "base"


def output_cache_prefix(cfg):
    prefix = run_name(cfg)
    path = adapter_path(cfg)
    if not path:
        return prefix
    manifest_path = Path(path) / "composition.json"
    if not manifest_path.is_file():
        return prefix
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    manifest_hash = manifest.get("manifest_hash")
    if not manifest_hash:
        raise ValueError(
            f"Composition manifest has no manifest_hash: {manifest_path}"
        )
    return f"{prefix}_{manifest_hash[:12]}"


def prompts_path(cfg):
    configured = cfg.get("prompts_file")
    if configured:
        return Path(configured).expanduser()
    return Path(cfg.dataset_dir) / "evals" / "prompts.json"


def output_directory(evals_dir, prefix):
    existing = sorted(
        path for path in evals_dir.glob(prefix + "_*") if path.is_dir()
    )
    if existing:
        return existing[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return evals_dir / f"{prefix}_{timestamp}"


def generate_samples(cfg, prompts, reference_dir, output_dir):
    reference_ready = required_samples_exist(
        reference_dir, len(prompts), cfg.num_seeds
    )
    output_ready = required_samples_exist(
        output_dir, len(prompts), cfg.num_seeds
    )
    if reference_ready and output_ready:
        return

    pipe = get_sana_pipeline(
        model_name_or_path=cfg.model_name_or_path,
        cache_dir=cfg.get("cache_dir"),
    )
    prompt_text = [item["prompt"] for item in prompts]
    if not reference_ready:
        log.info("Generating missing frozen-model reference samples")
        sample_prompts(
            pipe,
            prompt_text,
            reference_dir,
            seed=0,
            num_seeds=cfg.num_seeds,
        )
    if not output_ready:
        path = adapter_path(cfg)
        if path:
            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer, path
            )
        log.info("Generating missing adapted-model evaluation samples")
        sample_prompts(
            pipe,
            prompt_text,
            output_dir,
            seed=0,
            num_seeds=cfg.num_seeds,
        )
    del pipe
    torch.cuda.empty_cache()


def identity_prompts(prompts, references):
    return [
        {
            **prompt,
            "subjects": [
                subject for subject in prompt["subjects"]
                if subject in references
            ],
        }
        for prompt in prompts
    ]


def run_eval(cfg: DictConfig):
    log.info("Running evaluation")
    subjects = evaluation_subjects(cfg)
    prompts = load_prompts(
        prompts_path(cfg),
        max_subjects=cfg.get("max_subjects"),
    )
    fingerprint = prompt_fingerprint(
        prompts,
        cfg.num_seeds,
        model=str(cfg.model_name_or_path),
    )
    evals_dir = Path(cfg.dataset_dir) / "evals"
    reference_dir = evals_dir / f"reference_{fingerprint[:12]}"
    prefix = f"{output_cache_prefix(cfg)}_{fingerprint[:12]}"
    output_dir = output_directory(evals_dir, prefix)
    generate_samples(cfg, prompts, reference_dir, output_dir)

    face_app = get_arcface_app(cfg.get("arcface_root", cfg.cache_dir))
    subject_directories = {
        subject: Path(cfg.dataset_dir) / subject
        for subject in subjects
    }
    references, reference_counts = build_reference_embeddings(
        face_app, subject_directories
    )
    for subject, count in reference_counts.items():
        log.info(
            "Built %s identity reference from %d detected faces",
            subject,
            count,
        )

    primary_reference = (
        references[subjects[0]] if len(subjects) == 1 else None
    )
    samples = evaluate_samples(
        prompts,
        output_dir,
        reference_dir,
        subjects,
        face_app,
        cfg.num_seeds,
        image_path,
        device,
        primary_reference,
    )
    identities = evaluate_identity_assignments(
        face_app,
        identity_prompts(prompts, references),
        output_dir,
        cfg.num_seeds,
        references,
        image_path,
        cfg.get("identity_threshold", 0.3),
    )
    sample_summary = summarize_samples(
        samples, cfg.get("num_bootstrap", 5000)
    )
    identity_summary = summarize_identities(identities)
    if any(len(prompt["subjects"]) > 1 for prompt in prompts):
        headline = composition_headline_metrics(samples, identities)
    else:
        headline = headline_metrics(samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_dir / "sample_metrics.csv", index=False)
    identities.to_csv(output_dir / "identity_metrics.csv", index=False)
    sample_summary.to_csv(output_dir / "metrics.csv", index=False)
    identity_summary.to_csv(
        output_dir / "identity_summary.csv", index=False
    )
    headline.to_csv(output_dir / "headline_metrics.csv", index=False)
    print(headline.to_string(index=False))
    log.info("Saved evaluation metrics to %s", output_dir)


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(cfg: DictConfig):
    run_eval(cfg)


if __name__ == "__main__":
    main()
