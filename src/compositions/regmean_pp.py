import logging
import math
import re

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from peft import set_peft_model_state_dict

from recipes.flow_matching import FlowMatchingInputs, predict

from .common import (
    install_adapter,
    load_sources,
    resolved_configuration,
    source_scale,
)

log = logging.getLogger(__name__)
requires_pipeline = True


def _load_sources(cfg):
    return load_sources(
        cfg, method="RegMean++", minimum=2, require_prompts=True
    )


def _block_index(key):
    match = re.search(r"transformer_blocks\.(\d+)", key)
    if match is None:
        raise ValueError(f"Could not parse transformer block from {key}")
    return int(match.group(1))


def _rank_scale(config, rank):
    denominator = math.sqrt(rank) if config.get("use_rslora") else rank
    return config["lora_alpha"] / denominator


def _build_scheduler(base_model, device):
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model, subfolder="scheduler"
    )
    return {
        "timesteps": scheduler.timesteps.to(device),
        "sigmas": scheduler.sigmas.to(device),
        "num_train_timesteps": scheduler.config.num_train_timesteps,
    }


def _sample_batch(transformer, prompt_embeds, attention_mask):
    latent = torch.randn(
        1,
        transformer.config.in_channels,
        transformer.config.sample_size,
        transformer.config.sample_size,
    )
    return {
        "latent": latent,
        "prompt_embeds": prompt_embeds,
        "attention_mask": attention_mask,
    }


def _prepare_inputs(transformer, batch, scheduler, timesteps_mode):
    """Flow-matching inputs; `late` samples the low-noise half of the schedule."""
    device = transformer.device
    dtype = transformer.dtype
    latents = batch["latent"].to(device, dtype=dtype)
    batch_size = latents.size(0)
    noise = torch.randn_like(latents)
    if timesteps_mode == "late":
        u = 0.5 + 0.5 * torch.rand(batch_size, device=device)
    else:
        u = torch.rand(batch_size, device=device)
    indices = (u * scheduler["num_train_timesteps"]).long()
    timesteps = scheduler["timesteps"][indices].to(dtype=dtype)
    sigmas = scheduler["sigmas"][indices].to(dtype=dtype).view(
        batch_size, 1, 1, 1
    )
    return FlowMatchingInputs(
        noisy_latents=(1.0 - sigmas) * latents + sigmas * noise,
        target=noise - latents,
        timesteps=timesteps * transformer.config.timestep_scale,
    )


def _calibrate(peft, embeds, scheduler, cfg):
    seeds = int(cfg.get("calibration_seeds", 2))
    steps = int(cfg.get("calibration_steps", 2))
    base_seed = int(cfg.get("seed", 42))
    timesteps_mode = str(cfg.get("calibration_timesteps", "uniform"))
    count = 0
    for prompt_embeds, attention_mask in embeds:
        for seed_offset in range(seeds):
            torch.manual_seed(base_seed + seed_offset)
            for _ in range(steps):
                batch = _sample_batch(peft, prompt_embeds, attention_mask)
                inputs = _prepare_inputs(
                    peft, batch, scheduler, timesteps_mode
                )
                with torch.no_grad():
                    predict(
                        peft, inputs, batch["prompt_embeds"],
                        batch["attention_mask"],
                    )
                count += 1
    if not count:
        raise ValueError("No RegMean calibration samples were generated")
    return count


def _gram_hook(grams):
    def hook(module, inputs, _output):
        name = module._regmean_module_name
        features = inputs[0].detach().float()
        features = features.reshape(-1, features.shape[-1])
        if grams[name] is None:
            grams[name] = torch.zeros(
                features.shape[1], features.shape[1],
                dtype=torch.float32, device=features.device,
            )
        grams[name] += features.t() @ features

    return hook


def _capture_grams(peft, module_names, embeds, scheduler, cfg):
    named = dict(peft.named_modules())
    grams, handles = {}, []
    for name in module_names:
        if name not in named:
            raise ValueError(f"Could not find target module {name}")
        module = named[name]
        module._regmean_module_name = name
        grams[name] = None
        handles.append(module.register_forward_hook(_gram_hook(grams)))
    try:
        count = _calibrate(peft, embeds, scheduler, cfg)
    finally:
        for handle in handles:
            handle.remove()
        for name in module_names:
            module = named[name]
            if hasattr(module, "_regmean_module_name"):
                delattr(module, "_regmean_module_name")
    missing = [name for name, gram in grams.items() if gram is None]
    if missing:
        raise ValueError(f"No RegMean features captured for {missing[0]}")
    return {name: gram.cpu() / count for name, gram in grams.items()}, count


def _pad_factors(A, B, scale, rank):
    """Represent a rank-r LoRA delta at the probe's rank, scaling folded into B."""
    r = A.shape[0]
    if rank == r:
        return A, B * scale
    a_pad = torch.zeros(rank, A.shape[1], dtype=A.dtype, device=A.device)
    a_pad[:r] = A
    b_pad = torch.zeros(B.shape[0], rank, dtype=B.dtype, device=B.device)
    b_pad[:, :r] = B * scale
    return a_pad, b_pad


def _initial_factors(sources, rank):
    first = sources[0]
    factors = {}
    for key, tensor in first["tensors"].items():
        if key.endswith(".lora_A.weight"):
            factors[key] = torch.zeros(rank, tensor.shape[1])
        elif key.endswith(".lora_B.weight"):
            factors[key] = torch.zeros(tensor.shape[0], rank)
    return factors


def _candidate_factors(source, rank):
    """Rank-`rank` probe factors for every module of one candidate."""
    scale = _rank_scale(source["config"], int(source["config"]["r"]))
    factors = {}
    for key, tensor in source["tensors"].items():
        if not key.endswith(".lora_A.weight"):
            continue
        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        a_pad, b_pad = _pad_factors(
            tensor, source["tensors"][b_key], scale, rank
        )
        factors[key] = a_pad
        factors[b_key] = b_pad
    return factors


def _set_factors(peft, factors):
    result = set_peft_model_state_dict(peft, factors)
    missing = [key for key in result.missing_keys if ".lora_" in key]
    if missing or result.unexpected_keys:
        raise ValueError(
            "Could not set RegMean probe factors: "
            f"{missing}, {result.unexpected_keys}"
        )


def _effective_delta(source, a_key, b_key):
    """Dense, weight-scaled effective LoRA delta of one module."""
    down = source["tensors"][a_key].float()
    up = source["tensors"][b_key].float()
    rank = int(source["config"]["r"])
    return source_scale(source, rank) * (up @ down)


def _solve(grams, deltas, alpha, regularization, device):
    """RegMean closed form for one module: W = (Σ W_i Ĝ_i)(Σ Ĝ_i)^{-1}."""
    total = None
    rhs = None
    for gram, delta in zip(grams, deltas):
        gram = gram.to(device)
        diagonal = torch.diag(gram)
        gram_hat = alpha * gram + (1 - alpha) * torch.diag(diagonal)
        contribution = delta.to(device) @ gram_hat
        total = gram_hat if total is None else total + gram_hat
        rhs = contribution if rhs is None else rhs + contribution
    if total is None:
        raise ValueError("No Gram matrices for RegMean solve")
    if regularization > 0:
        total = total + torch.eye(
            total.shape[0], dtype=total.dtype, device=total.device
        ) * regularization
    try:
        return torch.linalg.solve(total, rhs.t()).t()
    except torch.linalg.LinAlgError as error:
        raise ValueError("Could not solve RegMean merge") from error


def _factorize(delta, rank, seed=42):
    """Return LoRA B/A factors for a rank-constrained dense delta."""
    maximum_rank = min(delta.shape)
    rank = min(rank, maximum_rank)
    if rank == maximum_rank:
        left, singular, right = torch.linalg.svd(
            delta.double(), full_matrices=False
        )
        left, singular, right = left.float(), singular.float(), right.float()
    else:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            left, singular, right_v = torch.svd_lowrank(
                delta, q=rank, niter=4
            )
        right = right_v.t()
    root = singular[:rank].sqrt()
    return left[:, :rank] * root, root.unsqueeze(1) * right[:rank]


def compose(transformer, cfg, base_model=None, pipeline=None):
    if pipeline is None:
        raise ValueError("RegMean++ requires a calibration pipeline")
    alpha = float(cfg.get("alpha", 0.9))
    regularization = float(cfg.get("regularization", 1e-6))
    features_mode = str(cfg.get("features_mode", "merged"))
    timesteps_mode = str(cfg.get("calibration_timesteps", "uniform"))
    if not 0 < alpha <= 1:
        raise ValueError("RegMean++ alpha must be in (0, 1]")
    if regularization < 0:
        raise ValueError("RegMean++ regularization must be non-negative")
    if features_mode not in ("merged", "candidate"):
        raise ValueError(
            "RegMean++ features_mode must be 'merged' or 'candidate'"
        )
    if timesteps_mode not in ("uniform", "late"):
        raise ValueError(
            "RegMean++ calibration_timesteps must be 'uniform' or 'late'"
        )
    if int(cfg.get("calibration_steps", 2)) < 1:
        raise ValueError("RegMean++ calibration_steps must be at least one")
    if int(cfg.get("calibration_seeds", 2)) < 1:
        raise ValueError("RegMean++ calibration_seeds must be at least one")
    if int(cfg.get("max_sequence_length", 300)) < 1:
        raise ValueError("RegMean++ max_sequence_length must be at least one")

    sources = _load_sources(cfg)
    first = sources[0]
    rank = int(
        cfg.get("rank", sum(int(source["config"]["r"]) for source in sources))
    )
    if rank < 1:
        raise ValueError("RegMean++ rank must be at least one")

    max_sequence_length = int(cfg.get("max_sequence_length", 300))
    embeds = []
    for source in sources:
        prompt_embeds, attention_mask, _, _ = pipeline.encode_prompt(
            source["prompt"],
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )
        embeds.append((prompt_embeds.cpu(), attention_mask.cpu()))

    scheduler = _build_scheduler(base_model, transformer.device)
    probe = install_adapter(
        transformer, _initial_factors(sources, rank),
        first["config"], rank, base_model,
    )
    pipeline.transformer = probe

    a_keys = sorted(
        key for key in first["tensors"] if key.endswith(".lora_A.weight")
    )
    blocks = sorted({_block_index(key) for key in a_keys})
    candidate_factors = [
        _candidate_factors(source, rank) for source in sources
    ]
    merged_factors = _initial_factors(sources, rank)
    calibration_samples = {}
    device = transformer.device
    try:
        for block in blocks:
            module_a_keys = [
                key for key in a_keys if _block_index(key) == block
            ]
            module_names = [
                key.removesuffix(".lora_A.weight")
                for key in module_a_keys
            ]
            grams_by_module = {name: [] for name in module_names}
            for index, source in enumerate(sources):
                if features_mode == "candidate":
                    probe_factors = candidate_factors[index]
                else:
                    probe_factors = dict(merged_factors)
                    for a_key in module_a_keys:
                        b_key = a_key.replace(
                            ".lora_A.weight", ".lora_B.weight"
                        )
                        probe_factors[a_key] = candidate_factors[index][a_key]
                        probe_factors[b_key] = candidate_factors[index][b_key]
                _set_factors(probe, probe_factors)
                grams, count = _capture_grams(
                    probe, module_names, [embeds[index]], scheduler, cfg
                )
                calibration_samples[source["path"]] = count
                for name in module_names:
                    grams_by_module[name].append(grams[name])
            for a_key in module_a_keys:
                b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
                module = a_key.removesuffix(".lora_A.weight")
                deltas = [
                    _effective_delta(source, a_key, b_key)
                    for source in sources
                ]
                merged = _solve(
                    grams_by_module[module], deltas, alpha,
                    regularization, device,
                )
                up, down = _factorize(merged, rank)
                merged_factors[a_key], merged_factors[b_key] = down, up
    finally:
        pipeline.transformer = probe.unload()
        torch.cuda.empty_cache()

    adapter = install_adapter(
        pipeline.transformer, merged_factors, first["config"],
        rank, base_model,
    )
    manifest = {
        "base_model": base_model,
        "method": "regmean_pp",
        "sources": [
            {
                "path": source["path"],
                "sha256": source["sha256"],
                "weight": source["weight"],
                "prompt": source["prompt"],
            }
            for source in sources
        ],
        "effective_rank": rank,
        "calibration_samples": calibration_samples,
        "configuration": resolved_configuration(cfg),
        "artifact_format": "adapter",
    }
    return adapter, manifest