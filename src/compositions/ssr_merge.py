import logging

import torch
from peft import PeftModel

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
        cfg, method="SSR-Merge", minimum=2, require_prompts=True
    )


def _build_statistics(sources):
    statistics = {}
    keys = sources[0]["tensors"]
    for a_key in sorted(key for key in keys if key.endswith(".lora_A.weight")):
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        downs, ups, ranks = [], [], []
        input_size = sources[0]["tensors"][a_key].shape[1]
        output_size = sources[0]["tensors"][b_key].shape[0]
        for source in sources:
            down = source["tensors"][a_key].float()
            up = source["tensors"][b_key].float()
            rank = down.shape[0]
            if down.shape[1] != input_size or up.shape != (output_size, rank):
                raise ValueError(f"Incompatible factors for {a_key}")
            config = source["config"]
            if (
                rank != config["r"]
                or config.get("rank_pattern")
                or config.get("alpha_pattern")
            ):
                raise ValueError(
                    "Per-module ranks are not supported by SSR-Merge"
                )
            downs.append(down)
            ups.append(up * source_scale(source, rank))
            ranks.append(rank)

        combined_down = torch.cat(downs, dim=0)
        total_rank = combined_down.shape[0]
        module_name = a_key.removesuffix(".lora_A.weight")
        statistics[module_name] = {
            "a_key": a_key,
            "b_key": b_key,
            "down": combined_down,
            "up": torch.cat(ups, dim=1),
            "ranks": ranks,
            "correlation": torch.zeros(total_rank, total_rank),
            "guide": torch.zeros(total_rank, total_rank),
            "count": 0,
        }
    return statistics


def _statistics_hook(statistics, calibration):
    def hook(module, inputs, _output):
        name = module._ssr_module_name
        current = statistics[name]
        task = calibration["task"]
        start = sum(current["ranks"][:task])
        end = start + current["ranks"][task]
        features = inputs[0].detach()
        input_size = current["down"].shape[1]
        if features.shape[-1] != input_size:
            raise ValueError(f"Unexpected input shape for {name}: {features.shape}")
        features = features.reshape(-1, input_size).t().float()
        projected = current["down"].to(features.device) @ features
        correlation = projected @ projected.t()
        guide = projected[start:end] @ projected.t()
        current["correlation"] += correlation.cpu()
        current["guide"][start:end] += guide.cpu()
        current["count"] += features.shape[1]

    return hook


def _attach_hooks(transformer, statistics, calibration):
    modules = dict(transformer.named_modules())
    missing = sorted(set(statistics) - set(modules))
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"Could not find {len(missing)} target modules in transformer: {preview}"
        )

    handles = []
    hook = _statistics_hook(statistics, calibration)
    for name, current in statistics.items():
        module = modules[name]
        module._ssr_module_name = name
        handles.append(module.register_forward_hook(hook))
    return handles


def _remove_hooks(handles, statistics, transformer):
    for handle in handles:
        handle.remove()
    modules = dict(transformer.named_modules())
    for name in statistics:
        if name in modules and hasattr(modules[name], "_ssr_module_name"):
            delattr(modules[name], "_ssr_module_name")


def _set_source_weights(transformer, sources):
    for index, source in enumerate(sources):
        adapter_name = f"source_{index}"
        for module in transformer.modules():
            scaling = getattr(module, "scaling", None)
            if scaling is not None and adapter_name in scaling:
                scaling[adapter_name] *= source["weight"]


def _calibrate(pipeline, sources, statistics, cfg):
    transformer = PeftModel.from_pretrained(
        pipeline.transformer,
        sources[0]["path"],
        adapter_name="source_0",
    )
    for index, source in enumerate(sources[1:], start=1):
        transformer.load_adapter(source["path"], adapter_name=f"source_{index}")
    _set_source_weights(transformer, sources)
    pipeline.transformer = transformer

    calibration = {"task": 0}
    handles = _attach_hooks(transformer, statistics, calibration)
    try:
        for index, source in enumerate(sources):
            calibration["task"] = index
            transformer.set_adapter(f"source_{index}", inference_mode=True)
            generator = torch.Generator(device="cpu").manual_seed(
                int(cfg.get("seed", 42))
            )
            log.info(
                "Calibrating SSR-Merge source %d/%d with %r",
                index + 1,
                len(sources),
                source["prompt"],
            )
            with torch.inference_mode():
                pipeline(
                    prompt=source["prompt"],
                    generator=generator,
                    num_inference_steps=int(cfg.get("calibration_steps", 1)),
                    height=int(cfg.get("height", 1024)),
                    width=int(cfg.get("width", 1024)),
                    guidance_scale=float(cfg.get("guidance_scale", 3.8)),
                    output_type="latent",
                )
    finally:
        _remove_hooks(handles, statistics, transformer)
    pipeline.transformer = transformer.unload()
    return pipeline.transformer


def _solve(statistics, regularization):
    merged, ranks, samples = {}, set(), {}
    for name, current in statistics.items():
        count = current["count"]
        if not count:
            raise ValueError(f"No calibration activations collected for {name}")
        dimension = current["down"].shape[0]
        correlation = current["correlation"].double() / count
        correlation += torch.eye(dimension, dtype=torch.float64) * regularization
        guide = current["guide"].double() / count
        try:
            router = torch.linalg.solve(correlation, guide.t()).t().float()
        except torch.linalg.LinAlgError as error:
            raise ValueError(f"Could not solve SSR router for {name}") from error
        merged[current["a_key"]] = current["down"]
        merged[current["b_key"]] = current["up"] @ router
        ranks.add(dimension)
        samples[name] = count

    if len(ranks) != 1:
        raise ValueError("All merged modules must have the same rank")
    return merged, ranks.pop(), samples


def compose(transformer, cfg, base_model=None, pipeline=None):
    if pipeline is None:
        raise ValueError("SSR-Merge requires a calibration pipeline")
    regularization = float(cfg.get("regularization", 1e-4))
    calibration_steps = int(cfg.get("calibration_steps", 1))
    if regularization < 0:
        raise ValueError("SSR-Merge regularization must be non-negative")
    if calibration_steps < 1:
        raise ValueError("SSR-Merge calibration_steps must be at least one")

    sources = _load_sources(cfg)
    statistics = _build_statistics(sources)
    transformer = _calibrate(pipeline, sources, statistics, cfg)
    merged, effective_rank, samples = _solve(statistics, regularization)
    adapter = install_adapter(
        transformer, merged, sources[0]["config"], effective_rank, base_model
    )
    manifest = {
        "base_model": base_model,
        "method": "ssr_merge",
        "sources": [
            {
                "path": source["path"],
                "sha256": source["sha256"],
                "weight": source["weight"],
                "prompt": source["prompt"],
            }
            for source in sources
        ],
        "effective_rank": effective_rank,
        "calibration_samples": samples,
        "configuration": resolved_configuration(cfg),
        "artifact_format": "adapter",
    }
    return adapter, manifest
