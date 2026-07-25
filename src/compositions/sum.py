import math

import torch
from omegaconf import OmegaConf

from .common import install_adapter, load_adapter


def compose(transformer, cfg, base_model=None):
    sources = [load_adapter(source) for source in cfg.get("sources", [])]
    if not sources:
        raise ValueError("Sum composition requires at least one source")

    first = sources[0]
    keys = set(first["tensors"])
    structure = (
        first["config"].get("target_modules"),
        first["config"].get("fan_in_fan_out", False),
    )
    for source in sources[1:]:
        other_structure = (
            source["config"].get("target_modules"),
            source["config"].get("fan_in_fan_out", False),
        )
        if set(source["tensors"]) != keys or other_structure != structure:
            raise ValueError("Source adapters target different modules")

    merged, ranks = {}, set()
    for a_key in sorted(key for key in keys if key.endswith(".lora_A.weight")):
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        downs, ups = [], []
        input_size = first["tensors"][a_key].shape[1]
        output_size = first["tensors"][b_key].shape[0]
        for source in sources:
            down, up = source["tensors"][a_key], source["tensors"][b_key]
            rank = down.shape[0]
            if down.shape[1] != input_size or up.shape != (output_size, rank):
                raise ValueError(f"Incompatible factors for {a_key}")
            config = source["config"]
            if (
                rank != config["r"]
                or config.get("rank_pattern")
                or config.get("alpha_pattern")
            ):
                raise ValueError("Per-module ranks are not supported by sum")
            alpha = config["lora_alpha"]
            scale = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)
            downs.append(down)
            ups.append(up * source["weight"] * scale)
        merged[a_key] = torch.cat(downs, dim=0)
        merged[b_key] = torch.cat(ups, dim=1)
        ranks.add(merged[a_key].shape[0])

    if len(ranks) != 1:
        raise ValueError("All merged modules must have the same rank")
    effective_rank = ranks.pop()
    adapter = install_adapter(
        transformer, merged, first["config"], effective_rank, base_model
    )
    configuration = (
        OmegaConf.to_container(cfg, resolve=True)
        if OmegaConf.is_config(cfg) else dict(cfg)
    )
    manifest = {
        "base_model": base_model,
        "method": "sum",
        "sources": [
            {key: source[key] for key in ("path", "sha256", "weight")}
            for source in sources
        ],
        "effective_rank": effective_rank,
        "configuration": configuration,
        "artifact_format": "adapter",
    }
    return adapter, manifest
