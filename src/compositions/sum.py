import torch

from .common import (
    install_adapter,
    load_sources,
    resolved_configuration,
    source_scale,
)


def compose(transformer, cfg, base_model=None):
    sources = load_sources(cfg, method="Sum")
    first = sources[0]
    keys = set(first["tensors"])

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
            downs.append(down)
            ups.append(up * source_scale(source, rank))
        merged[a_key] = torch.cat(downs, dim=0)
        merged[b_key] = torch.cat(ups, dim=1)
        ranks.add(merged[a_key].shape[0])

    if len(ranks) != 1:
        raise ValueError("All merged modules must have the same rank")
    effective_rank = ranks.pop()
    adapter = install_adapter(
        transformer, merged, first["config"], effective_rank, base_model
    )
    manifest = {
        "base_model": base_model,
        "method": "sum",
        "sources": [
            {key: source[key] for key in ("path", "sha256", "weight")}
            for source in sources
        ],
        "effective_rank": effective_rank,
        "configuration": resolved_configuration(cfg),
        "artifact_format": "adapter",
    }
    return adapter, manifest
