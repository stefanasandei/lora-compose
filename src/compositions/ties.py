import torch

from .common import (
    install_adapter,
    load_sources,
    resolved_configuration,
    source_scale,
)


def _trim(task_vectors, density):
    """Keep the largest-magnitude entries of every task vector."""
    if density == 1:
        return task_vectors
    keep = max(1, int(task_vectors.shape[1] * density))
    indices = task_vectors.abs().topk(keep, dim=1, sorted=False).indices
    mask = torch.zeros_like(task_vectors, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return task_vectors * mask


def _disjoint_mean(task_vectors):
    """Elect signs by total mass, then average only agreeing nonzero entries."""
    elected = task_vectors.sum(dim=0).sign()
    selected = (task_vectors.sign() == elected) & (task_vectors != 0)
    count = selected.sum(dim=0)
    total = (task_vectors * selected).sum(dim=0)
    return torch.where(count > 0, total / count.clamp_min(1), 0.0)


def _factorize(delta, rank, seed=42):
    """Return LoRA B/A factors for a rank-constrained dense task vector."""
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


def compose(transformer, cfg, base_model=None):
    density = float(cfg.get("density", 0.2))
    merge_scale = float(cfg.get("merge_scale", 1.0))
    seed = int(cfg.get("seed", 42))
    if not 0 < density <= 1:
        raise ValueError("TIES density must be in (0, 1]")
    if merge_scale < 0:
        raise ValueError("TIES merge_scale must be non-negative")
    if cfg.get("rank") is not None and int(cfg.rank) < 1:
        raise ValueError("TIES rank must be at least one")

    sources = load_sources(cfg, method="TIES", minimum=2)
    configured_rank = cfg.get("rank")
    merged, ranks = {}, set()
    retained = {}
    errors = {}
    first = sources[0]
    for a_key in sorted(
        key for key in first["tensors"] if key.endswith(".lora_A.weight")
    ):
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        task_vectors = []
        source_ranks = []
        for source in sources:
            down = source["tensors"][a_key].float()
            up = source["tensors"][b_key].float()
            rank = down.shape[0]
            if up.shape[1] != rank:
                raise ValueError(f"Incompatible factors for {a_key}")
            config = source["config"]
            if (
                rank != config["r"]
                or config.get("rank_pattern")
                or config.get("alpha_pattern")
            ):
                raise ValueError("Per-module ranks are not supported by TIES")
            task_vectors.append(
                (up @ down * source_scale(source, rank)).flatten()
            )
            source_ranks.append(rank)

        task_vectors = _trim(torch.stack(task_vectors), density)
        dense = _disjoint_mean(task_vectors).reshape(
            first["tensors"][b_key].shape[0],
            first["tensors"][a_key].shape[1],
        )
        dense *= merge_scale
        output_rank = int(configured_rank or sum(source_ranks))
        up, down = _factorize(dense, output_rank, seed=seed)
        merged[a_key], merged[b_key] = down, up
        ranks.add(down.shape[0])
        module = a_key.removesuffix(".lora_A.weight")
        retained[module] = (task_vectors != 0).float().mean().item()
        denominator = torch.linalg.vector_norm(dense)
        approximation = up @ down
        errors[module] = (
            (torch.linalg.vector_norm(dense - approximation) / denominator).item()
            if denominator else 0.0
        )

    if len(ranks) != 1:
        raise ValueError(
            "TIES output rank differs between modules; choose a rank no larger "
            "than the smallest target matrix dimension"
        )
    effective_rank = ranks.pop()
    adapter = install_adapter(
        transformer, merged, first["config"], effective_rank, base_model
    )
    manifest = {
        "base_model": base_model,
        "method": "ties",
        "sources": [
            {key: source[key] for key in ("path", "sha256", "weight")}
            for source in sources
        ],
        "effective_rank": effective_rank,
        "retained_density": retained,
        "relative_approximation_error": errors,
        "configuration": resolved_configuration(cfg),
        "artifact_format": "adapter",
    }
    return adapter, manifest
