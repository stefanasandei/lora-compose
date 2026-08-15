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
    sources = load_sources(cfg, method="IterIS", minimum=2)
    for source, configured in zip(sources, cfg.get("sources", [])):
        prompts = configured.get("prompts")
        if prompts is None:
            prompts = [configured.get("prompt")]
        elif isinstance(prompts, str):
            prompts = [prompts]
        source["prompts"] = [
            str(prompt).strip() for prompt in prompts if str(prompt).strip()
        ]
        if not source["prompts"]:
            raise ValueError("Each IterIS source requires calibration prompts")
        source["prompt"] = source["prompts"][0]
    return sources


def _build_layers(sources):
    layers = {}
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
                    "Per-module ranks are not supported by IterIS"
                )
            downs.append(down)
            ups.append(up * source_scale(source, rank))
            ranks.append(rank)

        name = a_key.removesuffix(".lora_A.weight")
        joint_down = torch.cat(downs, dim=0)
        basis = torch.linalg.qr(
            joint_down.t().double(), mode="reduced"
        ).Q.t().float()
        layers[name] = {
            "a_key": a_key,
            "b_key": b_key,
            "downs": downs,
            "up": torch.cat(ups, dim=1),
            "basis": basis,
            "ranks": ranks,
            "input_size": input_size,
        }
    return layers


def _capture_hook(features, offsets, size):
    def hook(module, inputs, _output):
        name = module._iteris_module_name
        current = inputs[0].detach()
        input_size = current.shape[-1]
        current = current.reshape(-1, input_size)
        if features[name] is None:
            features[name] = torch.zeros(
                size, input_size, device=current.device, dtype=torch.float32
            )

        positions = torch.arange(
            offsets[name], offsets[name] + current.shape[0],
            device=current.device,
        )
        hashed = (positions ^ (positions >> 16)) * 73244475
        hashed = (hashed ^ (hashed >> 16)) * 73244475
        hashed ^= hashed >> 16
        buckets = hashed.remainder(size)
        signs = torch.where(
            (hashed >> 32).bitwise_and(1).bool(),
            1.0, -1.0,
        )
        features[name].index_add_(
            0, buckets, current.float() * signs.unsqueeze(1)
        )
        offsets[name] += current.shape[0]

    return hook


def _capture_features(pipeline, layers, prompts, cfg):
    features = dict.fromkeys(layers)
    offsets = dict.fromkeys(layers, 0)
    limit = int(cfg.get("feature_samples", 128))
    modules = dict(pipeline.transformer.named_modules())
    missing = sorted(set(layers) - set(modules))
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"Could not find {len(missing)} target modules in transformer: {preview}"
        )

    handles = []
    seeds = int(cfg.get("calibration_seeds", 2))
    steps = int(cfg.get("calibration_steps", 8))
    hook = _capture_hook(features, offsets, limit)
    for name in layers:
        module = modules[name]
        module._iteris_module_name = name
        handles.append(module.register_forward_hook(hook))

    try:
        for prompt in prompts:
            for offset in range(seeds):
                generator = torch.Generator(device="cpu").manual_seed(
                    int(cfg.get("seed", 42)) + offset
                )
                with torch.inference_mode():
                    pipeline(
                        prompt=prompt,
                        generator=generator,
                        num_inference_steps=steps,
                        height=int(cfg.get("height", 1024)),
                        width=int(cfg.get("width", 1024)),
                        guidance_scale=float(
                            cfg.get("calibration_guidance_scale", 3.8)
                        ),
                        output_type="latent",
                    )
    finally:
        for handle in handles:
            handle.remove()
        for name in layers:
            module = modules[name]
            if hasattr(module, "_iteris_module_name"):
                delattr(module, "_iteris_module_name")

    missing = [name for name, values in features.items() if values is None]
    if missing:
        raise ValueError(f"No IterIS features collected for {missing[0]}")
    return {
        name: values.t().cpu() for name, values in features.items()
    }


def _source_features(pipeline, sources, layers, cfg):
    transformer = PeftModel.from_pretrained(
        pipeline.transformer, sources[0]["path"], adapter_name="source_0"
    )
    for index, source in enumerate(sources[1:], start=1):
        transformer.load_adapter(source["path"], adapter_name=f"source_{index}")
    for index, source in enumerate(sources):
        adapter_name = f"source_{index}"
        for module in transformer.modules():
            scaling = getattr(module, "scaling", None)
            if scaling is not None and adapter_name in scaling:
                scaling[adapter_name] *= source["weight"]
    pipeline.transformer = transformer

    captured = []
    try:
        log.info("Collecting IterIS source features from %d sources", len(sources))
        for index, source in enumerate(sources):
            transformer.set_adapter(f"source_{index}", inference_mode=True)
            captured.append(
                _capture_features(pipeline, layers, source["prompts"], cfg)
            )
    finally:
        pipeline.transformer = transformer.unload()
        torch.cuda.empty_cache()
    return captured


def _adaptive_weight(down, up, features):
    projected = down @ features
    output_norm = ((up.t() @ up) * (projected @ projected.t()).t()).sum()
    if not output_norm:
        return 0.0
    weight_norm = ((up.t() @ up) * (down @ down.t()).t()).sum()
    return (weight_norm / output_norm).item()


def _product_norm(left, right):
    """Frobenius norm of left @ right.T without a large dense product."""
    return ((left.t() @ left) * (right.t() @ right)).sum().sqrt()


def _subspace_solve(numerator, features, basis, ridge):
    """Solve the IterIS objective inside the joint source-LoRA row space."""
    if ridge <= 0:
        raise ValueError("IterIS requires positive effective regularization")
    projected = basis @ features
    denominator = projected @ projected.t()
    denominator += ridge * (basis @ basis.t())
    right = numerator @ basis.t()
    coefficients = torch.linalg.solve(
        denominator.t(), right.t()
    ).t()
    return coefficients @ basis


def _solve(layers, source_features, unified_features, alpha):
    merged, ranks, samples = {}, set(), {}
    for name, layer in layers.items():
        numerators = []
        weighted_features = []
        ridge = 0.0
        count = None
        for down, up, original, unified in zip(
            layer["downs"], layer["up"].split(layer["ranks"], dim=1),
            source_features, unified_features,
        ):
            x, merged_x = original[name].double(), unified[name].double()
            if x.shape != merged_x.shape:
                raise ValueError(f"Mismatched IterIS features for {name}")
            count = x.shape[1]
            weight = _adaptive_weight(down.double(), up.double(), x)
            cross_ridge = alpha * _product_norm(merged_x, x)
            gram_ridge = alpha * _product_norm(merged_x, merged_x)
            numerator = (down.double() @ x) @ merged_x.t()
            numerator += cross_ridge * down.double()
            numerators.append(weight * numerator)
            weighted_features.append(merged_x * weight**0.5)
            ridge += weight * gram_ridge

        numerator = torch.cat(numerators, dim=0)
        basis = layer["basis"].double()
        try:
            merged_down = _subspace_solve(
                numerator, torch.cat(weighted_features, dim=1), basis, ridge
            )
        except torch.linalg.LinAlgError as error:
            raise ValueError(f"Could not solve IterIS merge for {name}") from error
        merged_down = merged_down.float()
        merged[layer["a_key"]] = merged_down
        merged[layer["b_key"]] = layer["up"]
        ranks.add(merged_down.shape[0])
        samples[name] = count

    if len(ranks) != 1:
        raise ValueError("All merged modules must have the same rank")
    return merged, ranks.pop(), samples


def compose(transformer, cfg, base_model=None, pipeline=None):
    if pipeline is None:
        raise ValueError("IterIS requires a calibration pipeline")
    alpha = float(cfg.get("regularization", 1e-4))
    iterations = int(cfg.get("iterations", 5))
    if alpha < 0:
        raise ValueError("IterIS regularization must be non-negative")
    if iterations < 1:
        raise ValueError("IterIS iterations must be at least one")
    if int(cfg.get("calibration_steps", 1)) < 1:
        raise ValueError("IterIS calibration_steps must be at least one")
    if int(cfg.get("calibration_seeds", 1)) < 1:
        raise ValueError("IterIS calibration_seeds must be at least one")
    if int(cfg.get("feature_samples", 128)) < 1:
        raise ValueError("IterIS feature_samples must be at least one")

    sources = _load_sources(cfg)
    layers = _build_layers(sources)
    original = _source_features(pipeline, sources, layers, cfg)
    unified = original
    samples = {}
    for iteration in range(iterations):
        log.info("Solving IterIS iteration %d/%d", iteration + 1, iterations)
        merged, effective_rank, samples = _solve(
            layers, original, unified, alpha
        )
        if iteration + 1 < iterations:
            adapter = install_adapter(
                pipeline.transformer, merged, sources[0]["config"],
                effective_rank, base_model,
            )
            pipeline.transformer = adapter
            try:
                unified = [
                    _capture_features(
                        pipeline, layers, source["prompts"], cfg
                    )
                    for source in sources
                ]
            finally:
                pipeline.transformer = adapter.unload()
                torch.cuda.empty_cache()

    adapter = install_adapter(
        pipeline.transformer, merged, sources[0]["config"],
        effective_rank, base_model,
    )
    manifest = {
        "base_model": base_model,
        "method": "iteris",
        "sources": [
            {
                "path": source["path"],
                "sha256": source["sha256"],
                "weight": source["weight"],
                "prompt": source["prompt"],
                "prompts": source["prompts"],
            }
            for source in sources
        ],
        "effective_rank": effective_rank,
        "calibration_samples": samples,
        "configuration": resolved_configuration(cfg),
        "artifact_format": "adapter",
    }
    return adapter, manifest
