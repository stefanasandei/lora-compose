import hashlib
import json
import math
import os
from pathlib import Path

from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file


CONFIG_FILE = "adapter_config.json"
WEIGHTS_FILE = "adapter_model.safetensors"


def _hash_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_adapter(source, method="Sum"):
    """Load and minimally validate one standard PEFT LoRA."""
    path = Path(source.get("path")).expanduser().resolve()
    config_path, weights_path = path / CONFIG_FILE, path / WEIGHTS_FILE
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Invalid PEFT adapter directory: {path}")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    if config.get("peft_type") != "LORA" or any(
        config.get(key) for key in ("use_dora", "use_qalora", "target_parameters")
    ):
        raise ValueError(f"{method} only supports standard LoRA adapters: {path}")
    if config.get("bias", "none") != "none" or config.get("modules_to_save"):
        raise ValueError(
            f"{method} does not support extra adapter parameters: {path}"
        )

    tensors = load_file(str(weights_path), device="cpu")
    a_keys = {key.removesuffix(".lora_A.weight") for key in tensors
              if key.endswith(".lora_A.weight")}
    b_keys = {key.removesuffix(".lora_B.weight") for key in tensors
              if key.endswith(".lora_B.weight")}
    if not a_keys or a_keys != b_keys or len(tensors) != 2 * len(a_keys):
        raise ValueError(f"Invalid LoRA factor pairs: {path}")
    return {
        "path": str(path), "weight": float(source.get("weight", 1.0)),
        "config": config, "tensors": tensors,
        "sha256": _hash_files((config_path, weights_path)),
    }


def load_sources(cfg, method, minimum=1, require_prompts=False):
    sources = []
    for configured in cfg.get("sources", []):
        source = load_adapter(configured, method=method)
        if require_prompts:
            source["prompt"] = str(configured.get("prompt", "")).strip()
            if not source["prompt"]:
                raise ValueError(f"Each {method} source requires a prompt")
        sources.append(source)
    if len(sources) < minimum:
        raise ValueError(
            f"{method} composition requires at least {minimum} sources"
        )

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
    return sources


def source_scale(source, rank):
    config = source["config"]
    denominator = math.sqrt(rank) if config.get("use_rslora") else rank
    return source["weight"] * config["lora_alpha"] / denominator


def resolved_configuration(cfg):
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def install_adapter(transformer, tensors, source_config, rank, base_model):
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=source_config["target_modules"],
        fan_in_fan_out=source_config.get("fan_in_fan_out", False),
        bias="none",
        init_lora_weights=False,
        inference_mode=True,
    )
    adapter = get_peft_model(transformer, config)
    result = set_peft_model_state_dict(adapter, tensors)
    missing = [key for key in result.missing_keys if ".lora_" in key]
    if missing or result.unexpected_keys:
        raise ValueError(
            f"Could not install factors: {missing}, {result.unexpected_keys}"
        )
    adapter.peft_config["default"].base_model_name_or_path = base_model
    adapter.eval()
    return adapter


def save_composition(adapter, manifest, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    adapter.save_pretrained(output_dir, safe_serialization=True)
    artifact_paths = (
        Path(output_dir) / CONFIG_FILE,
        Path(output_dir) / WEIGHTS_FILE,
    )
    manifest["artifact_bytes"] = sum(path.stat().st_size for path in artifact_paths)
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    with open(
        os.path.join(output_dir, "composition.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    return manifest
