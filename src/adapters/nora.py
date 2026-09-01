import json
import re
from pathlib import Path
from types import MethodType

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

CONFIG_FILE = "adapter_config.json"
WEIGHTS_FILE = "adapter_model.safetensors"


class NoraLinear(nn.Module):
    """A frozen linear layer with a rank-normalized LoRA update."""

    def __init__(self, base_layer, rank, alpha, dropout, eps):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("NoRA currently supports nn.Linear layers only")

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.lora_A = nn.Linear(
            base_layer.in_features, rank, bias=False,
            device=base_layer.weight.device, dtype=base_layer.weight.dtype,
        )
        self.lora_B = nn.Linear(
            rank, base_layer.out_features, bias=False,
            device=base_layer.weight.device, dtype=base_layer.weight.dtype,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.scaling = alpha / rank
        self.eps = eps

    def forward(self, x):
        column_norms = self.lora_A.weight.norm(dim=0, keepdim=True)
        normalized_A = self.lora_A.weight / column_norms.clamp_min(self.eps)
        update = F.linear(self.dropout(x), normalized_A)
        update = self.lora_B(update) * self.scaling
        return self.base_layer(x) + update


def _target_names(transformer, target_modules):
    patterns = [target_modules] if isinstance(target_modules, str) else target_modules
    names = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear) and any(
            re.fullmatch(pattern, name) or name.endswith(f".{pattern}")
            for pattern in patterns
        ):
            names.append(name)
    if not names:
        raise ValueError("No NoRA target modules matched the transformer")
    return names


def _replace_module(root, name, replacement):
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def apply(transformer, cfg_nora):
    target_modules = cfg_nora["target_modules"]
    rank = int(cfg_nora.get("r", 16))
    alpha = float(cfg_nora.get("alpha", rank))
    dropout = float(cfg_nora.get("dropout", 0.0))
    eps = float(cfg_nora.get("eps", 1.0e-6))
    if rank < 1:
        raise ValueError("NoRA rank must be positive")
    if eps <= 0:
        raise ValueError("NoRA eps must be positive")

    for name in _target_names(transformer, target_modules):
        base_layer = transformer.get_submodule(name)
        _replace_module(
            transformer, name,
            NoraLinear(base_layer, rank, alpha, dropout, eps),
        )
    transformer.save_pretrained = MethodType(
        lambda model, output_dir, **_: save_pretrained(
            model, output_dir, cfg_nora
        ),
        transformer,
    )
    return transformer


def _configuration(cfg_nora):
    return {
        "peft_type": "LORA",
        "nora": True,
        "r": int(cfg_nora.get("r", 16)),
        "lora_alpha": float(cfg_nora.get("alpha", cfg_nora.get("r", 16))),
        "lora_dropout": float(cfg_nora.get("dropout", 0.0)),
        "target_modules": cfg_nora["target_modules"],
        "bias": "none",
        "init_lora_weights": False,
        "nora_eps": float(cfg_nora.get("eps", 1.0e-6)),
    }


def save_pretrained(transformer, output_dir, cfg_nora):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, module in transformer.named_modules():
        if isinstance(module, NoraLinear):
            norms = module.lora_A.weight.norm(dim=0, keepdim=True)
            normalized_A = module.lora_A.weight / norms.clamp_min(module.eps)
            prefix = f"base_model.model.{name}"
            tensors[f"{prefix}.lora_A.weight"] = normalized_A.detach().cpu()
            tensors[f"{prefix}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    if not tensors:
        raise ValueError("No NoRA layers found to save")
    save_file(tensors, str(output_dir / WEIGHTS_FILE))
    with (output_dir / CONFIG_FILE).open("w", encoding="utf-8") as file:
        json.dump(_configuration(cfg_nora), file, indent=2)
        file.write("\n")


def load(transformer, adapter_dir):
    adapter_dir = Path(adapter_dir)
    with (adapter_dir / CONFIG_FILE).open(encoding="utf-8") as file:
        config = json.load(file)
    if not config.get("nora"):
        raise ValueError(f"Not a NoRA adapter: {adapter_dir}")

    transformer = apply(transformer, {
        "target_modules": config["target_modules"],
        "r": config["r"],
        "alpha": config["lora_alpha"],
        "dropout": config.get("lora_dropout", 0.0),
        "eps": config.get("nora_eps", 1.0e-6),
    })
    tensors = load_file(str(adapter_dir / WEIGHTS_FILE), device="cpu")
    expected = {
        f"base_model.model.{name}.lora_{factor}.weight": getattr(
            module, "lora_" + factor
        ).weight
        for name, module in transformer.named_modules()
        if isinstance(module, NoraLinear)
        for factor in ("A", "B")
    }
    if set(tensors) != set(expected):
        raise ValueError("NoRA checkpoint factors do not match the transformer")
    with torch.no_grad():
        for key, parameter in expected.items():
            parameter.copy_(tensors[key].to(parameter))
    return transformer
