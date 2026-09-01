import torch
from peft import get_peft_model

from . import coftv2, dora, loha, lokr, lora, nora, oftv2, peanut, pissa

methods = {
    "coftv2": coftv2,
    "dora": dora,
    "loha": loha,
    "lokr": lokr,
    "lora": lora,
    "nora": nora,
    "oftv2": oftv2,
    "peanut": peanut,
    "pissa": pissa,
}


def apply_adapter(transformer, cfg_adapter: dict, **prepare_kwargs):
    method = methods[cfg_adapter.method]
    if hasattr(method, "apply"):
        transformer = method.apply(transformer, cfg_adapter)
        if hasattr(transformer, "print_trainable_parameters"):
            transformer.print_trainable_parameters()
        return transformer
    if hasattr(method, "prepare"):
        method.prepare(transformer, cfg_adapter, **prepare_kwargs)

    peft_config = method.create_config(cfg_adapter)
    transformer = get_peft_model(transformer, peft_config)
    if hasattr(method, "setup"):
        method.setup(transformer, cfg_adapter)

    transformer.print_trainable_parameters()
    return transformer


def finalize_adapter(transformer, cfg_adapter: dict):
    method = methods[cfg_adapter.method]
    if hasattr(method, "finalize"):
        method.finalize(transformer)


def create_optimizer(transformer, cfg_adapter: dict, cfg_train: dict):
    method = methods[cfg_adapter.method]
    if hasattr(method, "create_optimizer"):
        return method.create_optimizer(transformer, cfg_adapter, cfg_train)

    params = filter(lambda param: param.requires_grad, transformer.parameters())
    return torch.optim.AdamW(params, lr=cfg_train.lr, weight_decay=cfg_train.weight_decay)
