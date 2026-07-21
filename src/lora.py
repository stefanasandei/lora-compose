import logging

from peft import LoraConfig, get_peft_model


log = logging.getLogger(__name__)


def apply_lora(transformer, cfg_lora: dict):
    # apply a PEFT LoRA adapter to the transformer and return it
    method = cfg_lora.get("method", "lora")
    target_modules = cfg_lora.get("target_modules", ["to_q", "to_k", "to_v", "to_out"])
    if not isinstance(target_modules, str):
        target_modules = list(target_modules)

    config = LoraConfig(
        r=cfg_lora.get("r", 16),
        lora_alpha=cfg_lora.get("alpha", 32),
        target_modules=target_modules,
        lora_dropout=cfg_lora.get("dropout", 0.0),
        bias=cfg_lora.get("bias", "none"),
    )

    transformer = get_peft_model(transformer, config)
    transformer.print_trainable_parameters()
    return transformer
