from peft import LoraConfig


def create_config(cfg_lora):
    target_modules = cfg_lora["target_modules"]

    config = LoraConfig(
        r=cfg_lora.get("r", 16),
        lora_alpha=cfg_lora.get("alpha", 32),
        target_modules=target_modules,
        lora_dropout=cfg_lora.get("dropout", 0.0),
        bias=cfg_lora.get("bias", "none"),
    )

    return config
