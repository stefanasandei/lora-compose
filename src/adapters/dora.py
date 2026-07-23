from peft import LoraConfig


def create_config(cfg_dora):
    target_modules = cfg_dora["target_modules"]

    config = LoraConfig(
        r=cfg_dora.get("r", 16),
        lora_alpha=cfg_dora.get("alpha", 32),
        target_modules=target_modules,
        lora_dropout=cfg_dora.get("dropout", 0.0),
        bias=cfg_dora.get("bias", "none"),
        use_dora=True,
    )

    return config
