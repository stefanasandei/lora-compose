from peft import LoraConfig


def create_config(cfg_pissa):
    target_modules = cfg_pissa["target_modules"]

    config = LoraConfig(
        r=cfg_pissa.get("r", 16),
        lora_alpha=cfg_pissa.get("alpha", 32),
        target_modules=target_modules,
        lora_dropout=cfg_pissa.get("dropout", 0.0),
        bias=cfg_pissa.get("bias", "none"),
        init_lora_weights=cfg_pissa.get("init_lora_weights", "pissa_niter_16"),
    )

    return config
