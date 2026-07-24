from peft import HRAConfig


def create_config(cfg_hra):
    return HRAConfig(
        r=cfg_hra.get("r", 64),
        apply_GS=cfg_hra.get("apply_gs", False),
        target_modules=cfg_hra["target_modules"],
        bias=cfg_hra.get("bias", "none"),
        init_weights=cfg_hra.get("init_weights", True),
    )
