from peft import LoHaConfig


def create_config(cfg_loha):
    target_modules = cfg_loha["target_modules"]

    config = LoHaConfig(
        r=cfg_loha.get("r", 16),
        alpha=cfg_loha.get("alpha", 32),
        target_modules=target_modules,
        rank_dropout=cfg_loha.get("rank_dropout", 0.0),
        module_dropout=cfg_loha.get("module_dropout", 0.0),
    )

    return config
