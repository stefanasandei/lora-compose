from peft import LoHaConfig


def create_config(cfg_loha):
    return LoHaConfig(
        r=cfg_loha.get("r", 32),
        alpha=cfg_loha.get("alpha", 32),
        target_modules=cfg_loha["target_modules"],
        rank_dropout=cfg_loha.get("rank_dropout", 0.0),
        module_dropout=cfg_loha.get("module_dropout", 0.0),
        init_weights=cfg_loha.get("init_weights", True),
    )
