from peft import LoKrConfig


def create_config(cfg_lokr):
    target_modules = cfg_lokr["target_modules"]

    config = LoKrConfig(
        r=cfg_lokr.get("r", 16),
        alpha=cfg_lokr.get("alpha", 32),
        target_modules=target_modules,
        rank_dropout=cfg_lokr.get("rank_dropout", 0.0),
        module_dropout=cfg_lokr.get("module_dropout", 0.0),
        decompose_both=cfg_lokr.get("decompose_both", False),
        decompose_factor=cfg_lokr.get("decompose_factor", -1),
    )

    return config
