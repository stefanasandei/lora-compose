from peft import PeanutConfig


def create_config(cfg_peanut):
    return PeanutConfig(
        r=cfg_peanut.get("r", 32),
        depth=cfg_peanut.get("depth", 0),
        act_fn=cfg_peanut.get("act_fn", "relu"),
        scaling=cfg_peanut.get("scaling", 1.0),
        target_modules=cfg_peanut["target_modules"],
        init_weights=cfg_peanut.get("init_weights", True),
    )
