from peft import BOFTConfig


def create_config(cfg_boft):
    return BOFTConfig(
        boft_block_size=cfg_boft.get("block_size", 16),
        boft_n_butterfly_factor=cfg_boft.get("num_butterfly_factors", 4),
        target_modules=cfg_boft["target_modules"],
        boft_dropout=cfg_boft.get("dropout", 0.0),
        bias=cfg_boft.get("bias", "none"),
        init_weights=cfg_boft.get("init_weights", True),
    )


def setup(transformer, cfg_boft):
    expected = cfg_boft.get("num_butterfly_factors", 4)
    observed = {
        parameter.shape[0]
        for module in transformer.modules()
        if hasattr(module, "boft_R")
        for parameter in module.boft_R.values()
    }
    if observed != {expected}:
        raise RuntimeError(
            "PEFT did not initialize the requested BOFT butterfly factors "
            f"(expected {expected}, observed {sorted(observed)}). Check the "
            "preceding CUDA-extension warning and ensure that CUDA and ninja "
            "are available before training."
        )
