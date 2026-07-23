from peft import OFTConfig


def create_config(cfg_oftv2):
    target_modules = cfg_oftv2["target_modules"]

    config = OFTConfig(
        oft_block_size=cfg_oftv2.get("block_size", 32),
        target_modules=target_modules,
        module_dropout=cfg_oftv2.get("dropout", 0.0),
        bias=cfg_oftv2.get("bias", "none"),
        use_cayley_neumann=True,
        num_cayley_neumann_terms=cfg_oftv2.get("num_cayley_neumann_terms", 5),
    )

    return config
