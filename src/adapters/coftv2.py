from peft import OFTConfig


def create_config(cfg_coftv2):
    target_modules = cfg_coftv2["target_modules"]

    config = OFTConfig(
        oft_block_size=cfg_coftv2.get("block_size", 32),
        target_modules=target_modules,
        module_dropout=cfg_coftv2.get("dropout", 0.0),
        bias=cfg_coftv2.get("bias", "none"),
        coft=True,
        eps=cfg_coftv2.get("eps", 6.0e-5),
        use_cayley_neumann=True,
        num_cayley_neumann_terms=cfg_coftv2.get("num_cayley_neumann_terms", 5),
    )

    return config
