from peft import get_peft_model

from . import lora


methods = {
    "lora": lora,
}


def apply_adapter(transformer, cfg_adapter: dict):
    peft_config = methods[cfg_adapter.method].create_config(cfg_adapter)

    transformer = get_peft_model(transformer, peft_config)
    transformer.print_trainable_parameters()
    return transformer
