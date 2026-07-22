from peft import get_peft_model

from lora import apply_lora

methods = {"lora": apply_lora}


def apply_adapter(transformer, cfg_adapter: dict):
    peft_config = methods[cfg_adapter.method](transformer, cfg_adapter)

    transformer = get_peft_model(transformer, peft_config)
    transformer.print_trainable_parameters()
    return transformer
