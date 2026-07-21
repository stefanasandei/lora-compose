from diffusers import SanaPipeline
import torch
from typing import Optional
import logging


def get_sana_pipeline(model_name_or_path: str, cache_dir: Optional[str] = None):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    pipe = SanaPipeline.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    pipe.text_encoder.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)

    return pipe
