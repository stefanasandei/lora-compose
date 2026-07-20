from diffusers import SanaPipeline
import torch
from typing import Optional
import logging


def get_sana_pipeline(cache_dir: Optional[str] = None):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    pipe = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/SANA_600M_1024px_diffusers",
        cache_dir=cache_dir,
        variant="fp16",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)

    pipe.set_progress_bar_config(disable=True)

    return pipe
