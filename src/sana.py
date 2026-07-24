from contextlib import contextmanager
import logging
from diffusers import SanaPipeline
import torch
from typing import Optional


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
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.set_progress_bar_config(disable=True)

    return pipe


def offload_frozen_components(pipe: SanaPipeline) -> None:
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()


@contextmanager
def sampling_components(pipe: SanaPipeline):
    device = pipe.transformer.device
    pipe.text_encoder.to(device)
    pipe.vae.to(device)
    try:
        yield
    finally:
        offload_frozen_components(pipe)
