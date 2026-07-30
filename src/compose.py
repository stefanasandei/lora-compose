import logging

import hydra
import torch
from diffusers import SanaTransformer2DModel
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from compositions import compose, requires_pipeline
from compositions.common import save_composition
from sana import get_sana_pipeline


log = logging.getLogger(__name__)


def load_transformer(cfg):
    return SanaTransformer2DModel.from_pretrained(
        cfg.model_name_or_path,
        subfolder="transformer",
        cache_dir=cfg.get("cache_dir"),
        torch_dtype=torch.float32,
    )


def run_composition(cfg: DictConfig):
    composition_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.composition, resolve=True)
    )
    for source in composition_cfg.sources:
        source.path = to_absolute_path(source.path)
    pipeline = None
    if requires_pipeline(composition_cfg.method):
        pipeline = get_sana_pipeline(
            model_name_or_path=cfg.model_name_or_path,
            cache_dir=cfg.get("cache_dir"),
        )
        pipeline.vae.to("cpu")
        transformer = pipeline.transformer
    else:
        transformer = load_transformer(cfg)
    composition_kwargs = {"base_model": str(cfg.model_name_or_path)}
    if pipeline is not None:
        composition_kwargs["pipeline"] = pipeline
    adapter, manifest = compose(
        transformer, composition_cfg, **composition_kwargs
    )
    output_dir = to_absolute_path(cfg.output_dir)
    manifest = save_composition(adapter, manifest, output_dir)
    log.info(
        "Saved %s composition %s to %s",
        manifest["method"],
        manifest["manifest_hash"],
        output_dir,
    )
    return manifest


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="composition/sum",
)
def main(cfg: DictConfig):
    run_composition(cfg)


if __name__ == "__main__":
    main()
