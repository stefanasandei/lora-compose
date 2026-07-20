import hydra
from omegaconf import DictConfig
import logging
import json
import os

from models.sana import get_sana_pipeline
from reference import gen_reference_dataset

log = logging.getLogger(__name__)


def run_eval(cfg: DictConfig) -> None:
    # pipe = get_sana_pipeline(cache_dir=cfg.cache_dir)
    log.info("Running evaluation.")

    with open(f"{cfg.dataset_dir}/evals/prompts.json") as f:
        prompts = json.loads(f.read())

    if not os.path.isdir(f"{cfg.dataset_dir}/evals/reference"):
        log.info("Reference dataset not present, generating...")
        gen_reference_dataset()

    log.info("Evaluation done.")


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
