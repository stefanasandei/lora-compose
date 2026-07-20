import hydra
from omegaconf import DictConfig
import logging

# from models.sana import get_sana_pipeline

log = logging.getLogger(__name__)


def run_training(cfg: DictConfig) -> None:
    # pipe = get_sana_pipeline(cache_dir=cfg.cache_dir)
    log.info("Pipeline loaded, starting training...")


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()