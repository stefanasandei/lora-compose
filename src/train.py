import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


def run_training(cfg: DictConfig) -> None:
    log.info("hi")

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: DictConfig) -> None:
    run_training(cfg)

if __name__ == "__main__":
    main()
