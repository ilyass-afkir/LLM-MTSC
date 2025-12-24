"""
Description.
"""

import logging
import hydra
from omegaconf import DictConfig

from src.training_and_evalaution.pipeline import TrainingEvaluationPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@hydra.main(config_path="configs", config_name="main_config", version_base=None)
def main(cfg: DictConfig):
    
    if cfg.run_flags.run_training:
        logger.info("Running training and evalaution pipeline")
        pipeline = TrainingEvaluationPipeline(cfg)
        if cfg.training.use_uea:
            pipeline.run_uea()
            pipeline.plot_cd_all_metrics()
 
if __name__ == "__main__":
    main()
