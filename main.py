"""
Description.
"""

import logging
import hydra
from omegaconf import DictConfig

from src.data_understanding.pipeline import DataUnderstandingPipeline
from src.data_preparation.pipeline import DataPreparationPipeline
from src.training_and_evalaution.pipeline import TrainingEvaluationPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@hydra.main(config_path="configs", config_name="main_config", version_base=None)
def main(cfg: DictConfig):
    
    if cfg.run_flags.data_understanding:
        logger.info("Running data understanding pipeline")
        pipeline = DataUnderstandingPipeline(cfg)
        pipeline.run()


    if cfg.run_flags.data_preparation:
        logger.info("Running data preparation pipeline")
        pipeline = DataPreparationPipeline(cfg)
        pipeline.run()

    if cfg.run_flags.run_training:
        logger.info("Running training and evalaution pipeline")
        pipeline = TrainingEvaluationPipeline(cfg)
        if cfg.training.use_uea:
            pipeline.run_uea()
            pipeline.plot_cd_all_metrics()
        elif cfg.training.use_truck_data:
            pipeline.run_truck_data()
        elif cfg.training.use_benchmark:
            pipeline.run_benchmarks()
        else:
            logger.warning("Select an option UEA or truck data or benchmark")

if __name__ == "__main__":
    main()
