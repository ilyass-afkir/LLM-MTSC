"""
Description
"""

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import gc
import json
import logging
import time

import numpy as np
from aeon.visualisation import plot_critical_difference
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

from src.data_preparation.modules.constants import UEA
from src.data_preparation.modules.uea_data_loaders import UEADataFactory
from src.model.one_fits_all import OneFitsAll
from src.model.llm_few import LLMFew
from src.model.time_llm import TimeLLM
from src.model.s2ip_tempo import S2IPTempo
from src.model.deep_range import DeepRange
from src.model.deep_range_v2 import DeepRangeV2
from src.training_and_evalaution.trainer import Trainer
from src.training_and_evalaution.evaluator import UEAEvaluator
from src.utils.model_summarizer import ModelSummarizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class TrainingEvaluationPipeline:
    def __init__(self, cfg:DictConfig):
        self.cfg = cfg

    def _setup_dirs(self):
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def run_uea(self):

        model_summarizer = ModelSummarizer(self.cfg, self.paths["main"])
        evaluator = UEAEvaluator(self.cfg)
        training_times = {} 
   
        def run_single_dataset(name, info, train_loader, test_loader):
 
            if self.cfg.model.name == "one_fits_all":
                model = OneFitsAll(self.cfg)
            elif self.cfg.model.name == "llm_few":
                model = LLMFew(self.cfg)
            elif self.cfg.model.name == "time_llm":
                model = TimeLLM(self.cfg)
            elif self.cfg.model.name == "s2ip_tempo":
                model = S2IPTempo(self.cfg)
            elif self.cfg.model.name == "deep_range":
                model = DeepRange(self.cfg)
            elif self.cfg.model.name == "deep_range_v2":
                model = DeepRangeV2(self.cfg)
            else:
                raise ValueError(f"Unknown model: {self.cfg.model.name}")

            # Summarize model parameters
            _ = model_summarizer.summarize_model_parameters(model, name)
          
            # Train
            trainer = Trainer(
                cfg=self.cfg,
                model=model,
                train_loader=train_loader,
                val_loader=None,
                save_dir=self.paths["main"]
            )

            model, time = trainer.fit()
            training_times[name] = time
            
            # Evaluate
            evaluator.evaluate_model(model=model, test_loader=test_loader, dataset_name=name)

            # Clean up
            self._cleanup_objects([model.llm, model, trainer, train_loader, test_loader])           
           
        # Run all datasets
        for name, info in UEA.items():

            OmegaConf.update(self.cfg, "training.num_channels", info["num_channels"])
            OmegaConf.update(self.cfg, "training.num_classes", info["num_classes"])
            OmegaConf.update(self.cfg, "training.sequence_length", info["sequence_length"])
            OmegaConf.update(self.cfg, "training.batch_size", info["batch_size"])
            OmegaConf.update(self.cfg, "model.patch_stride", info["patch_stride"])
            OmegaConf.update(self.cfg, "model.patch_length", info["patch_length"])

            uea_factory = UEADataFactory(
                name,
                Path(self.cfg.training.uea_data_dir_path),
                self.cfg.training.batch_size,
                self.cfg.training.num_workers
            )

            train_loader, test_loader = uea_factory.build_train_test_loaders()
        
            if train_loader is None or test_loader is None:
                logger.warning(f"Skipping dataset {name} due to loading issues")
                continue

            run_single_dataset(name, info, train_loader, test_loader)

        # Save results
        _ = model_summarizer.average_model_summaries()
       
        training_times_output = {
            "individual_training_times": training_times,
            "total_training_time": sum(training_times.values())
        }

        with open(self.paths["main"] / "training_times.json", "w") as f:
            json.dump(training_times_output, f, indent=4)

        evaluator.save_evaluation_report()

    @staticmethod
    def plot_cd_all_metrics():
        
        font_path_normal = "llm-erange/src/utils/times.ttf"
        font_path_bold   = "llm-erange/src/utils/times_bold.ttf"
        fm.fontManager.addfont(font_path_normal)
        fm.fontManager.addfont(font_path_bold)
        prop_normal = fm.FontProperties(fname=font_path_normal)
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12

        base_path = Path("/mnt/nvme3/ilafkir/results/training/uea_normalized/final")
        json_files = [
            base_path / "one_fits_all/evaluation_report.json",
            base_path / "time_llm/evaluation_report.json",
            base_path /  "llm_few/evaluation_report.json",
            base_path /  "s2ip_tempo/evaluation_report.json",
            base_path / "dp_v1/evaluation_report.json",
            base_path / "dp_v2/evaluation_report.json",
        ]

        # Load all results
        all_results = {}
        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f)
                all_results[data['model_name']] = data['per_dataset_results']

        # Get common datasets
        datasets = set.intersection(*[set(r.keys()) for r in all_results.values()])
        datasets = sorted(datasets)

        labels = list(all_results.keys())
        labels_for_plot = ["One Fits All", "Time-LLM", "LLM-Few", "S$^2$IP-TEMPO", "Deep Range V1", "Deep Range V2"]
        metrics = ['accuracy', 'f1_score', 'b_accuracy']

        all_pvals = {}

        for metric in metrics:
            scores_matrix = []
            for dataset in datasets:
                row = [all_results[clf][dataset][metric] for clf in labels]
                scores_matrix.append(row)

            scores = np.array(scores_matrix)

            fig, _, p_vals = plot_critical_difference(
                scores=scores,
                labels=labels_for_plot,
                return_p_values=True
            )

            # Save p-values and classifier order
            all_pvals[metric] = {
                "labels": labels,
                "p_values": p_vals.tolist()
            }

            png_path = base_path / f'comp_uea_{metric}_with_dp.png'
            pdf_path = base_path / f'comp_uea_{metric}_with_dp.pdf'
            fig.savefig(pdf_path, bbox_inches='tight', facecolor='white', dpi=300)
            fig.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=300)

        # Save everything in one JSON
        with open(base_path / "p_values_with_dp.json", "w") as f:
            json.dump(all_pvals, f, indent=4)

    def _cleanup_objects(self, objs):
        for o in objs:
            try:
                # Move model to CPU first
                if hasattr(o, "to"):  # Check if it's a model/tensor
                    o.to("cpu")
                
                # Move nested LLM to CPU if it exists
                if hasattr(o, "llm") and hasattr(o.llm, "to"):
                    o.llm.to("cpu")
                
                del o
            except Exception as e:
                # Consider logging the exception for debugging
                pass

        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()


    
