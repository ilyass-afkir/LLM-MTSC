"""
Comprehensive model evaluator for test set evaluation.
"""

import logging
import json
from typing import Dict
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, 
    cohen_kappa_score, 
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class UEAEvaluator:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.model_name = self.cfg.model.name
        self.device = self.cfg.training.device
        self.save_results_dir_path = Path(self.cfg.training.save_results_dir_path)
        self.save_results_dir_path.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def evaluate_model(self, model, test_loader, dataset_name):
        
        model.to(self.device)
        model.eval()

        preds_list = []
        targets_list = []

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device, dtype=torch.bfloat16)
                targets = targets.to(self.device)

                outputs = model(data)
                preds = outputs.argmax(dim=-1)

                preds_list.append(preds.cpu())
                targets_list.append(targets.cpu())

        preds = torch.cat(preds_list).numpy()
        targets = torch.cat(targets_list).numpy()

        acc = accuracy_score(targets, preds)
        b_acc = balanced_accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="weighted")

        self.results[dataset_name] = {
            "accuracy": float(acc),
            "b_accuracy": float(b_acc),
            "f1_score": float(f1),
            "num_samples": len(targets),
        }

        logger.info(f"{dataset_name}: Accuracy={acc:.4f}, Balanced Accuracy={b_acc:.4f}, F1={f1:.4f}")
        
        return self.results[dataset_name]
    
    def compute_average_metrics(self) -> Dict[str, float]:
        
        accuracies = [r['accuracy'] for r in self.results.values()]
        f1_scores = [r['f1_score'] for r in self.results.values()]
        b_accuracies = [r['b_accuracy'] for r in self.results.values()]
        
        avg_metrics = {
            'avg_accuracy': float(np.mean(accuracies)),
            'avg_f1_score': float(np.mean(f1_scores)),
            "avg_b_accuracy": float(np.mean(b_accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'std_f1_score': float(np.std(f1_scores)),
            "std_b_accuracy": float(np.std(b_accuracies)),
            'num_datasets': len(self.results)
        }
        
        return avg_metrics
    
    def save_evaluation_report(self):
   
        save_path = self.save_results_dir_path / "evaluation_report.json"
        
        avg_metrics = self.compute_average_metrics()
        
        output = {
            'model_name': self.model_name,
            'average_metrics': avg_metrics,
            'per_dataset_results': self.results,
            'num_datasets': len(self.results)
        }
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        print(f"Evaluation report saved to {save_path}")
        return save_path
    



    