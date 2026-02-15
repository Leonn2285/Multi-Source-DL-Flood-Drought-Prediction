"""
Utility functions for Natural Disaster Prediction
Includes: metrics, early stopping, logging, checkpointing, and training helpers
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import LOG_DIR, MODEL_DIR


# ========================== LOGGING ==========================
def setup_logger(
    name: str = "disaster_prediction",
    log_dir: str = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logger with file and console handlers"""
    log_dir = log_dir or LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    
    return logger


# ========================== METRICS ==========================
class MetricsCalculator:
    """Calculate and track metrics for multi-task classification"""
    
    def __init__(self, task_names: List[str] = ["flood", "drought"]):
        self.task_names = task_names
        self.reset()
        
    def reset(self):
        """Reset all accumulated predictions and labels"""
        self.predictions = {task: [] for task in self.task_names}
        self.labels = {task: [] for task in self.task_names}
        self.probabilities = {task: [] for task in self.task_names}
        
    def update(
        self,
        task: str,
        preds: torch.Tensor,
        labels: torch.Tensor,
        probs: torch.Tensor = None
    ):
        """Update with new batch of predictions"""
        self.predictions[task].extend(preds.detach().cpu().numpy().tolist())
        self.labels[task].extend(labels.detach().cpu().numpy().tolist())
        if probs is not None:
            self.probabilities[task].extend(probs.detach().cpu().numpy().tolist())
    
    def compute(self, task: str = None) -> Dict[str, float]:
        """Compute all metrics for a task or all tasks"""
        if task:
            return self._compute_task_metrics(task)
        
        # Compute for all tasks
        all_metrics = {}
        for task in self.task_names:
            task_metrics = self._compute_task_metrics(task)
            for key, value in task_metrics.items():
                all_metrics[f"{task}_{key}"] = value
        
        # Average metrics across tasks
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = [all_metrics.get(f"{task}_{metric}", 0) for task in self.task_names]
            all_metrics[f"avg_{metric}"] = np.mean(values)
        
        return all_metrics
    
    def _compute_task_metrics(self, task: str) -> Dict[str, float]:
        """Compute metrics for a single task"""
        preds = np.array(self.predictions[task])
        labels = np.array(self.labels[task])
        
        if len(preds) == 0:
            return {}
        
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average='weighted', zero_division=0),
            "recall": recall_score(labels, preds, average='weighted', zero_division=0),
            "f1": f1_score(labels, preds, average='weighted', zero_division=0),
        }
        
        # Binary classification specific metrics
        if len(np.unique(labels)) == 2:
            metrics["precision_pos"] = precision_score(labels, preds, pos_label=1, zero_division=0)
            metrics["recall_pos"] = recall_score(labels, preds, pos_label=1, zero_division=0)
            metrics["f1_pos"] = f1_score(labels, preds, pos_label=1, zero_division=0)
            
            # ROC-AUC if probabilities available
            if self.probabilities[task]:
                probs = np.array(self.probabilities[task])
                if probs.ndim > 1:
                    probs = probs[:, 1]  # Probability of positive class
                try:
                    metrics["roc_auc"] = roc_auc_score(labels, probs)
                    metrics["avg_precision"] = average_precision_score(labels, probs)
                except:
                    pass
        
        return metrics
    
    def get_confusion_matrix(self, task: str) -> np.ndarray:
        """Get confusion matrix for a task"""
        preds = np.array(self.predictions[task])
        labels = np.array(self.labels[task])
        return confusion_matrix(labels, preds)
    
    def get_classification_report(self, task: str) -> str:
        """Get detailed classification report"""
        preds = np.array(self.predictions[task])
        labels = np.array(self.labels[task])
        return classification_report(labels, preds)


# ========================== EARLY STOPPING ==========================
class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",  # "max" for metrics like F1, "min" for loss
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. Best epoch: {self.best_epoch}")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


# ========================== CHECKPOINTING ==========================
class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(
        self,
        model_dir: str = None,
        experiment_name: str = "disaster_prediction",
        save_best_only: bool = True,
        mode: str = "max"
    ):
        self.model_dir = model_dir or MODEL_DIR
        self.experiment_name = experiment_name
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_score = None
        
        os.makedirs(self.model_dir, exist_ok=True)
        
    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        epoch: int,
        metrics: Dict[str, float],
        score: float
    ) -> Optional[str]:
        """Save checkpoint if improved"""
        
        # Check if should save
        if self.save_best_only:
            if self.best_score is not None:
                if self.mode == "max" and score <= self.best_score:
                    return None
                if self.mode == "min" and score >= self.best_score:
                    return None
            self.best_score = score
        
        # Create checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "best_score": self.best_score
        }
        
        # Save checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.save_best_only:
            filename = f"{self.experiment_name}_best.pt"
        else:
            filename = f"{self.experiment_name}_epoch{epoch}_{timestamp}.pt"
        
        filepath = os.path.join(self.model_dir, filename)
        torch.save(checkpoint, filepath)
        
        print(f"Checkpoint saved: {filepath}")
        return filepath
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer = None,
        scheduler: _LRScheduler = None,
        filepath: str = None
    ) -> Dict[str, Any]:
        """Load checkpoint"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"{self.experiment_name}_best.pt")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        # weights_only=False for PyTorch 2.6+ compatibility with numpy arrays
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded checkpoint from {filepath} (epoch {checkpoint['epoch']})")
        
        return checkpoint


# ========================== TRAINING HISTORY ==========================
class TrainingHistory:
    """Track and visualize training history"""
    
    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "learning_rates": [],
            "epoch_times": []
        }
        
    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        epoch_time: float
    ):
        """Update history with epoch results"""
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_metrics"].append(train_metrics)
        self.history["val_metrics"].append(val_metrics)
        self.history["learning_rates"].append(lr)
        self.history["epoch_times"].append(epoch_time)
        
    def save(self, filepath: str):
        """Save history to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {filepath}")
        
    def load(self, filepath: str):
        """Load history from JSON file"""
        with open(filepath, "r") as f:
            self.history = json.load(f)
        print(f"Training history loaded from {filepath}")
        
    def plot(self, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        # Loss plot
        ax = axes[0, 0]
        ax.plot(epochs, self.history["train_loss"], label="Train Loss", marker='o')
        ax.plot(epochs, self.history["val_loss"], label="Val Loss", marker='s')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 scores plot
        ax = axes[0, 1]
        train_f1_flood = [m.get("flood_f1", 0) for m in self.history["train_metrics"]]
        val_f1_flood = [m.get("flood_f1", 0) for m in self.history["val_metrics"]]
        train_f1_drought = [m.get("drought_f1", 0) for m in self.history["train_metrics"]]
        val_f1_drought = [m.get("drought_f1", 0) for m in self.history["val_metrics"]]
        
        ax.plot(epochs, val_f1_flood, label="Val Flood F1", marker='o')
        ax.plot(epochs, val_f1_drought, label="Val Drought F1", marker='s')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.set_title("Validation F1 Scores")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax = axes[1, 0]
        ax.plot(epochs, self.history["learning_rates"], marker='o', color='green')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax = axes[1, 1]
        train_acc = [m.get("avg_accuracy", 0) for m in self.history["train_metrics"]]
        val_acc = [m.get("avg_accuracy", 0) for m in self.history["val_metrics"]]
        ax.plot(epochs, train_acc, label="Train Acc", marker='o')
        ax.plot(epochs, val_acc, label="Val Acc", marker='s')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training and Validation Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        return fig


# ========================== VISUALIZATION ==========================
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: str = None,
    normalize: bool = True
):
    """Plot confusion matrix"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return plt.gcf()


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    title: str = "ROC Curve",
    save_path: str = None
):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    return plt.gcf()


def plot_precision_recall_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: str = None
):
    """Plot precision-recall curve"""
    precision, recall, _ = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    
    return plt.gcf()


# ========================== MISC UTILITIES ==========================
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds to human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


class ProgressBar:
    """Simple progress bar wrapper for training"""
    
    def __init__(self, dataloader, desc: str = ""):
        self.dataloader = dataloader
        self.desc = desc
        self.pbar = None
        
    def __iter__(self):
        self.pbar = tqdm(self.dataloader, desc=self.desc, ncols=100)
        for batch in self.pbar:
            yield batch
            
    def set_postfix(self, **kwargs):
        if self.pbar:
            self.pbar.set_postfix(**kwargs)


if __name__ == "__main__":
    # Test utilities
    logger = setup_logger()
    logger.info("Testing utilities")
    
    # Test device
    device = get_device()
    
    # Test seed
    set_seed(42)
    
    # Test metrics
    metrics = MetricsCalculator()
    preds = torch.randint(0, 2, (100,))
    labels = torch.randint(0, 2, (100,))
    metrics.update("flood", preds, labels)
    print(metrics.compute("flood"))
    
    # Test early stopping
    es = EarlyStopping(patience=3)
    for i, score in enumerate([0.5, 0.6, 0.65, 0.64, 0.63, 0.62]):
        stop = es(score, i)
        print(f"Epoch {i}, Score: {score}, Stop: {stop}")
