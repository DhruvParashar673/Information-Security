import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.dataset import build_loaders
from src.model import MalwareMLP
from src.backdoor import BackdoorConfig, compute_asr
from src.purification import PurificationConfig, purify


def load_model(checkpoint_path: str, input_dim: int, device: torch.device) -> nn.Module:
    """
    Load a saved model checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved model weights.
    input_dim : int
        Dimensionality of the input features.
    device : torch.device
        Compute device.

    Returns
    -------
    nn.Module
        The loaded MalwareMLP model.
    """
    model = MalwareMLP(input_dim=input_dim)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using initialized weights.")
    
    model.to(device)
    model.eval()
    return model


def evaluate_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate precision, recall, f1-score, and accuracy on a given dataset.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        DataLoader for the evaluation dataset.
    device : torch.device
        Compute device.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the computed metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = model(X_batch).squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu()
            
            all_preds.extend(preds.numpy())
            all_targets.extend(y_batch.numpy())
            
    return {
        "Clean Accuracy": accuracy_score(all_targets, all_preds),
        "Precision": precision_score(all_targets, all_preds, zero_division=0),
        "Recall": recall_score(all_targets, all_preds, zero_division=0),
        "F1-score": f1_score(all_targets, all_preds, zero_division=0)
    }


def plot_comparisons(
    poisoned_metrics: Dict[str, float], 
    purified_metrics: Dict[str, float], 
    output_dir: str
) -> None:
    """
    Generate and save comparison bar charts for the evaluated metrics.

    Parameters
    ----------
    poisoned_metrics : Dict[str, float]
        Metrics for the poisoned model.
    purified_metrics : Dict[str, float]
        Metrics for the purified model.
    output_dir : str
        Directory to save the generated graphs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = list(poisoned_metrics.keys())
    poisoned_vals = [poisoned_metrics[m] for m in metrics]
    purified_vals = [purified_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, poisoned_vals, width, label='Poisoned Model', color='salmon')
    ax.bar(x + width/2, purified_vals, width, label='Purified Model', color='skyblue')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison: Poisoned vs Purified')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Annotate bars with values
    for i, v in enumerate(poisoned_vals):
        ax.text(i - width/2, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(purified_vals):
        ax.text(i + width/2, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()


def print_comparison_table(
    poisoned_metrics: Dict[str, float], 
    purified_metrics: Dict[str, float]
) -> None:
    """
    Print a formatted comparison table of the metrics.

    Parameters
    ----------
    poisoned_metrics : Dict[str, float]
        Metrics for the poisoned model.
    purified_metrics : Dict[str, float]
        Metrics for the purified model.
    """
    print("\n" + "="*65)
    print(f"{'Metric':<25} | {'Poisoned Model':<15} | {'Purified Model':<15}")
    print("-" * 65)
    for metric in poisoned_metrics.keys():
        pm = poisoned_metrics[metric]
        pu = purified_metrics[metric]
        print(f"{metric:<25} | {pm:<15.4f} | {pu:<15.4f}")
    print("="*65 + "\n")


def main() -> None:
    """
    Main evaluation pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths and configurations
    data_path = "data/malware.csv"
    poisoned_ckpt = "models/poisoned_model.pt"
    graphs_dir = "outputs/graphs/"
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please generate or provide the dataset.")
        return
        
    # Build data loaders
    train_loader, test_loader, scaler, input_dim = build_loaders(
        csv_path=data_path,
        batch_size=256
    )
    
    # Extract raw dataset features for ASR calculation
    X_test = test_loader.dataset.features
    y_test = test_loader.dataset.labels
    
    # 1. Load and Evaluate Poisoned Model
    print("Loading poisoned model...")
    poisoned_model = load_model(poisoned_ckpt, input_dim, device)
    
    print("Evaluating poisoned model...")
    poisoned_metrics = evaluate_metrics(poisoned_model, test_loader, device)
    
    backdoor_config = BackdoorConfig()
    poisoned_asr = compute_asr(
        model=poisoned_model,
        X=X_test,
        y=y_test,
        config=backdoor_config,
        device=device
    )
    poisoned_metrics["Attack Success Rate (ASR)"] = poisoned_asr
    
    # 2. Purify Model
    print("\nPurifying model...")
    purification_config = PurificationConfig(finetune_epochs=5)
    purified_model = purify(
        model=poisoned_model,
        clean_loader=train_loader,
        device=device,
        config=purification_config,
        val_loader=test_loader
    )
    
    # 3. Evaluate Purified Model
    print("\nEvaluating purified model...")
    purified_metrics = evaluate_metrics(purified_model, test_loader, device)
    
    purified_asr = compute_asr(
        model=purified_model,
        X=X_test,
        y=y_test,
        config=backdoor_config,
        device=device
    )
    purified_metrics["Attack Success Rate (ASR)"] = purified_asr
    
    # 4. Output Results
    print_comparison_table(poisoned_metrics, purified_metrics)
    
    print(f"Generating and saving comparison graphs to {graphs_dir}...")
    plot_comparisons(poisoned_metrics, purified_metrics, graphs_dir)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
