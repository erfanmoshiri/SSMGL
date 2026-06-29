"""
Utilities for organizing training runs with timestamped folders and logging.
"""
import os
import json
from datetime import datetime
import torch


def create_run_folder(dataset, model_type="learnable"):
    """
    Create a timestamped run folder.

    Returns:
        run_dir: Path to the run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset}_{model_type}_{timestamp}"
    run_dir = os.path.join("..", "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(run_dir, args):
    """Save training configuration."""
    config_path = os.path.join(run_dir, "config.json")
    config_dict = vars(args)

    # Convert non-serializable types
    for key, value in config_dict.items():
        if isinstance(value, torch.device):
            config_dict[key] = str(value)
        elif not isinstance(value, (int, float, str, bool, list, dict, type(None))):
            config_dict[key] = str(value)

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {config_path}")


def save_epoch_metrics(run_dir, epoch, metrics):
    """Append epoch metrics to log file."""
    log_path = os.path.join(run_dir, "training_log.jsonl")

    log_entry = {"epoch": epoch, **metrics}

    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def save_final_results(run_dir, results):
    """Save final test results."""
    results_path = os.path.join(run_dir, "final_results.json")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal results saved to {results_path}")


def save_model_weights(run_dir, model, feature_learner=None, is_best=False):
    """Save model weights."""
    suffix = "_best" if is_best else "_final"

    model_path = os.path.join(run_dir, f"model{suffix}.pkl")
    torch.save(model.state_dict(), model_path)

    if feature_learner is not None:
        learner_path = os.path.join(run_dir, f"feature_learner{suffix}.pkl")
        torch.save(feature_learner.state_dict(), learner_path)

    if is_best:
        print(f"Best model saved to {run_dir}")


def save_loss_history(run_dir, loss_history):
    """Save complete loss history."""
    loss_path = os.path.join(run_dir, "loss_history.pkl")
    torch.save(loss_history, loss_path)
