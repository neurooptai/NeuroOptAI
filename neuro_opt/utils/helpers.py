import time
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Return available device.
    """

    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model):
    """
    Count trainable model parameters.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_accuracy(predictions, targets):
    """
    Compute classification accuracy.
    """

    predicted = torch.argmax(predictions, dim=1)
    correct = (predicted == targets).sum().item()

    return correct / len(targets)


def measure_inference_time(model, sample_input, device="cpu"):
    """
    Measure forward pass inference time.
    """

    model.eval()
    sample_input = sample_input.to(device)

    start = time.time()

    with torch.no_grad():
        _ = model(sample_input)

    end = time.time()

    return end - start


def model_summary(model):
    """
    Return simple model summary.
    """

    summary = {
        "parameters": count_parameters(model),
        "trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }

    return summary


def save_checkpoint(model, optimizer, epoch, path):
    """
    Save training checkpoint.
    """

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    """
    Load training checkpoint.
    """

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"]


def compute_training_efficiency(loss, compute_cost):
    """
    Simple efficiency metric.

    Higher is better.
    """

    if compute_cost <= 0:
        return 0.0

    return 1.0 / (loss * compute_cost + 1e-8)
