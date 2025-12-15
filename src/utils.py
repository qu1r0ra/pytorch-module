"""
Contains various utility functions for PyTorch model training and saving.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_model(model: torch.nn.Module, target_dir: Path, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pt")
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pt"), "model_name should end with '.pt'"

    MODEL_SAVE_PATH = target_dir / model_name

    print(f"[INFO] Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)


def plot_train_test_curves(results: dict[str, list[float]]):
    plt.figure(figsize=(15, 10))

    epochs = range(len(results))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], "Train Loss")
    plt.plot(epochs, results["test_loss"], "Test Loss")
    plt.title("Train Loss vs. Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_accuracy"], "Train Accuracy")
    plt.plot(epochs, results["test_accuracy"], "Test Accuracy")
    plt.title("Train Accuracy vs. Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
