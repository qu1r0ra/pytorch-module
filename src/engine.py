"""
Contains functions for training and testing a PyTorch model.
"""

import torch
import torchmetrics
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    accuracy_fn: torchmetrics.Metric,
    device: torch.device,
) -> tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        accuracy_fn: A function for the model's accuracy to ne evaluated on.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    model.train()

    train_loss, train_accuracy = 0.0, 0.0
    num_batches = len(dataloader)

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_logits = model(X)
        y_preds = y_logits.softmax(dim=1).argmax(dim=1)

        loss = loss_fn(y_logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        train_accuracy += accuracy_fn(y_preds, y)

    train_loss /= num_batches
    train_accuracy /= num_batches

    return train_loss, train_accuracy


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    accuracy_fn: torchmetrics.Metric,
    device: torch.device,
) -> tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
      model: A PyTorch model to be tested.
      dataloader: A DataLoader instance for the model to be tested on.
      loss_fn: A PyTorch loss function to calculate loss on the test data.
      accuracy_fn: A function for the model's accuracy to be evaluated on.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A tuple of testing loss and testing accuracy metrics.
      In the form (test_loss, test_accuracy). For example:

      (0.0223, 0.8985)
    """
    model.eval()

    test_loss, test_accuracy = 0.0, 0.0
    num_batches = len(dataloader)

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = y_logits.softmax(dim=1).argmax(dim=1)

            test_loss += loss_fn(y_logits, y)
            test_accuracy += accuracy_fn(y_preds, y)

        test_loss /= num_batches
        test_accuracy /= num_batches

    return test_loss, test_accuracy


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    accuracy_fn: torchmetrics.Metric,
    epochs: int,
    device: torch.device,
) -> dict[str, list[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    accuracy_fn: A function to calculate accuracy on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}
    For example if training for epochs=2:
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]}
    """
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
        )
        test_loss, test_accuracy = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"Train loss: {train_loss:.5f} | "
            f"Train accuracy: {train_accuracy:.2f} | "
            f"Test loss: {test_loss:.5f} | "
            f"Test accuracy: {test_accuracy:.2f}"
        )

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)

    return results
