"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os

import torch
from torch import nn
from torchmetrics import Accuracy
from torchvision.transforms import v2

from src.config import FOOD_DATA_TEST_DIR, FOOD_DATA_TRAIN_DIR, MODELS_DIR
from src.data_builder import create_dataloaders
from src.engine import train
from src.model_builder import TinyVGG
from src.utils import plot_train_test_curves, save_model

BATCH_SIZE = 32
EPOCHS = 5
HIDDEN_UNITS = 10
IMAGE_LENGTH = 64
LEARNING_RATE = 0.001
NUM_WORKERS = os.cpu_count()

assert isinstance(NUM_WORKERS, int), "NUM_WORKERS must be an int."

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
data_transform = v2.Compose(
    [v2.Resize(size=(IMAGE_LENGTH, IMAGE_LENGTH)), v2.ToTensor()]
)

# Get dataloaders
train_dataloader, test_dataloader, classes = create_dataloaders(
    train_dir=FOOD_DATA_TRAIN_DIR,
    test_dir=FOOD_DATA_TEST_DIR,
    transform=data_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

# Initialize model, loss function, and optimizer
model = TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(classes),
    image_length=IMAGE_LENGTH,
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
accuracy_fn = Accuracy(task="multiclass", num_classes=len(classes))

# Get results
results = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    accuracy_fn=accuracy_fn,
    epochs=EPOCHS,
    device=device,
)

# Visualize train and test loss and accuracy curves
plot_train_test_curves(results=results)

# Save model
save_model(model=model, target_dir=MODELS_DIR, model_name="tinyvgg_model.pt")
