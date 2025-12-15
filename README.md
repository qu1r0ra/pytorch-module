# pytorch-module <!-- omit from toc -->

![title](./readme/title.jpg)

<!-- Refer to https://shields.io/badges for usage -->

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff) ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)

A simple modularized PyTorch project for personal reference.

## Project Structure

```bash
pytorch-module/
├── data/                   # Dataset storage
├── models/                 # Saved models
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── config.py           # Configuration and paths
│   ├── data_builder.py     # Data loading and transforms
│   ├── engine.py           # Training and testing loops
│   ├── model_builder.py    # TinyVGG model architecture
│   ├── train.py            # Main training script
│   └── utils.py            # Utility functions (plotting, saving)
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

## Usage

This project uses `uv` for dependency management.

### Default Training

Run the training script with default hyperparameters (Batch size: 32, Epochs: 5, Hidden units: 10, LR: 0.001):

```bash
uv run src/train.py
```

### Custom Training

You can customize the training configuration using CLI flags:

```bash
uv run src/train.py --batch-size 64 --epochs 10 --hidden-units 20 --learning-rate 0.005
```

### CLI Arguments

| Argument          | Type    | Default | Description                                 |
| :---------------- | :------ | :------ | :------------------------------------------ |
| `--batch-size`    | `int`   | `32`    | Batch size for training                     |
| `--epochs`        | `int`   | `5`     | Number of epochs to train                   |
| `--hidden-units`  | `int`   | `10`    | Number of hidden units in the TinyVGG model |
| `--learning-rate` | `float` | `0.001` | Learning rate for the optimizer             |

### Inference

To be implemented.
