import sys
from pathlib import Path

# Directories
ROOT_DIR = Path(__file__).resolve(True).parents[1]

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

FOOD_DATA_DIR = DATA_DIR / "pizza_sushi_steak"

FOOD_DATA_TRAIN_DIR = FOOD_DATA_DIR / "train"
FOOD_DATA_TEST_DIR = FOOD_DATA_DIR / "test"
FOOD_DATA_ZIP = FOOD_DATA_DIR / "pizza_sushi_steak.zip"


# Initialization
def init_directories():
    for directory in [DATA_DIR, MODELS_DIR, FOOD_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    try:
        sys.exit(init_directories())
    except Exception as e:
        print(f"An exception occurred: {e}")
        sys.exit(1)
