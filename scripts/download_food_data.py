import os
import sys
import zipfile

import requests

from src.config import FOOD_DATA_ZIP


def download_food_data():
    if FOOD_DATA_ZIP.exists():
        print(f"File {FOOD_DATA_ZIP} exists. Skipping download.")
        return 0

    FOOD_DATA_ZIP.parent.mkdir(parents=True)

    with open(FOOD_DATA_ZIP, "wb") as f:
        request = requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        )
        print(f"Downloading food data to {FOOD_DATA_ZIP}...")
        f.write(request.content)

    with zipfile.ZipFile(FOOD_DATA_ZIP) as zip_ref:
        zip_ref.extractall(FOOD_DATA_ZIP.parent)

    os.remove(FOOD_DATA_ZIP)
    print(f"Extraction complete. File {FOOD_DATA_ZIP} removed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(download_food_data())
    except Exception as e:
        print(f"An exception occurred: {e}")
        sys.exit(1)
