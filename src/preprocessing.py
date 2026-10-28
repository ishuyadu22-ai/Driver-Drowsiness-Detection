
from pathlib import Path
from collections import Counter

# Project's data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATASETS = {
    "MRL-Eye-Dataset": {
        "sleepy": "drowsy",
        "awake": "alert",
    },
    "NTHU-DDD-Images": {
        "drowsy": "drowsy",
        "notdrowsy": "alert",
    },
    "UTA-RLDD-Images": {
        "fatigue": "drowsy",
        "active": "alert",
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def inspect_dataset(dataset_name, label_map):
    dataset_path = DATA_DIR / dataset_name

    if not dataset_path.exists():
        print(f"\nDataset folder not found: {dataset_path}")
        return

    print(f"\n{'=' * 45}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 45}")

    counts = Counter()
    total = 0

    # Supports datasets organized as train/val/test
    # or directly into class folders.
    for folder in sorted(dataset_path.iterdir()):
        if not folder.is_dir():
            continue

        if folder.name in {"train", "val", "test"}:
            class_folders = [
                item for item in folder.iterdir()
                if item.is_dir()
            ]
        else:
            class_folders = [folder]

        for class_folder in class_folders:
            original_label = class_folder.name

            if original_label not in label_map:
                print(f"Skipping unknown folder: {class_folder}")
                continue

            label = label_map[original_label]

            image_count = sum(
                1 for file in class_folder.rglob("*")
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            )

            counts[label] += image_count
            total += image_count

            print(
                f"{folder.name}/{original_label}: "
                f"{image_count} images -> {label}"
            )

    print(f"Total labelled images: {total}")
    print(f"Label summary: {dict(counts)}")


if __name__ == "__main__":
    import cv2

    for dataset_name, label_map in DATASETS.items():
        inspect_dataset(dataset_name, label_map)

    print("\n--- Image preprocessing test ---")

    dataset_path = DATA_DIR / "MRL-Eye-Dataset" / "train"

    sample_file = next(
        (
            file
            for file in dataset_path.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ),
        None,
    )

    if sample_file is None:
        print("No sample image found.")
    else:
        image = cv2.imread(str(sample_file))

        if image is None:
            print("Could not read sample image:", sample_file)
        else:
            resized = cv2.resize(image, (224, 224))

            print("Sample image:", sample_file.name)
            print("Original image shape:", image.shape)
            print("Resized image shape:", resized.shape)
            print("Image preprocessing test successful!")


