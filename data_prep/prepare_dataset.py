"""
Data preparation for Satellite VQA project.
Run this FIRST in Colab, in its own cell block or as a script.

Loads UC Merced Land Use dataset, builds a balanced subset across
chosen classes, and formats it into (image, instruction, caption)
pairs ready for LoRA fine-tuning.
"""

from datasets import load_dataset, concatenate_datasets
from collections import Counter

# ---------------------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------------------
def load_uc_merced():
    dataset = load_dataset("blanchon/UC_Merced", split="train")
    print(dataset)
    print(dataset[0])
    return dataset


LABEL_NAMES = {
    0: "agricultural", 1: "airplane", 2: "baseballdiamond", 3: "beach", 4: "buildings",
    5: "chaparral", 6: "denseresidential", 7: "forest", 8: "freeway", 9: "golfcourse",
    10: "harbor", 11: "intersection", 12: "mediumresidential", 13: "mobilehomepark",
    14: "overpass", 15: "parkinglot", 16: "river", 17: "runway", 18: "sparseresidential",
    19: "storagetanks", 20: "tenniscourt",
}

# Classes matching the SIH / disaster-management demo narrative
TARGET_LABELS = {0, 4, 6, 7, 16}  # agricultural, buildings, denseresidential, forest, river

CAPTION_TEMPLATES = {
    "agricultural": "This satellite image shows agricultural land with visible field patterns.",
    "buildings": "This satellite image shows a cluster of buildings and built-up structures.",
    "denseresidential": "This satellite image shows a dense residential area with closely packed housing.",
    "forest": "This satellite image shows a forested area with dense tree cover.",
    "river": "This satellite image shows a river or waterway running through the landscape.",
}


# ---------------------------------------------------------------------------
# Step 2: Balanced subset (shuffled per class, NOT a naive slice — a naive
# slice on a class-sorted dataset returns only one class, as we found out
# the hard way earlier in this project)
# ---------------------------------------------------------------------------
def build_balanced_subset(dataset, target_labels=TARGET_LABELS, per_class_target=80, seed=42):
    balanced_parts = []
    for label_id in target_labels:
        class_subset = dataset.filter(lambda x: x["label"] == label_id).shuffle(seed=seed)
        balanced_parts.append(class_subset.select(range(min(per_class_target, len(class_subset)))))

    subset = concatenate_datasets(balanced_parts).shuffle(seed=seed)
    print(f"Subset size: {len(subset)}")
    print(Counter(subset["label"]))
    return subset


# ---------------------------------------------------------------------------
# Step 3: Format into (image, instruction, caption) pairs
# ---------------------------------------------------------------------------
def format_example(example):
    label_name = LABEL_NAMES[example["label"]]
    return {
        "image": example["image"],
        "instruction": "Describe the land cover shown in this satellite image.",
        "caption": CAPTION_TEMPLATES.get(label_name, f"This satellite image shows {label_name}."),
        "label_name": label_name,
    }


def build_train_val_split(subset, test_size=0.15, seed=42):
    formatted = subset.map(format_example)
    formatted = formatted.shuffle(seed=seed)
    split = formatted.train_test_split(test_size=test_size)
    train_ds, val_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Optional: visual sanity check (run in a notebook cell, not headless)
# ---------------------------------------------------------------------------
def plot_sample_grid(subset, n=5):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    for i, ax in enumerate(axes.flat):
        img = subset[i]["image"]
        label = LABEL_NAMES[subset[i]["label"]]
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ds = load_uc_merced()
    subset = build_balanced_subset(ds)
    train_ds, val_ds = build_train_val_split(subset)
