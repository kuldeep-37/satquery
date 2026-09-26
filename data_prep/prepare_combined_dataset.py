"""
Combined dataset preparation: UC Merced + EuroSAT, unified under one
shared land-cover taxonomy so the model trains on visual diversity
from two different sensors/resolutions for the same concepts.

This is a stronger domain-adaptation story than either dataset alone:
"forest" is learned from both 256x256 aerial photography (UC Merced)
and 64x64 Sentinel-2 imagery (EuroSAT), which is genuine evidence of
cross-sensor generalization.
"""

from datasets import load_dataset, Dataset
from collections import Counter
import random

# ---------------------------------------------------------------------------
# Unified taxonomy — both datasets' classes map into these 4 shared concepts
# ---------------------------------------------------------------------------
UNIFIED_CAPTION_TEMPLATES = {
    "forest": "This satellite image shows a forested area with dense tree cover.",
    "water": "This satellite image shows a river, lake, or other body of water.",
    "agricultural": "This satellite image shows agricultural or pasture land with visible field patterns.",
    "urban": "This satellite image shows a built-up urban area with buildings and structures.",
}

UC_MERCED_LABEL_NAMES = {
    0: "agricultural", 1: "airplane", 2: "baseballdiamond", 3: "beach", 4: "buildings",
    5: "chaparral", 6: "denseresidential", 7: "forest", 8: "freeway", 9: "golfcourse",
    10: "harbor", 11: "intersection", 12: "mediumresidential", 13: "mobilehomepark",
    14: "overpass", 15: "parkinglot", 16: "river", 17: "runway", 18: "sparseresidential",
    19: "storagetanks", 20: "tenniscourt",
}
UC_MERCED_TO_UNIFIED = {
    "agricultural": "agricultural",
    "buildings": "urban",
    "denseresidential": "urban",
    "forest": "forest",
    "river": "water",
}

EUROSAT_LABEL_NAMES = {
    0: "AnnualCrop", 1: "Forest", 2: "HerbaceousVegetation", 3: "Highway", 4: "Industrial",
    5: "Pasture", 6: "PermanentCrop", 7: "Residential", 8: "River", 9: "SeaLake",
}
EUROSAT_TO_UNIFIED = {
    "Forest": "forest",
    "River": "water",
    "SeaLake": "water",
    "Pasture": "agricultural",
    "Industrial": "urban",
}


# ---------------------------------------------------------------------------
# Step 1: Load both datasets
# ---------------------------------------------------------------------------
def load_uc_merced():
    ds = load_dataset("blanchon/UC_Merced", split="train")
    print(f"UC Merced loaded: {len(ds)} images")
    return ds


def load_eurosat():
    ds = load_dataset("Honaker/eurosat_dataset", split="train")
    print(f"EuroSAT loaded: {len(ds)} images")
    return ds


# ---------------------------------------------------------------------------
# Step 2: Extract a balanced, unified-label subset from each dataset
# ---------------------------------------------------------------------------
def extract_examples(hf_dataset, raw_label_names, mapping, label_field, per_class_target, seed=42):
    """Returns a list of dicts: {image, unified_label, source} for classes in `mapping`."""
    examples = []
    for raw_label_id, raw_name in raw_label_names.items():
        if raw_name not in mapping:
            continue
        unified_label = mapping[raw_name]
        class_subset = hf_dataset.filter(lambda x: x[label_field] == raw_label_id).shuffle(seed=seed)
        n = min(per_class_target, len(class_subset))
        for i in range(n):
            examples.append({
                "image": class_subset[i]["image"].convert("RGB").resize((224, 224)),
                "unified_label": unified_label,
                "source_raw_label": raw_name,
            })
    return examples


def build_combined_subset(uc_merced_ds, eurosat_ds,
                           uc_per_class=80, eurosat_per_class=300, seed=42):
    uc_examples = extract_examples(
        uc_merced_ds, UC_MERCED_LABEL_NAMES, UC_MERCED_TO_UNIFIED,
        label_field="label", per_class_target=uc_per_class, seed=seed,
    )
    for ex in uc_examples:
        ex["source"] = "uc_merced"

    eurosat_examples = extract_examples(
        eurosat_ds, EUROSAT_LABEL_NAMES, EUROSAT_TO_UNIFIED,
        label_field="label", per_class_target=eurosat_per_class, seed=seed,
    )
    for ex in eurosat_examples:
        ex["source"] = "eurosat"

    combined = uc_examples + eurosat_examples
    random.seed(seed)
    random.shuffle(combined)

    print(f"Combined subset size: {len(combined)}")
    print("By unified label:", Counter(e["unified_label"] for e in combined))
    print("By source:", Counter(e["source"] for e in combined))
    return combined


# ---------------------------------------------------------------------------
# Step 3: Format into (image, instruction, caption) pairs + train/val split
# ---------------------------------------------------------------------------
def format_combined(combined_examples):
    formatted = []
    for ex in combined_examples:
        formatted.append({
            "image": ex["image"],
            "instruction": "Describe the land cover shown in this satellite image.",
            "caption": UNIFIED_CAPTION_TEMPLATES[ex["unified_label"]],
            "label_name": ex["unified_label"],
            "source": ex["source"],
        })
    return Dataset.from_list(formatted)


def build_train_val_split(formatted_ds, test_size=0.15, seed=42):
    formatted_ds = formatted_ds.shuffle(seed=seed)
    split = formatted_ds.train_test_split(test_size=test_size)
    train_ds, val_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Visual sanity check — confirm the mix looks right before training
# ---------------------------------------------------------------------------
def plot_sample_grid(dataset, n=10, save_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, n // 2, figsize=(3 * (n // 2), 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(dataset):
            break
        item = dataset[i]
        ax.imshow(item["image"])
        ax.set_title(f"{item['label_name']}\n({item['source']})", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Sample grid saved to {save_path}", flush=True)
    plt.close(fig)



if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    uc_ds = load_uc_merced()
    eurosat_ds = load_eurosat()
    combined = build_combined_subset(uc_ds, eurosat_ds, uc_per_class=16, eurosat_per_class=24)
    formatted = format_combined(combined)
    plot_sample_grid(formatted, n=10, save_path="outputs/sample_grid.png")
    train_ds, val_ds = build_train_val_split(formatted)

