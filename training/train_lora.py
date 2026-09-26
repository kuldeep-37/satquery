"""
LoRA fine-tuning of BLIP on satellite imagery captions.
Run this in Google Colab with a GPU runtime (T4 is sufficient).

Prerequisites:
    pip install -q transformers accelerate peft pillow torchao datasets

Usage (in Colab):
    from data_prep.prepare_dataset import load_uc_merced, build_balanced_subset, build_train_val_split
    ds = load_uc_merced()
    subset = build_balanced_subset(ds)
    train_ds, val_ds = build_train_val_split(subset)

    from training.train_lora import load_base_model, attach_lora, train, save_adapter
    model, processor = load_base_model()
    model = attach_lora(model)
    train(model, processor, train_ds, num_epochs=3)
    save_adapter(model, processor)
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model

MODEL_ID = "Salesforce/blip-image-captioning-base"
ADAPTER_SAVE_PATH = "outputs/blip-lora-satellite-adapter"


# ---------------------------------------------------------------------------
# Step 1: Load base model
# ---------------------------------------------------------------------------
def load_base_model(model_id=MODEL_ID, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    print(f"Base model loaded on {device} ({dtype}).")
    return model, processor


# ---------------------------------------------------------------------------
# Step 2: Attach LoRA adapter (only ~0.25% of params become trainable)
# ---------------------------------------------------------------------------
def attach_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Step 3: Dataset wrapper for the training loop
# ---------------------------------------------------------------------------
class CaptionDataset(Dataset):
    def __init__(self, hf_dataset, processor, max_length=30):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        caption = item["caption"]
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


# ---------------------------------------------------------------------------
# Step 4: Train
# ---------------------------------------------------------------------------
def train(model, processor, train_ds, num_epochs=3, batch_size=4, lr=5e-5, device=None, save_dir="outputs"):
    if device is None:
        device = next(model.parameters()).device
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    dataset = CaptionDataset(train_ds, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Batches per epoch: {len(loader)} (device: {device}, dtype: {dtype})")

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    loss_history = []
    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(loader, start=1):
            batch = {
                k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device))
                for k, v in batch.items()
            }
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 5 == 0 or step == len(loader):
                print(f"  [Epoch {epoch + 1}/{num_epochs}] Batch {step}/{len(loader)} - Current Loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"==> Epoch {epoch + 1}/{num_epochs} Complete — Avg Loss: {avg_loss:.4f}\n", flush=True)


    model.eval()

    # Save loss history and curve
    os.makedirs(save_dir, exist_ok=True)
    loss_json_path = os.path.join(save_dir, "training_loss.json")
    with open(loss_json_path, "w") as f:
        json.dump({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "loss_history": [round(l, 4) for l in loss_history],
        }, f, indent=2)
    print(f"Loss history saved to {loss_json_path}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        epochs = list(range(1, num_epochs + 1))
        ax.plot(epochs, loss_history, marker="o", color="#00bcd4", linewidth=2, label="LoRA Loss")
        ax.set_title("BLIP LoRA Training Loss (UC Merced + EuroSAT)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Cross-Entropy Loss")
        ax.set_xticks(epochs)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to {plot_path}")
    except Exception as e:
        print(f"Could not render loss plot: {e}")

    return model, loss_history


# ---------------------------------------------------------------------------
# Step 5: Evaluate — before vs after comparison, on the SAME val images
# ---------------------------------------------------------------------------
def generate_caption(model, processor, image, device=None):
    if device is None:
        device = next(model.parameters()).device
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    inputs = processor(image.convert("RGB"), return_tensors="pt")
    inputs = {
        k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device))
        for k, v in inputs.items()
    }
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(output[0], skip_special_tokens=True).strip()


def compare_before_after(model, processor, val_ds, n=8, device=None, save_dir="outputs"):
    print("=== BEFORE (LoRA adapter disabled) vs AFTER (LoRA adapter active) ===\n")
    comparisons = []
    
    # Try to pick samples covering all available classes and sources
    chosen_indices = []
    seen_combos = set()
    for idx, item in enumerate(val_ds):
        combo = (item.get("label_name"), item.get("source"))
        if combo not in seen_combos:
            seen_combos.add(combo)
            chosen_indices.append(idx)
        if len(chosen_indices) >= n:
            break
    # Fill up to n if needed
    for idx in range(len(val_ds)):
        if len(chosen_indices) >= n:
            break
        if idx not in chosen_indices:
            chosen_indices.append(idx)

    for i in chosen_indices:
        sample = val_ds[i]
        image = sample["image"].convert("RGB")
        true_label = sample.get("label_name", "unknown")
        source = sample.get("source", "unknown")

        with model.disable_adapter():
            before = generate_caption(model, processor, image, device)
        after = generate_caption(model, processor, image, device)

        comp = {
            "index": i,
            "true_label": true_label,
            "source": source,
            "before_blip_base": before,
            "after_blip_lora": after,
            "reference_template": sample.get("caption", ""),
        }
        comparisons.append(comp)

        print(f"Sample {i} | Label: {true_label} | Source: {source}")
        print(f"  Before (Base BLIP): {before}")
        print(f"  After  (LoRA BLIP): {after}")
        print("-" * 60)

    # Save to artifacts directory
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "before_after_comparisons.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)
    print(f"Comparison JSON artifact saved to {json_path}")

    txt_path = os.path.join(save_dir, "before_after_comparisons.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== BLIP vs LoRA Domain Adaptation Comparison Evidence ===\n\n")
        for c in comparisons:
            f.write(f"Sample #{c['index']} — Class: {c['true_label']} (Source: {c['source']})\n")
            f.write(f"  Base BLIP (no adapter):    {c['before_blip_base']}\n")
            f.write(f"  LoRA-adapted (Satellite):  {c['after_blip_lora']}\n")
            f.write(f"  Reference caption:         {c['reference_template']}\n")
            f.write("-" * 60 + "\n")
    print(f"Comparison TXT artifact saved to {txt_path}")

    return comparisons


# ---------------------------------------------------------------------------
# Step 6: Save adapter
# ---------------------------------------------------------------------------
def save_adapter(model, processor, path=ADAPTER_SAVE_PATH):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"Adapter saved to {path}")


def backup_to_drive(path=ADAPTER_SAVE_PATH, drive_folder="/content/drive/MyDrive/"):
    """Optional: persist the adapter past a Colab session reset."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        import shutil
        shutil.copytree(path, drive_folder + path.split("/")[-1], dirs_exist_ok=True)
        print("Backed up to Drive.")
    except ImportError:
        print(f"Not running in Colab; adapter saved locally in '{path}'.")

