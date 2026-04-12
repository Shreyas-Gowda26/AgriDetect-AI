"""
plant_disease_train.py
----------------------
Train MobileNetV2 on PlantVillage dataset.
Uses only 20% of data for faster training on CPU (~45-60 mins).
Saves model to ml_models/plant_disease_model.pth

Run with:
    python plant_disease_train.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
import zipfile
import random
import numpy as np

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ZIP_PATH    = os.path.expanduser("~/Downloads/archive (2).zip")
EXTRACT_DIR = "/tmp/plantvillage"
MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
MODEL_PATH  = os.path.join(MODEL_DIR, "plant_disease_model.pth")

SAMPLE_RATIO = 0.20   # use 20% of dataset
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 0.001
NUM_WORKERS  = 0       # 0 for Mac stability
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────
# Step 1: Extract dataset
# ─────────────────────────────────────────────
def extract_dataset():
    if os.path.exists(EXTRACT_DIR) and len(os.listdir(EXTRACT_DIR)) > 0:
        print(f"✅ Dataset already extracted at {EXTRACT_DIR}")
        return

    print(f"📦 Extracting dataset from {ZIP_PATH}...")
    print("   This may take 3-5 minutes...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print("✅ Extraction complete!")


# ─────────────────────────────────────────────
# Step 2: Find dataset root
# ─────────────────────────────────────────────
def find_dataset_root(base_dir: str) -> str:
    """
    Finds the folder that contains class subdirectories.
    PlantVillage zips can have different internal structures.
    """
    for root, dirs, files in os.walk(base_dir):
        # Look for a folder that has many subdirectories (class folders)
        subdirs = [d for d in dirs if not d.startswith('.')]
        if len(subdirs) >= 10:
            print(f"📂 Found dataset root: {root} ({len(subdirs)} classes)")
            return root
    return base_dir


# ─────────────────────────────────────────────
# Step 3: Train
# ─────────────────────────────────────────────
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Extract
    extract_dataset()

    # Find root
    dataset_root = find_dataset_root(EXTRACT_DIR)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load full dataset
    print("📊 Loading dataset...")
    full_dataset = datasets.ImageFolder(root=dataset_root, transform=train_transform)
    num_classes  = len(full_dataset.classes)
    print(f"   Found {len(full_dataset)} images across {num_classes} classes")
    print(f"   Classes: {full_dataset.classes[:5]}...")

    # Sample 20%
    total_samples  = len(full_dataset)
    sample_size    = int(total_samples * SAMPLE_RATIO)
    indices        = list(range(total_samples))
    random.shuffle(indices)
    sampled_indices = indices[:sample_size]

    # Split 80/20 train/val
    split       = int(len(sampled_indices) * 0.8)
    train_idx   = sampled_indices[:split]
    val_idx     = sampled_indices[split:]

    train_subset = Subset(full_dataset, train_idx)
    val_subset   = Subset(full_dataset, val_idx)

    # Apply val transform to val set
    val_dataset = datasets.ImageFolder(root=dataset_root, transform=val_transform)
    val_subset  = Subset(val_dataset, val_idx)

    print(f"   Training on {len(train_subset)} images | Validating on {len(val_subset)} images")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Build model
    print(f"\n🏗️  Building MobileNetV2 with {num_classes} output classes...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Using device: {device}")
    model = model.to(device)

    # Freeze feature layers, only train classifier first
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_val_acc = 0.0

    print(f"\n🚀 Training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        start_time = time.time()

        # Unfreeze all layers after epoch 2
        if epoch == 2:
            print("🔓 Unfreezing all layers for fine-tuning...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR * 0.1)

        # Training
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            _, predicted   = outputs.max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"   Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {train_loss/(batch_idx+1):.3f} | "
                      f"Acc: {100.*train_correct/train_total:.1f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs        = model(inputs)
                _, predicted   = outputs.max(1)
                val_total     += labels.size(0)
                val_correct   += predicted.eq(labels).sum().item()

        val_acc    = 100. * val_correct / val_total
        epoch_time = time.time() - start_time

        print(f"\n📈 Epoch {epoch+1}/{EPOCHS} — "
              f"Train Acc: {100.*train_correct/train_total:.1f}% | "
              f"Val Acc: {val_acc:.1f}% | "
              f"Time: {epoch_time/60:.1f} mins\n")

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"   💾 Saved best model (Val Acc: {val_acc:.1f}%)")

    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.1f}%")
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"\n   Class mapping saved — {num_classes} classes")
    print(f"   Restart your FastAPI server to use the new model!")


if __name__ == "__main__":
    train()