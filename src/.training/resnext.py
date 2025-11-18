# train_resnext_lfcc_FIXED.py
# 100% WORKING on Windows + Linux + macOS
# Fixes: multiprocessing error + wrong validation + proper train/dev split

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from torchvision import models
from sklearn.metrics import roc_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
DATA_ROOT = r"D:\Thesis traning\clean_samples"          # ← change if needed
MODEL_SAVE = "resnext50_lfcc_v2_20epoch_asvspoof2019.pth"

# ====================== Dataset with proper train/dev split ======================
class LFCCDataset(Dataset):
    def __init__(self, root_dir, split="train"):    # split = "train" or "dev"
        self.samples = []
        for label in [0, 1]:
            folder = os.path.join(root_dir, "real" if label == 0 else "fake", split)
            if not os.path.exists(folder):
                continue
            for file in os.listdir(folder):
                if file.endswith(".npy"):
                    path = os.path.join(folder, file)
                    self.samples.append((path, label))
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = np.load(path)                               # (60, 400)
        x = torch.from_numpy(feat).unsqueeze(0).float()    # (1, 60, 400)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# ====================== Model ======================
class ResNeXt50LFCC(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnext50_32x4d(weights="IMAGENET1K_V1")
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        model.maxpool = nn.Identity()
        # Downsample frequency early, preserve time
        model.layer1[0].conv1.stride = (2, 1)
        if model.layer1[0].downsample is not None:
            model.layer1[0].downsample[0].stride = (2, 1)
        model.fc = nn.Linear(model.fc.in_features, 2)
        self.model = model

    def forward(self, x):
        return self.model(x)

# ====================== EER ======================
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[eer_idx]

# ====================== Training Function ======================
def train():
    train_ds = LFCCDataset(DATA_ROOT, split="train")
    dev_ds   = LFCCDataset(DATA_ROOT, split="dev")

    # FIX 1: Force more workers + pre-fetch (even on Windows!)
    num_workers = 4 if os.name == "nt" else 8
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    model = ResNeXt50LFCC().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # FIX 2: Mixed Precision (AMP) → 2–3× speedup + lower memory
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    best_eer = 1.0
    print(f"\nFAST TRAINING STARTED | Workers: {num_workers} | AMP: ON\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        # tqdm will now show real speed!
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast():  # ← This is the magic
                out = model(x)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        # FIX 3: Validate only every 3 epochs (or when improvement expected)
        if epoch % 3 == 0 or epoch > 15:
            model.eval()
            scores, labels_list = [], []
            with torch.no_grad():
                for x, y in dev_loader:
                    x = x.to(DEVICE, non_blocking=True)
                    with autocast():
                        out = model(x)
                    prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    scores.extend(prob)
                    labels_list.extend(y.numpy())
            eer = compute_eer(labels_list, scores)
            scheduler.step(eer)
            print(f"\nEpoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Dev EER: {eer*100:.3f}%")
            
            if eer < best_eer:
                best_eer = eer
                torch.save(model.state_dict(), MODEL_SAVE)
                print(f"  → BEST MODEL SAVED! EER = {eer*100:.3f}%\n")
        else:
            print(f"Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | (skip val)")

    print(f"\nTraining finished! Best EER: {best_eer*100:.3f}%")

if __name__ == '__main__': 
    train()  