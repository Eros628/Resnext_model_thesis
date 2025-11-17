import os
import sys
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import soundfile as sf
import librosa
from scipy.fftpack import dct
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_class_weight

# -------------------------------
# CONFIG (ASVspoof2019 ONLY)
# -------------------------------
TRAIN_AUDIO_DIR = r"D:\Thesis traning\asvspoof2019\train"
DEV_AUDIO_DIR   = r"D:\Thesis traning\asvspoof2019\dev"
PROTOCOLS_DIR   = r"D:\Thesis traning\protocols"

TRAIN_PROTO = os.path.join(PROTOCOLS_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTO   = os.path.join(PROTOCOLS_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")

OUT_MODEL = "models/resnet50_lfcc_asvspoof2019_400.pth"
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)

# LFCC params
SR = 16000
N_FFT = 512
N_FILTERS = 40
N_CEPS = 20
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.01
MAX_FRAMES = 400  # 7.5 sec

# Training
BATCH_SIZE = 8
NUM_EPOCHS = 12
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4 if DEVICE == "cuda" else 2
SEED = 42

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# -------------------------------
# Protocol Loader (2019 only)
# -------------------------------
def load_protocol(protocol_path, audio_dir):
    paths, labels = [], []
    if not os.path.isfile(protocol_path):
        raise FileNotFoundError(f"Protocol not found: {protocol_path}")
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            file_id = parts[1]  # Standard: speaker_id utterance_id ...
            wav_path = os.path.join(audio_dir, file_id + ".wav")
            if not os.path.isfile(wav_path): continue
            label_token = parts[-1].lower()
            label = 0 if label_token in ("bonafide", "genuine") else 1
            paths.append(wav_path)
            labels.append(label)
    print(f"Loaded {len(paths)} samples from {protocol_path}")
    return paths, labels

# -------------------------------
# LFCC Extraction
# -------------------------------
def extract_lfcc_from_waveform(y, sr=SR):
    if y.ndim > 1: y = np.mean(y, axis=1)
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    frame_length = int(round(FRAME_SIZE * sr))
    frame_step = int(round(FRAME_STRIDE * sr))
    signal_length = len(y)
    num_frames = 1 if signal_length < frame_length else int(np.ceil((signal_length - frame_length) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.pad(y, (0, pad_signal_length - signal_length), mode='constant')

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices]
    frames *= np.hamming(frame_length)

    mag_frames = np.abs(np.fft.rfft(frames, N_FFT))
    pow_frames = (1.0 / N_FFT) * (mag_frames ** 2)

    low_freq, high_freq = 0, sr / 2
    linear_points = np.linspace(low_freq, high_freq, N_FILTERS + 2)
    bins = np.floor((N_FFT + 1) * linear_points / sr).astype(int)

    fbank = np.zeros((N_FILTERS, int(np.floor(N_FFT / 2 + 1))))
    for m in range(1, N_FILTERS + 1):
        f_m_minus, f_m, f_m_plus = bins[m-1], bins[m], bins[m+1]
        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    lfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :N_CEPS]
    return lfcc  # (T, 20)

def load_audio_and_extract_lfcc(path, max_frames=MAX_FRAMES):
    try:
        data, fs = sf.read(path)
    except:
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)

    if fs != SR:
        try:
            data = librosa.resample(data.astype(np.float32), orig_sr=fs, target_sr=SR)
        except:
            return np.zeros((N_CEPS, max_frames), dtype=np.float32)
    if data.ndim > 1: data = np.mean(data, axis=1)

    duration = len(data) / SR
    if duration < 1.0 or duration > 20.0 or np.max(np.abs(data)) < 1e-5 or np.any(np.isnan(data)):
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)
    if duration > 20.0:
        data = data[:int(SR * 20)]

    try:
        lfcc = extract_lfcc_from_waveform(data, SR)  # (T, 20)
    except:
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)

    # Pad / truncate
    if lfcc.shape[0] < max_frames:
        pad = max_frames - lfcc.shape[0]
        lfcc = np.pad(lfcc, ((0, pad), (0, 0)), mode='constant')
    else:
        lfcc = lfcc[:max_frames, :]

    # Normalize
    mean = lfcc.mean()
    std = lfcc.std() if lfcc.std() > 0 else 1.0
    lfcc = (lfcc - mean) / std

    # Transpose: (T, 20) → (20, T)
    lfcc = lfcc.T

    return lfcc  # (20, max_frames)

# -------------------------------
# Dataset
# -------------------------------
class LFCCDataset(Dataset):
    def __init__(self, file_paths, labels, max_frames=MAX_FRAMES):
        self.file_paths = file_paths
        self.labels = labels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        lfcc = load_audio_and_extract_lfcc(self.file_paths[idx], self.max_frames)  # (20, T)
        lfcc = torch.from_numpy(lfcc).unsqueeze(0).float()  # (1, 20, T)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return lfcc, label

# -------------------------------
# Modified ResNet-50 (for LFCC)
# -------------------------------
class ResNet50LFCC(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.resnet50(weights='IMAGENET1K_V1')

        # 1. 1-channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

        # 2. Remove maxpool
        model.maxpool = nn.Identity()

        # 3. Reduce stride in layer1 (height)
        model.layer1[0].conv1.stride = (1, 2)
        model.layer1[0].downsample[0].stride = (1, 2)

        # 4. Final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)

# -------------------------------
# EER
# -------------------------------
def compute_eer(y_true, y_scores):
    fpr, tpr, thr = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2.0

# -------------------------------
# Training
# -------------------------------
def train_model(model, train_loader, val_loader, device, epochs=NUM_EPOCHS, lr=LR, save_path=OUT_MODEL):
    # === FIXED: classes must be np.array ===
    all_labels = [y for _, y in train_loader.dataset]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),   # ← FIXED: np.array
        y=np.array(all_labels)      # ← Also convert y to np.array (safer)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


    model.to(device)
    best_eer = 1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=running_loss/(pbar.n+1), acc=100*correct/total)

        # Validation
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                all_scores.extend(probs)
                all_labels.extend(y.numpy())

        val_eer = compute_eer(np.array(all_labels), np.array(all_scores))
        scheduler.step(val_eer)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_eer)
        if optimizer.param_groups[0]['lr'] != old_lr:
            print(f"LR reduced: {old_lr:.2e} → {optimizer.param_groups[0]['lr']:.2e}")

        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
              f"Acc={100*correct/total:.2f}%, Val EER={val_eer:.4f}")

        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best EER: {best_eer:.4f} | Model saved!")

    print(f"Training finished. Best EER: {best_eer:.4f}")

# -------------------------------
# Main
# -------------------------------
def main():
    print("ASVspoof2019-Only Pipeline")
    if not os.path.isfile(TRAIN_PROTO):
        print(f"Train protocol not found: {TRAIN_PROTO}")
        sys.exit(1)
    if not os.path.isfile(DEV_PROTO):
        print(f"Dev protocol not found: {DEV_PROTO}")
        sys.exit(1)

    print("Loading train set...")
    train_files, train_labels = load_protocol(TRAIN_PROTO, TRAIN_AUDIO_DIR)
    print("Loading dev set...")
    dev_files, dev_labels = load_protocol(DEV_PROTO, DEV_AUDIO_DIR)

    print(f"Train: {len(train_files)} | Dev: {len(dev_files)}")

    train_ds = LFCCDataset(train_files, train_labels)
    dev_ds = LFCCDataset(dev_files, dev_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = ResNet50LFCC(num_classes=2)
    print(f"Model: ResNet50-LFCC | Device: {DEVICE}")
    print(f"Input shape: (1, 20, {MAX_FRAMES})")

    train_model(model, train_loader, dev_loader, DEVICE)

if __name__ == "__main__":
    main()