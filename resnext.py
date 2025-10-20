import os
import sys
import math
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import soundfile as sf
import librosa
from scipy.fftpack import dct
from sklearn.metrics import roc_curve


TRAIN_AUDIO_DIR = r"D:\Thesis traning\asvspoof2019\train"
DEV_AUDIO_DIR   = r"D:\Thesis traning\asvspoof2019\dev"
ASVSPOOF5_AUDIO_DIR = r"D:\Thesis traning\asvspoof5\wav_T"
PROTOCOLS_DIR = r"D:\Thesis traning\protocols"

TRAIN_PROTO = os.path.join(PROTOCOLS_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTO   = os.path.join(PROTOCOLS_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")

OUT_MODEL = "models/resnext_with_spoof5_and_spoof2019.pth"
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)

# LFCC / preprocessing params
SR = 16000
N_FFT = 512
N_FILTERS = 40
N_CEPS = 20
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.01
MAX_FRAMES = 224   # number of time frames (width) to fix to for CNN input
# Training hyperparams
BATCH_SIZE = 2
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2 if torch.cuda.is_available() else 0
SEED = 42

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ------------------------
# Utilities: protocol loader
# ------------------------
def load_protocol(protocol_path, audio_dir):
    """Return list of absolute audio paths and labels (0 bonafide, 1 spoof).
    Tolerant to several protocol formats: tries to find a filename token (wav/flac)
    or falls back to the 2nd column as file id. Maps common label tokens."""
    paths, labels = [], []
    if not os.path.isfile(protocol_path):
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # try to locate a token that looks like a filename
            file_id = None
            for p in parts:
                low = p.lower()
                if low.endswith(".wav") or low.endswith(".flac"):
                    file_id = os.path.splitext(os.path.basename(p))[0]
                    break
            # fallback to 2nd column if available, else first column
            if file_id is None:
                if len(parts) >= 2:
                    file_id = parts[1]
                else:
                    file_id = parts[0]
            # label token (use last token as common)
            label_token = parts[-1].lower()
            # construct path
            wav_name = file_id + ".wav"
            wav_path = os.path.join(audio_dir, wav_name)
            if not os.path.isfile(wav_path):
                    # skip missing file
                    continue
            # map label robustly
            if label_token in ("bonafide", "genuine", "0"):
                label = 0
            elif label_token in ("spoof", "attack", "1"):
                label = 1
            else:
                # try numeric
                try:
                    n = int(label_token)
                    label = 0 if n == 0 else 1
                except Exception:
                    label = 0 if "bonafide" in label_token else 1
            paths.append(wav_path)
            labels.append(label)
    print(f"✅ Loaded {len(paths)} samples from {protocol_path} using audio_dir={audio_dir}")
    return paths, labels

def find_train_protocols(protocols_dir):
    """Return train protocol files found in protocols_dir (sorted)."""
    protos = []
    if not os.path.isdir(protocols_dir):
        return protos
    for fname in sorted(os.listdir(protocols_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        if "train" in fname.lower():
            protos.append(os.path.join(protocols_dir, fname))
    return protos

# ------------------------
# LFCC extraction functions
# ------------------------
def extract_lfcc_from_waveform(y, sr=SR, n_fft=N_FFT, n_filters=N_FILTERS, n_ceps=N_CEPS,
                              frame_size=FRAME_SIZE, frame_stride=FRAME_STRIDE):
    # ensure 1d
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    # pre-emphasis
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    frame_length = int(round(frame_size * sr))
    frame_step = int(round(frame_stride * sr))
    signal_length = len(y)
    if signal_length < frame_length:
        num_frames = 1
    else:
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(y, z)

    indices = (np.tile(np.arange(0, frame_length), (num_frames, 1)) +
               np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T)
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)

    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

    # linear filterbanks
    low_freq = 0
    high_freq = sr / 2
    linear_points = np.linspace(low_freq, high_freq, n_filters + 2)
    bins = np.floor((n_fft + 1) * linear_points / sr).astype(int)

    fbank = np.zeros((n_filters, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_filters + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        # guard zeros
        if f_m == f_m_minus:
            f_m = f_m_minus + 1
        if f_m_plus == f_m:
            f_m_plus = f_m + 1
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    lfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_ceps]  # (num_frames, n_ceps)
    return lfcc  # shape (num_frames, n_ceps)

def load_audio_and_extract_lfcc(path, sr=SR, max_frames=MAX_FRAMES):
    try:
        data, fs = sf.read(path)
    except Exception as e:
        print(f"⚠️ Skipping unreadable file: {path} ({e})")
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)

    # Convert to mono and resample if needed
    if fs != sr:
        try:
            data = librosa.resample(data.astype(np.float32), orig_sr=fs, target_sr=sr)
        except Exception as e:
            print(f"⚠️ Resample failed for {path}: {e}")
            return np.zeros((N_CEPS, max_frames), dtype=np.float32)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # --- Safety cleaning rules ---
    num_samples = len(data)
    duration_sec = num_samples / sr

    if duration_sec < 1.0:
        print(f"⚠️ Skipping too short (<1s): {path}")
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)
    if duration_sec > 20.0:
        print(f"⚠️ Trimming long file (>20s): {path}")
        data = data[:int(sr * 20)]  # trim instead of skip
    if np.max(np.abs(data)) < 1e-5:
        print(f"⚠️ Skipping silent file: {path}")
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)
    if np.any(np.isnan(data)):
        print(f"⚠️ Skipping NaN file: {path}")
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)
    # ------------------------------

    try:
        lfcc = extract_lfcc_from_waveform(data, sr=sr)
    except MemoryError:
        print(f"💥 MemoryError — skipping heavy file: {path}")
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)
    except Exception as e:
        print(f"⚠️ LFCC extraction failed for {path}: {e}")
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)

    lfcc = lfcc.T
    if lfcc.shape[1] < max_frames:
        pad_width = max_frames - lfcc.shape[1]
        lfcc = np.pad(lfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        lfcc = lfcc[:, :max_frames]

    mean, std = lfcc.mean(), lfcc.std() if lfcc.std() > 0 else 1
    lfcc = (lfcc - mean) / std
    return lfcc
# ------------------------
# Dataset
# ------------------------
class LFCCDataset(Dataset):
    def __init__(self, file_paths, labels, max_frames=MAX_FRAMES):
        self.file_paths = file_paths
        self.labels = labels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        lfcc = load_audio_and_extract_lfcc(path, max_frames=self.max_frames)  # (n_ceps, max_frames)
        # add channel dimension -> (1, n_ceps, max_frames)
        lfcc = lfcc[np.newaxis, :, :].astype(np.float32)
        return torch.from_numpy(lfcc), torch.tensor(label, dtype=torch.long)

# ------------------------
# Model (ResNeXt teacher, adapted)
# ------------------------
class ResNeXtTeacher(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # load pretrained ResNeXt
        try:
            self.model = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        except TypeError:
            # older torchvision
            self.model = models.resnext50_32x4d(pretrained=True)
        # change first conv to accept 1-channel LFCC
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # replace final fc
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ------------------------
# EER computation
# ------------------------
def compute_eer(y_true, y_scores):
    # y_true: 0 bonafide, 1 spoof. pos_label=1
    fpr, tpr, thr = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # find threshold where abs(fpr - fnr) minimal
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer

# ------------------------
# Training loop
# ------------------------
def train_model(model, train_loader, val_loader, device, epochs=NUM_EPOCHS, lr=LR, save_path=OUT_MODEL):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_eer = 1.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=running_loss/ ( (pbar.n+1) if (pbar.n+1)>0 else 1 ), acc=100*correct/total)
        train_acc = 100*correct/total
        avg_loss = running_loss / len(train_loader)

        # validation -> compute scores for EER
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()  # spoof probability
                all_scores.extend(probs.tolist())
                all_labels.extend(y.numpy().tolist())
        val_eer = compute_eer(np.array(all_labels), np.array(all_scores))
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Val EER={val_eer:.4f}")
        # checkpoint
        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best EER {best_eer:.4f}. Model saved to {save_path}")
    print("Training finished. Best EER:", best_eer)

# ------------------------
# Main
# ------------------------
def main():
    print("Scanning protocols in:", PROTOCOLS_DIR)
    train_proto_paths = find_train_protocols(PROTOCOLS_DIR)

    # ensure explicit TRAIN_PROTO is first if present
    if os.path.isfile(TRAIN_PROTO):
        if TRAIN_PROTO in train_proto_paths:
            train_proto_paths.remove(TRAIN_PROTO)
        train_proto_paths.insert(0, TRAIN_PROTO)

    # categorize protocols: 2019 first, then asvspoof5, unknown => 2019 group
    proto_2019 = []
    proto_5 = []
    for p in train_proto_paths:
        fn = os.path.basename(p).lower()
        if "2019" in fn or "asvspoof2019" in fn or "la" in fn:
            proto_2019.append(p)
        elif "5" in fn or "asvspoof5" in fn:
            proto_5.append(p)
        else:
            proto_2019.append(p)

    all_train_files = []
    all_train_labels = []

    # load ASVspoof2019 train protocols first (using TRAIN_AUDIO_DIR)
    for proto in proto_2019:
        try:
            paths, labels = load_protocol(proto, TRAIN_AUDIO_DIR)
        except FileNotFoundError:
            continue
        if paths:
            all_train_files.extend(paths)
            all_train_labels.extend(labels)

    # then load ASVspoof5 train protocols (using ASVSPOOF5_AUDIO_DIR)
    for proto in proto_5:
        try:
            paths, labels = load_protocol(proto, ASVSPOOF5_AUDIO_DIR)
        except FileNotFoundError:
            continue
        if paths:
            all_train_files.extend(paths)
            all_train_labels.extend(labels)

    # fallback: if still empty, try scanning protocols and attempt both audio dirs for each proto
    if len(all_train_files) == 0:
        print("No train files found in prioritized pass; attempting both audio dirs per protocol.")
        for proto in train_proto_paths:
            for cand_dir in (TRAIN_AUDIO_DIR, ASVSPOOF5_AUDIO_DIR):
                try:
                    paths, labels = load_protocol(proto, cand_dir)
                except FileNotFoundError:
                    continue
                if paths:
                    all_train_files.extend(paths)
                    all_train_labels.extend(labels)
                    break

    # final fallback to explicit TRAIN_PROTO + TRAIN_AUDIO_DIR if nothing
    if len(all_train_files) == 0 and os.path.isfile(TRAIN_PROTO):
        print("Final fallback: loading TRAIN_PROTO with TRAIN_AUDIO_DIR")
        all_train_files, all_train_labels = load_protocol(TRAIN_PROTO, TRAIN_AUDIO_DIR)

    # dev protocol (unchanged)
    print("Loading dev protocol:", DEV_PROTO)
    dev_files, dev_labels = load_protocol(DEV_PROTO, DEV_AUDIO_DIR)

    print(f"Aggregated train samples: {len(all_train_files)}  Dev samples: {len(dev_files)}")
    if len(all_train_files) == 0 or len(dev_files) == 0:
        print("No data found. Check audio/protocol paths.")
        sys.exit(1)

    # Create datasets / loaders
    train_ds = LFCCDataset(all_train_files, all_train_labels, max_frames=MAX_FRAMES)
    dev_ds = LFCCDataset(dev_files, dev_labels, max_frames=MAX_FRAMES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"))
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"))

    # model
    model = ResNeXtTeacher(num_classes=2)
    print("Model created. Device:", DEVICE)
    # print(model)  # optional: prints model summary (long)

    # train
    train_model(model, train_loader, dev_loader, device=DEVICE, epochs=NUM_EPOCHS, lr=LR, save_path=OUT_MODEL)

if __name__ == "__main__":
    main()
# ...existing code...