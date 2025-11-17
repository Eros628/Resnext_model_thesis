"""
evaluate_asvspoof2021_modified.py
Evaluate the **modified ResNet-50 LFCC** (MAX_FRAMES=750) on ASVspoof2021.
Matches the training script you posted.
"""

import os
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

# ----------------------------------------------------------------------
# CONFIG – EDIT THESE PATHS
# ----------------------------------------------------------------------
MODEL_PATH = r"D:\Thesis traning\models\resnet50_lfcc_asvspoof2019_400.pth"   # <-- your trained checkpoint
PROTO_PATH = r"D:\Thesis traning\protocols\trial_metadata.txt"
AUDIO_DIR  = r"D:\Thesis traning\asvspoof2021-eval\wav_files"
N_SAMPLES  = 100000                    # set to int to evaluate a random subset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LFCC hyper-parameters – **must be identical to training**
SR           = 16000
N_FFT        = 512
N_FILTERS    = 40
N_CEPS       = 20
FRAME_SIZE   = 0.025
FRAME_STRIDE = 0.01
MAX_FRAMES   =  400                  # <-- 7.5 s, same as training

# Windows-safe workers
NUM_WORKERS = 0 if os.name == "nt" else 4

# ----------------------------------------------------------------------
# 1. Protocol loader (ASVspoof 2021 style)
# ----------------------------------------------------------------------
def load_asvspoof2021_protocol(proto_path, audio_dir, max_samples=None):
    """
    Example line:
    LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
    """
    paths, labels = [], []
    with open(proto_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            file_id = parts[1]
            label_str = parts[5].lower()                     # "bonafide" or "spoof"
            wav_path  = os.path.join(audio_dir, file_id + ".wav")
            flac_path = os.path.join(audio_dir, file_id + ".flac")
            if os.path.isfile(wav_path):
                path = wav_path
            elif os.path.isfile(flac_path):
                path = flac_path
            else:
                continue
            label = 0 if label_str == "bonafide" else 1
            paths.append(path)
            labels.append(label)

    if max_samples and len(paths) > max_samples:
        idx = random.sample(range(len(paths)), max_samples)
        paths  = [paths[i]  for i in idx]
        labels = [labels[i] for i in idx]

    print(f"Loaded {len(paths)} samples from protocol.")
    return paths, labels


# ----------------------------------------------------------------------
# 2. LFCC extraction – **identical to training**
# ----------------------------------------------------------------------
def extract_lfcc_from_waveform(y, sr=SR):
    if y.ndim > 1: y = np.mean(y, axis=1)
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    frame_len = int(round(FRAME_SIZE * sr))
    frame_step = int(round(FRAME_STRIDE * sr))
    signal_len = len(y)

    # number of frames
    num_frames = 1 if signal_len < frame_len else int(np.ceil((signal_len - frame_len) / frame_step)) + 1
    pad_len = num_frames * frame_step + frame_len
    pad_signal = np.pad(y, (0, pad_len - signal_len), mode='constant')

    # frame indexing
    indices = (np.tile(np.arange(0, frame_len), (num_frames, 1)) +
               np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T)
    frames = pad_signal[indices]
    frames *= np.hamming(frame_len)

    # power spectrum
    mag = np.abs(np.fft.rfft(frames, N_FFT))
    pow_spec = (1.0 / N_FFT) * (mag ** 2)

    # mel-filter bank (linear spacing)
    low_freq, high_freq = 0, sr / 2
    lin_pts = np.linspace(low_freq, high_freq, N_FILTERS + 2)
    bins = np.floor((N_FFT + 1) * lin_pts / sr).astype(int)

    fbank = np.zeros((N_FILTERS, int(np.floor(N_FFT / 2 + 1))))
    for m in range(1, N_FILTERS + 1):
        f_m_minus, f_m, f_m_plus = bins[m-1], bins[m], bins[m+1]
        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    filter_banks = np.dot(pow_spec, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # DCT → LFCC
    lfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :N_CEPS]   # (T, 20)
    return lfcc


def load_audio_and_extract_lfcc(path, max_frames=MAX_FRAMES):
    """Return (20, max_frames) – same shape as training."""
    try:
        data, fs = sf.read(path)
    except Exception:
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)

    # resample if needed
    if fs != SR:
        try:
            data = librosa.resample(data.astype(np.float32), orig_sr=fs, target_sr=SR)
        except Exception:
            return np.zeros((N_CEPS, max_frames), dtype=np.float32)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # sanity checks (same as training)
    duration = len(data) / SR
    if duration < 1.0 or duration > 20.0 or np.max(np.abs(data)) < 1e-5 or np.any(np.isnan(data)):
        return np.zeros((N_CEPS, max_frames), dtype=np.float32)
    if duration > 20.0:
        data = data[:int(SR * 20)]

    lfcc = extract_lfcc_from_waveform(data, SR)          # (T, 20)

    # pad / truncate on time axis
    if lfcc.shape[0] < max_frames:
        pad = max_frames - lfcc.shape[0]
        lfcc = np.pad(lfcc, ((0, pad), (0, 0)), mode='constant')
    else:
        lfcc = lfcc[:max_frames, :]

    # per-utterance normalisation (same as training)
    mean = lfcc.mean()
    std  = lfcc.std() if lfcc.std() > 0 else 1.0
    lfcc = (lfcc - mean) / std

    # (20, T)  →  ready for torch
    return lfcc.T


# ----------------------------------------------------------------------
# 3. Dataset
# ----------------------------------------------------------------------
class LFCCEvalDataset(Dataset):
    def __init__(self, file_paths, labels, max_frames=MAX_FRAMES):
        self.file_paths = file_paths
        self.labels     = labels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        lfcc = load_audio_and_extract_lfcc(self.file_paths[idx], self.max_frames)   # (20, 750)
        lfcc = torch.from_numpy(lfcc).unsqueeze(0).float()                         # (1, 20, 750)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return lfcc, label


# ----------------------------------------------------------------------
# 4. Modified ResNet-50 (exactly the same as training)
# ----------------------------------------------------------------------
class ResNet50LFCC(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.resnet50(weights='IMAGENET1K_V1')

        # 1-channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

        # remove max-pool
        model.maxpool = nn.Identity()

        # keep height, stride only on width
        model.layer1[0].conv1.stride = (1, 2)
        model.layer1[0].downsample[0].stride = (1, 2)

        # classifier
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------------------
# 5. EER
# ----------------------------------------------------------------------
def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2.0


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------
def main():
    print("=== ASVspoof2021 Evaluation (modified ResNet-50 LFCC) ===")
    test_files, test_labels = load_asvspoof2021_protocol(PROTO_PATH, AUDIO_DIR, max_samples=N_SAMPLES)

    dataset = LFCCEvalDataset(test_files, test_labels, max_frames=MAX_FRAMES)
    loader  = DataLoader(dataset, batch_size=8, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    print(f"Loading checkpoint: {MODEL_PATH}")
    model = ResNet50LFCC(num_classes=2).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print("Running inference...")
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval"):
            x = x.to(DEVICE)                                   # (B,1,20,750)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(y.tolist())

    eer = compute_eer(np.array(all_labels), np.array(all_scores))
    print(f"\nEER on {len(all_scores)} samples = {eer:.4%}")

if __name__ == "__main__":
    main()