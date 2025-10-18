"""
evaluate_asvspoof2021.py
Evaluate trained ResNeXt LFCC model on ASVspoof2021 dataset (with labeled protocol).
"""

import os
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

# ----------------------------
# CONFIG — EDIT THESE PATHS
# ----------------------------
MODEL_PATH = r"D:\Thesis traning\models\resnext_lfcc_asvspoof2019_best.pth"
PROTO_PATH = r"D:\Thesis traning\trial_metadata.txt"
AUDIO_DIR  = r"D:\Thesis traning\wav_files"  # folder where LA_E_*.wav are located
N_SAMPLES  = 20000  # how many samples to test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LFCC params (match training)
SR = 16000
N_FFT = 512
N_FILTERS = 40
N_CEPS = 20
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.01
MAX_FRAMES = 224

# ----------------------------
# Protocol Loader (for 2021)
# ----------------------------
def load_asvspoof2021_protocol(protocol_path, audio_dir, max_samples=None):
    """
    Load ASVspoof2021 style protocol:
    LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
    """
    paths, labels = [], []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            file_id = parts[1]
            label_str = parts[5].lower()
            wav_name = file_id + ".wav"
            flac_name = file_id + ".flac"

            wav_path = os.path.join(audio_dir, wav_name)
            flac_path = os.path.join(audio_dir, flac_name)

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
        paths = [paths[i] for i in idx]
        labels = [labels[i] for i in idx]

    print(f"✅ Loaded {len(paths)} labeled samples from protocol.")
    return paths, labels

# ----------------------------
# LFCC extraction (same as training)
# ----------------------------
def extract_lfcc_from_waveform(y, sr=SR, n_fft=N_FFT, n_filters=N_FILTERS, n_ceps=N_CEPS,
                              frame_size=FRAME_SIZE, frame_stride=FRAME_STRIDE):
    if y.ndim > 1:
        y = np.mean(y, axis=1)
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
    low_freq = 0
    high_freq = sr / 2
    linear_points = np.linspace(low_freq, high_freq, n_filters + 2)
    bins = np.floor((n_fft + 1) * linear_points / sr).astype(int)
    fbank = np.zeros((n_filters, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_filters + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    lfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_ceps]
    return lfcc

def load_audio_and_extract_lfcc(path):
    data, fs = sf.read(path)
    if fs != SR:
        data = librosa.resample(data.astype(np.float32), orig_sr=fs, target_sr=SR)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    lfcc = extract_lfcc_from_waveform(data, sr=SR)
    lfcc = lfcc.T
    if lfcc.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - lfcc.shape[1]
        lfcc = np.pad(lfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        lfcc = lfcc[:, :MAX_FRAMES]
    lfcc = (lfcc - lfcc.mean()) / (lfcc.std() if lfcc.std() > 0 else 1.0)
    return lfcc[np.newaxis, :, :].astype(np.float32)

# ----------------------------
# Dataset
# ----------------------------
class LFCCDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        lfcc = load_audio_and_extract_lfcc(self.file_paths[idx])
        label = self.labels[idx]
        return torch.from_numpy(lfcc), torch.tensor(label, dtype=torch.long)

# ----------------------------
# Model (same as training)
# ----------------------------
class ResNeXtTeacher(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        try:
            self.model = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        except:
            self.model = models.resnext50_32x4d(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ----------------------------
# EER computation
# ----------------------------
def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2.0

# ----------------------------
# MAIN EVALUATION
# ----------------------------
def main():
    print("Loading ASVspoof2021 protocol...")
    test_files, test_labels = load_asvspoof2021_protocol(PROTO_PATH, AUDIO_DIR, max_samples=N_SAMPLES)
    print(f"Loaded {len(test_files)} test samples.")

    dataset = LFCCDataset(test_files, test_labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    print("Loading model...")
    model = ResNeXtTeacher(num_classes=2).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print("Evaluating...")
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(DEVICE)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())

    eer = compute_eer(np.array(all_labels), np.array(all_scores))
    print(f"\n✅ Evaluation complete: EER on {len(all_scores)} samples = {eer:.4%}")

if __name__ == "__main__":
    main()