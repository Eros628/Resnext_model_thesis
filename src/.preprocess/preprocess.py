# preprocess_all_in_one_NO_ARGS.py
# DOUBLE-CLICK AND RUN — NO COMMAND LINE NEEDED
# Just change the 5 paths below and press F5 / run the file

import os
import numpy as np
import soundfile as sf
import librosa
from scipy.fftpack import dct
from tqdm import tqdm
from pathlib import Path

import torch
torch.set_num_threads(1)

# ====================== EDIT THESE 5 LINES ONLY ======================
TRAIN_PROTO = r"D:\Thesis traning\protocols\ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTO   = r"D:\Thesis traning\protocols\ASVspoof2019.LA.cm.dev.trl.txt"
TRAIN_AUDIO = r"D:\Thesis traning\asvspoof2019\train"
DEV_AUDIO   = r"D:\Thesis traning\asvspoof2019\dev"
OUTPUT_ROOT = r"D:\Thesis traning\clean_samples"      # Where features will be saved
# =====================================================================

# Silero VAD (auto download on first run)
print("Loading Silero VAD model...")
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              trust_repo=True)
(get_speech_timestamps, _, read_audio, _, _) = utils

def apply_vad(wav, sr=16000):
    try:
        speech_ts = get_speech_timestamps(wav, model, sampling_rate=sr,
                                         threshold=0.5,
                                         min_speech_duration_ms=250,
                                         min_silence_duration_ms=100)
        if not speech_ts:
            return wav
        return np.concatenate([wav[s['start']:s['end']] for s in speech_ts])
    except:
        return wav

# 60-dim LFCC + delta + delta-delta
SR = 16000
MAX_FRAMES = 400

def extract_lfcc_60(y):
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    S = librosa.stft(y, n_fft=512, hop_length=160, win_length=400, window='hamming')
    pow_spec = np.abs(S)**2

    # Linear freq filterbank
    freqs = np.linspace(0, SR/2, 42)
    bins = np.floor((513) * freqs / SR).astype(int)
    fbank = np.zeros((40, 257))
    for m in range(1, 41):
        l, c, r = bins[m-1], bins[m], bins[m+1]
        if c > l: fbank[m-1, l:c] = np.linspace(0, 1, c-l)
        if r > c: fbank[m-1, c:r] = np.linspace(1, 0, r-c)

    fbanks = np.dot(pow_spec.T, fbank.T)
    fbanks = np.where(fbanks == 0, np.finfo(float).eps, fbanks)
    fbanks = 20 * np.log10(fbanks)

    lfcc = dct(fbanks, type=2, axis=1, norm='ortho')[:, 1:21]  # skip c0
    delta = librosa.feature.delta(lfcc)
    ddelta = librosa.feature.delta(lfcc, order=2)
    return np.concatenate([lfcc, delta, ddelta], axis=1).astype(np.float32)

def process_one_protocol(proto_path, audio_dir, split_name):
    real_dir = Path(OUTPUT_ROOT) / "real" / split_name
    fake_dir = Path(OUTPUT_ROOT) / "fake" / split_name
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    with open(proto_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"\n=== Processing {split_name.upper()} set ({len(lines)} files) ===")

    for line in tqdm(lines, desc=split_name):
        parts = line.split()
        if len(parts) < 5: continue
        uid = parts[1]
        label = 0 if parts[-1] == "bonafide" else 1

        wav_path = os.path.join(audio_dir, uid + ".wav")
        if not os.path.exists(wav_path):
            wav_path = wav_path.replace(".wav", ".flac")

        if not os.path.exists(wav_path):
            continue

        try:
            y, sr = sf.read(wav_path)
            if y.ndim > 1: y = y.mean(axis=1).astype(np.float32)

            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)

            y = apply_vad(y, SR)
            if len(y) < SR * 0.6:  # skip too short after VAD
                continue

            feats = extract_lfcc_60(y)  # (T, 60)

            # pad / truncate
            if feats.shape[0] > MAX_FRAMES:
                feats = feats[:MAX_FRAMES]
            else:
                feats = np.pad(feats, ((0, MAX_FRAMES - feats.shape[0]), (0,0)), 'constant')

            # normalize
            feats = (feats - feats.mean()) / (feats.std() + 1e-8)

            out_dir = real_dir if label == 0 else fake_dir
            np.save(out_dir / f"{uid}.npy", feats.T)  # (60, 400)

        except Exception as e:
            print(f"Skip {uid}: {e}")

# ========================= RUN =========================
if __name__ == "__main__":
    print("Starting full preprocessing (VAD + 60-dim LFCC)...")
    process_one_protocol(TRAIN_PROTO, TRAIN_AUDIO, "train")
    process_one_protocol(DEV_PROTO,   DEV_AUDIO,   "dev")
    print("\nFINISHED!")
    print(f"Features saved in: {os.path.abspath(OUTPUT_ROOT)}")
    print("   clean_samples/real/train/*.npy")
    print("   clean_samples/fake/train/*.npy")
    print("   clean_samples/real/dev/*.npy")
    print("   clean_samples/fake/dev/*.npy")
    print("Now you can run your training script directly — super fast!")
    input("\nPress Enter to exit...")