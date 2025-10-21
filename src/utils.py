import os
import random
import numpy as np
import soundfile as sf
import librosa
from scipy.fftpack import dct
from sklearn.metrics import roc_curve
import config

def compute_eer(y_true, y_scores):
    # y_true: 0 bonafide, 1 spoof. pos_label=1
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    # find threshold where abs(fpr - fnr) minimal
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer

def extract_lfcc_from_waveform(y, sr=config.SR, n_fft=config.N_FFT, n_filters=config.N_FILTERS, 
                               n_ceps=config.N_CEPS, frame_size=config.FRAME_SIZE, 
                               frame_stride=config.FRAME_STRIDE):
    
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
        num_frames = int(np.ceil(float(np.abs(len(y) - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(y, np.zeros((pad_signal_length - len(y))))
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)

    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

    # Linear filterbanks
    low_freq, high_freq = 0, sr / 2
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

    lfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_ceps]
    return lfcc

def load_audio_and_extract_lfcc(path, sr=config.SR, max_frames=config.MAX_FRAMES):
    """Loads an audio file, cleans it, and extracts normalized LFCCs."""
    try:
        data, fs = sf.read(path)
    except Exception as e:
        print(f"⚠️ Skipping unreadable file: {path} ({e})")
        return np.zeros((config.N_CEPS, max_frames), dtype=np.float32)

    # Convert to mono and resample if needed
    if fs != sr:
        try:
            data = librosa.resample(data.astype(np.float32), orig_sr=fs, target_sr=sr)
        except Exception as e:
            print(f"⚠️ Resample failed for {path}: {e}")
            return np.zeros((config.N_CEPS, max_frames), dtype=np.float32)

    if data.ndim > 1: data = np.mean(data, axis=1)
    if len(data) / sr < 1.0: return np.zeros((config.N_CEPS, max_frames), dtype=np.float32)
    if len(data) / sr > 20.0: data = data[:int(sr * 20)]
    if np.max(np.abs(data)) < 1e-5: return np.zeros((config.N_CEPS, max_frames), dtype=np.float32)
    if np.any(np.isnan(data)): return np.zeros((config.N_CEPS, max_frames), dtype=np.float32)

    try:
        lfcc = extract_lfcc_from_waveform(data, sr=sr)
    except MemoryError:
        print(f"💥 MemoryError — skipping heavy file: {path}")
        return np.zeros((config.N_CEPS, max_frames), dtype=np.float32)
    except Exception as e:
        print(f"⚠️ LFCC extraction failed for {path}: {e}")
        return np.zeros((config.N_CEPS, max_frames), dtype=np.float32)

    lfcc = lfcc.T
    if lfcc.shape[1] < max_frames:
        lfcc = np.pad(lfcc, ((0, 0), (0, max_frames - lfcc.shape[1])), mode="constant")
    else:
        lfcc = lfcc[:, :max_frames]

    mean, std = lfcc.mean(), lfcc.std() if lfcc.std() > 0 else 1
    lfcc = (lfcc - mean) / std
    return lfcc


def load_asvspoof2021_protocol(protocol_path, audio_dir, max_samples=None):
    """
    Loads the ASVspoof 2021 evaluation protocol.
    Example line: LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
    """   
    paths, labels = [], []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            file_id = parts[1]
            label_str = parts[5].lower()
            
            # Check for both .wav and .flac extensions
            wav_path = os.path.join(audio_dir, file_id + ".wav")
            flac_path = os.path.join(audio_dir, file_id + ".flac")

            if os.path.isfile(wav_path):
                path = wav_path
            elif os.path.isfile(flac_path):
                path = flac_path
            else:
                continue # Skip if audio file is not found

            label = 1 if label_str == "spoof" else 0
            paths.append(path)
            labels.append(label)

    # Subsample if a maximum number of samples is specified
    if max_samples and len(paths) > max_samples:
        indices = random.sample(range(len(paths)), max_samples)
        paths = [paths[i] for i in indices]
        labels = [labels[i] for i in indices]

    print(f"✅ Loaded {len(paths)} samples from the ASVspoof 2021 protocol.")
    return paths, labels