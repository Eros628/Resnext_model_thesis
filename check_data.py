import os
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- PATH CONFIGURATION ----------
asv2019_train_protocol = r"D:\Thesis traning\protocols\ASVspoof2019.LA.cm.train.trn.txt"
asv2019_dev_protocol   = r"D:\Thesis traning\protocols\ASVspoof2019.LA.cm.dev.trl.txt"
asv5_train_protocol    = r"D:\Thesis traning\protocols\ASVspoof5.train.tsv.txt"

asv2019_train_audio = r"D:\Thesis traning\asvspoof2019\train"
asv2019_dev_audio   = r"D:\Thesis traning\asvspoof2019\dev"
asv5_train_audio    = r"D:\Thesis traning\asvspoof5\wav_T"
# ----------------------------------------


def load_protocol_2019(protocol_path):
    """Read ASVspoof2019 protocol file"""
    data = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id = parts[1]
            label = parts[-1].lower()
            data.append({"file_id": file_id, "label": label})
    return pd.DataFrame(data)


def load_protocol_asv5(protocol_path):
    """Read ASVspoof5 protocol file"""
    data = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            file_id = parts[1]
            label = parts[-2].lower()  # 'bonafide' or 'spoof'
            data.append({"file_id": file_id, "label": label})
    return pd.DataFrame(data)


def get_audio_info(file_path):
    """Get audio duration and file format"""
    try:
        info = torchaudio.info(file_path)
        duration = info.num_frames / info.sample_rate
        ext = os.path.splitext(file_path)[1]
        return duration, ext
    except Exception:
        return 0, "unknown"


def analyze_dataset(df, audio_path, name, extension=".flac"):
    """Compute stats and visualize"""
    durations, exts = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {name}"):
        file_path = os.path.join(audio_path, f"{row.file_id}{extension}")
        dur, ext = get_audio_info(file_path)
        durations.append(dur)
        exts.append(ext)

    df["duration_sec"] = durations
    df["format"] = exts

    stats = {
        "dataset": name,
        "total_files": len(df),
        "real_count": (df.label == "bonafide").sum(),
        "spoof_count": (df.label == "spoof").sum(),
        "avg_duration": df["duration_sec"].mean(),
        "min_duration": df["duration_sec"].min(),
        "max_duration": df["duration_sec"].max(),
        "formats": df["format"].value_counts().to_dict()
    }

    # Plot class balance
    plt.figure(figsize=(5, 4))
    df["label"].value_counts().plot(kind="bar", color=["green", "red"])
    plt.title(f"{name} — Real vs Spoof Count")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(axis="y")
    plt.show()

    # Plot duration histogram
    plt.figure(figsize=(5, 4))
    df["duration_sec"].plot(kind="hist", bins=40, color="blue", alpha=0.7)
    plt.title(f"{name} — Duration Distribution (seconds)")
    plt.xlabel("Duration (s)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    return stats


# ---------- RUN ANALYSIS ----------
print("📂 Loading datasets...")

train2019_df = load_protocol_2019(asv2019_train_protocol)
dev2019_df = load_protocol_2019(asv2019_dev_protocol)
train5_df = load_protocol_asv5(asv5_train_protocol)

print(f"ASVspoof2019 train: {len(train2019_df)} samples")
print(f"ASVspoof2019 dev:   {len(dev2019_df)} samples")
print(f"ASVspoof5 train:    {len(train5_df)} samples")

# Process all datasets
stats_2019_train = analyze_dataset(train2019_df, asv2019_train_audio, "ASVspoof2019 Train", extension=".flac")
stats_2019_dev   = analyze_dataset(dev2019_df, asv2019_dev_audio, "ASVspoof2019 Dev", extension=".flac")
stats_asv5_train = analyze_dataset(train5_df, asv5_train_audio, "ASVspoof5 Train", extension=".wav")

# ---------- SUMMARY ----------
summary_df = pd.DataFrame([stats_2019_train, stats_2019_dev, stats_asv5_train])
print("\n========= DATASET SUMMARY =========")
print(summary_df.to_string(index=False))
