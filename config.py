import os
import torch

# -- Paths --
TRAIN_AUDIO_DIR = r"D:\Thesis traning\asvspoof2019\train"
DEV_AUDIO_DIR = r"D:\Thesis traning\asvspoof2019\dev"
ASVSPOOF5_AUDIO_DIR = r"D:\Thesis traning\asvspoof5\wav_T"
PROTOCOLS_DIR = r"D:\Thesis traning\protocols"

# Protocol file paths
TRAIN_PROTO = os.path.join(PROTOCOLS_DIR, "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTO = os.path.join(PROTOCOLS_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")

# Output model path
OUT_MODEL = "saved_models/resnext_teacher_best.pth"

# -- LFCC / Preprocessing Params --
SR = 16000
N_FFT = 512
N_FILTERS = 40
N_CEPS = 20
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.01
MAX_FRAMES = 224  # Fixed number of time frames for CNN input

# -- Training Hyperparams --
BATCH_SIZE = 2
NUM_EPOCHS = 10
LR = 1e-4
SEED = 42

# -- System --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2 if torch.cuda.is_available() else 0

# -- Evaluation Paths & Settings --
# Path to the trained model checkpoint you want to evaluate
EVAL_MODEL_PATH = r"D:\Thesis traning\models\resnext_with_spoof5_and_spoof2019.pth" 
# NOTE: In our new structure, this would point to 'saved_models/resnext_teacher_best.pth'

# Path to the ASVspoof 2021 evaluation protocol file
EVAL_PROTO_PATH = r"D:\Thesis traning\protocols\trial_metadata.txt"

# Directory containing the ASVspoof 2021 evaluation .wav files
EVAL_AUDIO_DIR = r"D:\Thesis traning\asvspoof2021-eval\wav_files"

# (Optional) Limit the number of samples for a quick test run
EVAL_N_SAMPLES = 20000