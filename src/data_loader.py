import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils import load_audio_and_extract_lfcc
import config

class LFCCDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Feature extraction is done here
        lfcc = load_audio_and_extract_lfcc(path, max_frames=config.MAX_FRAMES)
        
        # Add channel dimension for the CNN -> (1, n_ceps, max_frames)
        lfcc = lfcc[np.newaxis, :, :].astype(np.float32)
        
        return torch.from_numpy(lfcc), torch.tensor(label, dtype=torch.long)