import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import project modules
import config
from src.utils import load_protocol, find_train_protocols # Only protocol helpers needed here
from src.data_loader import LFCCDataset
from src.models.resnet_teacher import ResNeXtTeacher
from src.trainer import train_model

def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

def main():
    """Main function to run the training pipeline."""
    set_seed(config.SEED)
    os.makedirs(os.path.dirname(config.OUT_MODEL), exist_ok=True)
    
    # --- Data Loading and Preparation ---
    print("🔎 Loading training and validation protocols...")
    
    # Your original logic for aggregating multiple protocol files
    train_files, train_labels = load_protocol(config.TRAIN_PROTO, config.TRAIN_AUDIO_DIR)
    # Note: Add your logic here to include the ASVspoof5 data if needed
    
    dev_files, dev_labels = load_protocol(config.DEV_PROTO, config.DEV_AUDIO_DIR)

    if not train_files or not dev_files:
        print("❌ Error: No data found. Please check paths in config.py")
        sys.exit(1)
        
    print(f"✅ Found {len(train_files)} training samples and {len(dev_files)} validation samples.")

    # --- Create Datasets and DataLoaders ---
    train_dataset = LFCCDataset(file_paths=train_files, labels=train_labels)
    dev_dataset = LFCCDataset(file_paths=dev_files, labels=dev_labels)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS
    )
    
    # --- Model Initialization ---
    print(f"🧠 Initializing the ResNeXtTeacher model on device: {config.DEVICE}")
    model = ResNeXtTeacher(num_classes=2)
    
    # --- Start Training ---
    print("🚀 Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader,
        device=config.DEVICE,
        epochs=config.NUM_EPOCHS,
        lr=config.LR,
        save_path=config.OUT_MODEL
    )

if __name__ == "__main__":
    main()