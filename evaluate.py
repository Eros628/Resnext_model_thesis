# evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our project's modules
import config
from src.utils import load_asvspoof2021_protocol, compute_eer
from src.data_loader import LFCCDataset
from src.models.resnet_teacher import ResNeXtTeacher

def main():
    """Main function to run the evaluation pipeline."""
    
    # 1. Load the dataset using the specific protocol loader
    print(f"🔎 Loading evaluation data from: {config.EVAL_AUDIO_DIR}")
    test_files, test_labels = load_asvspoof2021_protocol(
        protocol_path=config.EVAL_PROTO_PATH,
        audio_dir=config.EVAL_AUDIO_DIR,
        max_samples=config.EVAL_N_SAMPLES
    )

    # 2. Create the Dataset and DataLoader
    # We reuse the *exact same* LFCCDataset class from training
    dataset = LFCCDataset(file_paths=test_files, labels=test_labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=config.NUM_WORKERS)

    # 3. Load the trained model
    # We reuse the *exact same* ResNeXtTeacher class definition
    print(f"🧠 Loading model weights from: {config.EVAL_MODEL_PATH}")
    model = ResNeXtTeacher(num_classes=2).to(config.DEVICE)
    model.load_state_dict(torch.load(config.EVAL_MODEL_PATH, map_location=config.DEVICE))
    model.eval() # Set the model to evaluation mode

    # 4. Run inference and collect scores
    print("🚀 Evaluating the model...")
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Inference"):
            inputs = inputs.to(config.DEVICE)
            
            outputs = model(inputs)
            scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() # Get spoof probability
            
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.numpy().tolist())

    # 5. Compute and display the final EER score
    # We reuse the *exact same* compute_eer function
    eer = compute_eer(np.array(all_labels), np.array(all_scores))
    print(f"\n🎉 Evaluation Complete!")
    print(f"Equal Error Rate (EER) on {len(all_scores)} samples: {eer:.4%}")

if __name__ == "__main__":
    main()