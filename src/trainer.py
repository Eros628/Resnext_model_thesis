import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.utils import compute_eer # Use relative import

def train_model(model, train_loader, val_loader, device, epochs, lr, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    best_eer = 1.0
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                # Get spoof probability scores
                scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_scores.extend(scores.tolist())
                all_labels.extend(labels.numpy().tolist())
        
        val_eer = compute_eer(np.array(all_labels), np.array(all_scores))
        
        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f} | Val EER: {val_eer:.4f}")

        # --- Checkpoint Saving ---
        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(model.state_dict(), save_path)
            print(f"🎉 New best EER! Model saved to {save_path}")

    print(f"\nTraining finished. Best validation EER: {best_eer:.4f}")