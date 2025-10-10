# src/train.py (korrigiert)
import os; import torch; from torch.utils.data import DataLoader; from torch.optim import Adam; import pandas as pd; import matplotlib.pyplot as plt; from pathlib import Path; import argparse; from tqdm import tqdm
from dataset import NasalSinusDataset3D
from model import AttentionUNet 
import numpy  as np
import random

def dice_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds) > 0.5
    intersection = (preds.float() * targets.float()).sum()
    union = preds.float().sum() + targets.float().sum()
    return (2. * intersection + smooth) / (union + smooth)

def train(args):

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_base_dir = Path(args.run_output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda")
    exp_dir = output_base_dir / f"exp_{len(os.listdir(output_base_dir)) + 1}"

    
    exp_dir.mkdir(parents=True); print(f"Speichere Ergebnisse in: {exp_dir}")
    model = AttentionUNet().to(device)
    dataset = NasalSinusDataset3D(args.data_path + '/train'); val_dataset = NasalSinusDataset3D(args.data_path + '/val')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True); val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []; best_val_loss = float('inf'); best_val_dice = 0
    
    for epoch in range(args.epochs):
        model.train(); train_loss, train_dice = 0, 0
        for images, masks in tqdm(loader, desc=f"Epoch {epoch+1} Training"):
            images, masks = images.to(device), masks.to(device); optimizer.zero_grad()
            outputs = model(images); loss = criterion(outputs, masks); loss.backward(); optimizer.step()
            train_loss += loss.item(); train_dice += dice_score(outputs, masks)
        
        model.eval(); val_loss, val_dice = 0, 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                images, masks = images.to(device), masks.to(device); outputs = model(images)
                val_loss += criterion(outputs, masks).item(); val_dice += dice_score(outputs, masks)
        
        res = {'epoch': epoch + 1, 'train_loss': train_loss / len(loader), 'train_dice': train_dice / len(loader),
               'val_loss': val_loss / len(val_loader), 'val_dice': val_dice / len(val_loader)}
        history.append(res); print(f"Epoch {res['epoch']}: Train Loss: {res['train_loss']:.4f}, Dice: {res['train_dice']:.4f} | Val Loss: {res['val_loss']:.4f}, Dice: {res['val_dice']:.4f}")
        
        if res['val_loss'] < best_val_loss: best_val_loss = res['val_loss']; torch.save(model.state_dict(), exp_dir / 'best_loss.pth')
        if res['val_dice'] > best_val_dice: best_val_dice = res['val_dice']; torch.save(model.state_dict(), exp_dir / 'best_dice.pth')
    
    pd.DataFrame(history).to_csv(exp_dir / 'results.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/unet_dataset_512')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--run_output_dir', default='runs/train')
    args = parser.parse_args()
    train(args)