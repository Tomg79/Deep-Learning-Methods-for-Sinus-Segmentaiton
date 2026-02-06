# src/dataset.py (Flexible Version)
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

class NasalSinusDataset3D(Dataset):
    def __init__(self, data_path, fixed_size=512, fixed_depth=32):
        self.image_path = os.path.join(data_path, 'images')
        self.mask_path = os.path.join(data_path, 'masks')
        self.image_files = sorted([f for f in os.listdir(self.image_path) if f.endswith('.npy')])
        self.fixed_size = fixed_size
        self.fixed_depth = fixed_depth

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        image_vol = np.load(os.path.join(self.image_path, self.image_files[idx]))
        mask_vol = np.load(os.path.join(self.mask_path, self.image_files[idx]))
        
        # Sicherheits-Check, der die Dateigröße mit der erwarteten Größe vergleicht
        if image_vol.shape[1] != self.fixed_size or image_vol.shape[2] != self.fixed_size:
            # Automatisches Resizing als Fallback, falls die .npy-Dateien eine andere Größe haben
            print(f"WARNUNG: Falsche Größe in .npy! Erwartet {self.fixed_size}x{self.fixed_size}, aber Datei '{self.image_files[idx]}' hat {image_vol.shape[1]}x{image_vol.shape[2]}. Skaliere neu...")
            
            resized_slices_img = [cv2.resize(s, (self.fixed_size, self.fixed_size), interpolation=cv2.INTER_AREA) for s in image_vol]
            resized_slices_mask = [cv2.resize(s, (self.fixed_size, self.fixed_size), interpolation=cv2.INTER_NEAREST) for s in mask_vol]
            image_vol = np.stack(resized_slices_img, axis=0)
            mask_vol = np.stack(resized_slices_mask, axis=0)

        current_depth = image_vol.shape[0]
        if current_depth < self.fixed_depth:
            pad = ((0, self.fixed_depth - current_depth), (0, 0), (0, 0))
            image_vol = np.pad(image_vol, pad, mode='constant')
            mask_vol = np.pad(mask_vol, pad, mode='constant')
        else:
            image_vol = image_vol[:self.fixed_depth, :, :]
            mask_vol = mask_vol[:self.fixed_depth, :, :]
        
        image_float = image_vol.astype(np.float32) / 255.0
        mask = mask_vol.astype(np.float32)
        return torch.from_numpy(image_float).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)