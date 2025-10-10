# src/evaluate_unet.py (Version 2.0 - mit dynamischem Voxel Spacing)
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from medpy.metric.binary import hd95, assd
import pandas as pd
from torch.nn import functional as F
import pydicom

from model import AttentionUNet
from dataset import NasalSinusDataset3D

def get_voxel_spacing(dicom_folder_path, patient_id):
    """Liest die Voxel-Größen aus der ersten DICOM-Datei eines Patienten."""
    try:
        patient_dicom_path = dicom_folder_path / patient_id
        if not patient_dicom_path.exists():
            print(f"WARNUNG: DICOM-Ordner für Patient {patient_id} nicht gefunden. Verwende Standard-Spacing (1,1,1).")
            return (1.0, 1.0, 1.0) # Fallback

        first_dicom = next(f for f in sorted(list(patient_dicom_path.rglob("*"))) if f.is_file() and not f.suffix)
        
        dcm = pydicom.dcmread(first_dicom)
        pixel_spacing = dcm.PixelSpacing
        slice_thickness = dcm.SliceThickness
        
        return (float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1]))
    except Exception as e:
        print(f"WARNUNG: Konnte Voxel Spacing für {patient_id} nicht lesen: {e}. Verwende Standard (1,1,1).")
        return (1.0, 1.0, 1.0)

def evaluate_metrics(pred_mask, true_mask, voxel_spacing):
    """Berechnet Dice, IoU, HD95 und ASSD für ein einzelnes 3D-Volumen."""
    pred_mask = pred_mask.astype(np.uint8)
    true_mask = true_mask.astype(np.uint8)
    
    if np.sum(true_mask) == 0 and np.sum(pred_mask) == 0:
        return {"Dice": 1.0, "IoU": 1.0, "HD95": 0.0, "ASSD": 0.0}
    if np.sum(pred_mask) == 0 or np.sum(true_mask) == 0:
        return {"Dice": 0.0, "IoU": 0.0, "HD95": np.nan, "ASSD": np.nan}

    intersection = np.sum(pred_mask & true_mask)
    union = np.sum(pred_mask | true_mask)
    dice = (2. * intersection) / (np.sum(pred_mask) + np.sum(true_mask))
    iou = intersection / union
    
    hd95_score = hd95(pred_mask, true_mask, voxelspacing=voxel_spacing)
    assd_score = assd(pred_mask, true_mask, voxelspacing=voxel_spacing)

    return {"Dice": dice, "IoU": iou, "HD95": hd95_score, "ASSD": assd_score}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AttentionUNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    val_path = Path(args.data_path) / "val"
    raw_dicom_path = Path(args.raw_dicom_path)
    val_dataset = NasalSinusDataset3D(str(val_path), fixed_size=args.size, fixed_depth=args.depth)
    
    all_patient_results = []

    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc="Evaluating Validation Set"):
            image_vol_tensor, true_mask_tensor = val_dataset[i]
            true_mask_np = true_mask_tensor.squeeze().numpy()
            
            input_tensor = image_vol_tensor.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            pred_mask_np = (torch.sigmoid(outputs) > 0.5).cpu().squeeze().numpy()

            true_depth_slices = np.where(np.any(true_mask_np, axis=(1, 2)))[0]
            original_depth = true_depth_slices[-1] + 1 if len(true_depth_slices) > 0 else 0
            
            pred_mask_np = pred_mask_np[:original_depth]
            true_mask_np = true_mask_np[:original_depth]

            # Extrahiere die Patienten-ID aus dem .npy-Dateinamen
            patient_id = val_dataset.image_files[i].split('.npy')[0]
            
            voxel_spacing = get_voxel_spacing(raw_dicom_path, patient_id)
            
            metrics = evaluate_metrics(pred_mask_np, true_mask_np, voxel_spacing)
            metrics['patient_id'] = patient_id
            all_patient_results.append(metrics)

    df = pd.DataFrame(all_patient_results)
    print("\n--- 3D U-Net Evaluation Results (per Patient) ---")
    print(df.to_string())
    
    print("\n--- Average Metrics ---")
    print(df.drop(columns=['patient_id']).mean())
    
    df.to_csv("evaluation_results_unet.csv", index=False)
    print("\nResults saved to evaluation_results_unet.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates a trained 3D U-Net on its validation set.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pth).")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the U-Net dataset directory.")
    parser.add_argument('--raw_dicom_path', type=str, required=True, help="Path to the directory with the original DICOM patient folders.")
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--depth', type=int, default=32)
    args = parser.parse_args()
    main(args)