# src/evaluate_yolo.py (Version 2.0 - mit dynamischem Voxel Spacing)
import torch
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from medpy.metric.binary import hd95, assd
from collections import defaultdict
import pandas as pd
import pydicom

def create_mask_from_yolo_txt(label_path, height, width):
    """Liest eine YOLO .txt-Datei und erstellt eine binäre Ground-Truth-Maske."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        points_normalized = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        points_denormalized = points_normalized * np.array([width, height])
        points_int = points_denormalized.astype(np.int32)
        cv2.fillPoly(mask, [points_int], 1)
    return mask

def get_voxel_spacing(dicom_folder_path, patient_id):
    """Liest die Voxel-Größen aus der ersten DICOM-Datei eines Patienten."""
    try:
        patient_dicom_path = dicom_folder_path / patient_id
        if not patient_dicom_path.exists():
            print(f"WARNUNG: DICOM-Ordner für Patient {patient_id} nicht gefunden. Verwende Standard-Spacing (1,1,1).")
            return (1.0, 1.0, 1.0) # Fallback

        # Finde die erste Datei ohne Endung im Patientenordner (rekursiv)
        first_dicom = next(f for f in sorted(list(patient_dicom_path.rglob("*"))) if f.is_file() and not f.suffix)
        
        dcm = pydicom.dcmread(first_dicom)
        pixel_spacing = dcm.PixelSpacing # Ist eine Liste [row_spacing, col_spacing] -> (y, x)
        slice_thickness = dcm.SliceThickness
        
        # Format für medpy: (z, y, x) -> (Tiefe, Höhe, Breite)
        return (float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1]))
    except Exception as e:
        print(f"WARNUNG: Konnte Voxel Spacing für {patient_id} nicht lesen: {e}. Verwende Standard (1,1,1).")
        return (1.0, 1.0, 1.0)

def evaluate_metrics(pred_mask, true_mask, voxel_spacing):
    """Berechnet Dice, IoU, HD95 und ASSD für ein 3D-Volumen."""
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
    
    model = YOLO(args.model_path)
    
    val_images_path = Path(args.data_path) / "images" / "val"
    val_labels_path = Path(args.data_path) / "labels" / "val"
    raw_dicom_path = Path(args.raw_dicom_path)
    
    image_files = list(val_images_path.glob("*.png"))
    
    patient_slices = defaultdict(list)
    for img_path in image_files:
        patient_id = img_path.stem.split('_')[0]
        patient_slices[patient_id].append(img_path)

    all_patient_results = []
    
    for patient_id, slice_paths in tqdm(patient_slices.items(), desc="Evaluating Patients"):
        patient_pred_masks, patient_true_masks = [], []
        
        sorted_paths = sorted(slice_paths, key=lambda p: int(p.stem.split('_')[-1]))

        for img_path in sorted_paths:
            image = cv2.imread(str(img_path))
            h, w, _ = image.shape
            
            results = model.predict(img_path, verbose=False, device=device)
            result = results[0]
            
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            if result.masks is not None:
                for mask_tensor in result.masks.data:
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    if mask_np.shape != (h, w):
                         mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_mask = np.maximum(pred_mask, mask_np)
            
            label_path = val_labels_path / f"{img_path.stem}.txt"
            true_mask = create_mask_from_yolo_txt(label_path, h, w)
            
            patient_pred_masks.append(pred_mask)
            patient_true_masks.append(true_mask)
            
        pred_volume = np.stack(patient_pred_masks, axis=0)
        true_volume = np.stack(patient_true_masks, axis=0)
        
        voxel_spacing = get_voxel_spacing(raw_dicom_path, patient_id)
        
        metrics = evaluate_metrics(pred_volume, true_volume, voxel_spacing)
        metrics['patient_id'] = patient_id
        all_patient_results.append(metrics)

    df = pd.DataFrame(all_patient_results)
    print("\n--- YOLOv8 Evaluation Results (per Patient) ---")
    print(df.to_string()) # .to_string() um sicherzustellen, dass die Tabelle komplett angezeigt wird
    
    print("\n--- Average Metrics ---")
    print(df.drop(columns=['patient_id']).mean())
    
    df.to_csv("evaluation_results_yolo.csv", index=False)
    print("\nResults saved to evaluation_results_yolo.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates a trained YOLOv8 model on its validation set with dynamic voxel spacing.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (best.pt).")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the YOLO dataset directory (containing images/ and labels/).")
    parser.add_argument('--raw_dicom_path', type=str, required=True, help="Path to the directory with the original DICOM patient folders.")
    args = parser.parse_args()
    main(args)