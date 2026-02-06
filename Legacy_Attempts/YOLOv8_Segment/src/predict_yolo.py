# predict_yolo.py 
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import shutil

def get_processed_patient_ids(yolo_dataset_path):
    
    ids = set()
    for split in ['train', 'val']:
        image_dir = yolo_dataset_path / 'images' / split
        if image_dir.exists():
            for img_file in image_dir.iterdir():
                patient_id = img_file.name.split('_')[0]
                ids.add(patient_id)
    return ids

def robust_imread(file_path):
    
    try:
        with open(file_path, 'rb') as f:
            numpy_array = np.frombuffer(f.read(), np.uint8)
        image = cv2.imdecode(numpy_array, cv2.IMREAD_ANYCOLOR)
        return image
    except Exception as e:
        print(f"WARNUNG: Konnte Bild nicht laden {file_path}: {e}")
        return None

def predict(args):
    model_path = Path(args.model_path)
    inference_data_path = Path(args.inference_data_path)
    yolo_dataset_path = Path(args.yolo_dataset_path)
    
    
    output_dir = Path(args.output_dir) / model_path.parent.parent.name
    if output_dir.exists():
        print(f"Lösche alten Vorhersage-Ordner: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    print(f"Lade trainiertes YOLOv8-Modell von: {model_path}")
    model = YOLO(model_path)
    
    
    seen_patient_ids = get_processed_patient_ids(yolo_dataset_path)
    print(f"{len(seen_patient_ids)} Patienten aus Trainings-/Validierungsset gefunden. Diese werden übersprungen.")

    
    all_test_patients = [p for p in inference_data_path.iterdir() if p.is_dir()]
    patients_to_predict = [p for p in all_test_patients if p.name not in seen_patient_ids]
    
    print(f"\n{len(all_test_patients)} Patienten im Testordner gefunden.")
    print(f"Starte Vorhersage für {len(patients_to_predict)} NEUE Patienten...")

    for patient_folder in tqdm(patients_to_predict, desc="Patienten-Vorhersage"):
        image_files = sorted(list(patient_folder.rglob("*.png")))
        
        for image_path in image_files:
            original_image = robust_imread(str(image_path))
            if original_image is None: continue
                
            results = model.predict(image_path, verbose=False)
            result = results[0]
            
            if result.masks is not None and len(result.masks.data) > 0:
                image_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR) if len(original_image.shape) == 2 else original_image.copy()
                
                for mask_tensor in result.masks.data:
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    
                    if np.any(mask_np):
                        red_mask_overlay = np.zeros_like(image_color)
                        
                        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(red_mask_overlay, contours, -1, (0, 0, 255), thickness=cv2.FILLED)
                        
                        image_color = cv2.addWeighted(image_color, 0.7, red_mask_overlay, 0.3, 0)
                
                relative_path = image_path.relative_to(inference_data_path)
                save_path = output_dir / relative_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(str(save_path), image_color)

    print(f"\nAlle Vorhersagen abgeschlossen. Ergebnisse gespeichert in: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Macht Vorhersagen auf neuen Bildern mit einem trainierten YOLOv8-Modell.")
    parser.add_argument('--model_path', type=str, required=True, help='Pfad zum trainierten Modell (.pt-Datei).')
    parser.add_argument('--inference_data_path', type=str, required=True, help='Pfad zum Test-Ordner mit ungelabelten Patientendaten.')
    parser.add_argument('--yolo_dataset_path', type=str, required=True, help='Pfad zum aufbereiteten YOLO-Datensatz (um gesehene Patienten auszuschließen).')
    parser.add_argument('--output_dir', type=str, default='runs/predict_yolo', help='Ordner zum Speichern der Vorhersagen.')
    
    args = parser.parse_args()
    predict(args)