# build_yolo_dataset.py 
import os
import json
import cv2
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

def convert_json_to_yolo_segmentation(json_path, image_width, image_height):


    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

    yolo_lines = []
    for shape in data.get('shapes', []):
        points = shape.get('points', [])
        if not points:
            continue
        
        normalized_points = []
        for x, y in points:
            x_norm = x / image_width
            y_norm = y / image_height
            normalized_points.extend([x_norm, y_norm])
        
        class_index = 0
        line = f"{class_index} " + " ".join(map(str, normalized_points))
        yolo_lines.append(line)
        
    return "\n".join(yolo_lines)

def process_split(source_split_path, image_dest_path, label_dest_path):
    
    print(f"Verarbeite Split: {source_split_path.name}...")
    image_dest_path.mkdir(parents=True, exist_ok=True)
    label_dest_path.mkdir(parents=True, exist_ok=True)

    patient_folders = [p for p in source_split_path.iterdir() if p.is_dir()]
    
    slice_counter = 0
    for patient_folder in tqdm(patient_folders, desc=f"Patienten in {source_split_path.name}"):
        for root, _, files in os.walk(patient_folder):
            png_files = sorted([f for f in files if f.endswith('.png')])
            for png_file in png_files:
                image_path = Path(root) / png_file
                json_path = image_path.with_suffix('.json')
                
                dest_filename_base = f"{patient_folder.name}_{slice_counter}"
                
                shutil.copy(image_path, image_dest_path / f"{dest_filename_base}.png")
                
                if json_path.exists():
                    image = cv2.imread(str(image_path))
                    if image is None: continue
                    height, width, _ = image.shape
                    
                    yolo_data = convert_json_to_yolo_segmentation(json_path, width, height)
                    
                    if yolo_data:
                        label_file_path = label_dest_path / f"{dest_filename_base}.txt"
                        with open(label_file_path, 'w') as f:
                            f.write(yolo_data)
                
                slice_counter += 1

def create_dataset_yaml(output_dir, train_img_path, val_img_path):
    """Erstellt die für YOLO notwendige dataset.yaml Datei."""
    yaml_content = {
        'train': str(train_img_path.relative_to(output_dir)),
        'val': str(val_img_path.relative_to(output_dir)),
        'nc': 1,
        'names': ['nasal_sinus']
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"dataset.yaml erfolgreich erstellt in: {yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bereitet aufgeteilte CT-Daten für YOLOv8-Segmentierungstraining auf.")
    parser.add_argument('--base_path', type=str, default='C:/Users/Thomas/nasal_sinus_segmentation_250907_yolov8-seg', help='Hauptordner des Projekts')
    parser.add_argument('--source_split_dir', type=str, default='data/raw/data_split_final', help='Quellordner mit train/val/test Unterordnern')
    parser.add_argument('--output_dir', type=str, default='data/yolo_dataset', help='Zielordner für das YOLO-Dataset')
    args = parser.parse_args()

    source_path = Path(args.base_path) / args.source_split_dir
    output_path = Path(args.base_path) / args.output_dir

    if output_path.exists():
        print(f"Lösche alten Ordner: {output_path}")
        shutil.rmtree(output_path)
    
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_source = source_path / 'train'
    val_source = source_path / 'val'
    
    train_images_dest = output_path / 'images' / 'train'
    val_images_dest = output_path / 'images' / 'val'
    
    train_labels_dest = output_path / 'labels' / 'train'
    val_labels_dest = output_path / 'labels' / 'val'

    if train_source.exists():
        process_split(train_source, train_images_dest, train_labels_dest)
    if val_source.exists():
        process_split(val_source, val_images_dest, val_labels_dest)
        
    create_dataset_yaml(output_path, train_images_dest, val_images_dest)

    print("\nAlle Daten erfolgreich für YOLOv8 aufbereitet!")