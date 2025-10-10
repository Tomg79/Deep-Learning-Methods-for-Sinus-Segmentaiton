# train_yolo.py
import argparse
from ultralytics import YOLO

def train(args):
    
    model = YOLO(args.model_name)


    print("Starte YOLOv8-Training...")
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=args.project,
        name=args.run_name,
        device=0  
    )
    print("Training abgeschlossen.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainiert ein YOLOv8-Segmentierungsmodell.")
    parser.add_argument('--data_yaml', type=str, required=True, help='Pfad zur dataset.yaml Datei.')
    parser.add_argument('--model_name', type=str, default='yolov8n-seg.pt', help='Name des Basis-Modells (z.B. yolov8n-seg.pt, yolov8s-seg.pt).')
    parser.add_argument('--epochs', type=int, default=100, help='Anzahl der Trainingsepochen.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch-Größe. Anpassen je nach VRAM.')
    parser.add_argument('--imgsz', type=int, default=512, help='Bildgröße für das Training.')
    parser.add_argument('--project', type=str, default='runs/train_yolo', help='Ordner, in dem die Trainingsläufe gespeichert werden.')
    parser.add_argument('--run_name', type=str, default='exp', help='Name des spezifischen Trainingslaufs.')
    
    args = parser.parse_args()
    train(args)