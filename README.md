# Paranasal Sinus Segmentation: A Comparative Study of 3D U-Net and YOLOv8

This repository contains the source code, results, and trained models for a project focused on the automatic segmentation of paranasal sinuses from CT scans. It provides a comparative analysis of a volumetric **3D Attention U-Net** and a slice-based **2D YOLOv8-Segment** model.

---

## 🚀 Key Results & Conclusion

This study demonstrates that for this specific task, a pragmatic **2D YOLOv8-Segment model significantly outperforms the more complex 3D Attention U-Net** across all evaluated metrics. The final comparison on the validation set is summarized below:

| Metric (Average) | **YOLOv8-Segment (2D)** | **3D Attention U-Net** | Note |
| :--- | :---: | :---: | :--- |
| **Dice Score** | **0.920** | 0.871 | *Higher is better* |
| **IoU (Jaccard)** | **0.852** | 0.774 | *Higher is better* |
| **HD95 (mm)** | **0.990** | 3.049 | *Lower is better* |
| **ASSD (mm)** | **0.214** | 0.532 | *Lower is better* |

The superior performance of YOLOv8, combined with its significantly faster training time and lower computational cost, makes it the recommended architecture for this use case.

---

##  Trained Models

The final trained model weights can be downloaded from the links below.

- **[Download YOLOv8-Segment Model (best.pt)](DEIN_DOWNLOAD_LINK_HIER)**
- **[Download 3D Attention U-Net Model (best_dice_model.pth)](DEIN_ANDERER_LINK_HIER)**

---

## 📂 Repository Structure

The project is organized into two self-contained sub-projects:

- **/3D_Attention_UNet/**: All assets for the 3D U-Net experiment, including source code (`/src`), final evaluation metrics (`/results`), and package dependencies (`requirements.txt`).
- **/YOLOv8_Segment/**: All assets for the YOLOv8 experiment, structured similarly.

---

## 🛠️ How to Use

To run the evaluation or training scripts for either model:

1.  Navigate into the respective project directory (e.g., `cd YOLOv8_Segment`).
2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the desired script from the `/src` folder. For example, to re-run the evaluation for YOLOv8:
    ```bash
    python src/evaluate_yolo.py --model_path "path/to/downloaded/best.pt" --data_path "path/to/your/yolo_data" --raw_dicom_path "path/to/your/dicom_data"
    ```
*(Note: Datasets are not included in this repository due to their size and sensitive nature.)*