# Automated Paranasal Sinus Segmentation: From YOLO to nnU-Net

![Status](https://img.shields.io/badge/Status-SOTA_Achieved-success)
![Accuracy](https://img.shields.io/badge/Dice_Score-94.6%25-brightgreen)
![Framework](https://img.shields.io/badge/Framework-nnU--Net_V2-blue)

**Project Overview:** This repository documents the evolution of an automated segmentation pipeline for paranasal sinuses in CT scans. It compares initial deep learning attempts (YOLO, Custom U-Net) against a final, high-performance implementation using **nnU-Net V2**.

---

## Final Solution: nnU-Net V2

After evaluating multiple architectures, the **nnU-Net framework** (trained on a high-performance **4x NVIDIA A100** cluster for 1000 epochs) was selected as the final solution due to its superior 3D-context understanding and robustness.

### Dataset & Rigorous Split Strategy
The model was developed using a total dataset of **426 high-resolution CT scans**. To prevent data leakage and ensure a fair evaluation, a strictly stratified split was applied:

1.  **Training & Validation (n=307):** Used for model optimization (utilizing 5-fold cross-validation).
2.  **Internal Test Set (n=77):** Held out from training to monitor progress.
3.  **External Test Set (n=42):** A completely isolated "black box" dataset (10%) used **only** for the final metric calculation.

### Key Results
The model achieved state-of-the-art performance, with the external test set scoring slightly higher than the internal set, indicating excellent generalization.

| Dataset | Count | Mean Dice | HD95 (mm) |
| :--- | :--- | :--- | :--- |
| **Internal Test** | 77 | 94.1% | 1.68 |
| **External Test** | 42 | **94.6%** | **1.51** |

#### Visual Evaluation
![Results Training](nnUNet_SOTA_Approach/results/progress_dice.png)


---

## Project Evolution & Comparison

This project started with lightweight 2D models and evolved into a full 3D segmentation pipeline. Here is how the approaches compare:

| Approach | Type | Performance / Verdict | Status |
| :--- | :--- | :--- | :--- |
| **1. YOLOv8** | Object Detection (Bounding Box) | **Detection only.** Good for finding the sinus, but could not outline the precise shape/volume. |  [Archived](Legacy_Attempts/YOLO_Approach/) |
| **2. Custom 2D U-Net** | Segmentation (Slice-by-Slice) | **~80-85% Dice.** Struggled with 3D consistency between slices (z-axis). |  [Archived](Legacy_Attempts/Custom_UNet/) |
| **3. nnU-Net V2** | 3D Segmentation (Volumetric) | **94.6% Dice.** Solved the z-axis consistency and generalisation issues. |  **Current SOTA** |

*(Detailed code for the legacy approaches can still be found in the `Legacy_Attempts` folder for research purposes.)*

---

## Server Workflow (nnU-Net)

The final training was executed on a high-performance cluster:

1.  **Preprocessing:** `nnUNetv2_plan_and_preprocess -d 003`
2.  **Training:** `nnUNetv2_train 003 3d_fullres 0` (1000 Epochs)
3.  **Inference:** Validated on 77 internal and 42 external cases.

For the full evaluation logic, see [nnUNet_SOTA_Approach/evaluation_scripts/](nnUNet_SOTA_Approach/evaluation_scripts/).

