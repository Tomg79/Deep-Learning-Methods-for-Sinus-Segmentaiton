# Server Commands Reference

## 1. Preprocessing
```bash
nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity
```

## 2. Training
```bash
nnUNetv2_train 003 3d_fullres 0 --npz
```
## 3. Prediction
```bash
nnUNetv2_predict -i /data/imagesTs -o /data/predictions -d 003 -c 3d_fullres -f 0
nnUNetv2_predict -i /data/external_test -o /data/ext_predictions -d 003 -c 3d_fullres -f 0
```