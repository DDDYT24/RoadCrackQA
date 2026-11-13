# Ablation on V2 val

Every row is scored on the V2 val tiles (ambiguous tiles left out), each model at its own threshold and the OpenCV baseline at its CRACK500 val threshold. False alarm rate: no-crack tiles flagged / all no-crack tiles. SDNET2018 val holds no crack tiles, so only its false alarm rate is defined. Val also chooses each model's epoch and threshold (V1: on CRACK500 val), so these numbers are optimistic; the V2 test split reports once.

## V1: CRACK500 only, 8000 train tiles

Model `resnet18-43412d20c3be`, threshold 0.6563; label rule `centerline` (32 px); val manifest `2b9c27c116ef`.

| Dataset | Tiles | Crack | Precision | Recall | F1 | PR-AUC | False alarm rate | Baseline F1 | Baseline false alarm rate |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 22757 | 1783 | 0.421 | 0.679 | 0.520 | 0.500 | 0.079 | 0.428 | 0.166 |
| AEL | 99 | 22 | 0.882 | 0.682 | 0.769 | 0.852 | 0.026 | 0.597 | 0.325 |
| CFD | 113 | 72 | 1.000 | 0.347 | 0.515 | 0.962 | 0.000 | 0.813 | 0.098 |
| CRACK500 | 3123 | 966 | 0.871 | 0.892 | 0.881 | 0.954 | 0.059 | 0.784 | 0.157 |
| GAPS384 | 1549 | 352 | 0.633 | 0.372 | 0.469 | 0.573 | 0.064 | 0.536 | 0.341 |
| SDNET2018 | 16956 | 0 | - | - | - | - | 0.086 | - | 0.154 |
| cracktree200 | 917 | 371 | 0.989 | 0.480 | 0.646 | 0.885 | 0.004 | 0.678 | 0.185 |
