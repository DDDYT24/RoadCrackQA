# Ablation on V2 val

Every row is scored on the V2 val tiles (ambiguous tiles left out), each model at its own threshold and the OpenCV baseline at its CRACK500 val threshold. False alarm rate: no-crack tiles flagged / all no-crack tiles. SDNET2018 val holds no crack tiles, so only its false alarm rate is defined. Val also chooses each model's epoch and threshold (V1: on CRACK500 val), so these numbers are optimistic; the V2 test split reports once.

## Summary

Columns after PR-AUC: recall / false alarm rate at the threshold where each model flags 67.9% of all crack tiles (V1's pooled recall), so false alarms are compared at equal recall.

| Model | Pooled F1 | Pooled PR-AUC | Pooled | AEL | CFD | CRACK500 | GAPS384 | SDNET2018 | cracktree200 |
|---|---|---|---|---|---|---|---|---|---|
| V1: CRACK500 only, 8000 train tiles | 0.520 | 0.500 | 0.679 / 0.079 | 0.682 / 0.026 | 0.347 / 0.000 | 0.892 / 0.059 | 0.372 / 0.063 | - / 0.086 | 0.480 / 0.004 |
| v1_recipe | 0.539 | 0.456 | 0.679 / 0.067 | 0.773 / 0.013 | 0.458 / 0.000 | 0.830 / 0.022 | 0.398 / 0.076 | - / 0.074 | 0.590 / 0.026 |
| all_tiles | 0.566 | 0.502 | 0.679 / 0.062 | 0.682 / 0.013 | 0.375 / 0.000 | 0.856 / 0.013 | 0.406 / 0.069 | - / 0.070 | 0.536 / 0.011 |
| sdnet_negatives | 0.813 | 0.882 | 0.679 / 0.005 | 0.591 / 0.013 | 0.375 / 0.000 | 0.831 / 0.007 | 0.432 / 0.053 | - / 0.001 | 0.582 / 0.028 |
| mixed_sources | 0.674 | 0.683 | 0.679 / 0.031 | 0.591 / 0.013 | 0.639 / 0.000 | 0.784 / 0.006 | 0.366 / 0.009 | - / 0.037 | 0.717 / 0.000 |
| combined | 0.872 | 0.941 | 0.679 / 0.001 | 0.727 / 0.013 | 0.681 / 0.000 | 0.755 / 0.004 | 0.358 / 0.005 | - / 0.000 | 0.784 / 0.002 |

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

## v1_recipe

Model `resnet18-1effda9f6862`, threshold 0.4274; label rule `centerline` (32 px); val manifest `2b9c27c116ef`.

| Dataset | Tiles | Crack | Precision | Recall | F1 | PR-AUC | False alarm rate | Baseline F1 | Baseline false alarm rate |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 22757 | 1783 | 0.421 | 0.750 | 0.539 | 0.456 | 0.088 | 0.428 | 0.166 |
| AEL | 99 | 22 | 0.900 | 0.818 | 0.857 | 0.861 | 0.026 | 0.597 | 0.325 |
| CFD | 113 | 72 | 1.000 | 0.569 | 0.726 | 0.951 | 0.000 | 0.813 | 0.098 |
| CRACK500 | 3123 | 966 | 0.870 | 0.880 | 0.875 | 0.950 | 0.059 | 0.784 | 0.157 |
| GAPS384 | 1549 | 352 | 0.598 | 0.460 | 0.520 | 0.534 | 0.091 | 0.536 | 0.341 |
| SDNET2018 | 16956 | 0 | - | - | - | - | 0.092 | - | 0.154 |
| cracktree200 | 917 | 371 | 0.869 | 0.717 | 0.786 | 0.875 | 0.073 | 0.678 | 0.185 |

## all_tiles

Model `resnet18-874394ade59a`, threshold 0.6684; label rule `centerline` (32 px); val manifest `2b9c27c116ef`.

| Dataset | Tiles | Crack | Precision | Recall | F1 | PR-AUC | False alarm rate | Baseline F1 | Baseline false alarm rate |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 22757 | 1783 | 0.473 | 0.706 | 0.566 | 0.502 | 0.067 | 0.428 | 0.166 |
| AEL | 99 | 22 | 0.895 | 0.773 | 0.829 | 0.860 | 0.026 | 0.597 | 0.325 |
| CFD | 113 | 72 | 1.000 | 0.417 | 0.588 | 0.894 | 0.000 | 0.813 | 0.098 |
| CRACK500 | 3123 | 966 | 0.958 | 0.872 | 0.913 | 0.964 | 0.017 | 0.784 | 0.157 |
| GAPS384 | 1549 | 352 | 0.639 | 0.443 | 0.523 | 0.580 | 0.073 | 0.536 | 0.341 |
| SDNET2018 | 16956 | 0 | - | - | - | - | 0.075 | - | 0.154 |
| cracktree200 | 917 | 371 | 0.955 | 0.574 | 0.717 | 0.881 | 0.018 | 0.678 | 0.185 |

## sdnet_negatives

Model `resnet18-f6eda7568aac`, threshold 0.7425; label rule `centerline` (32 px); val manifest `2b9c27c116ef`.

| Dataset | Tiles | Crack | Precision | Recall | F1 | PR-AUC | False alarm rate | Baseline F1 | Baseline false alarm rate |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 22757 | 1783 | 0.848 | 0.781 | 0.813 | 0.882 | 0.012 | 0.428 | 0.166 |
| AEL | 99 | 22 | 0.824 | 0.636 | 0.718 | 0.864 | 0.039 | 0.597 | 0.325 |
| CFD | 113 | 72 | 1.000 | 0.431 | 0.602 | 0.909 | 0.000 | 0.813 | 0.098 |
| CRACK500 | 3123 | 966 | 0.941 | 0.903 | 0.921 | 0.969 | 0.025 | 0.784 | 0.157 |
| GAPS384 | 1549 | 352 | 0.653 | 0.551 | 0.598 | 0.654 | 0.086 | 0.536 | 0.341 |
| SDNET2018 | 16956 | 0 | - | - | - | - | 0.003 | - | 0.154 |
| cracktree200 | 917 | 371 | 0.857 | 0.757 | 0.804 | 0.891 | 0.086 | 0.678 | 0.185 |

## mixed_sources

Model `resnet18-85624acf6a9b`, threshold 0.7304; label rule `centerline` (32 px); val manifest `2b9c27c116ef`.

| Dataset | Tiles | Crack | Precision | Recall | F1 | PR-AUC | False alarm rate | Baseline F1 | Baseline false alarm rate |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 22757 | 1783 | 0.581 | 0.802 | 0.674 | 0.683 | 0.049 | 0.428 | 0.166 |
| AEL | 99 | 22 | 0.944 | 0.773 | 0.850 | 0.904 | 0.013 | 0.597 | 0.325 |
| CFD | 113 | 72 | 1.000 | 0.778 | 0.875 | 0.988 | 0.000 | 0.813 | 0.098 |
| CRACK500 | 3123 | 966 | 0.953 | 0.883 | 0.917 | 0.966 | 0.019 | 0.784 | 0.157 |
| GAPS384 | 1549 | 352 | 0.876 | 0.560 | 0.683 | 0.813 | 0.023 | 0.536 | 0.341 |
| SDNET2018 | 16956 | 0 | - | - | - | - | 0.057 | - | 0.154 |
| cracktree200 | 917 | 371 | 0.997 | 0.827 | 0.904 | 0.965 | 0.002 | 0.678 | 0.185 |

## combined

Model `resnet18-2bb90445b55f`, threshold 0.5279; label rule `centerline` (32 px); val manifest `2b9c27c116ef`.

| Dataset | Tiles | Crack | Precision | Recall | F1 | PR-AUC | False alarm rate | Baseline F1 | Baseline false alarm rate |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 22757 | 1783 | 0.883 | 0.861 | 0.872 | 0.941 | 0.010 | 0.428 | 0.166 |
| AEL | 99 | 22 | 0.800 | 0.909 | 0.851 | 0.914 | 0.065 | 0.597 | 0.325 |
| CFD | 113 | 72 | 0.982 | 0.778 | 0.868 | 0.970 | 0.024 | 0.813 | 0.098 |
| CRACK500 | 3123 | 966 | 0.922 | 0.913 | 0.917 | 0.968 | 0.035 | 0.784 | 0.157 |
| GAPS384 | 1549 | 352 | 0.782 | 0.705 | 0.741 | 0.817 | 0.058 | 0.536 | 0.341 |
| SDNET2018 | 16956 | 0 | - | - | - | - | 0.002 | - | 0.154 |
| cracktree200 | 917 | 371 | 0.930 | 0.889 | 0.909 | 0.967 | 0.046 | 0.678 | 0.185 |
