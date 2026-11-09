# Tile results: test

10684 labelled tiles, 4074 crack (38.1%); ambiguous tiles excluded. Thresholds were chosen on val.

| Method | Threshold | Precision | Recall | F1 | PR-AUC | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| resnet18 | 0.6563 | 0.835 | 0.824 | 0.829 | 0.918 | 3356 | 662 | 718 | 5948 |
| opencv_baseline | 0.0007935 | 0.761 | 0.788 | 0.774 | 0.777 | 3210 | 1010 | 864 | 5600 |
| majority_class | 0.5 | 0.000 | 0.000 | 0.000 | - | 0 | 0 | 4074 | 6610 |

Model: `resnet18-43412d20c3be`; commit: `9d273ef518bd498795350aa1335d72a2ce3accc1`.
