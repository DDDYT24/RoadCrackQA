# Tile results: val

3315 labelled tiles, 1158 crack (34.9%); ambiguous tiles excluded. Thresholds were chosen on val.

| Method | Threshold | Precision | Recall | F1 | PR-AUC | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| resnet18 | 0.6563 | 0.884 | 0.839 | 0.861 | 0.938 | 972 | 128 | 186 | 2029 |
| opencv_baseline | 0.0007935 | 0.746 | 0.856 | 0.797 | 0.815 | 991 | 338 | 167 | 1819 |
| majority_class | 0.5 | 0.000 | 0.000 | 0.000 | - | 0 | 0 | 1158 | 2157 |

Model: `resnet18-43412d20c3be`; commit: `848c4c841ecad0608cb230b32db56c4591977c1b`.
