# V1 and V2 on the V2 test split

20319 labelled tiles (957 crack) from AEL, GAPS384, SDNET2018, cracktree200; ambiguous tiles left out. Each model at its own val threshold (V1 0.6563, V2 0.5279), the OpenCV baseline at its CRACK500 val threshold. False alarm rate: no-crack tiles flagged / all no-crack tiles. SDNET2018 has no crack tiles, so only its false alarm rate is defined; it holds most of the no-crack tiles.

Acceptance (V2 lower false alarm rate, equal or higher recall, higher F1, pooled): **met** (false alarm rate lower, recall equal or higher, F1 higher).

| Dataset | Tiles | Crack | Method | Precision | Recall | F1 | PR-AUC | False alarm rate | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | 20319 | 957 | v1 | 0.185 | 0.469 | 0.265 | 0.191 | 0.102 | 449 | 1977 | 508 | 17385 |
| pooled | 20319 | 957 | v2 | 0.908 | 0.796 | 0.849 | 0.915 | 0.004 | 762 | 77 | 195 | 19285 |
| pooled | 20319 | 957 | opencv_baseline | 0.155 | 0.655 | 0.251 | 0.137 | 0.177 | 627 | 3421 | 330 | 15941 |
| AEL | 381 | 99 | v1 | 0.891 | 0.414 | 0.566 | 0.802 | 0.018 | 41 | 5 | 58 | 277 |
| AEL | 381 | 99 | v2 | 0.928 | 0.778 | 0.846 | 0.931 | 0.021 | 77 | 6 | 22 | 276 |
| AEL | 381 | 99 | opencv_baseline | 0.559 | 0.576 | 0.567 | 0.464 | 0.160 | 57 | 45 | 42 | 237 |
| GAPS384 | 1665 | 406 | v1 | 0.703 | 0.350 | 0.467 | 0.569 | 0.048 | 142 | 60 | 264 | 1199 |
| GAPS384 | 1665 | 406 | v2 | 0.828 | 0.675 | 0.744 | 0.820 | 0.045 | 274 | 57 | 132 | 1202 |
| GAPS384 | 1665 | 406 | opencv_baseline | 0.357 | 0.658 | 0.463 | 0.337 | 0.382 | 267 | 481 | 139 | 778 |
| SDNET2018 | 17240 | 0 | v1 | - | - | - | - | 0.111 | 0 | 1908 | 0 | 15332 |
| SDNET2018 | 17240 | 0 | v2 | - | - | - | - | 0.001 | 0 | 8 | 0 | 17232 |
| SDNET2018 | 17240 | 0 | opencv_baseline | - | - | - | - | 0.162 | 0 | 2791 | 0 | 14449 |
| cracktree200 | 1033 | 452 | v1 | 0.985 | 0.588 | 0.737 | 0.932 | 0.007 | 266 | 4 | 186 | 577 |
| cracktree200 | 1033 | 452 | v2 | 0.986 | 0.909 | 0.946 | 0.973 | 0.010 | 411 | 6 | 41 | 575 |
| cracktree200 | 1033 | 452 | opencv_baseline | 0.744 | 0.670 | 0.705 | 0.697 | 0.179 | 303 | 104 | 149 | 477 |

## V2 at V1's recall

V2 at threshold 0.994, where its pooled recall reaches V1's. This threshold is read off the test tiles, so it is a comparison only, not an operating point.

| Dataset | Recall | False alarm rate |
|---|---|---|
| pooled | 0.470 | 0.000 |
| AEL | 0.485 | 0.000 |
| GAPS384 | 0.195 | 0.002 |
| SDNET2018 | - | 0.000 |
| cracktree200 | 0.715 | 0.009 |

## 95% bootstrap intervals, pooled

2000 resamples of the 145 source photos, within each dataset, the same draw for every method.

| Method | F1 | Recall | False alarm rate |
|---|---|---|---|
| v1 | 0.265 (0.199 to 0.363) | 0.469 (0.419 to 0.516) | 0.102 (0.058 to 0.153) |
| v2 | 0.849 (0.823 to 0.871) | 0.796 (0.762 to 0.830) | 0.004 (0.003 to 0.005) |
| opencv_baseline | 0.251 (0.200 to 0.319) | 0.655 (0.611 to 0.699) | 0.177 (0.126 to 0.235) |
| v2_minus_v1 | 0.583 (0.484 to 0.648) | 0.327 (0.285 to 0.369) | -0.098 (-0.150 to -0.054) |

## Source photos

A source photo is flagged when any of its labelled tiles reaches the threshold. Crack-free photos: 5; photos with a crack tile: 125.

| Method | Crack-free photos flagged | Photos with crack flagged |
|---|---|---|
| v1 | 5 (1.000) | 100 (0.800) |
| v2 | 2 (0.400) | 125 (1.000) |
| opencv_baseline | 5 (1.000) | 125 (1.000) |

Figures: `reports/figures/confusion_test2.png`, `reports/figures/error_grid_test2.jpg` (V2's most confident errors; no GAPS384 tiles shown).

V1 `resnet18-43412d20c3be`, V2 `resnet18-2bb90445b55f` (combined); config `5d8fdbf387bc`; test lock `33a45ed47a12`; commit `504d7016e4f696c9e60d6e6a27c2e6cbc7266ca1`.
