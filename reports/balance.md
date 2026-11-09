# Tile class balance

Tiles of 128 px, stride 128. Crack: at least 50 crack pixels; no crack: 0; ambiguous: 1 to 49 (left out of training and metrics).

| Split | Images | Sources | Crack | No crack | Ambiguous | Crack share | Excluded tiles |
|---|---|---|---|---|---|---|---|
| train | 1896 | 250 | 7032 | 11909 | 299 | 0.3713 | 0 |
| val | 329 | 48 | 1158 | 2157 | 60 | 0.3493 | 190 |
| test | 1087 | 142 | 4074 | 6610 | 186 | 0.3813 | 370 |
