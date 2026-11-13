# V2 tile balance

Tiles of 128 px. Label rule `centerline`: crack: at least 32 pixels of the mask's centre line; no crack: no mask pixel; ambiguous: the rest, and every tile of a patch labelled crack without a mask. Ambiguous tiles are left out of training and metrics.

| Dataset | Split | Images | Sources | Groups | Sources without crack | Tiles | Crack | No crack | Ambiguous |
|---|---|---|---|---|---|---|---|---|---|
| AEL | train | 39 | 20 | 4 | 0 | 548 | 191 | 339 | 18 |
| AEL | val | 3 | 3 | 2 | 0 | 105 | 22 | 77 | 6 |
| AEL | test | 16 | 16 | 2 | 0 | 395 | 99 | 282 | 14 |
| CFD | train | 98 | 98 | 10 | 0 | 588 | 361 | 200 | 27 |
| CFD | val | 20 | 20 | 2 | 0 | 120 | 72 | 41 | 7 |
| CRACK500 | train | 1896 | 250 | 250 | 0 | 19240 | 5900 | 11909 | 1431 |
| CRACK500 | val | 329 | 48 | 48 | 0 | 3375 | 966 | 2157 | 252 |
| GAPS384 | train | 304 | 168 | 57 | 1 | 4968 | 1260 | 3506 | 202 |
| GAPS384 | val | 99 | 54 | 24 | 0 | 1636 | 352 | 1197 | 87 |
| GAPS384 | test | 106 | 64 | 30 | 0 | 1752 | 406 | 1259 | 87 |
| SDNET2018 | train | 14976 | 64 | 16 | 0 | 59904 | 0 | 52708 | 7196 |
| SDNET2018 | val | 4680 | 20 | 5 | 0 | 18720 | 0 | 16956 | 1764 |
| SDNET2018 | test | 4678 | 20 | 5 | 5 | 18712 | 0 | 17240 | 1472 |
| cracktree200 | train | 121 | 121 | 29 | 0 | 2904 | 1301 | 1472 | 131 |
| cracktree200 | val | 40 | 40 | 18 | 0 | 960 | 371 | 546 | 43 |
| cracktree200 | test | 45 | 45 | 3 | 0 | 1080 | 452 | 581 | 47 |

Images skipped (no mask, or outside the label folders): CFD: 37
