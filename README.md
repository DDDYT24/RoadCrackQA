# RoadCrackQA

A Python pipeline for road crack images: check a large image set for data problems, then
detect cracks with PyTorch and write organized results.

## Results

Tile level, CRACK500 test split: 10684 labelled tiles of 128 px, 38.1% crack, ambiguous tiles
excluded. Every threshold was chosen on val, and the test split was run once
([reports/test_results.md](reports/test_results.md)).

| Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| ResNet18, ImageNet weights, fine-tuned | 0.835 | 0.824 | **0.829** | 0.918 |
| OpenCV baseline (black-hat, tuned on val) | 0.761 | 0.788 | 0.774 | 0.777 |
| Majority class (never flags) | 0.000 | 0.000 | 0.000 | - |

- Training on CPU: head on a frozen backbone for 5 epochs, then `layer4` for 6 more, on 8000
  of the 18941 train tiles; 876 s on a 20 thread laptop CPU (torch used 14 threads)
  ([reports/train_report.json](reports/train_report.json)). Val F1 0.861, train F1 0.906.
- Batch scoring of 200 full 2560 x 1440 photos (54752 tiles): 262 s on CPU, 0.76 photos per
  second ([reports/batch_summary.json](reports/batch_summary.json)); in the Docker image, 50
  photos at 0.78 per second ([reports/batch_summary_docker.json](reports/batch_summary_docker.json)).

### Error analysis

From the saved test predictions, no retuning ([reports/error_analysis.json](reports/error_analysis.json)).

![Confusion matrices on the test tiles](reports/figures/confusion_test.png)

- 95% bootstrap CI of test F1, resampling the 142 test source photos 2000 times:
  ResNet18 0.810 to 0.847, OpenCV baseline 0.750 to 0.796.
- Photo level: a crop is flagged when any tile reaches the threshold; 993 of 1087 test crops
  are flagged (recall 0.914). Every CRACK500 crop holds a crack, so a false alarm rate cannot
  be measured at this level.
- The 12 most confident errors of each kind, inspected by eye
  ([reports/error_candidates.csv](reports/error_candidates.csv)). A1 to A12 are false
  positives, B1 to B12 false negatives; the scored tile is boxed in yellow and the labelled
  crack is outlined in green:

![Most confident false positives and false negatives](reports/figures/error_grid.jpg)

| Cause | False positives | False negatives |
|---|---|---|
| Gravel texture, dark voids between stones | 6 | 0 |
| Stain or dark patch | 2 | 0 |
| Shadow | 1 | 0 |
| Joint or groove | 1 | 0 |
| Label: visible crack not in the mask, or mask in doubt | 2 | 1 |
| Labelled crack only grazes the tile border | 0 | 10 |
| Thin, faint crack | 0 | 1 |

  Most missed tiles hold 50 or more crack pixels only because the mask of a crack next door
  reaches over the border; a wider ambiguous band or labels from the crack centre line
  would treat them more fairly.

### Second dataset (CFD), no retraining

The same model and val thresholds on the 118 labelled CrackForest images (683 tiles, 442
crack) ([reports/cross_cfd.json](reports/cross_cfd.json)):

| Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| ResNet18 | 1.000 | 0.534 | 0.696 | 0.972 |
| OpenCV baseline | 0.900 | 0.873 | 0.886 | 0.887 |

ResNet18 F1 drops from 0.829 to 0.696. Its ranking stays strong (PR-AUC 0.972) but the
threshold learnt on CRACK500 is too strict for CFD's thinner cracks, so recall falls. The
baseline's per image percentile rule happens to suit CFD, where every image holds a crack.
Photo level recall on CFD is 0.712.

### Pixel segmentation (SmallUNet)

Pixel level on the 1087 test crops (250.4 M pixels, 5.5% crack), pooled over all pixels;
the pixel threshold was chosen on val and the test split was run once
([reports/seg_test_results.json](reports/seg_test_results.json)).

| Method | IoU | F1 (Dice) | Precision | Recall |
|---|---|---|---|---|
| SmallUNet (1.94 M parameters) | **0.528** | 0.691 | 0.685 | 0.698 |
| OpenCV baseline mask | 0.249 | 0.399 | 0.575 | 0.306 |

- Trained from scratch on CPU with random 256 px crops, BCE plus Dice loss; best val epoch 8
  of 11 (early stopping), val IoU 0.636; 3787 s
  ([reports/seg_train_report.json](reports/seg_train_report.json)).
- `batch --mode seg` on 50 full photos: 0.51 photos per second on CPU, mean predicted crack
  pixel ratio 0.026 ([reports/batch_seg_summary.json](reports/batch_seg_summary.json)).
- Ten test crops picked with the seed, not by hand: predicted crack pixels in red, the
  labelled crack outlined in green, IoU per crop in the corner.

![SmallUNet on ten test crops](reports/figures/seg_examples.jpg)

## Setup

Python 3.11. Install the CPU build of PyTorch first, then the package:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Download the dataset by hand as described in [DATA.md](DATA.md).

## Data QA

```bash
crackqa qa --config configs/default.yaml
```

Writes `reports/qa_report.json` and `reports/qa_report.md` with:

- file inventory by split and extension
- corrupt files (zero bytes, or the image cannot be decoded)
- exact duplicates (SHA256) and near duplicates (dHash, Hamming distance 4 or less, under
  any rotation or flip), marked when a pair spans two splits
- image to mask pairing: missing masks, orphan masks, size mismatches
- masks that are not binary
- image size histogram and crack pixel ratio per split
- source photo ids per split and any id shared between splits
- what is left of the cross split findings after `data.exclude_sources`

On CRACK500 this found 8 photos stored in two official splits, rotated and sometimes
renamed; they are dropped from the later split (details in [DATA.md](DATA.md)).

```bash
crackqa qa-extra
```

Checks the other road datasets listed under `extra_data` (GAPS384, cracktree200, AEL, CFD
and the SDNET2018 pavement patches; see [DATA.md](DATA.md)) and writes
`reports/qa_extra.json` and `reports/qa_extra.md`: the same file, mask and source id checks
per dataset, labels from masks or from folders, source photos with no crack, and duplicates
against the CRACK500 crops and each other. A dHash candidate counts as a duplicate only when
the fine texture of the two aligned thumbnails also correlates, since plain road surfaces
under the same lighting share dHash bits without being the same picture. None of the 23806
candidates is a duplicate.

All settings live in `configs/default.yaml`.

## Tiles, split and baseline

```bash
crackqa tile        # data/processed/tiles.csv: every 128 px tile with its label
crackqa split       # per split manifests, reports/balance.md, reports/test_lock.txt
crackqa baseline    # OpenCV baseline, threshold tuned on val: reports/baseline_val.json
```

- A tile is **crack** with 50 or more crack pixels, **no crack** with none, and
  **ambiguous** in between. Ambiguous tiles are counted but left out of training and metrics.
- The split is the official one, checked by source photo: a test fails the run if any photo
  has tiles in two splits. The SHA256 of the test manifest is committed in
  `reports/test_lock.txt`.
- The classical baseline: CLAHE, blur, black-hat, a per image percentile threshold and small
  component removal, then the crack pixel ratio per tile. All of its settings and the tile
  threshold were chosen on val.

## Train, evaluate, score photos

```bash
crackqa train                       # reports/train_report.json, model in outputs/models/
crackqa eval --split val            # model vs baselines on val: reports/val_results.md
crackqa eval --split test --final   # once: reports/test_results.md, predictions_test.csv
crackqa batch --input PHOTOS --output OUT --include "*.jpg"
```

- `train` picks the epoch and the decision threshold by val F1; the ImageNet weights are
  downloaded by torchvision on first use.
- `eval --split test` only runs with `--final`, and refuses when the results already exist,
  the config differs from the one the model was trained with, the test manifest does not
  match `reports/test_lock.txt`, or the config or lock has uncommitted changes. It records
  the Git commit.
- `batch` writes `results.csv` (file, width, height, n_tiles, max_prob, mean_prob,
  crack_tile_ratio, flagged, status, model_version), one `overlays/*_overlay.jpg` per photo
  with tiles shaded by probability, `summary.json` with counts and runtime, and `batch.log`.
  A photo is flagged when any tile reaches the threshold. Corrupt files get
  `status=corrupt` and never stop the run.

## Segmentation

```bash
crackqa seg-train                       # reports/seg_train_report.json, outputs/models/smallunet.*
crackqa seg-eval --split val            # SmallUNet vs OpenCV baseline mask on val pixels
crackqa seg-eval --split test --final   # once, behind the same lock and Git checks
crackqa batch --mode seg --input PHOTOS --output OUT --include "*.jpg"
```

In `seg` mode, `results.csv` holds file, width, height, crack_pixel_ratio, status and
model_version, with a mask PNG in `masks/` and an overlay in `overlays/` for each photo.

## Analysis and Docker

```bash
crackqa analyze        # bootstrap CI, confusion figure, worst error grid from the saved test run
crackqa cross          # the frozen model on CFD (paths in cross_dataset)
crackqa seg-examples   # reports/figures/seg_examples.jpg, only after the final seg run
```

Causes of the worst errors are filled in by hand in `reports/error_candidates.csv` and kept
when `analyze` runs again.

```bash
docker build -t roadcrackqa .
```

PowerShell:

```powershell
docker run --rm -v "${PWD}\outputs:/app/outputs" -v "${PWD}\data:/app/data" roadcrackqa batch --input data/raw/CRACK500/valdata --output outputs/batch_docker --include "*.jpg"
```

Git Bash on Windows:

```bash
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)/outputs:/app/outputs" -v "$(pwd -W)/data:/app/data" roadcrackqa batch --input data/raw/CRACK500/valdata --output outputs/batch_docker --include "*.jpg"
```

On Linux or macOS, mount `$PWD/outputs` and `$PWD/data` the same way. The trained model is
read from the mounted `outputs/models/`, never baked into the image.

## Reproduce

```bash
crackqa qa && crackqa tile && crackqa split && crackqa baseline
crackqa train && crackqa eval --split val
crackqa seg-train && crackqa seg-eval --split val
```

The test lock and the config must be committed before either `--final` run. After the
final runs, `crackqa analyze` and `crackqa cross` read the saved results.

## Tests

```bash
ruff check .
pytest -q
```

Tests build synthetic images in a temporary folder, so they need no dataset.

## Limitations

CRACK500 holds RGB phone photos. Survey systems such as laser crack measurement produce 3D
range and intensity images, so this project shows the method, not a model ready for that data.

The baseline's percentile threshold assumes each image holds a crack, which is true for the
CRACK500 crops (they were cut around cracks) but not for arbitrary photos.

Both models expect road surface photos. They answer only crack or no crack and cannot say
that an image is not a road: on five ordinary desktop wallpapers (landscapes and abstract
art), `batch` in tile mode flagged four, marking tree lines and object edges as cracks with scores near
1.0. A dark line drawn on a plain texture is also flagged. Plain grey, white and noise images
are not. On real road tiles without a crack, the false alarm rate is 10% (662 of 6610 test
tiles).

## License

Code: MIT, see [LICENSE](LICENSE). Datasets keep their own terms, see [DATA.md](DATA.md).
