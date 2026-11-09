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
  second ([reports/batch_summary.json](reports/batch_summary.json)).

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

## Reproduce

```bash
crackqa qa && crackqa tile && crackqa split && crackqa baseline
crackqa train && crackqa eval --split val
```

The test lock and the config must be committed before `crackqa eval --split test --final`.

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

## License

Code: MIT, see [LICENSE](LICENSE). Datasets keep their own terms, see [DATA.md](DATA.md).
