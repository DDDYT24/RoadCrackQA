# RoadCrackQA

A Python pipeline for road crack images: check a large image set for data problems, then
detect cracks with PyTorch and write organized results.

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
