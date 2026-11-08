# RoadCrackQA

A Python pipeline for road crack images: check a large image set for data problems, then
detect cracks with PyTorch and write organized results.

## Setup

Python 3.11. Install the CPU build of PyTorch first, then the package:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
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
- exact duplicates (SHA256) and near duplicates (dHash, Hamming distance 4 or less),
  marked when a pair spans two splits
- image to mask pairing: missing masks, orphan masks, size mismatches
- masks that are not binary
- image size histogram and crack pixel ratio per split
- source photo ids per split and any id shared between splits

All settings live in `configs/default.yaml`.

## Tests

```bash
ruff check .
pytest -q
```

Tests build synthetic images in a temporary folder, so they need no dataset.

## Limitations

CRACK500 holds RGB phone photos. Survey systems such as laser crack measurement produce 3D
range and intensity images, so this project shows the method, not a model ready for that data.

## License

Code: MIT, see [LICENSE](LICENSE). Datasets keep their own terms, see [DATA.md](DATA.md).
