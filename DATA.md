# Data

No dataset image is stored in this repository. Download the data by hand and unpack it under
`data/raw/`, which Git ignores. Tests and CI use synthetic images generated at test time.

## CRACK500 (primary)

- Source: https://github.com/fyangneil/pavement-crack-detection
- Download: the repository README links one archive, `pavement crack datasets-20210103T153625Z-001.zip`,
  on [Google Drive](https://drive.google.com/file/d/13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD/view)
  and [Baidu Yunpan](https://pan.baidu.com/s/1JwJO96BOtJ50MykBcYKknQ) (extract code `jviq`).
  The archive holds several datasets; only CRACK500 is used here.
- Content: 500 pavement photos (about 2000 x 1500) with pixel binary crack masks, split by
  source photo into 250 train, 50 val and 200 test.
- Terms: no licence is stated. The authors ask users to cite the papers below. This project
  treats the data as all rights reserved and uses it for personal study only.

Citations:

```bibtex
@inproceedings{zhang2016road,
  title     = {Road crack detection using deep convolutional neural network},
  author    = {Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  pages     = {3708--3712},
  year      = {2016}
}

@article{yang2019feature,
  title   = {Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
  author  = {Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2019}
}
```

### What the archive holds

Inside the archive, `CRACK500/` contains one zip per folder:

| Zip | Content | Used |
|---|---|---|
| `traincrop.zip` | 1896 train crops | yes |
| `valcrop.zip` | 348 val crops | yes |
| `testcrop.zip` | 1124 test crops | yes |
| `valdata.zip` | 50 full val photos with `_mask.png` | batch demo |
| `testdata.zip` | 200 full test photos with `_mask.png` (52 use `.JPG`) | batch demo |

There is no `traindata.zip`: the 250 train photos are only shipped as crops, so every split
uses the crops. Crops are 640 x 360 (also 360 x 640 and 648 x 484), each paired with a
binary mask of the same name: `NAME.jpg` + `NAME.png`.

### Unpack

```bash
cd data/raw
unzip "pavement crack datasets/CRACK500/traincrop.zip" -d CRACK500
unzip "pavement crack datasets/CRACK500/valcrop.zip" -d CRACK500
unzip "pavement crack datasets/CRACK500/testcrop.zip" -d CRACK500
unzip "pavement crack datasets/CRACK500/valdata.zip" -d CRACK500
unzip "pavement crack datasets/CRACK500/testdata.zip" "testdata/*.jpg" "testdata/*.JPG" "testdata/*.png" -d CRACK500
unzip "pavement crack datasets/CFD/cfd_image.zip" -d CFD
unzip "pavement crack datasets/CFD/cfd_gt.zip" -d CFD
```

This gives `data/raw/CRACK500/{traincrop,valcrop,testcrop,valdata,testdata}/` and
`data/raw/CFD/{cfd_image,seg_gt}/`. The `_mask.txt` files in `testdata.zip` are skipped.

### Source photo id

A crop is named `SOURCE_<a>_<b>.jpg`, where `SOURCE` is `YYYYMMDD_HHMMSS` or `IMG_NNNN`
(`data.source_id_regex`). All crops of one photo share the id, and the split is checked by
this id, never by crop.

### Known leakage between the official splits

`crackqa qa` (rotation-aware dHash plus source ids) finds 8 photos that appear in two
splits, rotated by 90 or 180 degrees, sometimes saved under the next second's timestamp:

- train and val: `20160328_151056` / `20160328_151057`, `20160328_151243` / `20160328_151244`
- train and test: `20160328_154452`, `20160328_154454`, `20160329_103609`,
  `20160329_103800`, `20160329_111111`
- val and test: `20160329_103807`

Each stays in the earlier split and is dropped from the later one (`data.exclude_sources`):
56 crops in all. After that, no exact duplicate, near duplicate or source id crosses a split
(see `reports/qa_report.md`).

## CrackForest (CFD, cross dataset test only)

- Source: https://github.com/cuilimeng/CrackForest-dataset
- Content: 118 road images, 480 x 320, with pixel ground truth.
- Terms: non-commercial research use only.
- Citations: see the dataset repository.
