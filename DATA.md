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

### Layout expected by the config

`configs/default.yaml` reads `data/raw/CRACK500/` with one folder per split (`data.splits`).
An image `NAME.jpg` is paired with the mask `NAME_mask.png` in the same folder. The source
photo id is the leading `YYYYMMDD_HHMMSS` part of the file name (`data.source_id_regex`), so
all crops of one photo share an id. If the unpacked folder names differ, edit `data.splits`;
`crackqa qa` reports any file the pattern does not match.

## CrackForest (CFD, cross dataset test only)

- Source: https://github.com/cuilimeng/CrackForest-dataset
- Content: 118 road images, 480 x 320, with pixel ground truth.
- Terms: non-commercial research use only.
- Citations: see the dataset repository.
