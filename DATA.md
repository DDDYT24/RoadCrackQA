# Data

The dataset itself is not stored in this repository. Download it by hand and unpack it under
`data/raw/`, which Git ignores. Tests and CI use synthetic images generated at test time.
The only dataset pixels in the repository are two reduced figures used to illustrate results
in the README: `reports/figures/error_grid.jpg` (24 tiles) and
`reports/figures/seg_examples.jpg` (10 crops), both from CRACK500, cited below.

## CRACK500 (primary)

- Source: https://github.com/fyangneil/pavement-crack-detection
- Download: the repository README links one archive, `pavement crack datasets-20210103T153625Z-001.zip`,
  on [Google Drive](https://drive.google.com/file/d/13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD/view)
  and [Baidu Yunpan](https://pan.baidu.com/s/1JwJO96BOtJ50MykBcYKknQ) (extract code `jviq`).
  The archive holds several datasets; only CRACK500 is used here.
- Content: 500 pavement photos (about 2000 x 1500) with pixel binary crack masks, split by
  source photo into 250 train, 50 val and 200 test.
- Terms: no licence is stated. The authors ask users to cite the papers below. This project
  uses the data for study, credits the authors, and shows only the two reduced figures above.

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
- Content: 118 road images, 480 x 320, with pixel ground truth. The unpacked `cfd_image`
  folder holds 155 images (two of them 320 x 480) and an `image.rar`; only the 118 with a
  mask in `seg_gt` are used.
- Terms: non-commercial research use only.
- Citations: see the dataset repository.

## Extra datasets

`crackqa qa-extra` checks every dataset under `extra_data` in `configs/default.yaml`, CFD
included: inventory, corrupt files, image to mask pairing, mask values, labels per image and
per source photo, and duplicates against the CRACK500 crops and each other
(`reports/qa_extra.md`). Duplicate candidates come from the same rotation-aware dHash as
`crackqa qa`. A candidate counts only when the fine texture of the two aligned thumbnails
also correlates (`qa.confirm_min_ncc`): greyscale road photos under the same lighting share
dHash bits without being the same picture. The 56 known CRACK500 leak pairs score 0.999 or
more on this check.

Result: of 23806 candidates, none is confirmed, so no picture is shared between CRACK500,
CFD, GAPS384, cracktree200, AEL and SDNET2018 at any rotation or flip. The check finds the
same picture at another size, not a crop of a larger one.

The masks differ in how wide they draw a crack. One 3 x 3 erosion keeps 82% of the crack
pixels in CRACK500 train masks, 57% in GAPS384, 30% in CFD, 12% in AEL and none in
cracktree200, whose cracks are lines one pixel wide.

GAPS384, cracktree200 and AEL come in the same archive as CRACK500 (see above):

```bash
cd data/raw
unzip "pavement crack datasets/GAPS384/croppedimg.zip" -d GAPS384
unzip "pavement crack datasets/GAPS384/croppedgt.zip" -d GAPS384
unzip "pavement crack datasets/cracktree200/cracktree200rgb.zip" -d cracktree200
unzip "pavement crack datasets/cracktree200/cracktree200_gt.zip" -d cracktree200
unzip "pavement crack datasets/AEL/img.zip" -d AEL
unzip "pavement crack datasets/AEL/gt.zip" -d AEL
```

This gives `data/raw/GAPS384/{croppedimg,croppedgt}/`,
`data/raw/cracktree200/{cracktree200rgb,cracktree200_gt}/` and `data/raw/AEL/{img,gt}/`.

### GAPS384

- Source: 384 photos of the German Asphalt Pavement Distress dataset (GAPs, TU Ilmenau,
  https://www.tu-ilmenau.de/neurob/data-sets-code/gaps) with pixel masks, prepared by the
  CRACK500 authors. The cropped version is used: 509 greyscale crops, 540 x 640 or
  540 x 440, from 286 of those photos. Every crop holds a crack.
- Source photo id: `train_0554` in `train_0554_541_1.jpg` (GAPs photo, then the crop's y and
  x). Consecutive ids show neighbouring stretches of one road, with the same lane marking:
  not duplicates, but not independent either.
- Terms: TU Ilmenau gives GAPs logins for academic use only, and the archive states no
  licence. This project uses it for study and shows none of its pixels.

### cracktree200

- Source: CrackTree, in the same archive.
- Content: 206 pavement photos, 800 x 600, with pixel masks; every photo holds a crack.
  The photos are stored with three equal channels (greyscale), although the archive is named
  `cracktree200rgb`.
- Source photo id: the file number; one photo per file.
- Terms: no licence is stated; cite the paper below.

### SDNET2018 (pavement subset)

- Source: Utah State University Digital Commons,
  https://digitalcommons.usu.edu/all_datasets/48 (DOI 10.15142/T3TD19). The page's download
  button gives `DATA_Maguire_20180517_ALL.zip` (528286896 bytes), which holds a ReadMe and
  `SDNET2018.zip`. The site refuses scripted downloads; download it in a browser into
  `data/raw/SDNET2018/download/`.
- Content: 256 x 256 colour patches cut from photos of concrete roads and sidewalks on the
  USU campus, labelled per patch by folder: `P/CP` 2608 with a crack, `P/UP` 21726 without.
  There are no masks. The bridge deck (`D`) and wall (`W`) subsets are not used.
- Source photo id: `001` in `001-100.jpg` (photo, then patch). 104 photos, 232 to 234
  patches each; 5 photos (015, 016, 024, 025, 097) have no crack patch at all, the only
  crack-free road photos among these datasets. 13 patches of photo 027 carry a `_2` suffix;
  none duplicates another patch.
- Surface: concrete, while CRACK500 shows asphalt.
- Terms: Creative Commons Attribution 4.0; cite the dataset below.

Unpack only the pavement subset. Python is used for the inner zip because
`unzip SDNET2018.zip "P/*"` extracts only the empty `P` folder from it:

```bash
cd data/raw/SDNET2018/download
unzip DATA_Maguire_20180517_ALL.zip
cd ..
python -c "import zipfile; z = zipfile.ZipFile('download/DATA_Maguire_20180517_ALL/SDNET2018.zip'); z.extractall('.', [n for n in z.namelist() if n.startswith('P/')])"
```

This gives `data/raw/SDNET2018/P/{CP,UP}/`.

### AEL

- Source: images from three road survey systems (Aigle-RN, ESAR and LCMS) with crack ground
  truth, in the same archive.
- Content: 58 greyscale images. Aigle-RN: 19 images, each stored in two parts, `a`
  (991 x 462) and `b` (311 x 462); ESAR: 15 (768 x 512); LCMS: 5 (700 x 1000). One part,
  `AIGLE_RN_F03b`, holds no crack.
- Masks: named differently from their images (`Im_GT_AIGLE_RN_C18aor.jpg` has
  `GT_AIGLE_RN_C18a.png`, see `mask_name`) and inverted, cracks black on white
  (`invert_mask: true`).
- Source photo id: `AIGLE_RN_C18`, shared by both parts.
- Terms: no licence is stated; cite the paper below.

Citations:

```bibtex
@inproceedings{eisenbach2017how,
  title     = {How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach},
  author    = {Eisenbach, Markus and Stricker, Ronny and Seichter, Daniel and Amende, Karl and
               Debes, Klaus and Sesselmann, Maximilian and Ebersbach, Dirk and Stoeckert, Ulrike
               and Gross, Horst-Michael},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  pages     = {2039--2047},
  year      = {2017}
}

@article{zou2012cracktree,
  title   = {CrackTree: Automatic crack detection from pavement images},
  author  = {Zou, Qin and Cao, Yu and Li, Qingquan and Mao, Qingzhou and Wang, Song},
  journal = {Pattern Recognition Letters},
  volume  = {33},
  number  = {3},
  pages   = {227--238},
  year    = {2012}
}

@article{amhaz2016automatic,
  title   = {Automatic Crack Detection on Two-Dimensional Pavement Images: An Algorithm Based on
             Minimal Path Selection},
  author  = {Amhaz, Rabih and Chambon, Sylvie and Idier, J{\'e}r{\^o}me and Baltazart, Vincent},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume  = {17},
  number  = {10},
  pages   = {2718--2729},
  year    = {2016}
}

@misc{maguire2018sdnet,
  title     = {SDNET2018: A concrete crack image dataset for machine learning applications},
  author    = {Maguire, Marc and Dorafshan, Sattar and Thomas, Robert J.},
  publisher = {Utah State University},
  year      = {2018},
  doi       = {10.15142/T3TD19}
}
```
