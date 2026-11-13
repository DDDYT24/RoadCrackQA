# Extra data QA report

Reference: CRACK500 crops of every split (data.splits under paths.raw), 3368 images.
Pairs inside CRACK500 are in `qa_report.md`.

## Summary

| Check | GAPS384 | cracktree200 | AEL | CFD | SDNET2018 |
|---|---|---|---|---|---|
| images | 509 | 206 | 58 | 155 | 24334 |
| corrupt | 0 | 0 | 0 | 0 | 0 |
| greyscale_images | 509 | 206 | 58 | 0 | 0 |
| crack_images | 509 | 206 | 57 | 118 | 2608 |
| no_crack_images | 0 | 0 | 1 | 0 | 21726 |
| unlabelled | 0 | 0 | 0 | 0 | 0 |
| missing_mask | 0 | 0 | 0 | 37 | 0 |
| unpaired_mask_files | 0 | 0 | 0 | 0 | 0 |
| size_mismatch | 0 | 0 | 0 | 0 | 0 |
| non_binary_masks | 0 | 0 | 0 | 0 | 0 |
| sources | 286 | 206 | 39 | 155 | 104 |
| sources_without_crack | 0 | 0 | 0 | 0 | 5 |
| source_id_unmatched | 0 | 0 | 0 | 0 | 0 |
| duplicate_pairs_cross_dataset | 0 | 0 | 0 | 0 | 0 |
| duplicate_pairs_other_source | 0 | 0 | 0 | 0 | 0 |

## GAPS384

- Images: `data/raw/GAPS384/croppedimg`; labels from mask (`data/raw/GAPS384/croppedgt`)
- Files by extension: .jpg: 509
- Sizes (width x height): 540x640: 281, 540x440: 228
- Labels: {'crack': 509}; crack pixel ratio: 0.011913
- Source photos: 286 (1 to 4 images each); 286 with a crack, 0 without

## cracktree200

- Images: `data/raw/cracktree200/cracktree200rgb`; labels from mask (`data/raw/cracktree200/cracktree200_gt`)
- Files by extension: .jpg: 206
- Sizes (width x height): 800x600: 206
- Labels: {'crack': 206}; crack pixel ratio: 0.003196
- Source photos: 206 (1 to 1 images each); 206 with a crack, 0 without

## AEL

- Images: `data/raw/AEL/img`; labels from mask (`data/raw/AEL/gt`)
- Files by extension: .jpg: 58
- Sizes (width x height): 991x462: 19, 311x462: 19, 768x512: 15, 700x1000: 5
- Labels: {'crack': 57, 'no_crack': 1}; crack pixel ratio: 0.00674
- Source photos: 39 (1 to 2 images each); 39 with a crack, 0 without

## CFD

- Images: `data/raw/CFD/cfd_image`; labels from mask (`data/raw/CFD/seg_gt`)
- Files by extension: .jpg: 155, .rar: 1
- Sizes (width x height): 480x320: 153, 320x480: 2
- Labels: {'crack': 118}; crack pixel ratio: 0.016159
- Source photos: 155 (1 to 1 images each); 118 with a crack, 0 without

Images without a mask:

- `CFD/201.jpg`
- `CFD/202.jpg`
- `CFD/203.jpg`
- `CFD/204.jpg`
- `CFD/205.jpg`
- `CFD/206.jpg`
- `CFD/207.jpg`
- `CFD/208.jpg`
- `CFD/301.jpg`
- `CFD/302.jpg`
- `CFD/303.jpg`
- `CFD/304.jpg`
- `CFD/305.jpg`
- `CFD/306.jpg`
- `CFD/307.jpg`
- `CFD/308.jpg`
- `CFD/309.jpg`
- `CFD/310.jpg`
- `CFD/311.jpg`
- `CFD/312.jpg`
- `CFD/313.jpg`
- `CFD/314.jpg`
- `CFD/315.jpg`
- `CFD/316.jpg`
- `CFD/317.jpg`
- `CFD/318.jpg`
- `CFD/319.jpg`
- `CFD/320.jpg`
- `CFD/321.jpg`
- `CFD/322.jpg`
- `CFD/323.jpg`
- `CFD/324.jpg`
- `CFD/325.jpg`
- `CFD/326.jpg`
- `CFD/327.jpg`
- `CFD/328.jpg`
- `CFD/329.jpg`

Other files:

- `image.rar`

## SDNET2018

- Images: `data/raw/SDNET2018/P`; labels from folder
- Files by extension: .jpg: 24334
- Sizes (width x height): 256x256: 24334
- Labels: {'crack': 2608, 'no_crack': 21726}; crack pixel ratio: None
- Source photos: 104 (232 to 234 images each); 99 with a crack, 5 without

## Duplicates

Candidates: same SHA256, or dHash distance <= 4 under any rotation or flip. A candidate is a duplicate when the fine texture of the two aligned 128 px thumbnails correlates at least 0.9. Candidates checked: 23806.

Across datasets, CRACK500 included:

None.

Inside one dataset, between different source photos:

None.

Inside one source photo (overlapping crops are expected): 0.

Closest candidates that are not duplicates (up to 20, for review by eye):

- `GAPS384/train_0697_1_1.jpg` ~ `GAPS384/train_0698_1_1.jpg` (distance 0, rot0, texture correlation 0.5446)
- `GAPS384/train_0597_1_1.jpg` ~ `GAPS384/train_0597_1_641.jpg` (distance 1, rot0, texture correlation 0.5184)
- `GAPS384/train_0698_1_1.jpg` ~ `GAPS384/train_0698_1_641.jpg` (distance 2, rot0, texture correlation 0.496)
- `GAPS384/train_0598_1_1.jpg` ~ `GAPS384/train_0598_1_641.jpg` (distance 0, rot0, texture correlation 0.4807)
- `GAPS384/train_0697_1_1.jpg` ~ `GAPS384/train_0698_1_641.jpg` (distance 2, rot0, texture correlation 0.4453)
- `GAPS384/train_0514_541_1.jpg` ~ `GAPS384/train_0515_541_1.jpg` (distance 4, rot0, texture correlation 0.4253)
- `GAPS384/train_0597_1_1.jpg` ~ `GAPS384/train_0598_1_641.jpg` (distance 1, rot0, texture correlation 0.4225)
- `GAPS384/train_0698_1_1.jpg` ~ `GAPS384/train_0699_1_641.jpg` (distance 0, rot0, texture correlation 0.4179)
- `GAPS384/train_0697_1_1.jpg` ~ `GAPS384/train_0699_1_641.jpg` (distance 0, rot0, texture correlation 0.3983)
- `GAPS384/train_0576_1_641.jpg` ~ `GAPS384/train_0698_1_1.jpg` (distance 0, rot0, texture correlation 0.3766)
- `GAPS384/train_0598_1_1.jpg` ~ `GAPS384/train_0599_1_641.jpg` (distance 0, rot0, texture correlation 0.3691)
- `GAPS384/train_0576_1_641.jpg` ~ `GAPS384/train_0698_1_641.jpg` (distance 2, rot0, texture correlation 0.3501)
- `GAPS384/train_0576_1_641.jpg` ~ `GAPS384/train_0697_1_1.jpg` (distance 0, rot0, texture correlation 0.3489)
- `GAPS384/train_0576_1_1.jpg` ~ `GAPS384/train_0698_1_1.jpg` (distance 0, rot0, texture correlation 0.3453)
- `GAPS384/train_0576_1_1.jpg` ~ `GAPS384/train_0698_1_641.jpg` (distance 2, rot0, texture correlation 0.3445)
- `GAPS384/train_0698_1_641.jpg` ~ `GAPS384/train_0699_1_641.jpg` (distance 2, rot0, texture correlation 0.3422)
- `GAPS384/train_0696_1_641.jpg` ~ `GAPS384/train_0698_1_1.jpg` (distance 0, rot0, texture correlation 0.3397)
- `GAPS384/train_0576_1_1.jpg` ~ `GAPS384/train_0697_1_1.jpg` (distance 0, rot0, texture correlation 0.3365)
- `GAPS384/train_0696_1_641.jpg` ~ `GAPS384/train_0697_1_1.jpg` (distance 0, rot0, texture correlation 0.3113)
- `GAPS384/train_0599_1_641.jpg` ~ `CFD/308.jpg` (distance 3, rot180, texture correlation 0.3072)
