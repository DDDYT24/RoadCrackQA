# Data QA report

Dataset root: `data/raw/CRACK500`

## Summary

| Check | Count |
|---|---|
| images | 3368 |
| masks | 3368 |
| missing_split_dirs | 0 |
| corrupt | 0 |
| missing_mask | 0 |
| orphan_mask | 0 |
| size_mismatch | 0 |
| non_binary_masks | 0 |
| exact_duplicate_groups | 0 |
| exact_duplicate_groups_cross_split | 0 |
| near_duplicate_pairs | 76 |
| near_duplicate_pairs_cross_split | 56 |
| source_id_unmatched | 0 |
| source_id_cross_split_overlap | 6 |
| excluded_images | 56 |
| exact_duplicate_groups_cross_split_after_exclusions | 0 |
| near_duplicate_pairs_cross_split_after_exclusions | 0 |
| source_id_cross_split_overlap_after_exclusions | 0 |

## Inventory

| Split | Folder | Images | Masks | By extension |
|---|---|---|---|---|
| train | traincrop | 1896 | 1896 | .jpg: 1896, .png: 1896 |
| val | valcrop | 348 | 348 | .jpg: 348, .png: 348 |
| test | testcrop | 1124 | 1124 | .jpg: 1124, .png: 1124 |

## Image sizes (width x height)

- train: 640x360: 1754, 360x640: 86, 648x484: 56
- val: 640x360: 324, 648x484: 17, 360x640: 7
- test: 640x360: 1105, 360x640: 19

## Mask balance

| Split | Crack pixel ratio | Masks without crack |
|---|---|---|
| train | 0.063341 | 0 |
| val | 0.056467 | 0 |
| test | 0.055904 | 0 |

## Source ids

| Split | Unique sources | Images per source (min to max) |
|---|---|---|
| train | 250 | 3 to 16 |
| val | 50 | 4 to 12 |
| test | 148 | 1 to 15 |

Cross split overlap:

- test/train: 20160328_154452, 20160328_154454, 20160329_103609, 20160329_103800, 20160329_111111
- test/val: 20160329_103807

Files the source id pattern does not match:

None.

## Corrupt files

None.

## Mask pairing

Images without a mask:

None.

Masks without an image:

None.

Size mismatches:

None.

Non binary masks:

None.

## Exact duplicates (SHA256)

None.

## Near duplicates (dHash, distance <= 4)

- `traincrop/20160328_151056_1281_361.jpg` ~ `valcrop/20160328_151057_641_721.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151056_1_361.jpg` ~ `valcrop/20160328_151057_1921_721.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151056_641_1.jpg` ~ `valcrop/20160328_151057_1281_1081.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151056_641_1081.jpg` ~ `valcrop/20160328_151057_1281_1.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151056_641_721.jpg` ~ `valcrop/20160328_151057_1281_361.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_1281_1081.jpg` ~ `valcrop/20160328_151244_641_1.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_1281_361.jpg` ~ `valcrop/20160328_151244_641_721.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_1281_721.jpg` ~ `valcrop/20160328_151244_641_361.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_1921_1.jpg` ~ `valcrop/20160328_151244_1_1081.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_1921_1081.jpg` ~ `valcrop/20160328_151244_1_1.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_1_1.jpg` ~ `valcrop/20160328_151244_1921_1081.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_1_361.jpg` ~ `valcrop/20160328_151244_1921_721.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_1_721.jpg` ~ `valcrop/20160328_151244_1921_361.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_641_1.jpg` ~ `valcrop/20160328_151244_1281_1081.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_641_361.jpg` ~ `valcrop/20160328_151244_1281_721.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_151243_641_721.jpg` ~ `valcrop/20160328_151244_1281_361.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_154452_1281_721.jpg` ~ `testcrop/20160328_154452_641_361.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_154452_1921_721.jpg` ~ `testcrop/20160328_154452_1_361.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_154452_1_361.jpg` ~ `testcrop/20160328_154452_1921_721.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_154452_1_721.jpg` ~ `testcrop/20160328_154452_1921_361.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_154452_641_361.jpg` ~ `testcrop/20160328_154452_1281_721.jpg` (distance 0, rot180) (cross split)
- `traincrop/20160328_154454_361_1.jpg` ~ `testcrop/20160328_154454_1_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160328_154454_361_1921.jpg` ~ `testcrop/20160328_154454_1921_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160328_154454_721_1.jpg` ~ `testcrop/20160328_154454_1_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160328_154454_721_1281.jpg` ~ `testcrop/20160328_154454_1281_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160328_154454_721_1921.jpg` ~ `testcrop/20160328_154454_1921_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103609_1_1921.jpg` ~ `testcrop/20160329_103609_1921_1081.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103609_361_1.jpg` ~ `testcrop/20160329_103609_1_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103609_361_1281.jpg` ~ `testcrop/20160329_103609_1281_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103609_361_1921.jpg` ~ `testcrop/20160329_103609_1921_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103609_361_641.jpg` ~ `testcrop/20160329_103609_641_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103609_721_641.jpg` ~ `testcrop/20160329_103609_641_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103800_361_1.jpg` ~ `testcrop/20160329_103800_1_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103800_361_1281.jpg` ~ `testcrop/20160329_103800_1281_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103800_721_1281.jpg` ~ `testcrop/20160329_103800_1281_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103800_721_1921.jpg` ~ `testcrop/20160329_103800_1921_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_103800_721_641.jpg` ~ `testcrop/20160329_103800_641_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_111111_361_1281.jpg` ~ `testcrop/20160329_111111_1281_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_111111_361_1921.jpg` ~ `testcrop/20160329_111111_1921_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_111111_361_641.jpg` ~ `testcrop/20160329_111111_641_721.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_111111_721_1.jpg` ~ `testcrop/20160329_111111_1_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_111111_721_1281.jpg` ~ `testcrop/20160329_111111_1281_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_111111_721_1921.jpg` ~ `testcrop/20160329_111111_1921_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160329_111111_721_641.jpg` ~ `testcrop/20160329_111111_641_361.jpg` (distance 0, rot270) (cross split)
- `valcrop/20160329_103807_361_1.jpg` ~ `testcrop/20160329_103807_1_721.jpg` (distance 0, rot270) (cross split)
- `valcrop/20160329_103807_361_1281.jpg` ~ `testcrop/20160329_103807_1281_721.jpg` (distance 0, rot270) (cross split)
- `valcrop/20160329_103807_361_1921.jpg` ~ `testcrop/20160329_103807_1921_721.jpg` (distance 0, rot270) (cross split)
- `valcrop/20160329_103807_361_641.jpg` ~ `testcrop/20160329_103807_641_721.jpg` (distance 0, rot270) (cross split)
- `valcrop/20160329_103807_721_1.jpg` ~ `testcrop/20160329_103807_1_361.jpg` (distance 0, rot270) (cross split)
- `traincrop/20160328_151056_1281_1.jpg` ~ `valcrop/20160328_151057_641_1081.jpg` (distance 1, rot180) (cross split)
- ... and 26 more (see the JSON report)

## After exclusions

Sources dropped (`data.exclude_sources`):

- val: 20160328_151057, 20160328_151244
- test: 20160328_154452, 20160328_154454, 20160329_103609, 20160329_103800, 20160329_103807, 20160329_111111

Images dropped: 56

Cross split exact duplicates left: 0; near duplicates left: 0; source ids still shared: 0.
