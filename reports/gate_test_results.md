# Road gate on the test photos

A photo passes (is judged a road) when the mean road probability of its tiles reaches 0.8306, the threshold chosen on val so that 99% of the val road photos pass. 95% Wilson intervals.

| Photos | Road | Count | Passed | Pass rate | 95% CI |
|---|---|---|---|---|---|
| road photos | yes | 767 | 766 | 0.999 | 0.993 to 1.000 |
| non-road photos | no | 2104 | 6 | 0.003 | 0.001 to 0.006 |
| AEL | yes | 16 | 16 | 1.000 | 0.806 to 1.000 |
| CRACK500 full photos | yes | 200 | 199 | 0.995 | 0.972 to 0.999 |
| GAPS384 | yes | 106 | 106 | 1.000 | 0.965 to 1.000 |
| SDNET2018 | yes | 400 | 400 | 1.000 | 0.991 to 1.000 |
| cracktree200 | yes | 45 | 45 | 1.000 | 0.921 to 1.000 |
| COCO | no | 1000 | 0 | 0.000 | 0.000 to 0.004 |
| DTD | no | 1104 | 6 | 0.005 | 0.003 to 0.012 |

Non-road groups with the most photos passed:

- DTD/flecked: 2
- DTD/porous: 2
- DTD/stained: 1
- DTD/swirly: 1

Road photos stopped (1):

- CRACK500 full photos: `data/raw/CRACK500/testdata/20160308_074330.jpg` (0.712)

Model `resnet18-904a24f53acb`; config `a471139fddfb`; lock `255a83f708cd`; commit `ae0c2a31dfbf8ff265170925e835969435232bf2`.
