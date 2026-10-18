# Wisconsin Diagnostic Breast Cancer: two-feature teaching subset

Source: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993),
*Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository.
<https://doi.org/10.24432/C5DW2B>.

UCI distributes this dataset under **CC BY 4.0**:
<https://creativecommons.org/licenses/by/4.0/>. Credit belongs to the dataset
creators; this teaching adaptation does not imply their endorsement.

`two_features.csv` preserves all 569 records in the original `wdbc.data` order.
The adaptation selects original one-based field 9 (mean concavity) and field 4
(mean texture), renames them `concavity_mean` and `texture_mean`, omits the ID
and other predictors, and maps field 2 diagnosis to `malignant` (M=1, B=0).
Numerical predictor strings are unchanged. No rows are removed or imputed.
`concavity_mean` is UCI's `concavity1`; it is different from concave points.
`texture_mean` is UCI's `texture1`. `covavity_mean` is not a dataset variable.

The small subset is committed so notebooks run without a network connection.
See the adjacent `manifest.yml` for checksums, retrieval date, and split policy.
This is the identical CSV from HealthML/deep-learning-book commit
`7803e0bcbbe1b20d7563db09cee97f4976242e6b`. The portable loader and exact
seed-8 split live in `dlexercises.week02`; no book import is needed.
Synchronize the repository environment with `uv sync --locked --group notebook`,
then launch with `uv run --locked --group notebook jupyter lab`.
This historical two-feature classification exercise is not a validated clinical tool.

To reconstruct the subset, download the archive URL in the manifest and extract
`wdbc.data`. From the repository root run (replace the source path if needed):

```python
import csv
from pathlib import Path

source = Path('/tmp/wdbc.data')
with source.open() as original, Path('src/dlexercises/datasets/wdbc/two_features.csv').open('w', newline='') as target:
    writer = csv.writer(target, lineterminator='\n')
    writer.writerow(['concavity_mean', 'texture_mean', 'malignant'])
    for row in csv.reader(original):
        writer.writerow([row[8], row[3], int(row[1] == 'M')])
```
