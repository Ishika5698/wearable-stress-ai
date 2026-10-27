# Local WESAD data

No dataset files are included in this repository. Obtain the archive only through the download link on the [official University of Siegen WESAD page](https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/). Read and follow the current source and download-host terms. The source page describes scientific, non-commercial use and attribution in publications. Do not redistribute the archive or participant records.

Extract subject folders so files follow the expected pattern:

```text
data/raw/WESAD/S2/S2.pkl
data/raw/WESAD/S3/S3.pkl
...
```

Before running, check the release README for label codes, channel names, and any format differences. This project only loads trusted official pickle files; Python pickle files can execute code during loading. The loader checks top-level keys and the feature pipeline checks chest channel lengths against the label clock. Wrist channels run at different documented rates and are inventoried separately; do not align them to chest samples by row number.

`data/raw/` and `data/processed/` are git-ignored. Keep any extracted participant files and participant-level derived tables local.
