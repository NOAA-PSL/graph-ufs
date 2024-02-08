# graph-ufs

Repository for training and evaluating
[GraphCast](https://github.com/google-deepmind/graphcast) with UFS data.

## Installation

To install from source in development mode:

```bash
git clone --recursive git@github.com:NOAA-PSL/graph-ufs.git
cd graph-ufs
git checkout develop
conda env create -f conda/gpu-workaround.yaml
pip install -e --no-deps .
```

Using conda to install dependencies will make sure that packages work on the
GPU.
