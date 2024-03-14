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

Please note that you will need to replace the correct paths below and run this command for your python environment to work correctly:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/graph-ufs:/path/to/graph-ufs/graphcast:/path/to/ufs2arco
```

### Submodules
Currently `graphcast` (from Google), weatherbench2 (from Google), and `ufs2arco` (from NOAA) are pulled in as submodules if you do a recursive clone of the repository.
If you do not use the `--recursive` clone when you clone the repo, then you can do a `git submodule --init` later.
If you already have an existing installation of one of these that you would prefer to use, you can just create a symlink between the top
directory of `graphufs` and your installed location.  You will need to add the location to the `PYTHONPATH` environment variable as above,
but a symlink should ensure you can use the same paths as if the submodules were pulled recursively.

### WeatherBench2 evaluation
The official weatherbench2 evaluation results are computed with a script `scripts/evaluate.py`.
Command line scripts can be found here: https://weatherbench2.readthedocs.io/en/latest/official-evaluation.html
A sample evaluation script for Pangu weather data can be found in `evaluate-pangu.sh`.
