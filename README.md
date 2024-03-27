# graph-ufs

Repository for training and evaluating
[GraphCast](https://github.com/google-deepmind/graphcast) with UFS data.

## Installation

To install from source in development mode:

```bash
git clone --recursive git@github.com:NOAA-PSL/graph-ufs.git -b develop
cd graph-ufs
conda env create -f conda/gpu.yaml
```

Using conda to install dependencies will make sure that packages work on the
GPU.

Please note that you will need to replace the correct paths below and run this command for your python environment to work correctly:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/graph-ufs:/path/to/graph-ufs/graphcast:/path/to/ufs2arco
```

### Submodules
Currently the following repositories are pulled in as submodules if you do a recursive clone of the repository (as suggested in [Installation](#installation) above):
- [`graphcast`](https://github.com/google-deepmind/graphcast) from Google DeepMind
- [`weatherbench2`](https://github.com/google-research/weatherbench2) from Google Research
- [`ufs2arco`](https://github.com/NOAA-PSL/ufs2arco) from NOAA Physical Sciences Laboratory 

If you do not use the `--recursive` flag when you clone the repo, then you will need to do a `git submodule --init` later to be able to use the submodules.
If you already have an existing installation of one of these that you would prefer to use, you can just create a symlink between the top
directory of `graphufs` and your installed location.  You will need to add the location to the `PYTHONPATH` environment variable as above,
but a symlink should ensure you can use the same paths as if the submodules were pulled recursively.  However, it is recommended to use the
submodule versions checked into the repo for consistency.

### WeatherBench2 evaluation
The official weatherbench2 evaluation results are computed with a script `scripts/evaluate.py`.
Command line scripts can be found [here](https://weatherbench2.readthedocs.io/en/latest/official-evaluation.html).
A sample evaluation script for Pangu weather data can be found in `evaluate-pangu.sh`.

## Branches & PRs

For development, we recommend creating a branch off of `develop` following the below naming conventions:
- `documentation/user_branch_name`: Documenation additions and/or corrections
- `feature/user_branch_name`: Enhancements/upgrades
- `fix/user_branch_name`: Bug-type fixes
- `hotfix/user_branch_name`: Bug-type fixes which require immediate attention and are required to fix significant issues that compromise the integrity of the software

Once the desired contributions are complete in your branch, submit a pull request (PR) to merge your branch into `develop`.
Also note that all python coding additions should follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide.
We will use the "squash merge" feature on PRs that we merge to consolidate the commit history.

When prototypes are completed (or other significant milestones worthy of a release are reached), we will open a PR to `main` and tag the release.

## Disclaimer

The United States Department of Commerce (DOC) GitHub project code is
provided on an "as is" basis and the user assumes responsibility for
its use. The DOC has relinquished control of the information and no longer
has responsibility to protect the integrity, confidentiality, or
availability of the information.  Any claims against the Department of
Commerce stemming from the use of its GitHub project will be governed
by all applicable Federal law.  Any reference to specific commercial
products, processes, or services by service mark, trademark,
manufacturer, or otherwise, does not constitute or imply their
endorsement, recommendation or favoring by the Department of
Commerce.  The Department of Commerce seal and logo, or the seal and
logo of a DOC bureau, shall not be used in any manner to imply
endorsement of any commercial product or activity by DOC or the United
States Government.
