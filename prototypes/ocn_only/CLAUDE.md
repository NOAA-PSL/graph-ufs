# ocn_only prototype -- notes for future sessions

This directory builds an ocean-only emulator on top of the DeepMind GraphCast
codebase (JAX-based, not PyTorch). It is pre-AI-revolution-era code: expect
copy-pasted boilerplate across R-directories, minimal error handling, and long
debug sessions left uncommitted. Flag opportunities to simplify/harden when
you see them, but this file is about *what things are and how they connect*,
not a to-do list.

## Repo layout and where logic actually lives

- `prototypes/ocn_only/` (this dir): per-experiment configs (`R0`...`R12`),
  `emulator.py` (defines `OcnTrainer`/`OcnPreprocessor`/`OcnPreprocessed`/
  `OcnEvaluator`, all thin `OmegaConf`-merging wrappers), `train.py`,
  `inference.py`, `preprocess.py`.
- `../../graphufs/` (sibling package, imported as `graphufs`): the real
  emulator base classes (`FVCoupledEmulator` etc.), `datasets.py` (builds the
  training/validation `Dataset`, including loss-metric loading),
  `stacked_training.py` / `stacked_mpi_training.py` (the actual training
  loop, single-GPU and MPI variants), `utils.py` (covariance/Cholesky
  helpers, channel-index utilities).
- `../../graphcast/` (submodule, sibling package `graphcast`): DeepMind's
  original GraphCast code plus this project's "stacked" variants
  (`stacked_graphcast.py`, `losses.py`). **This is a git submodule with its
  own independent history/dirty-state** -- `git status` at the graph-ufs root
  can show it as modified without that being obvious from inside
  `prototypes/ocn_only`. Check `cd ../../graphcast && git status` separately.

Call chain for a training step: `R*/config.yaml` -> `emulator.py:OcnTrainer`
(merges with `base_config_ocn_only.yaml`) -> `graphufs/datasets.py:Dataset`
(builds input/target arrays, loads any Mahalanobis covariance file) ->
`graphufs/stacked_mpi_training.py:optimize()` -> `graphcast/stacked_graphcast.py:
StackedGraphCast.loss_coupled` -> `graphcast/losses.py:stacked_mse`.

## Experiment-directory conventions

- Each `R<n>/` is (nominally) one architecture/dataset variant: interfaces,
  input/target/forcing variables, `delta_t_model`. `config.yaml` sets `case`
  (-> default `local_store_path`), and `sub_expt` distinguishes repeated
  training runs *within* that architecture (different loss, hyperparams,
  etc.) sharing the same `local_store_path`. Outputs are namespaced by
  `sub_expt`: `models_<sub_expt>/`, `loss_<sub_expt>.nc`,
  `inference_<sub_expt>/`, `tensorboard/<sub_expt>/`.
- `README.md` inside each `R<n>/` is a running lab notebook -- read it before
  touching an experiment; it records what was tried, what broke, and why.
  Keep it updated when you resolve or extend something documented there.
- `local_store_path` does not have to equal `${case}`'s own directory: if a
  new experiment needs identical preprocessed data to an existing one (same
  interfaces/variables/delta_t, only the loss differs), point
  `local_store_path` directly at the existing experiment's path instead of
  reprocessing (preprocessed training data can be >1TB). Pick a `sub_expt`
  name that can't collide with anything already in that shared tree. `R12`
  does this: it reuses `R8`'s `/pscratch/.../ocn-only/R8` data via
  `sub_expt: T1_spacecov`, rather than duplicating R8's 1.1TB training set.
- `job-scripts/` per experiment aren't all trustworthy -- e.g.
  `R8/job-scripts/submit_evaluation.sh` actually points at an unrelated
  `cp1/R0` prototype (copy-paste leftover, never fixed). Don't assume a
  script does what its filename says; read it.
- `submit_training.sh` toggles between training and inference by
  commenting/uncommenting the `srun ... train.py` vs `srun ... inference.py`
  line -- check which is live before submitting, it's easy to fire the wrong
  one.

## The Mahalanobis loss (R8, R12)

`use_mahalanobis_loss: true` + `mah_metric_file: <path>.nc` (a NetCDF with a
`targets` DataArray) switches the loss from per-channel MSE to a Mahalanobis
distance using the inverse covariance of channel *tendencies*. Two shapes are
supported end-to-end:

- **Global**: `targets` shaped `(channels_x, channels_y)` -- one covariance
  for the whole domain (R8/T1M-T13M use this, e.g. `29x29` for the current
  channel set).
- **Location-based**: `targets` shaped `(lat, lon, channels_x, channels_y)`
  -- a separate covariance per grid cell. Files with this shape live in
  `tendency_correlations/*_space_dependent.nc`. `graphufs/utils.py:
  cholesky_decomp()` is dimension-agnostic (`xr.apply_ufunc` vectorizes over
  whatever leading dims exist), and `graphcast/graphcast/losses.py:
  stacked_mse`'s inner `mahalanobis_loss()` branches on `L.ndim == 4` to
  apply a per-grid-cell inverse via
  `jnp.einsum('bijk,ijlk->bijl', diff, L_inv)`. This wiring already existed
  before this session (commit `31bc042` in graph-ufs); this session added
  the numerical fix that made it usable (see below).

Data flow for the covariance: `graphufs/datasets.py:Dataset.__init__` opens
`mah_metric_file`, calls `graphufs/utils.py:cholesky_decomp()` once (not
inside the training loop), and stores both the raw covariance and its
Cholesky factor `L` on the `Dataset`. `scripts/train_mpi_no_init.py` passes
`tds.mah_metric_matrix` / `tds.mah_metric_matrix_cholesky_factor_L` into
`optimize()` as `covariance` / `covariance_cholesky_factor_L`. Note
`covariance` itself is only used as a not-None flag inside `stacked_mse`; the
actual math only ever touches the precomputed `covariance_cholesky_factor_L`.

`safe_cholesky()` in `graphufs/utils.py` returns a NaN matrix (instead of
raising) wherever the input has NaN/Inf or fails positive-definiteness --
this is deliberate, since a location-based covariance file can have literal
missing data (e.g. land, ~42% of grid cells in the current space-dependent
file). Downstream, `stacked_mse` builds `valid_loss_mask =
~jnp.isnan(loss).any(axis=-1)` and combines it with the ocean/land
`binary_mask`, zeroing out and excluding from the normalization denominator
any location where the loss came out NaN.

### The high-latitude ill-conditioning bug (R8/T9M -> R12)

`R8/T9M` (Jan, this year) first trained with the raw space-dependent
covariance and no regularization; it completed but the user's later
debugging (uncommitted work in `graphcast/losses.py`, heavy `jdb.print`
instrumentation, now removed) suggested numerical trouble. Diagnosis this
session: **NaN-masking alone doesn't catch everything.** Some high-latitude
/ polar grid cells have highly-correlated channels, making the per-cell
covariance technically positive-definite (Cholesky succeeds, no NaN) but
extremely ill-conditioned -- worst observed case at lat=-64.8 had condition
number ~1e6, vs ~55 for the global `(29,29)` matrix used elsewhere. Inverting
such a matrix in float32 (`jnp.linalg.inv(L)` inside `stacked_mse`) can
silently blow up to huge-but-finite values without ever tripping the
NaN-based mask, corrupting the loss/gradients at those locations.

Fix (this session, `graphufs/utils.py`): `shrink_covariance()` blends each
per-cell covariance towards its own diagonal before Cholesky --
`cov_shrunk = (1-alpha)*cov + alpha*diag(diag(cov))` (Ledoit & Wolf-style
shrinkage estimator: https://www.ledoit.net/honey.pdf and
https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance).
Verified numerically: alpha=0.05 drops the lat=-64.8 cell's condition number
from ~1e6 to ~130 and cuts the resulting inverse's max entry by >10x.
Wired through as `mah_covariance_shrinkage` in config ->
`emulator.py`/`emulator_for_statistics.py` -> `graphufs/datasets.py` ->
`cholesky_decomp(..., shrinkage=...)`. Default is `0.0` (no behavior change
for existing runs); `R12/config.yaml` sets it to `0.05`.

If you're asked to retune this: check condition number vs. alpha on the
worst grid cells first (search across `abs(lat) > 60`, finite-covariance
cells only -- the file also has literal NaNs at land/missing-data cells,
those are a separate, already-handled issue). Don't just pick a value
without checking it against the global matrix's condition number (~55) as a
sanity target.

## Cluster/environment gotchas (NERSC Perlmutter)

- Conda envs in use: `graphufs` (plain, CPU-ish analysis/xarray work),
  `graphufs-mpi` (training, needs GPU + CUDA-aware MPI), `graphufs-cpu`.
  `graphufs-mpi` has `omegaconf`; plain `graphufs` may not -- check before
  assuming a package is available in a given env.
- **`conda activate graphufs-mpi` silently downgrades the loaded
  `cudatoolkit` module (13.2 -> 12.9) via its own Lmod hooks.** This broke
  training as of Sept 2026 (worked fine as recently as June 2026 with no
  module output at all -- this is a NERSC-side default-module change, not
  anything in this repo). The downgrade orphans cray-mpich's CUDA-aware GTL
  plugin, which is linked against `libcudart.so.13`, causing `from mpi4py
  import MPI` to fail immediately with `ImportError: libcudart.so.13: cannot
  open shared object file`, before any training code even runs. Fix: add
  `module load cudatoolkit/13.2` immediately after `conda activate
  graphufs-mpi` in the sbatch script, to force it back. If this starts
  failing differently in the future, suspect another NERSC default-module
  change first -- compare against a `module -t list` / `Lmod is
  automatically replacing...` diff from a job that last succeeded.
- Debug-queue jobs on Perlmutter GPU nodes: `--qos=debug`, walltime capped at
  30 min (`-t 00:30:00`); regular training jobs use `--qos=regular` with much
  longer walltime (12h in these scripts) and checkpoint/resume across
  multiple submissions since 12h isn't enough for a full run.
- `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed` is the fast way to
  check whether a submitted job actually ran vs. crashed immediately --
  `squeue` only shows currently pending/running jobs, not history.

## Useful one-off numerical checks

Reading the `.nc` covariance/correlation files or running any project code
needs an activated conda env -- the bare `python3` on PATH has neither
`xarray` nor `omegaconf`. Use `source activate graphufs` (or `conda run -n
graphufs-mpi ...` for `omegaconf`) before any such check. `conda run`
triggers the same Lmod module-swap noise described above; it's harmless for
non-GPU checks, just noisy on stdout/stderr.
