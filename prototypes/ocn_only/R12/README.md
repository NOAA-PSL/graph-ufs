# Ocn-only R12
This extends the Mahalanobis-loss work in R8 to a location-based covariance
matrix, of size (nlat, nlon, nchannels, nchannels), instead of R8's
spatially-averaged (nchannels, nchannels) matrix. Everything else (interfaces,
input/target/forcing variables, hyperparameters, delta_t) matches R8's
current (T13M) configuration -- only the loss's covariance matrix differs, so
`local_store_path` in config.yaml points at R8's existing preprocessed
training/validation data instead of reprocessing it (~1.2TB). Model
checkpoints, loss, and tensorboard logs are controlled separately by
`output_dir`, which points at R12's own directory
(`/pscratch/sd/n/nagarwal/ocn-only/R12`) so they don't land in R8's tree.
`sub_expt` disambiguates repeated R12 runs within that output_dir, the same
way R8 uses it for its own T1M..T13M runs.

**Note on T1_spacecov/T1M:** the `output_dir` override above didn't exist
yet when the first run (below) was launched, so `checkpoint_dir`/loss.nc
followed `local_store_path` and were written into R8's directory as
`R8/models` and `R8/loss.nc`, alongside R8's own T13M outputs. They were
moved by hand afterwards into
`R12/models_T1M`/`R12/loss_T1M.nc`/`R12/tensorboard/T1M` and the run
renamed from its original sub_expt (`T1_spacecov`) to `T1M`, to match this
directory's own T-numbering and keep it out of R8's tree. `output_dir` now
makes this automatic for future R12 runs.

This was tried once before as R8/T9M using the raw space-dependent tendency
covariance (`tendency_correlation_ocn_only_24h_rm_seasonality_space_dependent.nc`),
with no regularization. Diagnosis (see below) found that some high-latitude
grid cells have highly-correlated channels, making the per-location
covariance matrix nearly singular there (condition number as high as ~1e6,
worst case at lat=-64.8, vs ~55 for R8's global matrix). Cholesky decomposition
of these matrices still technically succeeds (they remain positive-definite),
so it doesn't fail loudly, but inverting them amplifies float32 error and can
silently corrupt the loss/gradients at those locations, unlike a literal NaN,
which is already caught by the existing masking logic in
`graphcast/graphcast/losses.py`.

# Fix: covariance shrinkage
Added `shrink_covariance()`/`mah_covariance_shrinkage` (see
`graphufs/utils.py`), which blends each grid cell's covariance towards
`diag(diag(cov))` before Cholesky decomposition:
`cov_shrunk = (1 - alpha) * cov + alpha * diag(diag(cov))`.
This is a standard covariance-shrinkage estimator (Ledoit & Wolf 2004) for
fixing ill-conditioning from highly-correlated variables, without disturbing
well-conditioned locations much at small alpha. Verified on this dataset's
worst-case cell (lat=-64.8): alpha=0.05 drops the condition number from ~1e6
to ~130 (comparable to R8's global matrix) and drops the largest entry of the
resulting L^-1 by more than 10x.

The `mah_covariance_shrinkage` config value (currently 0.05) is applied
uniformly across all grid cells in `cholesky_decomp()`, which is called once
(offline, in `graphufs/datasets.py`) when the dataset is built, not inside
the training loop.

# Configurations:
[T1M] (originally launched as sub_expt T1_spacecov, renamed after the fact --
see note above) First training with the space-dependent covariance and
shrinkage regularization (alpha=0.05) applied. Uses R8/T13M's architecture
and R8's preprocessed data.

Completed 2026-09-14 (job 58212660, 4 nodes x 4 GPUs, ~4h49m of the 24h
walltime budget, all 40 epochs, exit code 0). Training loss dropped from
1.076 (epoch 1) to 0.437 (epoch 40); validation loss from 0.974 to 0.569,
plateauing somewhat over the last ~10 epochs as the LR schedule decayed to
~0 -- comparable in shape to R8's other Mahalanobis runs, nothing alarming.
Checkpoints (`model_0.npz`..`model_40.npz`) and the tensorboard log are in
`models_T1M/` and `tensorboard/T1M/` respectively.

One thing worth checking before trusting it for anything: `g_norm` logged
as NaN for every optimization step across all 40 epochs, not just at
ill-conditioned grid cells. Since `loss`/`loss_train`/`loss_valid` are all
finite and behaved normally throughout, this looks like a gradient-norm
*logging* issue rather than the shrinkage fix failing to do its job, but it
hasn't been root-caused yet.
