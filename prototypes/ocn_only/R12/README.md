# Ocn-only R12
This extends the Mahalanobis-loss work in R8 to a location-based covariance
matrix, of size (nlat, nlon, nchannels, nchannels), instead of R8's
spatially-averaged (nchannels, nchannels) matrix. Everything else (interfaces,
input/target/forcing variables, hyperparameters, delta_t) matches R8's
current (T13M) configuration -- only the loss's covariance matrix differs, so
`local_store_path` in config.yaml points at R8's existing preprocessed
data/model/inference/tensorboard tree instead of reprocessing it, and
`sub_expt` is used to keep R12's outputs from colliding with R8's own T*
runs.

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
[T1_spacecov] First training with the space-dependent covariance and
shrinkage regularization (alpha=0.05) applied. Uses R8/T13M's architecture
and R8's preprocessed data.
