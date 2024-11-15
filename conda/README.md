# Environment

Create with

```
conda env create -f gpu.yaml
```

## Perlmutter

On Perlmutter, make sure to have cudatoolkit loaded. It's a default, but just in
case:

```
module load cudatoolkit
```

Or similarly with `cpu.yaml`.

### Multi node on Perlmutter

I followed [these
instructions](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#installing-mpi4py-with-gpu-aware-cray-mpich) for mpi4py/mpi4jax, after creating the environment

```
module load PrgEnv-gnu cray-mpich craype-accel-nvidia80
MPICC="cc -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
```

And then, the [mpi4jax
instructions](https://github.com/mpi4jax/mpi4jax?tab=readme-ov-file#installation),
specifically the lines for when mpi4py is already installed

```
pip install cython
CUDA_ROOT=$CUDA_HOME pip install mpi4jax --no-build-isolation
```

Before running, and in slurm scripts, add

```
export MPICH_GPU_SUPPORT_ENABLED=1
export MPI4JAX_USE_CUDA_MPI=1
```

## Optional dependencies

It's best to install these after creating and activating the environment, i.e.
first:

```
conda activate graphufs
```

Then...


* To use the `graphufs.tensorstore` module, install `xarray-tensorstore` from pip:
  ```
  pip install xarray-tensorstore
  ```

* To use the `graphufs.torch` module, install pytorch as follows:
   ```
   conda install pytorch torchvision -c conda-forge -c pytorch
   ```
