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

Or similarly with `cpu.yaml`. Note for the CPU environment, fix the jax and
jaxlib versions to

```yaml
  - jax=0.4.26
  - jaxlib=0.4.26
```

## Perlmutter MPI installation

This loosely follows these
[these instructions](
https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#installing-mpi4py-with-gpu-aware-cray-mpich)
with some modifications.

Note that:
* I installed this in the common directory, hence the use of the `-p` flag in
  the `conda env create` call, where I have the environment variable
  ```
  export graphufs=/global/common/software/m4718/timothys/graphufs
  ```
* With this setup, using MPI and multithreading will not work
* I also tried the GNU version listed in those instructions. Training timing
  is essentially the same and multithreading + MPI still doesn't work.

### 1. Build the environment

```
module load conda
module load PrgEnv-nvidia cray-mpich cudatoolkit craype-accel-nvidia80 cudnn/8.9.3 nccl/2.21.5
conda env create -f perlmutter-gpu.yaml -p $graphufs
```


### 2. Install JAX and mpi4py


Make sure the modules in step 1 are loaded.

```
conda activate $graphufs
pip install --upgrade jax==0.4.26 jaxlib==0.4.26+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
MPICC="cc -shared" CC=nvc CFLAGS="-noswitcherror" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
```

Note that when built this way, `jaxlib` will not show up in any `conda list`
calls.
However, it will show up if you use `pip list`.
Maybe this is because of the `+` sign, or something? IDK.

### 3. Install mpi4jax

Make sure the modules in step 1 are loaded.

Had to clone [mpi4jax](https://github.com/mpi4jax/mpi4jax), comment out
lines 186-187 (that check if `"/opt/intel/oneapi/compiler/latest"` exists, and if
so try to build XPU extension which we don't need/want.

Then build that local version

```
pip install --no-cache-dir cython
cd /path/to/clone/mpi4jax/to
git clone git@github.com:mpi4jax/mpi4jax.git
# modify setup.py as noted above
CC=nvc CFLAGS="-noswitcherror" CUDA_ROOT=$CUDA_HOME pip install --force --no-cache-dir --no-binary=mpi4jax --no-deps --no-build-isolation ./mpi4jax
```

Note there's some discussion
[here](https://github.com/mpi4jax/mpi4jax/issues/245)
about a different workaround.

### 4. Install the rest of the JAX dependencies

Make sure the modules in step 1 are loaded.

```
pip install chex==0.1.86
pip install optax==0.2.2
pip install orbax-checkpoint==0.6.4 flax==0.8.2 dm-haiku==0.0.12
pip install jraph
```

### 5. Run code

Make sure to have the modules loaded and the following MPI flag in slurm
scripts, unless either are in a .bashrc or something like that.

```
module load conda
module load PrgEnv-nvidia cray-mpich cudatoolkit craype-accel-nvidia80 cudnn/8.9.3 nccl/2.21.5
conda activate graphufs
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
