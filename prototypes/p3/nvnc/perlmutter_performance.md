# Perlmutter I/O Performance

Note that the statement

```python
from mpi4py import MPI
```

has to exist in all of the main scripts (e.g. in train.py) for mpi4py to work at
all.

## MPI Training Timing

|                                                            | Batch Size = 16           |
|------------------------------------------------------------|---------------------------|
| MPI + Serial Data Loading<br>1 Node                        | 1.05 it/sec (2.04 it/sec) |
| Threads + Serial Data Loading<br>1 Node                    | 1.05 it/sec (1.68 it/sec) |
| Threads + Parallel Data Loading<br>1 Node                  | 1.15 it/sec (2.94 it/sec) |
| MPI + Serial Data Loading<br>2 Nodes (8221,8224)           | 1.89 it/sec (3.98 it/sec) |
| MPI + Serial Data Loading<br>4 Nodes (8241,8244,8301,8332) | 2.92 it/sec (7.38 it/sec) |


Some notes:
* using the slurm option `--nodelist` to select nearby nodes made no significant difference,
  i.e. 0.01 it/sec faster
* reducing communication by only communicating loss per channel, and summing to
  compute the loss function, has a minor impact. No impact with 2 nodes, and
  0.05 it/sec faster with 4 nodes.



## Single Node Training Timing

Time per iteration (validation iterations in parentheses), moving through 100 iterations.

| Batch Size | Sample Loss<br>1 GPU  | Sample Loss<br>4 GPUs | Batch Loss<br>4 GPUs |
|------------|-----------------------|-----------------------|----------------------|
| 16         | 1.45 s/it (1.15 s/it) | 1.45 s/it (2.32 it/s) | .88 s/it (2.3 it/s)  |

Notes:
* Sample loss means the "for loop" approach that Daniel introduced to reduce
  memory, which is actually super fast
* However, on Perlmutter, it takes ~.3 seconds to read a batch, we're now compute bound
  whereas on Azure we were I/O bound, so it was more economical to do the for
  loop (sample loss) approach.
* I verified this by loading a small number of samples into memory,
  and the training/validation iterations take the same amount of time.
* Batch size 32 I/O is fast, but the model doesn't fit in memory unless we use
  the "sample loss" approach, but this is 2x slower per optim iteration

## MPI I/O Timing


Tested by running

```bash
srun -n 4 python test_mpi_read.py # 1 node
srun -n 8 python test_mpi_read.py # 2 nodes
srun -n 16 python test_mpi_read.py # 4 nodes
```

See the submission scripts `job-scripts/submit_N.sh` for each node count.
By using the `--tasks-per-node` option, the `-n n_processes` option in the `srun` command was not needed.

Time to load, per batch
| Batch Size | Non MPI | 1 Node | 2 Nodes | 4 Nodes |
|------------|---------|--------|---------|---------|
| 16         | .35     | 0.23   | 0.12    | 0.07    |
| 32         | .69     | 0.46   | 0.23    | 0.11    |

Note that the MPI total walltime seems to add ~28 seconds, and it's not clear where this comes from...
Hopefully not from communication!
But no matter what the scaling sticks.

Note that running `nvidia-smi` shows that all of the processes on that node are running
on all 4 of the GPUs.
So, in all of these cases there were 4 processes running (4 unique PIDs), each
process is copied / running on each GPU, so it looks like 16 processes are
running.
However, this doesn't seem to matter, since we handle GPU data placement at the
code level with JAX (as is recommended).
Read time is the same with or without the `select_gpu_device` script that is
potentially suggested by Perlmutter [here](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#using-mpi4py-with-gpu-aware-cray-mpich)


## Initial P2 Light with 8 vertical levels

### How many dask threads to use during data preprocessing?

TL;DR it doesn't matter

Timing to read With batch size = 16
*  16 threads = 46 s
*  32 threads = 47 s
*  64 threads = 47 s
* 128 threads = 46 s
* 256 threads = 46 s


### How many input/target channels are there? What chunksize to use?

Here I was just testing what chunksize to create, it turns out not to matter.
Just use something ~1-10 MB and it'll be fine.

Timing to read from scratch, batch size 16

**chunk size = 5**
sec / batch
*  1 worker  = 1.0
*  2 worker  = .54
*  4 worker  = .36
*  8 workers = .26
* 16 workers = .25
* 32 workers = .24
* 64 workers = .26
* xarray-tensorstore = 0.16

**chunk size = 19/15**
*  1 worker  = .85
*  2 worker  = .49
*  4 worker  = .32
*  8 workers = .28
* 16 workers = .26
* 32 workers = .25
* 64 workers = .27
* xarray-tensorstore = 0.16
