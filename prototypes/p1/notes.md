# Stacked I/O and Threading Notes

## Time to read a single batch

### Local Data on Azure using Lustre


### Using custom loader, full batch is a single dask/zarr call

On gpu4
- `batch_size` = 4
    * 1  dask worker thread  = 1.23 sec / batch
    * 2  dask worker threads = 0.69 sec / batch
    * 4  dask worker threads = 0.49 sec / batch
    * 8  dask worker threads = 0.41 sec / batch
    * 16 dask worker threads = 0.37 sec / batch
    * 24 dask worker threads = 0.37 sec / batch
    * 32 dask worker threads = 0.37 sec / batch


- `batch_size` = 16
    * 1  dask worker thread  = 4.35 sec / batch
    * 2  dask worker threads = 2.70 sec / batch
    * 4  dask worker threads = 1.94 sec / batch
    * 8  dask worker threads = 1.50 sec / batch
    * 16 dask worker threads = 1.33 sec / batch
    * 24 dask worker threads = 1.31 sec / batch
    * 32 dask worker threads = 1.33 sec / batch

Note that
- **Most importantly** We have to ask for `--cpus-per-task=N` where N is 24, 48,
  96 for the GPU instances with 1, 2, 4 cards. This reduces the read time by a
  factor of ~2 for >=16 threads, and improves all read times except for the
  single threaded case.
- `shuffle`=True is a better test because presumably some values are
  stored in some sort of cache. I was getting that reading `batch_size=16` with 1 thread was the fastest,
  but this was right after running the `batch_size=4` tests.
- opening with `xr.open_zarr(..., chunks={"sample": 1, "lat": -1, "lon": -1, "channels:-1"})`
  is a bit faster, makes a big difference for the full dataset
- For some reason it takes a bit longer to read the full dataset vs the 1 year
  dataset. This could be related to the fact that no compression was used, and
  should be tested. For this case `xarray-tensorstore` was faster, since it had
  no difference in read speeds

### Using custom loader, but with `xarray-tensorstore`


On gpu4, when we do not specify `.compute(num_threads=...)` when converting to
numpy

- `batch_size` = 4, ~0.3 sec/batch
- `batch_size` = 16, ~1.3 sec/batch

We get a similar speedup as with dask when using `--cpus-per-tasks`>1.

On gpu4, when we do specify `.compute(num_threads=...)`
the timing is pretty bad.

#### Using PyTorch-like loader, each sample read is a separate dask/zarr call
- `batch_size` = 4, Takes 1 sec per batch with 1-16 threads.
- `batch_size` = 16  (on gpu4), using non-custom loader (i.e. each sample is
  separate)
    * 1  worker thread  = 5.8 sec / batch
    * 2  worker threads = 4.2 sec / batch
    * 4  worker threads = 3.8 sec / batch
    * 8  worker threads = 3.7 sec / batch
    * 16 worker threads = 3.6 sec / batch



### Remote Data on PSL GPU

Time to read a single batch of 4 samples took (note, see basically same testing samples not batch):

- `batch_size`=4
    * 1  worker thread  = 35 sec
    * 2  worker threads = 18 sec
    * 4  worker threads = 13 sec
    * 8  worker threads = 10 sec
    * 16 worker threads = 10 sec

Does `dask.cache.Cache` help?
No.
Time to read with 10 GB cache and 8 thread workers = 16.4 sec/batch
vs
Time to read without cache and 8 thread workers = 10 sec/batch


## Thread Data Queue Timing

On gpu4 using `batch_size`=16 and 16 dask worker threads with the
BatchLoader.

Basically all that matters is the `max_queue_size`, which
gets drained eventually and we're reduced to the I/O speed.

I saw no real difference with lock on or off, may as well keep it.

- `num_workers` = 0
    * 3.6 sec / iteration
- `num_workers` = 1
    * `max_queue_size` = 1: 1.2 sec / iteration, queue is cleared at iter 6
    * `max_queue_size` = 2: 1.2 sec / iteration, queue cleared after iter 11
    * `max_queue_size` = 3:  sec / iteration, queue cleared after iter
    * `max_queue_size` = 4:  sec / iteration, queue cleared after iter
    * `max_queue_size` = 8:  sec / iteration, queue cleared after iter
- `num_workers` = 2
    * `max_queue_size` = 1: 1.2 sec / iteration, queue cleared after iter 11
    * `max_queue_size` = 2: 1.2 sec / iteration, queue cleared after iter 17
    * `max_queue_size` = 4:  sec / iteration, queue cleared after iter
- `num_workers` = 4
    * `max_queue_size` = 4:  sec / iteration, queue cleared after iter
    * `max_queue_size` = 8:  sec / iteration, queue cleared after iter

## Training notes

On 1 GPU:
- Queue never fully clears, so we can push through at ~1.1 it/s
- Unclear why the validation queue was able to get to 100, when we had 61
  batches
- Need to test that queue is "cleared" after first epoch and properly refilled

On 4 GPUs:
- Cruise through first ~100 iterations at ~1.1 s/it (note slightly slower at 4
  samples per GPU vs 1 GPU with 4 samples per batch at 1.1it/s due to communication)
- Then queue is empty and each iteration takes ~2.7 s/it , i.e. the time to load
  a batch
- If that time can be reduced to < 1 sec then we are golden
- If we can truly hit 500 MB/s then best we can do is 1.8 s/it
- On second epoch we're slowing down to even 4 s/it ... what's up
