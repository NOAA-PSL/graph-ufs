# Process

1. run through 2 epochs of 1 year of data with 1, 2, 4 GPUs
    * check the `chunks_per_epoch`
2. check timing... run for full dataset
    * reset `chunks_per_epoch` -> 26 or whatever works for 1 year

## Lessons Learned

1. For some reason, the GPU cluster created for us by PW is unable to allocate
   CPU instances. Until this is resolved, it is required to have a separate CPU
   cluster to do the initial preprocessing.
2. Memory issues even with 1 year of data:
    * 1 GPU
    * 2 GPUs, `batch_size`=16 per GPU
    * 4 GPUs, `batch_size`=8 per GPU
   Have tried:
    * `chunks_per_epoch`=48 (so 2 chunks per month)
    * `latent_size`=256
   What worked?
    * 4 GPUs, `batch_size`=8 (per GPU), `latent_size`=256, `chunks_per_epoch`=48
    * Speed = 35 seconds per global batch (32 samples)
    * Num batches = 2,374, num. samples = 75,968
    * Time per epoch = 23 hours


# Stacked I/O and Threading Notes

## Time to read a single batch

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


### Local Data on Azure using Lustre

#### Using non-custom loader, each sample read is a separate dask/zarr call
- `batch_size` = 4, Takes 1 sec per batch with 1-16 threads.
- `batch_size` = 16  (on gpu4), using non-custom loader (i.e. each sample is
  separate)
    * 1  worker thread  = 5.8 sec / batch
    * 2  worker threads = 4.2 sec / batch
    * 4  worker threads = 3.8 sec / batch
    * 8  worker threads = 3.7 sec / batch
    * 16 worker threads = 3.6 sec / batch



### Using custom loader, full batch is a single dask/zarr call


On gpu4
- `batch_size` = 4
    * 1  dask worker thread  = 1.13 sec / batch
    * 2  dask worker threads = 0.82 sec / batch
    * 4  dask worker threads = 0.70 sec / batch
    * 8  dask worker threads = 0.70 sec / batch
    * 16 dask worker threads = 0.68 sec / batch
    * 24 dask worker threads = 0.67 sec / batch
    * 32 dask worker threads = 0.70 sec / batch

- `batch_size` = 16
    * 1  dask worker thread  = 4.35 sec / batch
    * 2  dask worker threads = 3.15 sec / batch
    * 4  dask worker threads = 2.90 sec / batch
    * 8  dask worker threads = 2.73 sec / batch
    * 16 dask worker threads = 2.66 sec / batch
    * 24 dask worker threads = 2.75 sec / batch
    * 32 dask worker threads = 2.73 sec / batch

Note that
- `shuffle`=True is a better test because presumably some values are
  stored in some sort of cache. I was getting that reading `batch_size=16` with 1 thread was the fastest,
  but this was right after running the `batch_size=4` tests.
- rechunking `x = x.chunk({"channels": -1})` right before loading hinders
  performance


## Thread Data Queue Timing

On gpu4 using `batch_size`=16 and 16 dask worker threads with the
DaskDataLoader.


Basically all that matters is the `max_queue_size`, which
gets drained eventually and we're reduced to the I/O speed.

I saw no real difference with lock on or off, may as well keep it.

- `num_workers` = 0
    * 3.6 sec / iteration
- `num_workers` = 1
    * `max_queue_size` = 1: 2.5 sec / iteration, queue is cleared at iter 0
    * `max_queue_size` = 2: 2.4 sec / iteration, queue cleared after iter 3
    * `max_queue_size` = 3: 2.3 sec / iteration, queue cleared after iter 4
    * `max_queue_size` = 4: 2.2 sec / iteration, queue cleared after iter 6
    * `max_queue_size` = 8: 1.9 sec / iteration, queue cleared after iter 13
- `num_workers` = 2
    * `max_queue_size` = 2: 2.4 sec / iteration, queue cleared after iter 3
    * `max_queue_size` = 4: 2.2 sec / iteration, queue cleared after iter 7
- `num_workers` = 4
    * `max_queue_size` = 4: 2.3 sec / iteration, queue cleared after iter 7
    * `max_queue_size` = 8: 2.0 sec / iteration, queue cleared after iter 11
