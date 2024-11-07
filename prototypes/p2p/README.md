# P2 Light

## The dataset

* Num. Samples = 75,968
* Num. Batches =
* Level centers =
  ```
  [219.71, 257.35, 298.49, 342.79,
   414.63, 522.39, 636.50, 745.35, 810.27,
   837.22, 862.13, 884.08, 910.94, 936.90, 962.09, 987.79]
  ```
* Num. Channels
  * Inputs = 175
  * Targets = 85
* Channel Chunksize = 5 = 1.5 MB

## Preprocessing


## Old numbers...

### How many dask threads to use?

Timing to read With batch size = 16
*  16 threads = 46 s
*  32 threads = 47 s
*  64 threads = 47 s
* 128 threads = 46 s
* 256 threads = 46 s


### How many input/target channels are there? What chunksize to use?

Timing to read from scratch, batch size 16, 32, 64

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
*  1 worker  = .85, 2.4,
*  2 worker  = .49, 1.0,
*  4 worker  = .32, .58,
*  8 workers = .28, .44,
* 16 workers = .26, .40,
* 32 workers = .25, .40,
* 64 workers = .27, .41,
* xarray-tensorstore = 0.16, .28


Steps
- [ ] Create the chunksize = 5 dataset
- [ ] Run the test read script, running up to a lot more threads
- [ ] create the chunksize = 19/15 dataset
- [ ] Run the test read script, running up to a lot more threads
