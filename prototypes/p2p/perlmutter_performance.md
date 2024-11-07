# Perlmutter I/O Performance

## P2 Comparison





## Initial P2 Light with 8 vertical levels

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

