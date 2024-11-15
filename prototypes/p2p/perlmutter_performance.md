# Perlmutter I/O Performance

## Training timing

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


Time to load, per batch
| Batch Size | Non MPI | 1 Node | 2 Nodes | 4 Nodes |
|------------|---------|--------|---------|---------|
| 16         | .35     | 0.23   | 0.12    | 0.07    |
| 32         | .69     | 0.46   | 0.23    | 0.11    |

Note that the MPI total walltime seems to add ~28 seconds, and it's not clear where this comes from...
Hopefully not from communication!
But no matter what the scaling sticks.


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
