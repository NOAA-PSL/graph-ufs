# Some quick tests


## Training

Time per optimization iteration.
It's actually the same whether we use 1 or 4 GPUs...
This actually makes sense because we are I/O bound, so it doesn't make a
difference whether we run through the batch on 1 or 4 GPUs once it's loaded.
Same with the XLA flags, the computation is fast enough so that these just don't
matter.

### 1 GPU

* `use_xla_flags=False`:
    * batch size = 16: 1.15 sec/iter
    * batch size = 32: 2.5 sec / iter
* `use_xla_flags=True`:
    * batch size = 16: 1.15 sec/iter
    * batch size = 32: 2.25 sec/iter

### 4 GPU

* `use_xla_flags=False`:
    * batch size = 16: 1.29 sec / iter
    * batch size = 32:
* `use_xla_flags=True`:
    * batch size = 16: 1.2 sec / iter
    * batch size = 32: 2.2 sec / iter

### 4 GPU non for loop way

* `use_xla_flags=False`:
    * batch size = 16: 1.2 sec / iter
    * batch size = 32: NaN
* `use_xla_flags=True`:
    * batch size = 16: 1.2 sec / iter

## Reading


### With dask

The dask implementation is faster than this by quite a bit when setting
`chunks={"channels":-1}`, but it's still slower than tensorstore.

* Num GPUs = 4 (1), batch size = 16:
        num_workers      avg seconds / batch
        8               2.5 (2.8)
        16              2.3 (2.5)
        24              2.2 (2.4)
        32              2.2 (2.4)

* Num GPUs = 4 (1), batch size = 32:
        num_workers      avg seconds / batch
        8               4.2 (4.4)
        16              3.5 (3.8)
        24              3.5 (3.5)
        32              3.4 (3.5)


### With xarray tensorstore

* Num GPUs = 4 (1), batch size = 16:
    * 1.2 (1.2)

* Num GPUs = 4 (1), batch size = 32:
    * 2.2 (2.2)
