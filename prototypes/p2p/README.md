# P2 on Perlmutter

A series of experiments to break into the machine.
The baseline experiment is `uvwc`, which in other contexts will be referred to
as `P2c` since it is exactly the P2 setup, but with a channel loss - it turns
out that this improves prediction skill quite a bit.

- `uvwc` = Uniform Vertical, With Clock. This is the baseline setup, P2 +
  channel loss.
- `uvnc` = Uniform Vertical, No Clock.
- `nvnc` = Nonuniform Vertical, No Clock.
- `uvncbs32` = `uvnc` + batch size = 32.
- `uvwcsic` = `uvwc` + Single Initial Condition
- `uvwcsicisp` = `uvwc` + Single Initial Condition, use Single Precision during
  Inference.
- `uvwcsicsp` = `uvwc` + Single Initial Condition, use Single Precision during
  Training and Inference.
- `dlwsltgh` = `uvwc` + Diagnosed Loss with
    * Wind Speed (10m horizontal and 3D 3 component)
    * Layer Thickness
    * Geopotential Height

## Preprocessing

To run preprocessing, first you'll want to figure out the inputs and targets
chunksize.
To do this, I would do the following:

* open up an interactive session (CPU or GPU doesn't matter)
* Navigate to the experiment specific directory, e.g.
  ```bash
  cd prototypes/p2p/uvwc
  ```
* Create the emulator and dataset object
  ```python
  from config import P2PTrainer
  from graphufs.datasets import Dataset

  gufs = P2PTrainer()
  tds = Dataset(gufs, mode="training")
  ```
* Then get a sample, inspect the size, and set the c
  ```python
  x,y = tds[0] # gets the first sample
  print(x.shape) # length of the last dimension is the number of channels
  print(y.shape)
  ```
* Pick a chunk size that the channel size is evenly divisible by, and is about
  1-5 MB or so
  * e.g. for one case, my inputs/targets channel sizes were 175 and 85
  * setting the channel chunksize to 5 results in a 1.5 MB chunk (since it
    includes all latitudes and longitudes)
* Set the `_input_channel_chunks` and `_target_channel_chunks` in
  `preprocess.py` to these values,
  lines 20 and 21 in e.g. `prototypes/p2p/uvwc/preprocess.py`

Lastly, if you want to run the script over multiple, short jobs then choose a
short walltime by setting `_walltime` at the top of preprocess.py script.
This corresponds to the length of each job.
Then the `_n_jobs` parameter sets the number of jobs to spread it all across.
Alternatively, you can do it in a single, long job. This actually worked for me.

## Training

Have to pin each MPI process to each GPU, using the `select_gpu_device` script:

```
srun ./select_gpu_device python train.py
```

See example job script that runs training and inference at
`prototypes/p2p/uvwc/job-scripts/submit_training.sh`

## Postprocessing

Make sure to have a separate CPU conda environment, using conda/cpu.yaml (with
`jax=0.4.26` and `jaxlib=0.4.26`).
Then, see `prototypes/p2p/job-scripts/submit_evaluation.sh` for a sample job
script.
