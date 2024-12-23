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

To run preprocessing, first you'll want to figure out the inputs and targets
chunksize.
To do this, I would do the following:

* open up an interactive session (CPU or GPU doesn't matter)
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
  * e.g. for me, my inputs/targets channel sizes were 175 and 85
  * setting the channel chunksize to 5 results in a 1.5 MB chunk (since it
    includes all latitudes and longitudes)
* Set the `_input_channel_chunks` and `_target_channel_chunks` to these values,
  lines 20 and 21 in preprocess.py


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
