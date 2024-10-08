# Environment

Create with

```
conda env create -f gpu.yaml
```

Or similarly with `cpu.yaml`.

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
