"""
Same as the LocalDataset and BatchLoader, but using xarray_tensorstore
"""
import numpy as np
import xarray_tensorstore

from .datasets import PackedDataset as BaseDataset
from .batchloader import BatchLoader as BaseBatchLoader

class PackedDataset(BaseDataset):
    """Same as the other PackedDatset, but use xarray_tensorstore instead of xarray/dask/zarr
    """

    def __init__(self, emulator, mode):
        self.emulator = emulator
        self.mode = mode
        self.inputs = xarray_tensorstore.open_zarr(self.local_inputs_path)
        self.targets = xarray_tensorstore.open_zarr(self.local_targets_path)

    def __getitem__(self, idx):
        if isinstance(idx, int):

            x = xarray_tensorstore.read(self.inputs["inputs"].sel(sample=idx))
            y = xarray_tensorstore.read(self.targets["targets"].sel(sample=idx))

        else:
            x = [xarray_tensorstore.read(self.inputs["inputs"].sel(sample=i)) for i in idx]
            y = [xarray_tensorstore.read(self.targets["targets"].sel(sample=i)) for i in idx]

        return x, y


class BatchLoader(BaseBatchLoader):

    def _next_data(self):

        if self.data_counter < len(self):
            st = self.data_counter * self.batch_size
            ed = st + self.batch_size
            batch_indices = self.sample_indices[st:ed]
            x, y = self.dataset[batch_indices]
            x = np.vstack([xi.values[None] for xi in x])
            y = np.vstack([yi.values[None] for yi in y])
            self.data_counter += 1
            return x, y
        else:
            raise StopIteration
