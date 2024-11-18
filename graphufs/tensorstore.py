"""
Same as the LocalDataset and BatchLoader, but using xarray_tensorstore
"""
import numpy as np
import logging
import xarray_tensorstore

from .datasets import PackedDataset as BaseDataset
from .batchloader import BatchLoader as BaseBatchLoader
from .mpi import MPITopology, _has_mpi

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
            return x, y
        else:
            raise StopIteration

class MPIBatchLoader(BaseBatchLoader):
    """Make sure mpi4py and mpi4jax is installed
    """
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        mpi_topo,
        drop_last=True,
        num_workers=0,
        max_queue_size=1,
        rng_seed=None,
        sample_stride=None,
        start=0,
    ):
        assert _has_mpi, f"MPIBatchLoader.__init__: Unable to import mpi4py or mpi4jax, cannot use this class"

        self.topo = mpi_topo
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            max_queue_size=max_queue_size,
            rng_seed=rng_seed,
            sample_stride=sample_stride,
            start=start,
        )

        self.data_per_device = batch_size // self.topo.size
        self.local_batch_index = self.topo.rank*self.data_per_device
        logging.info(str(self))
        if self.data_per_device*self.topo.size != batch_size:
            logging.warning(f"MPIBatchLoader.__init__: batch_size = {batch_size} not divisible by MPI Size = {self.topo.size}")
            logging.warning(f"MPIBatchLoader.__init__: some data will be skipped in each batch")

    def __str__(self):
        msg = "\ngraphufs.tensorstore.MPIBatchLoader\n" +\
            "-----------------------------------\n"
        for key in ["local_batch_index", "data_per_device", "batch_size"]:
            msg += f"{key:<18s}: {getattr(self, key):02d}\n"
        return msg


    def _next_data(self):
        if self.data_counter < len(self):
            st = (self.data_counter * self.batch_size) + self.local_batch_index
            ed = st + self.data_per_device
            batch_indices = self.sample_indices[st:ed]

            x, y = self.dataset[batch_indices]
            x = np.vstack([xi.values[None] for xi in x])
            y = np.vstack([yi.values[None] for yi in y])
            return x, y
        else:
            raise StopIteration

    def restart(self, idx=0, cancel=False, **kwargs):
        super().restart(idx=idx, cancel=cancel, **kwargs)
        self.sample_indices = self.topo.bcast(self.sample_indices)

class ExpandedBatchLoader(BaseBatchLoader):
    def _next_data(self):

        if self.data_counter < len(self):
            st = self.data_counter * self.batch_size
            ed = st + self.batch_size
            batch_indices = self.sample_indices[st:ed]
            data = self.dataset.get_batch_of_xarrays(batch_indices)
            return tuple(d.compute() for d in data)
        else:
            raise StopIteration
