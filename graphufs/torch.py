"""
Implementations of Torch Dataset and DataLoader
"""
from os.path import join
from typing import Optional
import numpy as np
import xarray as xr
import dask.array
import threading
import queue
import concurrent

from jax.tree_util import tree_map
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import default_collate

from xbatcher import BatchGenerator

from graphcast.data_utils import extract_inputs_targets_forcings
from graphcast.model_utils import dataset_to_stacked

from .emulator import ReplayEmulator

class Dataset(TorchDataset):
    """
    PyTorch dataset for Replay Data that should work with GraphCast
    """

    @property
    def xds(self) -> xr.Dataset:
        """
        Returns the xarray dataset.

        Returns:
            xds (xarray.Dataset): The xarray dataset.
        """
        return self.sample_generator.ds

    @property
    def local_inputs_path(self) -> str:
        return join(self.emulator.local_store_path, self.mode, "inputs.zarr")

    @property
    def local_targets_path(self) -> str:
        return join(self.emulator.local_store_path, self.mode, "targets.zarr")

    def __init__(
        self,
        emulator: ReplayEmulator,
        mode: str,
        preload_batch: bool = False,
    ):
        """
        Initializes the Dataset object.

        Args:
            emulator (ReplayEmulator): The emulator object.
            mode (str): "training", "validation", or "testing"
        """
        self.emulator = emulator
        self.mode = mode
        xds = self._open_dataset()
        self.sample_generator = BatchGenerator(
            ds=xds,
            input_dims={
                "datetime": emulator.n_forecast,
                "lon": len(xds["lon"]),
                "lat": len(xds["lat"]),
                "level": len(xds["level"]),
            },
            input_overlap={
                "datetime": emulator.n_input,
            },
            preload_batch=preload_batch,
        )

    def __len__(self) -> int:
        """
        Returns the number of sample forecasts in the dataset

        Returns:
            length (int): The length of the dataset.
        """
        return len(self.sample_generator)

    def __getitem__(self, idx) -> tuple[np.ndarray]:
        """
        Returns a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            X, y (np.ndarray): with inputs and targets
        """
        sample_input, sample_target, sample_forcing = self.get_xarrays(idx)

        X = self._stack(sample_input, sample_forcing)
        y = self._stack(sample_target)
        return X, y

    @staticmethod
    def _xstack(a: xr.DataArray, b: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Stack xarrays to form input tensors.

        Args:
            a (xarray.DataArray): First xarray.
            b (xarray.DataArray, optional): Second xarray.

        Returns:
            result (xarray.DataArray): Stacked xarray.
        """
        result = dataset_to_stacked(a)
        if b is not None:
            result = xr.concat(
                [result, dataset_to_stacked(b)],
                dim="channels",
            )
        result = result.transpose("batch", "lat", "lon", "channels")
        return result

    def _stack(self, a: xr.DataArray, b: Optional[xr.DataArray] = None) -> np.ndarray:
        """
        Stack xarrays to form input tensors.

        Args:
            a (xarray.DataArray): First xarray.
            b (xarray.DataArray): Second xarray.

        Returns:
            result (np.ndarray): Stacked tensor.
        """
        xresult = self._xstack(a, b)
        return xresult.values.squeeze()

    def _open_dataset(self) -> xr.Dataset:
        """
        Open, subsample, and rename variables in the dataset.

        Returns:
            xds (xarray.Dataset): Preprocessed xarray dataset.
        """
        xds = self.emulator.open_dataset()
        time = self.emulator.get_time(mode=self.mode)
        xds = self.emulator.subsample_dataset(xds, new_time=time)
        xds = xds.rename({
            "time": "datetime",
            "pfull": "level",
            "grid_yt": "lat",
            "grid_xt": "lon",
        })
        xds = xds.drop_vars(["cftime", "ftime"])
        return xds

    def _preprocess(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Preprocess the xarray dataset as necessary for GraphCast.

        Args:
            xds (xarray.Dataset): Input xarray dataset.

        Returns:
            xds (xarray.Dataset): Preprocessed xarray dataset.
        """
        xds["time"] = xds["datetime"] - xds["datetime"][0]
        xds = xds.swap_dims({"datetime": "time"}).reset_coords()
        xds = xds.set_coords(["datetime"])
        return xds

    def get_xds(self, idx: int) -> xr.Dataset:
        """
        Get a single dataset used to create inputs, targets, forcings for this sample index

        Args:
            idx (int): Index of the sample.

        Returns:
            xds (xarray.Dataset): Preprocessed xarray dataset.
        """
        sample = self.sample_generator[idx]
        sample = self._preprocess(sample)
        return sample

    def get_xarrays(self, idx: int) -> tuple:
        """
        Get input, target, and forcing xarrays.

        Args:
            idx (int): Index of the sample.

        Returns:
            xinput, xtarget, xforcing (xarray.DataArray): as from graphcast.data_utils.extract_inputs_targets_forcings
        """
        sample = self.get_xds(idx)

        xinput, xtarget, xforcing = extract_inputs_targets_forcings(
            sample,
            **self.emulator.extract_kwargs,
        )
        xinput = xinput.expand_dims({"batch": [idx]})
        xtarget = xtarget.expand_dims({"batch": [idx]})
        xforcing = xforcing.expand_dims({"batch": [idx]})
        return xinput, xtarget, xforcing

    def get_batch_of_xarrays(self, indices: list[int]) -> tuple:
        """
        Get batches of input, target, and forcing xarrays, convenience to mimic using the DataLoader.

        Args:
            indices (list[int]): List of sample indices.

        Returns:
            Tuple of input, target, and forcing xarrays.
        """
        xinputs = []
        xtargets = []
        xforcings = []
        for idx in indices:
            xi, xt, xf = self.get_xarrays(idx)
            xinputs.append(xi)
            xtargets.append(xt)
            xforcings.append(xf)

        xinputs = xr.concat(xinputs, dim="batch")
        xtargets = xr.concat(xtargets, dim="batch")
        xforcings = xr.concat(xforcings, dim="batch")
        return xinputs, xtargets, xforcings

    def get_xsample(self, idx: int) -> tuple[xr.DataArray]:
        """
        Same as __getitem__, except returns xarray.DataArrays

        Args:
            idx (int): Index of the sample.

        Returns:
            X, y (xarray.DataArray): inputs (forcings stacked) and targets
        """
        sample_input, sample_target, sample_forcing = self.get_xarrays(idx)

        X = self._xstack(sample_input, sample_forcing)
        y = self._xstack(sample_target)
        return X, y

    def _store_sample(self, idx: int, chunks: dict) -> None:
        x,y = self.get_xsample(idx)
        x = x.rename({"batch": "sample"})
        y = y.rename({"batch": "sample"})

        x = x.chunk(chunks)
        y = y.chunk(chunks)
        spatial_region = {k : slice(None, None) for k in x.dims if k != "sample"}
        region = {"sample": slice(idx, idx+1), **spatial_region}
        x.to_dataset(name="inputs").to_zarr(self.local_inputs_path, region=region)
        y.to_dataset(name="targets").to_zarr(self.local_targets_path, region=region)

    def _make_container(self, template: xr.Dataset, name: str, chunks: dict):

        if "batch" in template.dims:
            template = template.isel(batch=0, drop=True)

        xds = xr.Dataset()
        xds["sample"] = np.arange(len(self))
        for key in ["lat", "lon", "channels"]:
            xds[key] = template[key].copy()

        dims = ("sample",) + template.dims
        shape = (len(self),) + template.shape
        xds[name] = xr.DataArray(
            data=dask.array.zeros(
                shape=shape,
                chunks=tuple(chunks[k] for k in dims),
                dtype=template.dtype,
            ),
            dims=dims,
        )
        return xds


class LocalDataset(TorchDataset):

    def __init__(self, emulator, mode):
        self.emulator = emulator
        self.mode = mode
        self.inputs = xr.open_zarr(self.local_inputs_path)
        self.targets = xr.open_zarr(self.local_targets_path)

    def __len__(self):
        return len(self.inputs["sample"])

    def __getitem__(self, idx):
        x = self.inputs["inputs"].isel(sample=idx, drop=True).values
        y = self.targets["targets"].isel(sample=idx, drop=True).values
        return x, y

    @property
    def local_inputs_path(self) -> str:
        return join(self.emulator.local_store_path, self.mode, "inputs.zarr")

    @property
    def local_targets_path(self) -> str:
        return join(self.emulator.local_store_path, self.mode, "targets.zarr")


class DataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = collate_fn
        super().__init__(*args, **kwargs)


def collate_fn(batch):
    return tree_map(np.asarray, default_collate(batch))

class DataGenerator:
    """Data generator class"""

    def __init__(
        self,
        dataloader,
        num_workers: int = 1,
        max_queue_size: int = 1,
    ):
        # params for data queue
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        self.dataloader = dataloader

        # create a thread pool of workers for generating data
        if self.num_workers > 0:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            )
            self.futures = [
                self.executor.submit(self.generate) for i in range(self.num_workers)
            ]

    @property
    def drop_last(self):
        return self.dataloader.drop_last

    def __len__(self):
        return len(self.dataloader)

    def generate(self):
        """ Data generator function called by workers """
        while not self.stop_event.is_set():
            # get next batch
            x, y = next(iter(self.dataloader))

            # put data to queue
            self.data_queue.put((x,y))

    def get_data(self):
        """ Get data from queue """
        if self.num_workers > 0:
            return self.data_queue.get()
        else:
            return next(iter(self.dataloader))

    def stop(self):
        """ Stop generator at the end of training"""
        while not self.data_queue.empty():
            self.data_queue.get()
            self.data_queue.task_done()
        self.stop_event.set()
