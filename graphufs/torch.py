"""
Implementations of Torch Dataset and DataLoader
"""
from typing import Optional
import numpy as np
import xarray as xr

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

    def __init__(
        self,
        emulator: ReplayEmulator,
        mode: str,
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
            preload_batch=False,
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


class DataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = collate_fn
        super().__init__(*args, **kwargs)


def collate_fn(batch):
    return tree_map(np.asarray, default_collate(batch))
