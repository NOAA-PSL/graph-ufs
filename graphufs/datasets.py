"""
Implementations of Torch Dataset and DataLoader
"""
from os.path import join
from typing import Optional
import numpy as np
import xarray as xr
import dask.array
import pandas as pd
import logging
import os

from xbatcher import BatchGenerator

from graphcast.data_utils import extract_inputs_targets_forcings, extract_inputs_targets_forcings_coupled
from graphcast.model_utils import dataset_to_stacked

from .utils import get_channel_index
from .emulator import ReplayEmulator
from .coupledemulator import ReplayCoupledEmulator

class Dataset():
    """
    Dataset for Replay Data, in the style of pytorch, but does not require torch
    """
    def __init__(
        self,
        emulator: ReplayEmulator | ReplayCoupledEmulator,
        mode: str,
        preload_batch: bool = False,
        input_chunks: Optional[dict | None] = None,
        target_chunks: Optional[dict | None] = None,
    ):
        """
        Initializes the Dataset object.

        Args:
            emulator (ReplayEmulator): The emulator object.
            mode (str): "training", "validation", or "testing"
            preload_batch (bool, optional): If True, preload a sample before doing any processing, usually a good idea
            input_chunks, target_chunks (dict, optional): chunks used to store a local dataset
        """
        self.emulator = emulator
        self.mode = mode
        self.preload_batch = preload_batch
        self.input_chunks = input_chunks
        self.target_chunks = target_chunks
        xds, es_comp = self._open_dataset()
        self.es_comp = es_comp
        if hasattr(emulator, "delta_t_model") and hasattr(emulator, "delta_t_data"):
            self.dt_m_over_d = int(pd.Timedelta(emulator.delta_t_model)/pd.Timedelta(emulator.delta_t_data))
        else:
            self.dt_m_over_d = int(1.)
        input_dims = {
                "datetime": self.dt_m_over_d*emulator.n_forecast,
            }
        for key in ["lon", "lat", "level", "z_l"]:
            if key in xds.dims:
                input_dims[key] = xds.sizes[key]
        self.sample_generator = BatchGenerator(
            ds=xds,
            input_dims=input_dims,
            input_overlap={
                "datetime": int(self.dt_m_over_d*emulator.n_forecast-1),
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

    def __getitem__(self, idx) -> tuple[xr.DataArray]:
        """
        Returns a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            X, y (np.ndarray): with inputs and targets
        """
        if isinstance(idx, int):
            sample_input, sample_target, sample_forcing = self.get_xarrays(idx)
        else:
            sample_input, sample_target, sample_forcing = self.get_batch_of_xarrays(idx)
            
        x = self._stack(sample_input, sample_forcing)
        
        # add gaussian noise to inputs
        if self.emulator.add_gauss_noise:
            logging.info("Gaussian noise is being added to inputs")
            
            tmeta_inp = get_channel_index(sample_input)
            tmeta_forcing = get_channel_index(sample_forcing)
            tmeta_forcing_copy = {}
            for key, value in tmeta_forcing.items():
                tmeta_forcing_copy[key+len(tmeta_inp)] = value
            
            tmeta_x = {**tmeta_inp, **tmeta_forcing_copy}
                
            x = self._add_gaussian_noise(x, tmeta_x)

        y = self._stack(sample_target)
        return x, y
    
    def _add_gaussian_noise(self, x, tmeta_x):
        """
        Add Gaussian noise to xr.DataArray.
        
        Args:
            x (xr.DataArray): Input dataset to which noise would be added.
        
        Returns:
            xr.DataArray: Noisy array. 
        """
        stacked_norm_inputs_path = os.path.join(
                self.emulator.local_store_path,
                "stacked-normalization", "inputs",
                os.path.basename(self.emulator.norm_urls["atm"]["std"]),
        )

        if not os.path.isdir(stacked_norm_inputs_path):
            raise FileNotFoundError(
                f"Stacked normalization statistics directory not found: {stacked_norm_inputs_path}"
            )
        
        stddev_channels = xr.open_zarr(stacked_norm_inputs_path)
        stddev_x = stddev_channels["inputs"]

        # Build noise fraction mask
        omit_vars = [
            "landsea_mask", "land_static",
            "day_progress_cos", "day_progress_sin",
            "year_progress_cos", "year_progress_sin"
        ]

        fraction_da = xr.full_like(stddev_x, fill_value=self.emulator.noise_std_as_fraction)
        for key, meta in tmeta_x.items():
            if meta["varname"] in omit_vars:
                fraction_da.loc[dict(channels=key)] = 0.0

        rng = np.random.default_rng(self.emulator.gauss_noise_seed)
        scale = (stddev_x * fraction_da).broadcast_like(x)
        noise = xr.apply_ufunc(
            rng.normal,
            0.0,
            scale,
            input_core_dims=[[], x.dims],
            output_core_dims=[x.dims],
            vectorize=True,
            dask="allowed",
            output_dtypes=[x.dtype]
        )

        return x + noise

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

    @property
    def initial_times(self) -> list[np.datetime64]:
        """Returns dates of all initial conditions"""
        return [self.xds["datetime"].values[i + self.emulator.n_input - 1] for i in range(len(self))]


    @staticmethod
    def _stack(a: xr.DataArray, b: Optional[xr.DataArray] = None) -> xr.DataArray:
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

    def _open_dataset(self) -> xr.Dataset:
        """
        Open, subsample, and rename variables in the dataset.

        Returns:
            xds (xarray.Dataset): Preprocessed xarray dataset.
        """
        if isinstance(self.emulator, ReplayCoupledEmulator): 
            time = self.emulator.get_time(mode=self.mode)
            xds = self.emulator.open_and_subsample(all_new_time=time, mode=self.mode)
            es_comp = "coupled"
        elif isinstance(self.emulator, ReplayEmulator):
            xds = self.emulator.open_dataset()
            time = self.emulator.get_time(mode=self.mode)
            xds = self.emulator.subsample_dataset(xds, new_time=time)
            es_comp = "atm"
        xds = self.emulator.check_for_ints(xds)
        if "time" in xds.dims:
            xds = xds.rename({"time": "datetime"})
        if "grid_xt" in xds.dims:
            xds = xds.rename({"grid_xt": "lon"})
        if "grid_yt" in xds.dims:
            xds = xds.rename({"grid_yt": "lat"})
        if "pfull" in xds.dims:
            xds = xds.rename({"pfull": "level"})
        for x in ["cftime", "ftime"]:
            if x in xds.dims:
                xds = xds.drop_vars(x) 

        return xds, es_comp

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
        xds = xds.isel(time=slice(0, None, self.dt_m_over_d))
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
        if self.es_comp == "coupled":
            xinput, xtarget, xforcing = extract_inputs_targets_forcings_coupled(
                sample,
                drop_datetime=False,
                **self.emulator.extract_kwargs,
            )
        else:
            xinput, xtarget, xforcing = extract_inputs_targets_forcings(
                sample,
                drop_datetime=False,
                **self.emulator.extract_kwargs,
            )

        xinput = xinput.expand_dims({"batch": [idx]})
        xtarget = xtarget.expand_dims({"batch": [idx]})
        xforcing = xforcing.expand_dims({"batch": [idx]})
        return xinput, xtarget, xforcing

    def get_batch_of_xarrays(self, indices: list[int]) -> tuple:
        """
        Get batches of input, target, and forcing xarrays,
        convenience method to compare the "StackedGraphCast" and "GraphCast" implementations.

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

    def store_sample(self, idx: int) -> None:
        x,y = self[idx]

        x = x.load()
        y = y.load()
        x = x.expand_dims("batch").rename({"batch": "sample"})
        y = y.expand_dims("batch").rename({"batch": "sample"})

        x = x.chunk(self.input_chunks)
        y = y.chunk(self.target_chunks)
        spatial_region = {k : slice(None, None) for k in x.dims if k != "sample"}
        region = {"sample": slice(idx, idx+1), **spatial_region}
        for name, array, path in zip(
            ["inputs", "targets"],
            [x, y],
            [self.local_inputs_path, self.local_targets_path],
        ):
            if "batch" in array.coords:
                array = array.drop_vars("batch")
            array.to_dataset(name=name).to_zarr(
                path,
                region=region,
            )

    def get_container(self, template: xr.Dataset, name: str, chunks: dict):

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

    def store_containers(self):

        # get templates
        x, y = self[0]
        for name, template, chunks, path in zip(
            ["inputs", "targets"],
            [x, y],
            [self.input_chunks, self.target_chunks],
            [self.local_inputs_path, self.local_targets_path],
        ):
            xds = self.get_container(template=template, name=name, chunks=chunks)
            if "batch" in xds:
                xds = xds.drop_vars("batch")
            xds.to_zarr(path, compute=False, mode="w", consolidated=True)


class PackedDataset():
    """Similar in style to the Dataset class, and to PyTorch, but no torch dependency
    and relies on the dataset being local and ready to go for training.

    Note that this still returns xarray.DataArray with __getitem__, and this is so
    that BatchLoader can pull a full batch in a single dask/zarr call
    """

    def __init__(self, emulator, mode, missing_samples=None, meta_inputs=None, 
                 meta_targets=None, **kwargs):
        self.emulator = emulator
        self.mode = mode
        self.inputs = xr.open_zarr(self.local_inputs_path, **kwargs)
        self.targets = xr.open_zarr(self.local_targets_path, **kwargs)
        
        self.drop_missing(missing_samples)
        
        self.tmeta_inputs = meta_inputs
        self.tmeta_targets = meta_targets

    def __len__(self):
        return len(self.inputs["sample"])

    def __getitem__(self, idx):
        x = self.inputs["inputs"].isel(sample=idx, drop=True)
        y = self.targets["targets"].isel(sample=idx, drop=True)
        return x, y

    def drop_missing(self, missing_samples=None):
        if missing_samples is not None:
            if isinstance(missing_samples, int):
                missing_samples = [missing_samples]
            for idx in missing_samples:
                logging.info(f"PackedDataset: dropping missing sample at idx = {idx}")
                self.inputs = self.inputs.drop_sel(sample=idx)
                self.targets = self.targets.drop_sel(sample=idx)
    
    @property
    def local_inputs_path(self) -> str:
        return join(self.emulator.local_store_path, self.mode, "inputs.zarr")

    @property
    def local_targets_path(self) -> str:
        return join(self.emulator.local_store_path, self.mode, "targets.zarr")
