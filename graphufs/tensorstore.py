"""
Same as the LocalDataset and BatchLoader, but using xarray_tensorstore
"""
import numpy as np
import xarray as xr
import logging
import xarray_tensorstore
import os

from .datasets import PackedDataset as BaseDataset, Dataset as RawDataset
from .utils import get_channel_index, search_nested_dict
from .batchloader import BatchLoader as BaseBatchLoader, MPIBatchLoader as BaseMPIBatchLoader
from .mpi import MPITopology, _has_mpi

class PackedDataset(BaseDataset):
    """Same as the other PackedDatset, but use xarray_tensorstore instead of xarray/dask/zarr
    """

    def __init__(self, emulator, mode, missing_samples=None, meta_inputs=None, meta_targets=None):
        self.emulator = emulator
        self.mode = mode
        self.inputs = xarray_tensorstore.open_zarr(self.local_inputs_path)
        self.targets = xarray_tensorstore.open_zarr(self.local_targets_path)

        self.drop_missing(missing_samples)
        
        self.tmeta_inputs = meta_inputs if meta_inputs is not None else {}
        self.tmeta_targets = meta_targets if meta_targets is not None else {}

        self._init_gaussian_noise()
        self._init_mask()

    def __getitem__(self, idx):
        
        is_single = isinstance(idx, int)
        idx = np.atleast_1d(idx)
        
        x = xarray_tensorstore.read(self.inputs["inputs"].isel(sample=idx))
        y = xarray_tensorstore.read(self.targets["targets"].isel(sample=idx))

        # add gaussian noise to inputs
        if self.emulator.add_gauss_noise and self.mode.lower() == "training":
            x = self._add_gaussian_noise(x)

        if is_single:
            return x[0], y[0]
        
        return x, y

    def _init_gaussian_noise(self):
        if not self.emulator.add_gauss_noise:
            self.stddev_x = None
            self.fraction_da = None
            return

        logging.info("Initializing Gaussian noise...")

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
        self.stddev_x = stddev_channels["inputs"].load()

        # Build noise fraction mask
        omit_vars = [
            "landsea_mask", "land_static",
            "day_progress_cos", "day_progress_sin",
            "year_progress_cos", "year_progress_sin"
        ]

        self.fraction_da = xr.full_like(self.stddev_x, fill_value=self.emulator.noise_std_as_fraction)
        for key, meta in self.tmeta_inputs.items():
            if meta["varname"] in omit_vars:
                self.fraction_da.loc[dict(channels=key)] = 0.0

        self.fraction_da.load()
        logging.info("Gaussian noise fields preloaded.")

    def _add_gaussian_noise(self, x):
        """
        Add Gaussian noise to xr.DataArray.

        Args:
            x (xr.DataArray): Input dataset to which noise would be added.

        Returns:
            xr.DataArray: Noisy array.
        """
        rng = np.random.default_rng(self.emulator.gauss_noise_seed)
        
        scale = (self.stddev_x * self.fraction_da).broadcast_like(x)
        
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
        
        noise = noise * self.mask
       
        return x + noise
    
    def _init_mask(self):
        """
        Get masks for ocean, land, ice channels

        Returns:
            xr.DataArray: mask
        """
        # identify ocean, land, and ice variables in x
        all_variables_ocn = set(self.emulator.ocn_input_variables + self.emulator.ocn_forcing_variables)
        all_variables_ice = set(self.emulator.ice_input_variables + self.emulator.ice_forcing_variables)
        all_variables_land = set(self.emulator.land_input_variables + self.emulator.land_forcing_variables)

        # Get static masks
        _, landsea_mask_dict = search_nested_dict(self.tmeta_inputs, "varname", "landsea_mask")
        cidx_land_static, _ = search_nested_dict(self.tmeta_inputs, "varname", "land_static")
        land_static = self.inputs.inputs.isel(sample=0, channels=cidx_land_static).squeeze()

        # Create base mask = ones
        self.mask = xr.DataArray(
                np.ones((self.inputs.inputs.sizes["channels"], self.inputs.inputs.sizes["lat"], 
                         self.inputs.inputs.sizes["lon"])),
                dims=["channels", "lat", "lon"],
                coords={"channels": self.inputs.coords["channels"].values, 
                        "lat": self.inputs.coords["lat"].values,
                        "lon": self.inputs.coords["lon"].values},
        )
        
        ocean_2d_vars = ["ssh", "lw", "sw"]
        land_2d_vars = ["soilm", "snowc_ave", "veg"]
        all_2d_vars = ocean_2d_vars + land_2d_vars

        # Loop through channels and set masks
        for cidx, meta in self.tmeta_inputs.items():
            var = meta["varname"].lower()

            # Note: use land_static to mask ssh and other surface ocean variables
            if var in all_2d_vars or var.startswith("ice"):
                mask_ = xr.where(land_static > 0,
                                 0 if var in ocean_2d_vars or var.startswith("ice") else 1,
                                 1 if var in ocean_2d_vars or var.startswith("ice") else 0)
                self.mask.loc[dict(channels=cidx)] = mask_
                continue

            if "z_l" in meta:
                ch_vert, _ = search_nested_dict(landsea_mask_dict, "z_l", meta["z_l"])
                layer_mask = self.inputs.inputs.isel(sample=0, channels=ch_vert).squeeze()
                self.mask.loc[dict(channels=cidx)] = xr.where(layer_mask > 0, 0, 1)
        
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


class MPIBatchLoader(BaseMPIBatchLoader):

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
