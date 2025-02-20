import os
import logging
from typing import Optional
import numpy as np
import xarray as xr
import pandas as pd
from ufs2arco.timer import Timer

from graphcast import data_utils, solar_radiation
from .diagnostics import prepare_diagnostic_functions

class StatisticsComputer:
    """Class for computing normalization statistics.

    Attributes:
        path_in (str): Path to the original dataset.
        path_out (str): Path to save normalization statistics.
        start_date (str): Start date to subsample data.
        end_date (str): End date to subsample data, inclusive.
        time_skip (int): Integer used to skip in time.
        open_zarr_kwargs (dict): Keyword arguments for opening zarr dataset.
        to_zarr_kwargs (dict): Keyword arguments for saving to zarr.
        load_full_dataset (bool): Whether to load the full dataset.
    """
    dims = ("time", "grid_yt", "grid_xt")

    @property
    def name(self):
        return str(type(self).__name__)

    def __init__(
        self,
        path_in: str,
        path_out: str,
        comp: str = "atm",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        time_skip: Optional[int] = None,
        open_zarr_kwargs: Optional[dict] = None,
        to_zarr_kwargs: Optional[dict] = None,
        load_full_dataset: Optional[bool] = False,
        transforms: Optional[dict] = None,
    ):
        """Initializes StatisticsComputer with specified attributes.

        Args:
            path_in (str): Path to the original dataset.
            path_out (str): Path to save normalization statistics.
            start_date (str, optional): Start date to subsample data.
            end_date (str, optional): End date to subsample data, inclusive.
            time_skip (int, optional): Integer used to skip in time.
            open_zarr_kwargs (dict, optional): Keyword arguments for opening zarr dataset.
            to_zarr_kwargs (dict, optional): Keyword arguments for saving to zarr.
            load_full_dataset (bool, optional): Whether to load the full dataset.
            transforms (dict, optional): with a mapping from {variable_name : operation} e.g. {"spfh": np.log}
        """
        self.path_in = path_in
        self.path_out = path_out
        self.comp = comp
        self.start_date = start_date
        self.end_date = end_date
        self.time_skip = time_skip
        self.open_zarr_kwargs = open_zarr_kwargs if open_zarr_kwargs is not None else dict()
        self.to_zarr_kwargs = to_zarr_kwargs if to_zarr_kwargs is not None else dict()
        self.load_full_dataset = load_full_dataset
        if self.comp.lower() == "atm".lower():
            self.delta_t = f"{self.time_skip*3} hour" if self.time_skip is not None else "3 hour"
            self.dims = ("time", "grid_yt", "grid_xt")
        elif self.comp.lower() == "ocean".lower():
            self.delta_t = f"{self.time_skip*6} hour" if self.time_skip is not None else "6 hour"
            self.dims = ("time", "lat", "lon")
        else:
            raise ValueError("component can only be atm or ocean")
        self.transforms = transforms

        self.delta_t = f"{self.time_skip*3} hour" if self.time_skip is not None else "3 hour"

    def __call__(self, data_vars=None, diagnostics=None, **tisr_kwargs):
        """Processes the input dataset to compute normalization statistics.

        Args:
            data_vars (str or list of str, optional): Variables to select.
        """
        walltime = Timer()
        localtime = Timer()
        walltime.start()

        localtime.start("Setup")
        ds = self.open_dataset(data_vars=data_vars, diagnostics=diagnostics, **tisr_kwargs)
        self._transforms_warning(list(ds.data_vars.keys()))
        localtime.stop()


        logging.info(f"{self.name}: computing statistics for {data_vars}")
        if diagnostics is not None:
             logging.info(f"{self.name}: computing statistics for diagnostics: {diagnostics}")

        # load if not 3D
        if self.load_full_dataset:
            localtime.start("Loading the whole dataset...")
            ds = ds.load();
            localtime.stop()

        # do the computations
        localtime.start("Computing mean")
        self.calc_mean_by_level(ds)
        localtime.stop()

        localtime.start("Computing stddev")
        self.calc_stddev_by_level(ds)
        localtime.stop()

        if "time" in ds.dims:
            localtime.start("Computing diff stddev")
            self.calc_diffs_stddev_by_level(ds)
            localtime.stop()

        walltime.stop("Total Walltime")

    def open_dataset(self, data_vars=None, diagnostics=None, **tisr_kwargs):
        xds = xr.open_zarr(self.path_in, **self.open_zarr_kwargs)

        # subsample in time
        if "time" in xds.dims:
            xds = self.subsample_time(xds)

        logging.info(f"{self.name}: Adding any derived variables")
        xds = add_derived_vars(
            xds,
            compute_tisr=data_utils.TISR in data_vars if data_vars is not None else False,
            **tisr_kwargs,
        )

        logging.info(f"{self.name}: Adding any transformed variables")
        xds = add_transformed_vars(
            xds,
            transforms=self.transforms,
        )

        if diagnostics is not None:
            if isinstance(diagnostics, str):
                diagnostics = [diagnostics]
            logging.info("{self.name}: computing diagnostics {diagnostics}")
            mappings = prepare_diagnostic_functions(diagnostics)
            for key, func in mappings["functions"].items():
                xds[key] = func(xds)
        else:
            diagnostics = []

        # select variables
        if data_vars is not None:
            if isinstance(data_vars, str):
                data_vars = [data_vars]
            sel_vars = data_vars + diagnostics
            xds = xds[sel_vars]
        return xds

    def subsample_time(self, xds):
        """Selects a specific time period and frequency from the input dataset.

        Args:
            xds (xarray.Dataset): Input dataset.

        Returns:
            xarray.Dataset: Subsampled dataset.
        """
        with xr.set_options(keep_attrs=True):
            rds = xds.sel(time=slice(self.start_date, self.end_date))
            rds = rds.isel(time=slice(None, None, self.time_skip))
        return rds

    def calc_diffs_stddev_by_level(self, xds):
        """Computes the standard deviation of differences by level and stores the result in a Zarr file.

        Args:
            xds (xarray.Dataset): Input dataset.

        Returns:
            xarray.Dataset: Result dataset with standard deviation of differences by vertical level.
        """

        result = xr.Dataset()
        time_varying_vars = [key for key in xds.data_vars if "time" in xds[key].dims]
        for key in time_varying_vars:
            result[key] = self._local_op(
                xds[key],
                opstr="diffs_stddev",
                description=f"standard deviation of temporal {self.delta_t} difference over ",
            )

        result = self._add_coords(result, xds)

        this_path_out = os.path.join(
            self.path_out,
            "diffs_stddev_by_level.zarr",
        )
        result.to_zarr(this_path_out, **self.to_zarr_kwargs)
        logging.info(f"Stored result: {this_path_out}")
        return result

    def calc_stddev_by_level(self, xds):
        """Computes the standard deviation by level and stores the result in a Zarr file.

        Args:
            xds (xarray.Dataset): Input dataset.

        Returns:
            xarray.Dataset: Result dataset with standard deviation by vertical level.
        """

        result = xr.Dataset()
        for key in xds.data_vars:
            result[key] = self._local_op(
                xds[key],
                opstr="stddev",
                description=f"standard deviation over ",
            )

        result = self._add_coords(result, xds)

        this_path_out = os.path.join(
            self.path_out,
            "stddev_by_level.zarr",
        )
        result.to_zarr(this_path_out, **self.to_zarr_kwargs)
        logging.info(f"Stored result: {this_path_out}")
        return result

    def calc_mean_by_level(self, xds):
        """Computes the mean by level and stores the result in a Zarr file.

        Args:
            xds (xarray.Dataset): Input dataset.

        Returns:
            xarray.Dataset: Result dataset with mean by vertical level.
        """
        result = xr.Dataset()
        for key in xds.data_vars:
            result[key] = self._local_op(
                xds[key],
                opstr="mean",
                description=f"average over ",
            )

        result = self._add_coords(result, xds)

        this_path_out = os.path.join(
            self.path_out,
            "mean_by_level.zarr",
        )
        result.to_zarr(this_path_out, **self.to_zarr_kwargs)
        logging.info(f"Stored result: {this_path_out}")
        return result

    def _local_op(self, xda, opstr, description):

        # get appropriate dims, e.g. maybe not time varying
        dims = list(d for d in self.dims if d in xda.dims)

        with xr.set_options(keep_attrs=True):
            if opstr == "mean":
                result = xda.mean(dims)
            elif opstr == "stddev":
                result = xda.std(dims)
            elif opstr == "diffs_stddev":
                result = xda.diff("time").std(dims)

        result.attrs["description"] = description+str(dims)
        if "time" in xda.dims:
            result.attrs["stats_start_date"] = self._time2str(xda["time"][0])
            result.attrs["stats_end_date"] = self._time2str(xda["time"][-1])
        return result

    def _add_coords(self, result, xds):
        for key in xds.coords:
            if key not in result.coords:
                result[key] = xds[key]
                result = result.set_coords(key)
        for key in ["cftime", "ftime"]:
            if key in result:
                logging.info(f"{self.name}: dropping coordinate {key}")
                result = result.drop_vars(key)
        return result

    def _transforms_warning(self, data_vars):
        if self.transforms is not None:
            for key, mapping in self.transforms.items():
                transformed_key = f"{mapping.__name__}_{key}"
                if key not in data_vars:
                    logging.warn(f"{self.name}: '{transformed_key}' listed in transforms, but '{key}' stats not being stored. Make sure this gets computed.")
                if transformed_key not in data_vars:
                    logging.warn(f"{self.name}: '{transformed_key}' listed in transforms, but '{transformed_key}' stats not being stored. Make sure this gets computed.")

    @staticmethod
    def _time2str(xval):
        """Converts an xarray numpy.datetime64 object to a string representation at hourly resolution.

        Args:
            xval (xarray.DataArray[numpy.datetime64]): Input datetime object.

        Returns:
            str: String representation of the datetime object.
        """
        return str(xval.values.astype("M8[h]"))

def add_derived_vars(
    xds: xr.Dataset,
    component: str = "atm",
    compute_tisr: Optional[bool]=False,
    **tisr_kwargs,
) -> xr.Dataset:
    """Add derived variables to the dataset, including the clock variables and TISR from graphcast

    Note that here we store a separate variable for transformed variables because we may want to store
    both the original stats and transformed stats for comparison at some point.

    Args:
        xds (xr.Dataset): with original data
        compute_tisr (bool, optional): if true, add derived toa_incident_solar_radiation from graphcast code
        tisr_kwargs (optional args): passed to TISR computation, e.g. integration period

    Returns:
        xds (xr.Dataset): with added variables
    """
    with xr.set_options(keep_attrs=True):
        if component.lower() == "atm".lower():
            xds = xds.rename({"time": "datetime", "grid_xt": "lon", "grid_yt": "lat", "pfull": "level"})
            data_utils.add_derived_vars(xds)
            if compute_tisr:
                logging.info(f"Computing {data_utils.TISR}")
                xds[data_utils.TISR] = solar_radiation.get_toa_incident_solar_radiation_for_xarray(
                    xds,
                    **tisr_kwargs
                )
            xds = xds.rename({"datetime": "time", "lon": "grid_xt", "lat": "grid_yt", "level": "pfull"})

        elif component.lower() == "ocean".lower():
            xds = xds.rename({"time": "datetime"})
            data_utils.add_derived_vars(xds)
            xds = xds.rename({"datetime": "time"})

    return xds

def add_transformed_vars(
    xds: xr.Dataset,
    transforms: Optional[dict]=None,
) -> xr.Dataset:
    """Add any transformed variables, like e.g. log(spfh)
    Args:
        xds (xr.Dataset): with original data
        transforms (dict, optional): with a mapping from {variable_name : operation} e.g. {"spfh": np.log}

    Returns:
        xds (xr.Dataset): with transformed variables added
    """

    with xr.set_options(keep_attrs=True):
        if transforms is not None:
            for key, mapping in transforms.items():
                if key in xds:
                    logging.info(f"statistics.add_transformed_vars: transforming {key} -> {mapping.__name__}({key})")
                    transformed_key = f"{mapping.__name__}_{key}"
                    with xr.set_options(keep_attrs=True):
                        xds[transformed_key] = mapping(xds[key])
                    xds[transformed_key].attrs = xds[key].attrs.copy()
                    xds[transformed_key].attrs["long_name"] = f"{mapping.__name__} of {xds[key].attrs['long_name']}"
                    xds[transformed_key].attrs["transformation"] = f"this variable shows {mapping.__name__}({key})"
                    xds[transformed_key].attrs["units"] = ""

    return xds
