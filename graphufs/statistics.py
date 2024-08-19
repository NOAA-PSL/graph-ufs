import os
import logging
import numpy as np
import xarray as xr
from ufs2arco.timer import Timer

from graphcast import data_utils

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

    def __init__(
        self,
        path_in: str,
        path_out: str,
        start_date: str = None,
        end_date: str = None,
        time_skip: int = None,
        open_zarr_kwargs: dict = None,
        to_zarr_kwargs: dict = None,
        load_full_dataset: bool = False,
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
        """
        self.path_in = path_in
        self.path_out = path_out
        self.start_date = start_date
        self.end_date = end_date
        self.time_skip = time_skip
        self.open_zarr_kwargs = open_zarr_kwargs if open_zarr_kwargs is not None else dict()
        self.to_zarr_kwargs = to_zarr_kwargs if to_zarr_kwargs is not None else dict()
        self.load_full_dataset = load_full_dataset

        self.delta_t = f"{self.time_skip*3} hour" if self.time_skip is not None else "3 hour"

    def __call__(self, data_vars=None):
        """Processes the input dataset to compute normalization statistics.

        Args:
            data_vars (str or list of str, optional): Variables to select.
        """
        walltime = Timer()
        localtime = Timer()
        walltime.start()

        localtime.start("Setup")
        ds = xr.open_zarr(self.path_in, **self.open_zarr_kwargs)
        ds = add_derived_vars(ds)

        # select variables
        if data_vars is not None:
            if isinstance(data_vars, str):
                data_vars = [data_vars]
            ds = ds[data_vars]

        # subsample in time
        if "time" in ds.dims:
            ds = self.subsample_time(ds)
        localtime.stop()

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
        with xr.set_options(keep_attrs=True):
            result = xds.diff("time")
            dims = list(d for d in self.dims if d in result.dims)
            result = result.std(dims)

        for key in result.data_vars:
            result[key].attrs["description"] = f"standard deviation of temporal {self.delta_t} difference over lat, lon, time"
            result[key].attrs["stats_start_date"] = self._time2str(xds["time"][0])
            result[key].attrs["stats_end_date"] = self._time2str(xds["time"][-1])

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
        with xr.set_options(keep_attrs=True):
            dims = list(d for d in self.dims if d in xds.dims)
            result = xds.std(dims)

        for key in result.data_vars:
            result[key].attrs["description"] = f"standard deviation over {str(dims)}"
            if "time" in xds.dims:
                result[key].attrs["stats_start_date"] = self._time2str(xds["time"][0])
                result[key].attrs["stats_end_date"] = self._time2str(xds["time"][-1])

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
        with xr.set_options(keep_attrs=True):
            dims = list(d for d in self.dims if d in xds.dims)
            result = xds.mean(dims)

        for key in result.data_vars:
            result[key].attrs["description"] = f"average over {str(dims)}"
            if "time" in xds.dims:
                result[key].attrs["stats_start_date"] = self._time2str(xds["time"][0])
                result[key].attrs["stats_end_date"] = self._time2str(xds["time"][-1])

        this_path_out = os.path.join(
            self.path_out,
            "mean_by_level.zarr",
        )
        result.to_zarr(this_path_out, **self.to_zarr_kwargs)
        logging.info(f"Stored result: {this_path_out}")
        return result

    @staticmethod
    def _time2str(xval):
        """Converts an xarray numpy.datetime64 object to a string representation at hourly resolution.

        Args:
            xval (xarray.DataArray[numpy.datetime64]): Input datetime object.

        Returns:
            str: String representation of the datetime object.
        """
        return str(xval.values.astype("M8[h]"))

def add_derived_vars(xds):

    with xr.set_options(keep_attrs=True):
        xds = xds.rename({"time": "datetime", "grid_xt": "lon", "grid_yt": "lat", "pfull": "level"})
        data_utils.add_derived_vars(xds)
        xds = xds.rename({"datetime": "time", "lon": "grid_xt", "lat": "grid_yt", "level": "pfull"})

    return xds
